from ...data.fcs_experiments import FCSExperiment
from ..transforms import apply_transform
from .utilities import find_common_features
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np


def calculate_ref_sample_fast(experiment: FCSExperiment,
                              exclude_samples: list or None = None,
                              sample_n: int = 1000,
                              verbose: bool = True):
    """
    Given an FCS Experiment with multiple FCS files, calculate the optimal reference file.

    This is performed as described in Li et al paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860171/) on
    DeepCyTOF: for every 2 samples i, j compute the Frobenius norm of the difference between their covariance matrics
    and then select the sample with the smallest average distance to all other samples.

    This is an optimised version of supervised.ref.calculate_red_sample that leverages the multi-processing library
    to speed up operations

    Parameters
    ----------
    experiment: FCSExperiment
        Experiment to find reference sample for
    exclude_samples: list, optional
        If given, any samples in list will be excluded
    sample_n: int, (default=1000)
        Data is downsampled prior to running algorithm, this specifies how many observations to sample from each
    verbose: bool, (default=True)
        Feedback
    Returns
    -------
    str
        Sample ID of reference sample
    """
    vprint = print if verbose else lambda *a, **k: None
    if exclude_samples is None:
        exclude_samples = []
    vprint('-------- Calculating Reference Sample (Multi-processing) --------')
    # Calculate common features
    vprint('...match feature space between samples')
    features = find_common_features(experiment)
    # List samples
    all_samples = [x for x in experiment.list_samples() if x not in exclude_samples]
    vprint('...pulling data')
    # Fetch data
    pool = Pool(cpu_count())
    f = partial(pull_data_hashtable, experiment=experiment, features=features, sample_n=sample_n)
    all_data_ = pool.map(f, all_samples)
    vprint('...calculate covariance matrix for each sample')
    # Calculate covar for each
    all_data = dict()
    for d in all_data_:
        all_data.update(d)
    del all_data_
    all_data = {k: np.cov(v, rowvar=False) for k, v in all_data.items()}
    vprint('...search for sample with smallest average euclidean distance to all other samples')
    # Make comparisons
    n = len(all_samples)
    norms = np.zeros(shape=[n, n])
    ref_ind = None
    for i in range(0, n):
        cov_i = all_data[all_samples[i]]
        for j in range(0, n):
            cov_j = all_data[all_samples[j]]
            cov_diff = cov_i - cov_j
            norms[i, j] = np.linalg.norm(cov_diff, ord='fro')
            norms[j, i] = norms[i, j]
            avg = np.mean(norms, axis=1)
            ref_ind = np.argmin(avg)
    pool.close()
    pool.join()
    return all_samples[int(ref_ind)]


def pull_data_hashtable(sid: str,
                        experiment: FCSExperiment,
                        features: list,
                        sample_n: int):
    """
    Wrapper for pull_data that returns sample data as a dictionary where the key is the sample ID
    and the value is the sample DataFrame

    Parameters
    ----------
    sid: str
        Sample ID for data retrieval
    experiment: FCSExperiment
        Experiment for data retrieval
    features: list
        List of features to retrieve
    sample_n: int
        Total number of events to sample

    Returns
    -------
    dict
        {sample ID: Pandas.DataFrame}
    """
    return {sid: pull_data(sid, experiment, features, sample_n=sample_n)}


def pull_data(sid: str,
              experiment: FCSExperiment,
              features: list,
              sample_n: int or None = None,
              transform: str or None = 'logicle'):
    """
    Given a sample ID and experiment that the sample belongs too, fetch a DataFrame of associated single cell
    data including the given features and (if provided) down-sample the data to the value of sample_n.

    Parameters
    ----------
    sid: str
        Sample ID
    experiment: FCSExperiment
        Experiment to retrieve data from
    features: list
        List of valid features (variables measured e.g. CD4 or FSC-A)
    sample_n: int, optional
        If given, data is down-sampled to the given value
    transform: str, optional, (default='logicle')
        If given, data is transformed according to given method, see flow.transforms
    Returns
    -------
    Pandas.DataFrame
    """
    d = experiment.pull_sample_data(sample_id=sid, include_controls=False,
                                    sample_size=sample_n)
    if d is None:
        return None
    d = [x for x in d if x['typ'] == 'complete'][0]['data'][features]
    d = d[[x for x in d.columns if x != 'Time']]
    if transform is not None:
        return apply_transform(d, transform_method=transform)
    return d


def calculate_reference_sample(experiment: FCSExperiment,
                               exclude_samples: list) -> str:
    """
    Given an FCS Experiment with multiple FCS files, calculate the optimal reference file.

    This is performed as described in Li et al paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860171/) on
    DeepCyTOF: for every 2 samples i, j compute the Frobenius norm of the difference between their covariance matrics
    and then select the sample with the smallest average distance to all other samples.

    This is an optimised version of supervised.ref.calculate_red_sample that leverages the multi-processing library
    to speed up operations

    Parameters
    ----------
    experiment: FCSExperiment
        Experiment to find reference sample for
    exclude_samples: list, optional
        If given, any samples in list will be excluded
    Returns
    -------
    str
        Sample ID of reference sample
    """
    features = find_common_features(experiment)
    samples = experiment.list_samples()
    samples = [x for x in samples if x not in exclude_samples]
    if len(samples) == 0:
        raise ValueError('Error: no samples associated to given FCSExperiment')
    n = len(samples)
    norms = np.zeros(shape=[n, n])
    ref_ind = None
    for i, si in enumerate(samples):
        print(f'Running comparisons for {si}')
        data_i = pull_data(si, experiment, features)
        if data_i is None:
            print(f'Error: failed to fetch data for {si}. Skipping.')
            continue
        cov_i = np.cov(data_i, rowvar=False)
        for j, sj in enumerate(samples):
            data_j = pull_data(sj, experiment, features)
            if data_j is None:
                print(f'Error: failed to fetch data for {sj}. Skipping.')
                continue
            cov_j = np.cov(data_j, rowvar=False)
            cov_diff = cov_i - cov_j
            norms[i, j] = np.linalg.norm(cov_diff, ord='fro')
            norms[j, i] = norms[i, j]
            avg = np.mean(norms, axis=1)
            ref_ind = np.argmin(avg)
    if ref_ind is not None:
        return samples[int(ref_ind)]
    else:
        raise ValueError('Error: unable to calculate sample with minimum average distance. You must choose'
                         ' manually.')
