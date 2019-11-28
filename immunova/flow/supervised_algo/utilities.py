from immunova.data.fcs_experiments import FCSExperiment
from immunova.flow.gating.transforms import apply_transform
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from functools import partial
import numpy as np


def random_oversampling(x, y):
    ros = RandomOverSampler(random_state=42)
    return ros.fit_resample(x, y)


def __pull_features(sid, experiment):
    d = experiment.pull_sample_data(sample_id=sid, include_controls=False)
    d = [x for x in d if x['typ'] == 'complete'][0]['data']
    return list(d.columns)


def find_common_features(experiment: FCSExperiment, exclude: list or None = None):
    pool = Pool(cpu_count())
    if exclude is not None:
        samples = [f for f in experiment.list_samples() if f not in exclude]
    else:
        samples = experiment.list_samples()
    pull = partial(__pull_features, experiment=experiment)
    all_features = pool.map(pull, samples)
    common_features = set(all_features[0])
    for f in all_features[1:]:
        common_features.intersection_update(f)
    return list(common_features)


def standard_scale(data: np.array):
    data = data.copy()
    preprocessor = StandardScaler().fit(data)
    data = preprocessor.transform(data)
    return data, preprocessor


def norm_scale(data: np.array):
    data = data.copy()
    preprocessor = MinMaxScaler().fit(data)
    data = preprocessor.transform(data)
    return data, preprocessor


def predict_class(y_probs, threshold):
    """
    Returns the predicted class given the probabilities of each class. If threshold = None, the class with
    the highest probability is returned for each value in y, otherwise assumed to be multi-class prediction
    and converts output to one-hot-encoded multi-label output using the given threshold.
    :param y_probs:
    :param threshold:
    :return:
    """
    def convert_ml(y):
        if y > threshold:
            return 1
        return 0
    if threshold is not None:
        y_hat = list()
        for x in y_probs:
            y_hat.append(list(map(lambda i: convert_ml(i), x)))
        return y_hat
    return list(map(lambda j: np.argmax(j), y_probs))


def calculate_ref_sample_fast(experiment, exclude_samples, sample_n):
    # Calculate common features
    features = find_common_features(experiment)
    # List samples
    all_samples = [x for x in experiment.list_samples() if x not in exclude_samples]
    # Fetch data
    pool = Pool(cpu_count())
    f = partial(pull_data_hashtable, experiment=experiment, features=features, sample_n=sample_n)
    all_data_ = pool.map(f, all_samples)
    # Calculate covar for each
    all_data = dict()
    for d in all_data_:
        all_data.update(d)
    del all_data_
    all_data = {k: np.cov(v, rowvar=False) for k, v in all_data.items()}
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


def pull_data_hashtable(sid, experiment, features, sample_n):
    return {sid: pull_data(sid, experiment, features, sample_n=sample_n)}


def pull_data(sid, experiment, features, sample_n=None):
    d = experiment.pull_sample_data(sample_id=sid, data_type='raw', include_controls=False,
                                    sample_size=sample_n)
    if d is None:
        return None
    d = [x for x in d if x['typ'] == 'complete'][0]['data'][features]
    d = d[[x for x in d.columns if x != 'Time']]
    return apply_transform(d, transform_method='log_transform')


def calculate_reference_sample(experiment: FCSExperiment, exclude_samples: list, sample_n=1000) -> str:
    """
    Given an FCS Experiment with multiple FCS files, calculate the optimal reference file.

    This is performed as described in Li et al paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860171/) on
    DeepCyTOF: for every 2 samples i, j compute the Frobenius norm of the difference between their covariance matrics
    and then select the sample with the smallest average distance to all other samples.
    :param experiment: FCSExperiment with multiple FCS samples
    :return: sample ID for optimal reference sample
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
