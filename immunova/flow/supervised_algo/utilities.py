from immunova.data.fcs_experiments import FCSExperiment
from immunova.flow.gating.transforms import apply_transform
import numpy as np


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


def calculate_reference_sample(experiment: FCSExperiment, exclude_samples: list) -> str:
    """
    Given an FCS Experiment with multiple FCS files, calculate the optimal reference file.

    This is performed as described in Li et al paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860171/) on
    DeepCyTOF: for every 2 samples i, j compute the Frobenius norm of the difference between their covariance matrics
    and then select the sample with the smallest average distance to all other samples.
    :param experiment: FCSExperiment with multiple FCS samples
    :return: sample ID for optimal reference sample
    """
    def pull_data(sid):
        d = experiment.pull_sample_data(sample_id=sid, data_type='raw')
        if d is None:
            return None
        d = [x for x in d if x['typ'] == 'complete'][0]['data']
        d = d[[x for x in d.columns if x != 'Time']]
        return apply_transform(d, transform_method='log_transform')

    samples = experiment.list_samples()
    if len(samples) == 0:
        raise ValueError('Error: no samples associated to given FCSExperiment')
    n = len(samples)
    norms = np.zeros(shape=[n, n])
    ref_ind = None

    print('Running comparisons....')
    for i in range(0, n):
        print(f'----------------------- {samples[i]} -----------------------')
        if samples[i] in exclude_samples:
            print(f'Skipping {samples[i]}; found in exclude list')
            continue

        print('Estimating covariance matrix')
        data_i = pull_data(samples[i])
        if data_i is None:
            print(f'Error: failed to fetch data for {samples[i]}. Skipping.')
            continue
        cov_i = np.cov(data_i, rowvar=False)

        print('Make comparisons to other samples...')
        for j in range(0, n):
            if samples[j] in exclude_samples:
                continue
            print(f'Compare to {samples[j]}..')
            data_j = pull_data(samples[j])
            if data_i is None:
                print(f'Error: failed to fetch data for {samples[i]}. Skipping.')
                continue
            cov_j = np.cov(data_j, rowvar=False)

            cov_diff = cov_i - cov_j
            norms[i, j] = np.linalg.norm(cov_diff, ord='fro')
            norms[j, i] = norms[i, j]
            avg = np.mean(norms, axis=1)
            ref_ind = np.argmin(avg)
    if ref_ind is not None:
        print('Complete!')
        return samples[ref_ind[0]]
    else:
        raise ValueError('Error: unable to calculate sample with minimum average distance. You must choose'
                         ' manually.')
