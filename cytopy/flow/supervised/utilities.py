from ...data.fcs_experiments import FCSExperiment
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from imblearn.over_sampling import RandomOverSampler
from functools import partial
import numpy as np


def random_oversampling(x: np.array,
                        y: np.array):
    """
    Wrapper for imblearn.over_sampling.RandomOverSampler

    Parameters
    ----------
    x: Numpy.array
        Feature space
    y: Numpy.array
        Labels

    Returns
    -------
    (Numpy.array, Numpy.array)
        Re-sampled data with unrepresented classes randomly oversampled
    """
    ros = RandomOverSampler(random_state=42)
    return ros.fit_resample(x, y)


def _genetate_feature_list(channel_mappings: list):
    features = list()
    for cm in channel_mappings:
        if cm.marker:
            features.append(cm.marker)
        else:
            features.append(cm.channel)
    return features


def _get_features(experiment: FCSExperiment,
                  sample_id: str or None = None):
    """
    For a given sample in a given experiment, return the list of
    Parameters
    ----------
    sid
    experiment

    Returns
    -------

    """
    if sample_id is None:
        return _genetate_feature_list(experiment.panel.mappings)
    assert sample_id in experiment.list_samples(), f'{sample_id} not found in experiment {experiment.experiment_id}'
    sample = experiment.pull_sample(sample_id)
    return _genetate_feature_list(sample.files[0].channel_mappings)


def find_common_features(experiment: FCSExperiment, samples: list or None = None):
    if samples is None:
        samples = experiment.list_samples()
    assert all([s in experiment.list_samples() for s in samples]), \
        'One or more samples specified do not belong to experiment'
    features = [_get_features(experiment, sample_id=s) for s in samples]
    common_features = set(features[0])
    for f in features[1:]:
        common_features.intersection_update(f)
    return list(common_features)


def scaler(data: np.array, scale_method: str, **kwargs) -> np.array and callable:
    """
    Wrapper for Sklearn transformation methods
    :param data: data to transform; expects a numpy array
    :param method: type of transformation to perform
    :param kwargs: additional keyword arguments that can be passed to sklearn function
    :return: transformed data and sklearn transformer object
    """
    if scale_method == 'standard':
        preprocessor = StandardScaler(**kwargs).fit(data)
    elif scale_method == 'norm':
        preprocessor = MinMaxScaler(**kwargs).fit(data)
    elif scale_method == 'power':
        preprocessor = PowerTransformer(**kwargs).fit(data)
    elif scale_method == 'robust':
        preprocessor = RobustScaler(**kwargs).fit(data)
    else:
        raise ValueError('Method should be one of the following: [standard, norm, power, robust]')
    data = preprocessor.transform(data)
    return data, preprocessor


def predict_class(y_probs: np.array, threshold: float or None = None):
    """
    Returns the predicted class given the probabilities of each class. If threshold = None, the class with
    the highest probability is returned for each value in y, otherwise assumed to be multi-label prediction
    and converts output to binarised-encoded multi-label output using the given threshold.
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



