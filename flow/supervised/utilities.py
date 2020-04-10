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
    """
    Generate a list of features from a list of ChannelMap objects (see data.panel.ChannelMap). By default
    the ChannelMap marker value is used, but if missing will use channel value instead.

    Parameters
    ----------
    channel_mappings: list
        List of ChannelMap objects

    Returns
    -------
    List
    """
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
    Generate a list of features from either an experiment or a sample belonging to that experiment. If a value
    for sample_id is given, then features are extracted from this sample alone, if a value is not given for sample_id
    then the features are derived from the panel associated to the experiment.

    Parameters
    ----------
    experiment: FCSExperiment
        Experiment to extract features from
    sample_id: str, optional
        Sample to retrieve features from
    Returns
    -------
    List
    """
    if sample_id is None:
        return _genetate_feature_list(experiment.panel.mappings)
    assert sample_id in experiment.list_samples(), f'{sample_id} not found in experiment {experiment.experiment_id}'
    sample = experiment.pull_sample(sample_id)
    return _genetate_feature_list(sample.files[0].channel_mappings)


def find_common_features(experiment: FCSExperiment,
                         samples: list or None = None):
    """
    Generate a list of common features present in all given samples of an experiment. By 'feature' we mean
    a variable measured for a particular sample e.g. CD4 or FSC-A (forward scatter)

    Parameters
    ----------
    experiment: FCSExperiment
        Experiment to extract features from
    samples: list, optional
        List of samples to get common features of. If None, will search all samples in experiment.

    Returns
    -------
    List
    """
    if samples is None:
        samples = experiment.list_samples()
    assert all([s in experiment.list_samples() for s in samples]), \
        'One or more samples specified do not belong to experiment'
    features = [_get_features(experiment, sample_id=s) for s in samples]
    common_features = set(features[0])
    for f in features[1:]:
        common_features.intersection_update(f)
    return list(common_features)


def scaler(data: np.array,
           scale_method: str,
           **kwargs) -> np.array and callable:
    """
    Wrapper for Sklearn transformation methods

    Parameters
    -----------
    data: Numpy.array
        data to transform; expects a numpy array
    scale_method: str
        type of transformation to perform
    kwargs:
        additional keyword arguments that can be passed to sklearn function

    Returns
    --------
    (Numpy.array, callable)
        transformed data and sklearn transformer object
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


def predict_class(y_probs: np.array,
                  threshold: float or None = None):
    """
    Returns the predicted class given the probabilities of each class. If threshold = None, the class with
    the highest probability is returned for each value in y, otherwise assumed to be multi-label prediction
    and converts output to binarised-encoded multi-label output using the given threshold.

    Parameters
    -----------
    y_probs: Numpy.array
        List of probabilities for predicted labels
    threshold: float, optional
        Threshold for positivity

    Returns
    --------
    List
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



