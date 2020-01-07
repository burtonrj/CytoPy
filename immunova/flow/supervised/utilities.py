from immunova.data.fcs_experiments import FCSExperiment
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
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



