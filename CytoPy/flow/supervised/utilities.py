from ...data.fcs_experiments import FCSExperiment
from ..feedback import progress_bar
from ..gating.actions import Gating
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
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
    sample = experiment.get_sample(sample_id)
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


def build_labelled_dataset(experiment: FCSExperiment,
                           labels: dict,
                           target_population: str,
                           sample_n: int or float,
                           transform: str or None = 'logicle',
                           verbose: bool = False) -> pd.DataFrame:
    """
    Generate a DataFrame of concatenated single cell data where each row represents a single cell. Ech cell is labelled
    and this value is contained within a column named "label". The label is specified by the "label" argument. The
    label argument should be a dictionary where the key corresponds to the label and the values corresponds to
    a list of sample IDs to fetch and associate to that label.

    Parameters
    ----------
    experiment: FCSExperiment
        Experiment to fetch samples from. The DataFrame can only contain single cell data
        from the same experiment
    labels: dict
        Dictionary where each key is a label and its corresponding value is a list of associated sample IDs
    target_population: str
        Name of the population to fetch from each sample
    sample_n: int or float
        Number of cells to fetch from each sample
    transform: str (optional) (default='logicle')
        Transformation to be applied to data before returning the DataFrame. Set to None to return untransformed data
    verbose: bool (default=False)
        Set to True to print progress to screen

    Returns
    -------
    Pandas DataFrame
        Concatenated dataframe of single cell data where each row is labelled accordingly
    """
    vprint = print if verbose else lambda *a, **k: None
    data = pd.DataFrame()
    for label, sids in labels.items():
        if verbose:
            vprint(f"------- Fetching data for label: {label} -------")
        for s in progress_bar(sids, verbose=verbose):
            g = Gating(experiment=experiment, sample_id=s, include_controls=False)
            d = g.get_population_df(population_name=target_population,
                                    transform=transform is not None,
                                    transform_method=transform,
                                    transform_features="all").copy()
            if type(sample_n) == int:
                d = d.sample(sample_n)
            else:
                d = d.sample(frac=sample_n)
            d["sid"] = s
            d["label"] = label
            data = pd.concat([data, d])
    return data


