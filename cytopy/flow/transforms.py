from flowutilspd.transforms import logicle, hyperlog, log_transform, asinh
from .supervised.utilities import scaler
import pandas as pd


class TransformError(Exception):
    pass


def percentile_rank_transform(data: pd.DataFrame, features_to_transform: list) -> pd.DataFrame:
    """
    Calculate percentile rank transform of data-frame. Each event is ranked as the average according to the
    column, then divided by the total number of events and multiplied by 100 to give the percentile.
    :param data: pandas dataframe of events
    :param features_to_transform: features to perform transformation on
    :return: Transformed dataframe
    """
    data = data.copy()
    transform = data[features_to_transform].rank(axis=0, method='average')
    transform = (transform / transform.shape[0]) * 100
    data[features_to_transform] = transform
    return data


def apply_transform(data: pd.DataFrame, features_to_transform: list or str = 'all',
                    transform_method: str = 'logicle', prescale=1) -> pd.DataFrame:
    """
    Apply a transformation to the given dataset; valid transformation methods are:
    logicle, hyperlog, log_transform, or asinh
    :param data: pandas dataframe
    :param features_to_transform: a list of features to transform
    :param transform_method: string value indicating transformation method
    :param prescale: if using asinh transformaion this value is passed as the scalling argument
    :return: transformed pandas dataframe
    """

    if features_to_transform == 'all':
        features_to_transform = data.columns
    elif features_to_transform == 'fluorochromes':
        features_to_transform = [x for x in data.columns if all([y not in x for y in ['FSC', 'SSC', 'Time', 'label']])]
    elif type(features_to_transform) != list:
        print('Error: invalid argument provided for `features_to_transform`, expected one of: `all`, `fluorochromes`,'
              ' or list of valid column names, proceeding with transformation of entire dataframe as precaution.')
        features_to_transform = data.columns
    elif not all([x in data.columns for x in features_to_transform]):
        print('Error: invalid argument provided for `features_to_transform`, list must contain column names that '
              f'correspond to the provided dataframe. Valid input would be one or several of: {data.columns} '
              'proceeding with transformation of entire dataframe as precaution.')
        features_to_transform = data.columns

    if transform_method == 'logicle':
        return logicle(data, features_to_transform)
    if transform_method == 'hyperlog':
        return hyperlog(data, features_to_transform)
    if transform_method == 'log_transform':
        return log_transform(data, features_to_transform)
    if transform_method == 'asinh':
        return asinh(data, features_to_transform, prescale)
    if transform_method == 'percentile rank':
        return percentile_rank_transform(data, features_to_transform)
    if transform_method == 'Yeo-Johnson':
        return sklearn_scaler(data, features_to_transform, scale_method='power', method='yeo-johnson')
    if transform_method == 'RobustScale':
        return sklearn_scaler(data, features_to_transform, scale_method='robust')
    raise TransformError("Error: invalid transform_method, must be one of: 'logicle', 'hyperlog', 'log_transform',"
                         " 'asinh', 'percentile rank', 'Yeo-Johnson', 'RobustScale'")


def sklearn_scaler(data: pd.DataFrame, features_to_transform: list, scale_method: str, **kwargs) -> pd.DataFrame:
    """
    Wrapper function for transforming single cell data using Sklearn scaler functions
    :param data: dataframe of data to apply scale function too
    :param features_to_transform: list of features (columns) to scale
    :param scale_method: name of scaler method to use (see cytopy.flow.supervised.utilities.scaler_for available methods)
    :param kwargs: keyword arguments to pass to scaler function (see cytopy.flow.supervised.utilities.scaler)
    :return: DataFrame with scaler applied
    """
    data = data.copy()
    transform, _ = scaler(data[features_to_transform], scale_method=scale_method, **kwargs)
    data[features_to_transform] = transform
    return data
