from flowutils.transforms import logicle, hyperlog, log_transform, asinh
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
    not_transform = [f for f in data.columns if f not in features_to_transform]
    data_transform = data[features_to_transform]
    if not_transform:
        not_transform = data[not_transform]
    data_transform = data_transform.rank(axis=1, method='average')
    data_transform = (data_transform/data_transform.shape[0]) * 100
    if type(not_transform) is not list:
        return pd.concat([data_transform, not_transform])
    return data_transform


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
        features_to_transform = [x for x in data.columns if all([y not in x for y in ['FSC', 'SSC', 'Time']])]
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
    raise TransformError("Error: invalid transform_method, must be one of: 'logicle', 'hyperlog', 'log_transform',"
                         " 'asinh'")
