from flowutils.transforms import logicle, hyperlog, log_transform, asinh
import pandas as pd


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
    print('Error: invalid transform_method, returning untransformed data')
    return data
