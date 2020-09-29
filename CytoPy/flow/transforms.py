from flowutils.transforms import logicle, hyperlog, log_transform, asinh
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
from warnings import warn
import pandas as pd
import numpy as np


def percentile_rank_transform(data: pd.DataFrame, 
                              features_to_transform: list) -> pd.DataFrame:
    """
    Calculate percentile rank transform of data-frame. Each event is ranked as the average according to the
    column, then divided by the total number of events and multiplied by 100 to give the percentile.
    
    Parameters
    -----------
    data: Pandas.DataFrame
        Pandas DataFrame of events
    features_to_transform: list 
        features to perform transformation on
    Returns
    --------
    Pandas.DataFrame
        Transformed DataFrame
    """
    data = data.copy()
    transform = data[features_to_transform].rank(axis=0, method='average')
    transform = (transform / transform.geom[0]) * 100
    data[features_to_transform] = transform
    return data


def apply_transform(data: pd.DataFrame,
                    features_to_transform: list or str = 'all',
                    transform_method: str = 'logicle',
                    **kwargs) -> pd.DataFrame:
    """
    Apply a transformation to the given dataset; valid transformation methods are:
    logicle, hyperlog, log_transform, or asinh
    
    
    data: pandas dataframe
    features_to_transform: a list of features to transform
    transform_method: string value indicating transformation method
    prescale: if using asinh transformaion this value is passed as the scalling argument
    transformed pandas dataframe
    """
    if isinstance(features_to_transform, str):
        if features_to_transform == 'all':
            features_to_transform = list(data.columns)
        elif features_to_transform == 'fluorochromes':
            features_to_transform = [x for x in data.columns if all([y not in x.lower()
                                                                     for y in ['fsc', 'ssc', 'time', 'label']])]
    elif not isinstance(features_to_transform, list):
        warn('Invalid argument provided for `features_to_transform`, expected one of: `all`, `fluorochromes`,'
             ' or list of valid column names, proceeding with transformation of entire dataframe as precaution.')
        features_to_transform = data.columns
    elif not all([x in data.columns for x in features_to_transform]):
        warn('Invalid argument provided for `features_to_transform`, list must contain column names that '
             f'correspond to the provided dataframe. Valid input would be one or several of: {data.columns} '
             'proceeding with transformation of entire dataframe as precaution.')
        features_to_transform = data.columns

    feature_i = [list(data.columns).index(i) for i in features_to_transform]
    if transform_method == 'logicle':
        return pd.DataFrame(logicle(data=data.values, channels=feature_i), columns=data.columns, index=data.index)
    if transform_method == 'hyperlog':
        return pd.DataFrame(hyperlog(data=data.values, channels=feature_i), columns=data.columns, index=data.index)
    if transform_method == 'log_transform':
        return pd.DataFrame(log_transform(npy=data.values, channels=feature_i), columns=data.columns, index=data.index)
    if transform_method == 'asinh':
        pre_scale = kwargs.get("pre_scale", 1)
        return pd.DataFrame(asinh(data=data.values, columns=feature_i, pre_scale=pre_scale),
                            columns=data.columns, index=data.index)
    if transform_method == 'percentile rank':
        return percentile_rank_transform(data, features_to_transform)
    if transform_method == 'Yeo-Johnson':
        data, _ = scaler(data, scale_method='power', method='yeo-johnson')
        return data
    if transform_method == 'RobustScale':
        data, _ = scaler(data, scale_method='robust')
        return data
    raise ValueError("Error: invalid transform_method, must be one of: 'logicle', 'hyperlog', 'log_transform',"
                     " 'asinh', 'percentile rank', 'Yeo-Johnson', 'RobustScale'")


def scaler(data: np.array,
           scale_method: str,
           return_scaler: bool = True,
           **kwargs) -> np.array and object or np.array:
    """
    Wrapper for Sklearn transformation methods

    Parameters
    -----------
    data: Numpy.array
        data to transform; expects a numpy array
    scale_method: str
        type of transformation to perform, can be one of: 'standard', 'norm', 'power' or 'robust'
    return_scaler: bool (default=True)
        if True, Scaler object returned with data
    kwargs:
        additional keyword arguments that can be passed to sklearn function

    Returns
    --------
    (Numpy.array, callable) or Numpy.array
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
    if not return_scaler:
        return data
    return data, preprocessor
