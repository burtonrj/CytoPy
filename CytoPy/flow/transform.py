#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Cytometry data has to be transformed prior to analysis. There are multiple
techniques for transformation of data, the most popular being the biexponential
transform. CytoPy employs multiple methods using the FlowUtils package
(https://github.com/whitews/FlowUtils), including the Logicle transform, a modified
version of the biexponential transform.


Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from flowutils.transforms import logicle, hyperlog, log_transform, asinh
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
import pandas as pd
import numpy as np

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def percentile_rank_transform(data: pd.DataFrame, 
                              features_to_transform: list) -> pd.DataFrame:
    """
    Calculate percentile rank transform of data-frame. Each event is
    ranked as the average according to the column, then divided by
    the total number of events and multiplied by 100 to give the percentile.
    
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
    transform = (transform / transform.shape[0]) * 100
    data[features_to_transform] = transform
    return data


def _features(data: pd.DataFrame,
              features_to_transform: str):
    """
    Filter data to only required features, either all or just fluorochromes

    Parameters
    ----------
    data: Pandas.DataFrame
    features_to_transform: str

    Returns
    -------
    Pandas.DataFrame
    """
    if features_to_transform == 'all':
        return list(data.columns)
    elif features_to_transform == 'fluorochromes':
        return [x for x in data.columns if all([y not in x.lower() for y in ['fsc', 'ssc', 'time', 'label']])]
    raise ValueError(f"Expected one of: 'all' or 'fluorochromes', but got {features_to_transform}")


def _transform(data: pd.DataFrame,
               features: list,
               method: str or None,
               **kwargs):
    """
    Wrap the FlowUtils/Scikit-Learn functions for transformations and returning the transformed
    data as a Pandas Dataframe.

    Parameters
    ----------
    data: Pandas.DataFrame
        Data to transform
    features: list
        Features to be included in transformation
    method: str or None
        Method used for transformation (if None, returns original data unchanged)
        Available transforms:
        * logicle
        * hyperlog
        * log_transform
        * asinh
        * percentile_rank
        * Yeo-Johnson
        * RobustScale
    kwargs:
        Additional keyword arguments passed to transform function

    Returns
    -------
    Pandas.DataFrame
    """
    pre_scale = kwargs.get("pre_scale", 1)
    feature_i = [list(data.columns).index(i) for i in features]
    if method is None:
        return data
    if method == 'logicle':
        return pd.DataFrame(logicle(data=data.values, channels=feature_i, **kwargs), columns=data.columns, index=data.index)
    if method == 'hyperlog':
        return pd.DataFrame(hyperlog(data=data.values, channels=feature_i, **kwargs), columns=data.columns, index=data.index)
    if method == 'log_transform':
        return pd.DataFrame(log_transform(npy=data.values, channels=feature_i,), columns=data.columns, index=data.index)
    if method == 'asinh':
        return pd.DataFrame(asinh(data=data.values, columns=feature_i, pre_scale=pre_scale),
                            columns=data.columns, index=data.index)
    if method == 'percentile rank':
        return percentile_rank_transform(data, features)
    if method == 'Yeo-Johnson':
        data, _ = scaler(data, scale_method='power', method='yeo-johnson')
        return data
    if method == 'RobustScale':
        data, _ = scaler(data, scale_method='robust')
        return data
    raise ValueError("Error: invalid transform_method, must be one of: 'logicle', 'hyperlog', 'log_transform',"
                     " 'asinh', 'percentile rank', 'Yeo-Johnson', 'RobustScale'")


def individual_transforms(data: pd.DataFrame,
                          transforms: dict,
                          **kwargs):
    """
    Given a Pandas DataFrame and a dictionary of transformations to apply, where the
    key is the column to transform and the value the method for transformation, apply
    transforms to each specified column.

    Parameters
    ----------
    data: Pandas.DataFrame
    transforms: dict
    kwargs:
        Additional keyword arguments passed to transform function

    Returns
    -------
    Pandas.DataFrame
    """
    for feature, method in transforms.items():
        assert feature in data.columns, f"{feature} column not found for given DataFrame"
        data[feature] = _transform(data, [feature], method, **kwargs)[feature]
    return data


def apply_transform(data: pd.DataFrame,
                    features_to_transform: list or str or dict = 'all',
                    transform_method: str or None = 'logicle',
                    **kwargs) -> pd.DataFrame:
    """
    Apply a transformation to the given dataframe. The features_to_transform specified which
    columns in the dataframe to transform. This can be given as:
    * a string value of either 'all' or 'fluorochromes'; transform_method defines which transform
      to apply to columns
    * a list of columns to transform; transform_method defines which transform
      to apply to columns
    * alternatively, a dictionary where the key is the column name and the value is the
      transform method to apply to this column; transform_method is ignored
    
    Parameters
    -----------
    data: Pandas.DataFrame
    features_to_transform: list or str or dict (default="all")
    transform_method: str or None

    Returns
    --------
    Pandas.DataFrame
    """
    data = data.copy()
    if isinstance(features_to_transform, dict):
        return individual_transforms(data, features_to_transform, **kwargs)
    if isinstance(features_to_transform, str):
        features_to_transform = _features(data, features_to_transform)
    elif isinstance(features_to_transform, list):
        assert all([x in data.columns for x in features_to_transform]), \
            "One or more provided features does not exist for the given dataframe"
    return _transform(data=data, features=features_to_transform, method=transform_method, **kwargs)


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
    if len(data) == 0:
        return data
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
