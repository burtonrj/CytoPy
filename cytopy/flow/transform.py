#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Cytometry data has to be transformed prior to analysis. There are multiple
techniques for transformation of data, the most popular being the biexponential
transform. cytopy employs multiple methods using the FlowUtils package
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
from functools import partial
from flowutils import transforms
from sklearn import preprocessing
from warnings import warn
import pandas as pd
import numpy as np

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


class TransformError(Exception):
    pass


def _get_dataframe_column_index(data: pd.DataFrame,
                                features: list):
    """
    Given the features of interest (columns in data) return the index of these features

    Parameters
    ----------
    data: Pandas.DataFrame
    features: list

    Returns
    -------
    numpy.ndarray
        Index of columns of interest
    """
    return np.array([data.columns.get_loc(f) for f in features if f in data.columns])


class Transformer:
    """
    Base class for Transformer object.

    Parameters
    ----------
    transform: callable
        Transform function
    inverse: callable
        Inverse transformation function
    kwargs:
        Keyword arguments passed to transform/inverse transform

    Attributes
    ----------
    transform: callable
    inverse: callable
    kwargs: dict
    """

    def __init__(self,
                 transform_function: callable,
                 inverse_function: callable,
                 **kwargs):
        self.transform = transform_function
        self.inverse = inverse_function
        self.kwargs = kwargs or {}

    def scale(self,
              data: pd.DataFrame,
              features: list):
        """
        Scale features (columns) of given dataframe

        Parameters
        ----------
        data: Pandas.DataFrame
        features: list

        Returns
        -------
        Pandas.DataFrame

        Raises
        ------
        TransformError
            Chosen transform function is missing the arguments channel_indices or channels. cytopy uses
            the FlowUtils class for transformations. See FlowUtils documentation for details.
        """
        data = data.copy()
        original_index = data.index.values
        data[features] = data[features].astype(float)
        idx = _get_dataframe_column_index(data, features)
        if "channel_indices" in self.transform.__code__.co_varnames:
            data = pd.DataFrame(self.transform(data=data.values,
                                               channel_indices=idx,
                                               **self.kwargs),
                                columns=data.columns,
                                index=original_index)
        elif "channels" in self.transform.__code__.co_varnames:
            data = pd.DataFrame(self.transform(data=data.values,
                                               channels=idx,
                                               **self.kwargs),
                                columns=data.columns,
                                index=original_index)
        else:
            raise TransformError("Invalid transform function, missing argument 'channel_indices' or 'channels'")
        data[features] = data[features].astype(float)
        return data

    def inverse_scale(self,
                      data: pd.DataFrame,
                      features: list):
        """
        Apply inverse scale to features (columns) of given dataframe, under the assumption that
        these features have previously been transformed with this Transformer

        Parameters
        ----------
        data: Pandas.DataFrame
        features: list

        Returns
        -------
        Pandas.DataFrame

        Raises
        ------
        TransformError
            Chosen inverse transform function is missing the arguments channel_indices or channels.
            cytopy uses the FlowUtils class for transformations. See FlowUtils documentation for details.
        """
        data = data.copy()
        original_index = data.index.values
        data[features] = data[features].astype(float)
        idx = _get_dataframe_column_index(data, features)
        if "channel_indices" in self.inverse.__code__.co_varnames:
            data = pd.DataFrame(self.inverse(data=data.values,
                                             channel_indices=idx,
                                             **self.kwargs),
                                columns=data.columns,
                                index=original_index)
        elif "channels" in self.inverse.__code__.co_varnames:
            data = pd.DataFrame(self.inverse(data=data.values,
                                             channels=idx,
                                             **self.kwargs),
                                columns=data.columns,
                                index=original_index)
        else:
            raise TransformError("Invalid inverse transform function, missing argument 'channel_indices' or 'channels'")
        data[features] = data[features].astype(float)
        return data


class LogicleTransformer(Transformer):
    """
    Implementation of Logicle transform is authored by Scott White (FlowUtils v0.8).
    Logicle transformation, implemented as defined in the GatingML 2.0 specification:

    logicle(x, T, W, M, A) = root(B(y, T, W, M, A) − x)

    where B is a modified bi-exponential function defined as:

    B(y, T, W, M, A) = ae^(by) − ce^(−dy) − f

    The Logicle transformation was originally defined in the publication:
    Moore WA and Parks DR. Update for the logicle data scale including operational
    code implementations. Cytometry A., 2012:81A(4):273–277.

    Parameters
    ----------
    w: float (default=0.5)
        Approximate number of decades in the linear region
    m: float (default=4.5)
        Number of decades the true logarithmic scale approaches at the high end of the scale
    a: float (default=0)
        Additional number of negative decades
    t: int (default=262144)
        Top of the linear scale
    """

    def __init__(self,
                 w: float = 0.5,
                 m: float = 4.5,
                 a: float = 0.0,
                 t: int = 262144):
        super().__init__(transform_function=transforms.logicle,
                         inverse_function=transforms.logicle_inverse,
                         w=w,
                         m=m,
                         a=a,
                         t=t)


class HyperlogTransformer(Transformer):
    """
    Implementation of Hyperlog transform is authored by Scott White (FlowUtils v0.8).

    Hyperlog transformation, implemented as defined in the GatingML 2.0 specification:

    hyperlog(x, T, W, M, A) = root(EH(y, T, W, M, A) − x)

    where EH is defined as:

    EH(y, T, W, M, A) = ae^(by) + cy − f

    The Hyperlog transformation was originally defined in the publication:
    Bagwell CB. Hyperlog-a flexible log-like transform for negative, zero, and
    positive valued data. Cytometry A., 2005:64(1):34–42.

    Parameters
    ----------
    w: float (default=0.5)
        Approximate number of decades in the linear region
    m: float (default=4.5)
        Number of decades the true logarithmic scale approaches at the high end of the scale
    a: float (default=0)
        Additional number of negative decades
    t: int (default=262144)
        Top of the linear scale
    """

    def __init__(self,
                 w: float = 0.5,
                 m: float = 4.5,
                 a: float = 0.0,
                 t: int = 262144):
        super().__init__(transform_function=transforms.hyperlog,
                         inverse_function=transforms.hyperlog_inverse,
                         w=w,
                         m=m,
                         a=a,
                         t=t)


class AsinhTransformer(Transformer):
    """
    Implementation of inverse hyperbolic sine function, authored by Scott White (FlowUtils v0.8).

    Parameters
    ----------
    m: float (default=4.5)
        Number of decades the true logarithmic scale approaches at the high end of the scale
    a: float (default=0)
        Additional number of negative decades
    t: int (default=262144)
        Top of the linear scale
    """

    def __init__(self,
                 m: float = 4.5,
                 a: float = 0.0,
                 t: int = 262144):
        super().__init__(transform_function=transforms.asinh,
                         inverse_function=transforms.asinh_inverse,
                         m=m,
                         a=a,
                         t=t)


class LogTransformer(Transformer):
    """
    Apply log transform to data, either using parametrized log transform as defined in GatingML 2.0 specification
    (implemented by Scott White in FlowUtils v0.8) or using natural log, base 2 or base 10.

    Parameters
    ----------
    base: str or int (default="parametrized")
        Method to be used, should either be 'parametrized', 10, 2, or 'natural'
    m: float (default=4.5)
        Number of decades the true logarithmic scale approaches at the high end of the scale
    t: int (default=262144)
        Top of the linear scale

    Raises
    ------
    TransformError
        Invalid LogTransformer method
    """

    def __init__(self,
                 base: str or int = "parametrized",
                 m: float = 4.5,
                 t: int = 262144,
                 **kwargs):
        if base == "parametrized":
            super().__init__(transform_function=lambda x: (1. / m) * np.log10(x / t) + 1.,
                             inverse_function=lambda x: t * (10 ** ((x - 1) * m)))
        elif base == 10:
            super().__init__(transform_function=partial(np.log10, **kwargs),
                             inverse_function=lambda x: 10 ** x)
        elif base == 2:
            super().__init__(transform_function=partial(np.log2, **kwargs),
                             inverse_function=lambda x: 2 ** x)
        elif base == "natural":
            super().__init__(transform_function=partial(np.log, **kwargs),
                             inverse_function=np.exp)
        else:
            raise TransformError("Invalid LogTransformer method, expected one of:"
                                 "'parametrized', 10, 2, or 'natural'")

    def scale(self,
              data: pd.DataFrame,
              features: list):
        """
        Scale features (columns) of given dataframe using log transform

        Parameters
        ----------
        data: Pandas.DataFrame
        features: list

        Returns
        -------
        Pandas.DataFrame
        """
        data = remove_negative_values(data, features)
        data[features] = self.transform(data[features].values)
        return data

    def inverse_scale(self,
                      data: pd.DataFrame,
                      features: list):
        """
        Apply inverse of log transform to features (columns) of given dataframe,
        under the assumption that these features have previously been transformed with LogTransformer

        Parameters
        ----------
        data: Pandas.DataFrame
        features: list

        Returns
        -------
        Pandas.DataFrame
        """
        if data.shape[0] == 1 and data[features].iloc[0] == 0:
            return data
        data = remove_negative_values(data, features)
        data[features] = self.inverse(data[features].values)
        return data


class Normalise:
    """
    Normalise data using Scikit-Learn normalize function (https://bit.ly/2YBfe3o)
    Norms are stored in the attribute 'norms' and normalisation reversed
    by passing the transformed data to the 'inverse' method.

    Parameters
    ----------
    norm: Numpy Array
        An array of norms along given axis for X
    axis: int (default=1)
        Axis to apply normalisation along. If 1, independently normalize each sample, otherwise (if 0)
         normalize each feature.
    """

    def __init__(self,
                 norm: str = "l2",
                 axis: int = 1):
        self.norm = norm
        self.axis = axis
        self._norms = None
        self._shape = None

    def __call__(self,
                 data: pd.DataFrame,
                 features: list):
        """
        Normalise columns (features) for given dataframe. Returns copy of DataFrame
        with chosen columns normalised

        Parameters
        ----------
        data: Pandas.DataFrame
        features: List

        Returns
        -------
        Pandas.DataFrame
        """
        data = data.copy()
        self._shape = data[features].shape
        x, self._norms = preprocessing.normalize(data[features].values,
                                                 norm=self.norm,
                                                 axis=self.axis,
                                                 return_norm=True)
        data[features] = x
        return data

    def inverse(self,
                data: pd.DataFrame,
                features: list):
        """
        Perform inverse of normalisation to given dataframe. Returns copy of DataFrame
        with inverse normalisation applied to chosen columns (features)

        Parameters
        ----------
        data: Pandas.DataFrame
        features: List

        Returns
        -------
        Pandas.DataFrame

        Raises
        ------
        AssertionError
            Shape of given dataframe does not match the data
        ValueError
            Inverse called prior to normalisation
        """
        assert data[features].shape == self._shape, "Shape of given dataframe does not match the data " \
                                                    f"transformed originally: {self._shape} != {data.shape}"
        if self._norms is None:
            "Call Normalise object to first normalise target data prior to attempting inverse"
        data[features] = data[features] * self._norms
        return data


SCALERS = {"standard": preprocessing.StandardScaler,
           "minmax": preprocessing.MinMaxScaler,
           "robust": preprocessing.RobustScaler,
           "maxabs": preprocessing.MaxAbsScaler,
           "quantile": preprocessing.QuantileTransformer,
           "yeo_johnson": preprocessing.PowerTransformer,
           "box_cox": preprocessing.PowerTransformer}


class Scaler:
    """
    Utility object for applying Scikit-Learn transformers to a chosen
    dataset. Following transformations supported; method and corresponding
    Scikit-Learn class:

    * "standard" - sklearn.preprocessing.StandardScaler
    * "minmax" - sklearn.preprocessing.MinMaxScaler
    * "robust" - sklearn.preprocessing.RobustScaler
    * "maxabs" - sklearn.preprocessing.MaxAbsScaler
    * "quantile" - sklearn.preprocessing.QuantileTransformer
    * "yeo_johnson" - sklearn.preprocessing.PowerTransformer
    * "box_cox" - sklearn.preprocessing.PowerTransformer

    (PowerTransformer method argument will be 'yeo-johnson' or 'box-cox'
    according to the chosen method)

    See relevant Scikit-Learn documentation for guidance on a particular method:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing

    User should initialise object with 'method' according to the above and
    provide any additional keyword arguments, relevant to the chosen object,
    as kwargs.

    Attributes
    -----------
    method: str
        Name of scikit-learn transformer to use
    """

    def __init__(self,
                 method: str = "standard",
                 **kwargs):
        """
        Initialise object and create transformer

        Parameters
        ----------
        method: str
        kwargs:
            Additional keyword arguments used when initialising Scikit-Learn object
        """
        if method not in SCALERS.keys():
            raise TransformError(f"Method not supported, must be one of: {list(SCALERS.keys())}")
        kwargs = kwargs or {}
        if method == "yeo_johnson":
            kwargs["method"] = "yeo-johnson"
        if method == "box_cox":
            kwargs["method"] = "box_cox"
        self._scaler = SCALERS.get(method)(**kwargs)

    def __call__(self,
                 data: pd.DataFrame,
                 features: list):
        """
        Using a given dataframe and a list of columns (features) to transform,
        call 'fit_transform' on Scikit-Learn transformer. Returns copy of
        DataFrame with columns transformed

        Parameters
        ----------
        data: Pandas.DataFrame
        features: list

        Returns
        -------
        Pandas.DataFrame
        """
        data = data.copy()
        data[features] = self._scaler.fit_transform(data[features].values)
        return data

    def inverse(self, data: pd.DataFrame, features: list):
        """
        Given dataframe and a list of columns (features) that has been previously
        transformed, apply inverse transform. Returns copy of DataFrame with
        transformation reversed.

        Parameters
        ----------
        data: Pandas.DataFrame
        features: List

        Returns
        -------
        Pandas.DataFrame

        Raises
        -------
        TransformError
            If the chosen Scikit-Learn method does not support inverse transform
        """
        if getattr(self._scaler, "inverse_transform", None) is None:
            raise TransformError("Chosen scaler method does not support inverse transformation")
        else:
            data = data.copy()
            data[features] = self._scaler.inverse_transform(data[features].values)
            return data

    def set_params(self, **kwargs):
        """
        Sets parameters of underlying Scikit-Learn method

        Parameters
        ----------
        kwargs
            Additional keyword arguments passed to 'set_params' call

        Returns
        -------
        None
        """
        self._scaler.set_params(**kwargs)


TRANSFORMERS = {"logicle": LogicleTransformer,
                "hyperlog": HyperlogTransformer,
                "asinh": AsinhTransformer,
                "log": LogTransformer}


def apply_transform(data: pd.DataFrame,
                    features: list or dict,
                    method: str or None = "logicle",
                    return_transformer: bool = False,
                    **kwargs):
    """
    Apply a transformation to the given DataFrame and the chosen
    columns (features). Transformation method is specified using the
    'method' argument and should be one of:

    * logicle: see cytopy.flow.transform.LogicleTransformer
    * hyperlog: see cytopy.flow.transform.HyperlogTransformer
    * asinh: see cytopy.flow.transform.AsinhTransformer
    * log: see cytopy.flow.transform.LogTransformer

    Parameters
    ----------
    data: Pandas.DataFrame
    features: List or dict
        Column names to be transformed
    method: str (default='logicle')
        Transformation method
    return_transformer: bool (default=False)
        If True, Transformer object is also returned
    kwargs
        Additional keyword arguments passed to respective Transformer

    Returns
    -------
    Pandas.DataFrame or (Pandas.DataFrame and Transformer)
        Copy of the DataFrame with chosen features transformed and Transformer object if return_transformer is True

    Raises
    ------
    TransformError
        Raised if invalid transform method requested
    """
    if method is None:
        if return_transformer:
            return data, None
        return data
    if method not in TRANSFORMERS.keys():
        raise TransformError(f"Invalid transform, must be one of: {list(TRANSFORMERS.keys())}")
    method = TRANSFORMERS.get(method)(**kwargs)
    if return_transformer:
        x = method.scale(data=data, features=features)
        return x, method
    return method.scale(data=data, features=features)


def apply_transform_map(data: pd.DataFrame,
                        feature_method: dict,
                        kwargs: dict or None = None):
    """
    Wrapper function to cytopy.flow.transform.apply_transform; takes a dictionary (feature_method) where
    each key is the name of a feature and the value the transform to be applied to that feature.

    Parameters
    ----------
    data: Pandas.DataFrame
    feature_method: dict
    kwargs: dict
        Additional keyword arguments passed to apply_transform

    Returns
    -------
    Pandas.DataFrame
        DataFrame with feature transformed
    """
    data = data.copy()
    kwargs = kwargs or {}
    for feature, method in feature_method.items():
        transform_kwargs = kwargs.get(feature, {})
        data = apply_transform(data=data,
                               features=[feature],
                               method=method,
                               return_transformer=False,
                               **transform_kwargs)
    return data


def remove_negative_values(data: pd.DataFrame,
                           features: list):
    """
    For each feature (as given in 'features') in data, check for negative values and
    replace with minimum value in range for that feature.

    Parameters
    ----------
    data: Pandas.DataFrame
    features: list

    Returns
    -------
    Pandas.DataFrame
        Modified DataFrame without negative values

    Raises
    ------
    TransformError
        All values for a given feature are negative
    """
    data = data.copy()
    for f in features:
        if (data[f] <= 0).any():
            warn(f"Feature {f} contains negative values. Chosen Transformer requires values "
                 f">=0, all values <=0 will be forced to the minimum valid values in {f}")
            valid_range = data[data[f] > 0][f].values
            if len(valid_range) == 0:
                raise TransformError(f"All values for {f} <= 0")
            min_ = np.min(valid_range)
            data[f] = np.where(data[f].values <= 0, min_, data[f].values)
    return data


def safe_range(data: pd.DataFrame, x: str):
    """
    Return the minimum and maximum values in a range, ignore negative values

    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
        Column of interest

    Returns
    -------
    (float, float)
        Min, max

    Raises
    ------
    AssertionError
        If all values in x are negative
    """
    valid_range = data[data[x] > 0][x].values
    assert len(valid_range) > 0, f"All values for {x} <= 0"
    return np.min(valid_range), np.max(valid_range)
