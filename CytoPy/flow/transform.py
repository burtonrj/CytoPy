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
from functools import partial
from flowutils import transforms
from sklearn import preprocessing
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
    Numpy.Array
        Index of columns of interest
    """
    return np.array([data.columns.get_loc(f) for f in features if f in data.columns])


class LogicleTransformer:
    """
    Implementation of Logicle transform is authored by Scott White (FlowUtils v0.8).
    Logicle transformation, implemented as defined in the GatingML 2.0 specification:

    logicle(x, T, W, M, A) = root(B(y, T, W, M, A) − x)

    where B is a modified bi-exponential function defined as:

    B(y, T, W, M, A) = ae^(by) − ce^(−dy) − f

    The Logicle transformation was originally defined in the publication:
    Moore WA and Parks DR. Update for the logicle data scale including operational
    code implementations. Cytometry A., 2012:81A(4):273–277.

    Attributes
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
        self.w = w
        self.m = m
        self.a = a
        self.t = t

    def scale(self,
              data: pd.DataFrame,
              features: list):
        """
        Scale features (columns) of given dataframe using logicle transform

        Parameters
        ----------
        data: Pandas.DataFrame
        features: list

        Returns
        -------
        Pandas.DataFrame
        """
        data = data.copy()
        idx = _get_dataframe_column_index(data, features)
        data[features] = transforms.logicle(data=data,
                                            channel_indices=idx,
                                            t=self.t,
                                            m=self.m,
                                            w=self.w,
                                            a=self.a)
        return data

    def inverse(self,
                data: pd.DataFrame,
                features: list):
        """
        Apply inverse logicle scale to features (columns) of given dataframe, under the assumption that
        these features have previously been transformed with LogicleTransformer

        Parameters
        ----------
        data: Pandas.DataFrame
        features: list

        Returns
        -------
        Pandas.DataFrame
        """
        data = data.copy()
        idx = _get_dataframe_column_index(data, features)
        data[features] = transforms.logicle_inverse(data=data,
                                                    channel_indices=idx,
                                                    t=self.t,
                                                    m=self.m,
                                                    w=self.w,
                                                    a=self.a)
        return data


class HyperlogTransformer:
    """
    Implementation of Hyperlog transform is authored by Scott White (FlowUtils v0.8).

    Hyperlog transformation, implemented as defined in the GatingML 2.0 specification:

    hyperlog(x, T, W, M, A) = root(EH(y, T, W, M, A) − x)

    where EH is defined as:

    EH(y, T, W, M, A) = ae^(by) + cy − f

    The Hyperlog transformation was originally defined in the publication:
    Bagwell CB. Hyperlog-a flexible log-like transform for negative, zero, and
    positive valued data. Cytometry A., 2005:64(1):34–42.

    Attributes
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
        self.w = w
        self.m = m
        self.a = a
        self.t = t

    def scale(self,
              data: pd.DataFrame,
              features: list):
        """
        Scale features (columns) of given dataframe using hyperlog transform

        Parameters
        ----------
        data: Pandas.DataFrame
        features: list

        Returns
        -------
        Pandas.DataFrame
        """
        data = data.copy()
        idx = _get_dataframe_column_index(data, features)
        data[features] = transforms.hyperlog(data=data,
                                             channel_indices=idx,
                                             t=self.t,
                                             m=self.m,
                                             w=self.w,
                                             a=self.a)
        return data

    def inverse(self,
                data: pd.DataFrame,
                features: list):
        """
        Apply inverse hyperlog scale to features (columns) of given dataframe, under the assumption that
        these features have previously been transformed with HyperlogTransformer

        Parameters
        ----------
        data: Pandas.DataFrame
        features: list

        Returns
        -------
        Pandas.DataFrame
        """
        data = data.copy()
        idx = _get_dataframe_column_index(data, features)
        data[features] = transforms.hyperlog_inverse(data=data,
                                                     channels=idx,
                                                     t=self.t,
                                                     m=self.m,
                                                     w=self.w,
                                                     a=self.a)
        return data


class AsinhTransformer:
    """
    Implementation of inverse hyperbolic sine function, authored by Scott White (FlowUtils v0.8).

    Attributes
    ----------
    m: float (default=4.5)
        Number of decades the true logarithmic scale approaches at the high end of the scale
    a: float (default=0)
        Additional number of negative decades
    t: int (default=262144)
        Top of the linear scale
    """

    def __init__(self,
                 t: int = 262144,
                 m: float = 4.5,
                 a: float = 0):
        self.t = t
        self.m = m
        self.a = a

    def scale(self,
              data: pd.DataFrame,
              features: list):
        """
        Scale features (columns) of given dataframe using inverse hyperbolic sine transform

        Parameters
        ----------
        data: Pandas.DataFrame
        features: list

        Returns
        -------
        Pandas.DataFrame
        """
        data = data.copy()
        idx = _get_dataframe_column_index(data, features)
        data[features] = transforms.asinh(data=data,
                                          channel_indices=idx,
                                          t=self.t,
                                          m=self.m,
                                          a=self.a)
        return data

    def inverse(self,
                data: pd.DataFrame,
                features: list):
        """
        Apply inverse of parametrized hyperbolic sine function scale to features (columns) of given dataframe,
        under the assumption that these features have previously been transformed with AsinhTransformer

        Parameters
        ----------
        data: Pandas.DataFrame
        features: list

        Returns
        -------
        Pandas.DataFrame
        """
        data = data.copy()
        idx = _get_dataframe_column_index(data, features)
        data[features] = transforms.asinh_inverse(data=data,
                                                  channel_indices=idx,
                                                  t=self.t,
                                                  m=self.m,
                                                  a=self.a)
        return data


class LogTransformer:
    """
    Apply log transform to data, either using parametrized log transform as defined in GatingML 2.0 specification
    (implemented by Scott White in FlowUtils v0.8) or using natural log, base 2 or base 10.

    Attributes
    ----------
    base: str or int (default="parametrized")
        Method to be used, should either be 'parametrized', 10, 2, or 'natural'
    m: float (default=4.5)
        Number of decades the true logarithmic scale approaches at the high end of the scale
    t: int (default=262144)
        Top of the linear scale
    """
    def __init__(self,
                 base: str or int = "parametrized",
                 m: float = 4.5,
                 t: int = 262144,
                 **kwargs):
        if base == "parametrized":
            self._log = lambda x: (1./m) * np.log10(x/t) + 1.
            self._inverse = lambda x: t * (10 ** ((x-1) * m))
        elif base == 10:
            self._log = partial(np.log10, **kwargs)
            self._inverse = lambda x: 10**x
        elif base == 2:
            self._log = partial(np.log2, **kwargs)
            self._inverse = lambda x: 2**x
        elif base == "natural":
            self._log = partial(np.log, **kwargs)
            self._inverse = np.exp
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
        data = data.copy()
        data[features] = self._log(data[features].values)
        return data

    def inverse(self,
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
        data = data.copy()
        data[features] = self._inverse(data[features].values)
        return data


class Normalise:
    def __init__(self,
                 norm: str = "l2",
                 axis: int = 1):
        self.norm = norm
        self.axis = axis
        self._norms = None
        self._shape = None

    def __call__(self, data: pd.DataFrame, features: list):
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
        assert data[features].shape == self._shape, "Shape of given dataframe does not match the data " \
                                                    f"transformed originally: {self._shape} != {data.shape}"
        assert self._norms is not None, "Call Normalise object to first normalise target data prior to attempting " \
                                        "inverse"
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
    def __init__(self,
                 method: str = "standard",
                 **kwargs):
        if method not in SCALERS.keys():
            raise TransformError(f"Method not supported, must be one of: {list(SCALERS.keys())}")
        kwargs = kwargs or {}
        if method == "yeo_johnson":
            kwargs["method"] = "yeo-johnson"
        if method == "box_cox":
            kwargs["method"] = "box_cox"
        self._scaler = SCALERS.get(method)(**kwargs)

    def __call__(self, data: pd.DataFrame, features: list, **kwargs):
        data = data.copy()
        data[features] = self._scaler.fit_transform(data[features].values)
        return data

    def inverse(self, data: pd.DataFrame, features: list):
        if getattr(self._scaler, "inverse_transform", None) is None:
            raise TransformError("Chosen scaler method does not support inverse transformation")
        else:
            data = data.copy()
            data[features] = self._scaler.inverse_transform(data[features].values)
            return data

    def set_params(self, **kwargs):
        self._scaler.set_params(**kwargs)


TRANSFORMERS = {"logicle": LogicleTransformer,
                "hyperlog": HyperlogTransformer,
                "asinh": AsinhTransformer,
                "log": LogTransformer}


def apply_transform(data: pd.DataFrame,
                    features: list,
                    method: str = "logicle",
                    return_transformer: bool = False,
                    **kwargs):
    if method not in TRANSFORMERS.keys():
        raise TransformError(f"Invalid transform, must be one of: {list(TRANSFORMERS.keys())}")
    method = TRANSFORMERS.get(method)(**kwargs)
    if return_transformer:
        x = method.scale(data=data, features=features)
        return x, method
    return method.scale(data=data, features=features)
