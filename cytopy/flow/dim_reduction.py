#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
cytopy supports the following dimension reduction methods: UMAP, tSNE,
PCA, Kernel PCA, and PHATE. These are implemented through the dim_reduction
function. This takes a dataframe of single cell events and generates the
desired number of embeddings. These are returned as a matrix or
as appended columns to the given dataframe.

If you would like to contribute to cytopy to expand the supported dimension
reduction methods, please contact us at burtonrj@cardiff.ac.uk

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
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
import phate
import polars as pl
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from umap import UMAP

from ..feedback import progress_bar
from .sampling import sample_dataframe

logger = logging.getLogger(__name__)


class DimensionReduction:
    """
    Dimension reduction methods with in-built support for:

    * UMAP
    * t-SNE
    * PCA
    * Kernel PCA
    * Multidimensional scaling (MDS)
    * Isomap
    * PHATE

    You can provide your own custom method by providing a class to the 'method' parameter, so long as that
    class has a 'fit_transform' function defined, with optionally 'fit' and 'transform' also defined. These methods
    should accept a polars DataFrame and a list of columns (features). The 'transform' and 'fit_transform' functions
    must return a polars DataFrame with embeddings as new columns.

    Parameters
    -----------
    method: str or custom type
        Method to use for dimension reduction (see DimensionReduction.base_methods)
    n_components: int (Default=2)
        Number of embeddings to retain
    random_state: int (default=42)
    kwargs:
        Additional keyword arguments passed to base method

    Attributes
    ----------
    method: Object
        Reducer object with type of requested method
    embeddings: None or Numpy.Array
        Embeddings generated from fit_transform method
    """

    base_methods = {
        "UMAP": UMAP,
        "PCA": PCA,
        "tSNE": TSNE,
        "PHATE": phate.PHATE,
        "KernelPCA": KernelPCA,
        "MDS": MDS,
        "Isomap": Isomap,
    }

    def __init__(self, method: Union[str, Type], n_components: int = 2, **kwargs):
        params = dict(n_components=n_components)
        params = {**params, **kwargs}
        try:
            if isinstance(method, str):
                self.method = self.base_methods[method](**params)
        except KeyError:
            raise KeyError(
                f"Invalid method, must be one of: {self.base_methods.keys()} or a valid class with "
                f"method: fit_transform"
            )
        except TypeError as e:
            logger.error(f"Type error when initiating dim reduction method {method}; invalid argument")
            logger.exception(e)
            raise TypeError(
                f"Type error when initiating dim reduction method {method}; invalid argument",
                e,
            )
        self.embeddings = None
        self._method_name = type(self.method).__name__

    def fit(self, data: Union[pl.DataFrame, pd.DataFrame], features: List[str]) -> Union[None, pl.DataFrame]:
        """
        Fit the underlying method. Will call 'fit_transform' if fit is not supported.

        Parameters
        ----------
        data: polars.DataFrame
        features: List[str]
            List of features (columns) to use

        Returns
        -------
        None or polars.DataFrame
            If fit is not supported, will returns a polars DataFrame.
        """
        if not hasattr(self.method, "fit"):
            logger.warning(f"Method {self._method_name} has no method 'fit', calling 'fit_transform' instead.")
            return self.fit_transform(data=data, features=features)
        if isinstance(data, pl.DataFrame):
            self.method.fit(data[features].to_numpy())
        else:
            self.method.fit(data[features].values)

    def fit_transform(self, data: Union[pl.DataFrame, pd.DataFrame], features: List[str]) -> pl.DataFrame:
        """
        Fit the underlying method and generate transformed embeddings. Transformed embeddings are
        stored as new columns in the polars DataFrame. DataFrame is copied and not mutated.

        Parameters
        ----------
        data: polars.DataFrame
        features: List[str]
            List of features (columns) to use

        Returns
        -------
        polars.DataFrame
        """
        if isinstance(data, pl.DataFrame):
            x = data[features].to_numpy()
        else:
            x = data[features].values
        self.embeddings = self.method.fit_transform(x)
        for i, e in enumerate(self.embeddings.T):
            data[f"{self._method_name}{i + 1}"] = e
        return data

    def transform(self, data: Union[pl.DataFrame, pd.DataFrame], features: List[str]) -> pl.DataFrame:
        """
        Generate embeddings for the given DataFrame using the current fitted method. Transformed embeddings are
        stored as new columns in the polars DataFrame. DataFrame is copied and not mutated.

        Will call 'fit_transform' if fit is not supported.

        Parameters
        ----------
        data: polars.DataFrame
        features: List[str]
            List of features (columns) to use

        Returns
        -------
        polars.DataFrame
        """
        if not hasattr(self.method, "transform"):
            logger.warning(f"Method {self._method_name} has no method 'transform', calling 'fit_transform' instead.")
            return self.fit_transform(data=data, features=features)

        embeddings = self.method.transform(data[features].to_numpy())
        for i, e in enumerate(embeddings.T):
            data[f"{self._method_name}{i + 1}"] = e
        return data


def dimension_reduction_with_sampling(
    data: Union[pd.DataFrame, pl.DataFrame],
    features: List[str],
    method: str,
    sampling_size: Union[int, float],
    sampling_method: str = "uniform",
    sampling_kwargs: Optional[Dict] = None,
    verbose: bool = True,
    **kwargs,
) -> Union[pd.DataFrame, DimensionReduction]:
    sampling_kwargs = sampling_kwargs or {}
    sample = sample_dataframe(data=data, sample_size=sampling_size, method=sampling_method, **sampling_kwargs)
    reducer = DimensionReduction(method=method, **kwargs)
    reducer.fit(data=sample, features=features)
    data_with_embeddings = reducer.transform(data=sample, features=features)
    remaining_data = data[~data.index.isin(sample.index)]
    for _, df in progress_bar(
        remaining_data.groupby(np.arange(remaining_data.shape[0]) // sampling_size), verbose=verbose
    ):
        data_with_embeddings = pd.concat([data_with_embeddings, reducer.transform(data=df, features=features)])
    return data_with_embeddings, reducer
