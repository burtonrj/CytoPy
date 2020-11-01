#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
CytoPy supports the following dimension reduction methods: UMAP, tSNE,
PCA, Kernel PCA, and PHATE. These are implemented through the dim_reduction
function. This takes a dataframe of single cell events and generates the
desired number of embeddings. These are returned as a matrix or
as appended columns to the given dataframe.

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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from umap import UMAP
import numpy as np
import pandas as pd
import phate

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def dimensionality_reduction(data: pd.DataFrame,
                             features: list,
                             method: str,
                             n_components: int,
                             return_embeddings_only: bool = False,
                             return_reducer: bool = False,
                             **kwargs) -> pd.DataFrame or np.array:
    """
    Perform dimensionality reduction using either UMAP, PCA, tSNE, or PHATE. PCA and tSNE are implemented using
    the Scikit-Learn machine learning library.
    Documentation for UMAP can be found here: https://umap-learn.readthedocs.io/en/latest/
    Documentation for PHATE can be found here: https://phate.readthedocs.io/en/stable/

    Parameters
    -----------
    data: Pandas.DataFrame
        Events to perform dim reduction on
    features: list
        column names for feature space
    method: str
        method to use; either UMAP, PCA, tSNE, or PHATE
    n_components: int
        number of components to generate
    return_embeddings_only: bool, (default=True)
        if True, the embeddings are returned as a numpy array, otherwise original dataframe
        is returned modified with new columns, one for each embedding (column name of format {Method}_{i}
        where i = 0 to n_components)
    return_reducer: bool, (default=False)
        If True, returns instance of dimensionality reduction object
    kwargs:
        keyword arguments to pass to chosen dim reduction method

    Returns
    --------
    (Pandas.DataFrame or Numpy.array) or (Pandas.DataFrame or Numpy.array, Reducer)
        Embeddings as numpy array or original DataFrame with new columns for embeddings
    """
    data = data.copy()
    if method == 'UMAP':
        reducer = UMAP(random_state=42, n_components=n_components, **kwargs)
    elif method == 'PCA':
        reducer = PCA(random_state=42, n_components=n_components, **kwargs)
    elif method == 'tSNE':
        reducer = TSNE(random_state=42, n_components=n_components, **kwargs)
    elif method == 'PHATE':
        reducer = phate.PHATE(random_state=42, n_jobs=-2, n_components=n_components, **kwargs)
    elif method == 'KernelPCA':
        reducer = KernelPCA(random_state=42, n_components=n_components, **kwargs)
    else:
        raise ValueError("Error: invalid method given for plot clusters, "
                         "must be one of: 'UMAP', 'tSNE', 'PCA', 'PHATE', 'KernelPCA'")
    embeddings = reducer.fit_transform(data[features])
    if return_embeddings_only:
        if return_reducer:
            return embeddings, reducer
        return embeddings
    for i, e in enumerate(embeddings.T):
        data[f'{method}{i+1}'] = e
    if return_reducer:
        return data, reducer
    return data
