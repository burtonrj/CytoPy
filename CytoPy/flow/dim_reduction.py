from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
from umap import UMAP
import numpy as np
import pandas as pd
import phate


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
        return embeddings
    for i, e in enumerate(embeddings.T):
        data[f'{method}_{i}'] = e
    if return_reducer:
        if return_embeddings_only:
            return embeddings, reducer
        return data, reducer
    return data
