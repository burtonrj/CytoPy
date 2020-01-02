from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap. import UMAP
import numpy as np
import pandas as pd
import phate


def dimensionality_reduction(data: pd.DataFrame, features: list,
                             method: str, n_components: int, return_embeddings_only: bool = False,
                             **kwargs) -> pd.DataFrame or np.array:
    """
    Perform dimensionality reduction using either UMAP, PCA, tSNE, or PHATE. PCA and tSNE are implemented using
    the Scikit-Learn machine learning library.
    Documentation for UMAP can be found here: https://umap-learn.readthedocs.io/en/latest/
    Documentation for PHATE can be found here: https://phate.readthedocs.io/en/stable/
    :param data: Pandas DataFrame of events to perform dim reduction on
    :param features: column names for feature space
    :param method: method to use; either UMAP, PCA, tSNE, or PHATE
    :param n_components: number of components to generate
    :param return_embeddings_only: if True, the embeddings are returned as a numpy array, otherwise original dataframe
    is returned modified with new columns, one for each embedding
    (column name of format {Method}_{i} where i = 0 to n_components)
    :param kwargs: keyword arguments to pass to chosen dim reduction method
    :return: Embeddings as numpy array or original DataFrame with new columns for embeddings
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
    else:
        raise ValueError("Error: invalid method given for plot clusters, "
                         "must be one of: 'UMAP', 'tSNE', 'PCA', 'PHATE'")
    embeddings = reducer.fit_transform(data[features])
    if return_embeddings_only:
        return embeddings
    for i, e in enumerate(embeddings.T):
        data[f'{method}_{i}'] = e
    return data
