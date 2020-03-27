from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd


def make_example_date(n_samples=100, centers=3, n_features=2):
    blobs = make_blobs(n_samples=n_samples,
                       centers=centers,
                       n_features=n_features,
                       random_state=42)
    columns = [f'feature{i}' for i in range(n_features)]
    columns = columns + ['blobID']
    example_data = pd.DataFrame(np.hstack((blobs[0], blobs[1].reshape(-1, 1))),
                                columns=columns)
    return example_data
