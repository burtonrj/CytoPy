#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module houses some two convenient functions for wrapping the
Scikit-Learn implementation of K nearest neighbours classification
algorithm.

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
from typing import Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from ..data.read_write import polars_to_pandas


def calculate_optimal_neighbours(x: Union[pl.DataFrame, pd.DataFrame], y: np.array, scoring: str, **kwargs):
    """
    Calculate the optimal n_neighbours parameter for KNeighborsClassifier using GridSearchCV.
    Returns optimal n and highest score

    Parameters
    ----------
    x: polars.DataFrame
    y: np.array
    scoring: str
    kwargs: dict

    Returns
    -------
    int, float

    Raises
    ------
    AssertionError
        Less than 5 observations provided
    """
    x = x if isinstance(x, pd.DataFrame) else polars_to_pandas(data=x)
    assert x.shape[0] > 5, "Less than 5 observations in data provided for fit"
    max_ = 500
    if (max_ - 5) > x.shape[0]:
        max_ = 250
    n = np.arange(5, max_, 10, dtype=np.int64)
    knn_ = KNeighborsClassifier(**kwargs)
    grid_cv = GridSearchCV(knn_, {"n_neighbors": n}, scoring=scoring, n_jobs=-1, cv=10)
    grid_cv.fit(x.values, y)
    return grid_cv.best_params_.get("n_neighbors"), grid_cv.best_score_


def knn(
    data: Union[pl.DataFrame, pd.DataFrame],
    labels: np.array,
    features: list,
    n_neighbours: int,
    holdout_size: float = 0.2,
    random_state: int = 42,
    return_model: bool = False,
    **kwargs
):
    """
    Train a nearest neighbours classifier (scikit-learn implementation) and return
    the balanced accuracy score for both training and validation.

    Parameters
    ----------
    data: polars.DataFrame
    labels: numpy.ndarray
    features: list
    n_neighbours: int
    holdout_size: float (default=0.2)
    random_state: int (default=42)
    return_model: bool (default=False)
    kwargs: dict
        Keyword arguments passed to KNeighborsClassifier initialisation

    Returns
    -------
    (float, float) or (float, float, object)
        Training balanced accuracy score, Validation balanced accuracy score,
        Classifier (if return_model is True)
    """
    data = data if isinstance(data, pd.DataFrame) else polars_to_pandas(data=data)
    x_train, x_test, y_train, y_test = train_test_split(
        data[features].values, labels, test_size=holdout_size, random_state=random_state
    )
    knn_ = KNeighborsClassifier(n_neighbors=n_neighbours, **kwargs)
    knn_.fit(x_train, y_train)
    train_acc = balanced_accuracy_score(y_pred=knn_.predict(x_train), y_true=y_train)
    val_acc = balanced_accuracy_score(y_pred=knn_.predict(x_test), y_true=y_test)
    if return_model:
        return train_acc, val_acc, knn_
    return train_acc, val_acc
