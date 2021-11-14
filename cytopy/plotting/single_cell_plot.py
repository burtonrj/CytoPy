#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module houses plotting functions for global views of an Experiment data, for example
single cell or cluster centroid plots after dimension reduction has been performed.

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
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.colors as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.cm import ScalarMappable

from cytopy.data.read_write import polars_to_pandas

logger = logging.getLogger(__name__)


def discrete_palette(n: int) -> str:
    if n <= 10:
        return "tab10"
    if n <= 20:
        return "tab20"
    logger.warning(
        "Max number of unique labels is greater than the maximum number of colours (20) provided for "
        "scatterplot - may generate a misleading plot with colours duplicated!"
    )
    return "tab20"


def discrete_label(data: pd.DataFrame, label: str, discrete: Optional[bool] = None):
    if discrete is None:
        if pd.api.types.is_numeric_dtype(data[label]):
            return False
        discrete = True
    if discrete:
        data[label] = data[label].astype(str)
        return True
    return False


def scatterplot_defaults(**kwargs) -> Dict:
    updated_kwargs = {k: v for k, v in kwargs.items()}
    defaults = {"alpha": 0.75, "linewidth": 0, "edgecolors": None, "s": 5}
    for k, v in defaults.items():
        if k not in updated_kwargs.keys():
            updated_kwargs[k] = v
    return updated_kwargs


def single_cell_plot(
    data: Union[pl.DataFrame, pd.DataFrame],
    x: str,
    y: str,
    label: Optional[str] = None,
    discrete: Optional[bool] = None,
    hue_norm: Optional[Union[plt.Normalize, Tuple[int, int]]] = (0, 1),
    include_legend: bool = False,
    palette: Optional[str] = None,
    size: Optional[Union[int, str]] = 5,
    style: Optional[str] = None,
    legend_kwargs: Optional[Dict] = None,
    cbar_kwargs: Optional[Dict] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
    xscale: Optional[str] = None,
    yscale: Optional[str] = None,
    **kwargs,
):
    if isinstance(data, pl.DataFrame):
        data = polars_to_pandas(data=data)
    if label is not None:
        discrete = discrete_label(data=data, label=label, discrete=discrete)
    if palette is None:
        if discrete and label is not None:
            palette = discrete_palette(n=data[label].nunique())
        else:
            palette = "coolwarm"
    if size is not None:
        if isinstance(size, int):
            kwargs["s"] = size
        else:
            kwargs.pop("s", None)

    kwargs = scatterplot_defaults(**kwargs)
    cbar_kwargs = cbar_kwargs or {}
    legend_kwargs = legend_kwargs or {}

    data = data.dropna(axis=1, how="any")
    ax = ax or plt.subplots(figsize=figsize)[1]

    ax = sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=label,
        size=size if isinstance(size, str) else None,
        style=style,
        palette=palette,
        hue_norm=hue_norm,
        legend=include_legend,
        ax=ax,
        **kwargs,
    )

    if not discrete:
        if isinstance(hue_norm, tuple):
            hue_norm = cm.Normalize(*hue_norm)
        plt.gcf().colorbar(ScalarMappable(norm=hue_norm, cmap=palette), ax=ax, **cbar_kwargs)

    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_xscale(yscale)

    if discrete:
        if include_legend:
            ax.legend(*ax.get_legend_handles_labels(), **legend_kwargs)
        else:
            ax.legend().remove()

    return ax


def single_cell_density(
    data: Union[pl.DataFrame, pd.DataFrame],
    x: str,
    y: str,
    bins: Optional[int] = None,
    palette: str = "jet",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
    norm: Optional[plt.Normalize] = None,
    **kwargs,
):
    bins = bins if isinstance(bins, int) else int(np.sqrt(data.shape[0]))
    ax = ax or plt.subplots(figsize=figsize)[1]
    norm = norm or cm.LogNorm()
    ax.autoscale(enable=True)
    ax.hist2d(data[x], data[y], bins=bins, cmap=palette, norm=norm, **kwargs)
    return ax
