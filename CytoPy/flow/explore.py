#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Once the single cell populations of the FileGroups of an Experiment
have been classified, you might want to visualise and explore this
space to better understand the phenotype of your subjects. You might
also want to compare different classification/clustering techniques
when identifying single cell subtypes. Whilst doing so, it is helpful
to associate our data to experimental/clinical meta-data as to not loose
sight of the overarching question that is at the center of our analysis.
This functionality is provided by the Explorer class, which contains
methods for the visual exploration of data.

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
from ..data.subject import Subject, bugs, hmbpp_ribo, gram_status, biology
from ..data.experiment import load_data
from ..feedback import vprint, progress_bar
from .dim_reduction import dimensionality_reduction
from mongoengine.base.datastructures import EmbeddedDocumentList
from sklearn.preprocessing import MinMaxScaler
from warnings import warn
from matplotlib.colors import LogNorm
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scprep
import os

META_VARS = ["meta_label",
             "cluster_id",
             "population_label",
             "sample_id",
             "subject_id",
             "original_index"]

SEQ_COLOURS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
               'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
               'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
               'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
                                                           'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
               'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
               'hot', 'afmhot', 'gist_heat', 'copper']


def data_loaded(func):
    def wrapper(*args, **kwargs):
        assert args[0].data is not None, f"Dataframe has not been initialised"
        return func(*args, **kwargs)

    return wrapper


def assert_column(column_name: str, data: pd.DataFrame):
    assert column_name in data.columns, f"{column_name} missing from dataframe"


def _set_palette(discrete: bool,
                 palette: str):
    if discrete:
        if palette not in SEQ_COLOURS:
            warn("Palette invalid for discrete labelling, defaulting to 'inferno'")
            return "inferno"
    return palette


class Explorer:
    """
    Visualisation class for exploring the results of autonomous gate,
    cell classification, and clustering though dimension reduction,
    interactive scatter plots and customizable heatmaps.
    Explorer is freeform and only requires that data be provided in
    the form of a Pandas DataFrame, where the only requirement is
    that a column named "subject_id" is present; this should be a row
    identifier relating to a subjects unique ID, linking to their
    Subject document in the database.

    Attributes
    -----------
    data: Pandas.DataFrame (optional)
        A DataFrame for visualisation. If not provided, then a path to an existing dataframe should be provided.
    verbose: bool (default=True)
        Whether to provide feedback
    """

    def __init__(self,
                 data: pd.DataFrame or None = None,
                 verbose: bool = True):
        self.data = None
        if data is not None:
            self.data = data
        self.meta_vars = [i for i in META_VARS if i in data.columns]
        self.verbose = verbose
        self.print = vprint(verbose)

    def load_from_file(self,
                       key: str,
                       path: str,
                       **kwargs):
        """
        Load either primary data or summarised data saved to disk (specified by 'key'
        Parameters
        ----------
        key
        path
        kwargs

        Returns
        -------
        None
        """
        self.data = pd.read_csv(path, **kwargs)
        self.meta_vars = [i for i in META_VARS if i in self.data.columns]

    def load_data(self,
                  **kwargs):
        self.data = load_data(**kwargs)
        self.meta_vars = [i for i in META_VARS if i in self.data.columns]

    @data_loaded
    def mask_data(self,
                  mask: pd.DataFrame,
                  save: bool = True) -> None:
        """
        Update contained dataframe according to a given mask.

        Parameters
        ----------
        mask : Pandas.DataFrame
            Valid pandas dataframe mask
        Returns
        -------
        None
        """
        if not save:
            return self.data[mask]
        self.data = self.data[mask]
        return None

    @data_loaded
    def _summarise(self,
                   grp_keys: list,
                   features: list,
                   summary_method: str = "median"):
        if summary_method == "mean":
            return self.data.groupby(by=grp_keys)[features].mean().reset_index()
        if summary_method == "median":
            return self.data.groupby(by=grp_keys)[features].median().reset_index()
        raise ValueError("Summary method should be 'median' or 'mean'")

    def summarise_clusters(self,
                           features: list,
                           identifier: str = "sample_id",
                           summary_method: str = "median"):
        assert identifier in ["sample_id", "subject_id"], "identifier should be 'sample_id' or 'subject_id'"
        return self._summarise(grp_keys=[identifier, "cluster_id"], features=features, summary_method=summary_method)

    def summarise_metaclusters(self,
                               features: list,
                               summary_method: str = "median"):
        return self._summarise(grp_keys=["meta_label"], features=features, summary_method=summary_method)

    def summarise_populations(self,
                              features: list,
                              identifier: str = "sample_id",
                              summary_method: str = "median"):
        assert identifier in ["sample_id", "subject_id"], "identifier should be 'sample_id' or 'subject_id'"
        return self._summarise(grp_keys=[identifier, "population_label"], features=features,
                               summary_method=summary_method)

    @data_loaded
    def save(self,
             path: str) -> None:
        f"""
        Save the contained dataframe to a new csv file

        Parameters
        ----------
        path: str
            Output path for csv file
        Returns
        -------
        None
        """
        self.data.to_csv(path)

    @data_loaded
    def load_meta(self,
                  variable: str) -> None:
        """
        Load meta data for each subject. Must be provided with a variable that is a field with a single value
        NOT an embedded document. A column will be generated in the Pandas DataFrame stored in the attribute 'data'
        that pertains to the variable given and the value will correspond to that of the subjects.

        Parameters
        ----------
        variable : str
            field name to populate data with

        Returns
        -------
        None
        """
        self.data[variable] = None
        self.meta_vars.append(variable)
        for _id in progress_bar(self.data.subject_id.unique(),
                                verbose=self.verbose):
            if _id is None:
                continue
            p = Subject.objects(subject_id=_id).get()
            try:
                assert type(p[variable]) != EmbeddedDocumentList, \
                    'Chosen variable is an embedded document.'
                self.data.loc[self.data.subject_id == _id, variable] = p[variable]
            except KeyError:
                warn(f'{_id} is missing meta-variable {variable}')
                self.data.loc[self.data.subject_id == _id, variable] = None

    @data_loaded
    def load_infectious_data(self,
                             multi_org: str = 'list'):
        """
        Load the bug data from each subject and populate 'data' accordingly.
        As default variables will be created as follows:

        * organism_name = If 'multi_org' equals 'list' then multiple
          organisms will be stored as a comma separated list without
          duplicates, whereas if the value is 'mixed' then
          multiple organisms will result in a value of 'mixed'.
        * organism_type = value of either 'gram positive', 'gram negative', 'virus', 'mixed' or 'fungal'
        * hmbpp = True or False based on HMBPP status 
          (Note: it only takes one positive organism for this value to be True)
        * ribo = True or False based on Ribo status 
          (Note: it only takes one positive organism for this value to be True)
        
        Parameters
        ----------
        multi_org: str (Default value = 'list')
        
        Returns
        -------
        None
        """
        inf_vars = ['organism_name',
                    'gram_status',
                    'organism_name_short',
                    'hmbpp',
                    'ribo']
        for variable in inf_vars:
            self.meta_vars.append(variable)
            self.data[variable] = 'Unknown'

        for subject_id in progress_bar(self.data.subject_id.unique()):
            if subject_id is None:
                continue
            p = Subject.objects(subject_id=subject_id).get()
            self.data.loc[self.data.subject_id == subject_id, 'organism_name'] = bugs(subject=p, multi_org=multi_org)
            self.data.loc[self.data.subject_id == subject_id, 'organism_name_short'] = bugs(subject=p,
                                                                                            multi_org=multi_org,
                                                                                            short_name=True)
            self.data.loc[self.data.subject_id == subject_id, 'hmbpp'] = hmbpp_ribo(subject=p, field='hmbpp_status')
            self.data.loc[self.data.subject_id == subject_id, 'ribo'] = hmbpp_ribo(subject=p, field='ribo_status')
            self.data.loc[self.data.subject_id == subject_id, 'gram_status'] = gram_status(subject=p)

    @data_loaded
    def load_biology_data(self,
                          test_name: str,
                          summary_method: str = 'average'):
        """
        Load the pathology results of a given test from each subject and populate 'data' accordingly. As multiple
        results may exist for one particular test, a summary method should be provided, this should have a value as
        follows:
        
        * average - the average test result is generated and stored
        * max - the maximum value is stored
        * min - the minimum value is stored
        * median - the median test result is generated and stored

        Parameters
        ----------
        test_name: str
            Name of test to load
        summary_method: str, (Default value = 'average')
        
        Returns
        -------
        None
        """
        self.meta_vars.append(test_name)
        self.data[test_name] = self.data['subject_id'].apply(lambda x: biology(x, test_name, summary_method))

    @data_loaded
    def dimenionality_reduction(self,
                                method: str,
                                features: list,
                                data: pd.DataFrame or None = None,
                                n_components: int = 2,
                                **kwargs) -> None or pd.DataFrame:
        """
        Performs dimensionality reduction and saves the result to the contained dataframe. Resulting embeddings
        are saved to columns named as follows: {method}_{n} where n is an integer in range 0 to n_components

        Parameters
        ----------
        method : str
            method to use for dimensionality reduction; valid methods are 'tSNE', 'UMAP', 'PHATE' or 'PCA'
        features : list
            list of features to use for dimensionality reduction
        n_components : int, (default = 2)
            number of components to generate
        overwrite : bool, (default = False)
            If True, existing embeddings will be overwritten (default = False)
        kwargs :
            additional keyword arguments to pass to dim reduction algorithm (see flow.dim_reduction)

        Returns
        -------
        None or Pandas.DataFrame
        """
        if data is None:
            self.data = dimensionality_reduction(self.data,
                                                 features,
                                                 method,
                                                 n_components,
                                                 return_embeddings_only=False,
                                                 return_reducer=False,
                                                 **kwargs)
            for variable in [f"{method}{i + 1}" for i in range(n_components)]:
                if variable not in self.meta_vars:
                    self.meta_vars.append(variable)
            return None
        return dimensionality_reduction(data,
                                        features,
                                        method,
                                        n_components,
                                        return_embeddings_only=False,
                                        return_reducer=False,
                                        **kwargs)

    def _scatterplot_defaults(self,
                              **kwargs):
        updated_kwargs = {k: v for k, v in kwargs.items()}
        defaults = {"edgecolor": "black",
                    "alpha": 0.75,
                    "linewidth": 2,
                    "s": 10}
        for k, v in defaults.items():
            if k not in updated_kwargs.keys():
                updated_kwargs[k] = v
        return updated_kwargs

    def single_cell_plot(self,
                         label: str,
                         features: list,
                         discrete: bool,
                         n_components: int = 2,
                         method: str = "UMAP",
                         dim_reduction_kwargs: dict or None = None,
                         figsize: tuple = (12, 8),
                         palette: str = "tab20",
                         colourbar_kwargs: dict or None = None,
                         legend_kwargs: dict or None = None,
                         **kwargs):
        palette = _set_palette(discrete=discrete, palette=palette)
        colourbar_kwargs = colourbar_kwargs or {}
        legend_kwargs = legend_kwargs or {"bbox_to_anchor": (1.15, 1)}
        assert n_components in [2, 3], 'n_components must have a value of 2 or 3'
        assert label in self.data.columns, f'{label} is not valid, must be an existing column in linked dataframe'

        embeddings = [f"{method}{i + 1}" for i in range(n_components)]
        if not all([x in self.data.columns for x in embeddings]):
            dim_reduction_kwargs = dim_reduction_kwargs or {}
            self.dimenionality_reduction(method=method,
                                         n_components=n_components,
                                         features=features,
                                         **dim_reduction_kwargs)

        fig, ax = plt.subplots(figsize=figsize)
        kwargs = self._scatterplot_defaults(**kwargs)
        if not discrete:
            im = ax.scatter(self.data[f"{method}1"], self.data[f"{method}2"],
                            c=self.data[label], cmap=palette, **kwargs)
            ax.set_xlabel(f"{method}1")
            ax.set_ylabel(f"{method}2")
            fig.colorbar(im, ax=ax, **colourbar_kwargs)
            return ax
        colours = plt.get_cmap(palette)
        for i, (l, df) in enumerate(self.data.groupby(label)):
            ax.scatter(df[f"{method}1"], df[f"{method}2"], c=colours[i], label=l, **kwargs)
        ax.set_xlabel(f"{method}1")
        ax.set_ylabel(f"{method}2")
        ax.legend(**legend_kwargs)
        return ax

    def cluster_plot(self,
                     label: str,
                     features: list,
                     discrete: bool,
                     mask: pd.DataFrame or None = None,
                     n_components: int = 2,
                     method: str = "UMAP",
                     dim_reduction_kwargs: dict or None = None,
                     scale_factor: int = 100,
                     figsize: tuple = (12, 8),
                     palette: str = "tab20",
                     colourbar_kwargs: dict or None = None,
                     legend_kwargs: dict or None = None,
                     **kwargs):
        palette = _set_palette(discrete=discrete, palette=palette)
        colourbar_kwargs = colourbar_kwargs or {}
        legend_kwargs = legend_kwargs or {"bbox_to_anchor": (1.15, 1)}
        assert n_components in [2, 3], 'n_components must have a value of 2 or 3'
        assert label in self.data.columns, f'{label} is not valid, must be an existing column in linked dataframe'
        data = self.summarise_clusters(features=features, identifier="sample_id", summary_method="median")
        data = self.dimenionality_reduction(method=method,
                                            data=data,
                                            n_components=n_components,
                                            features=features,
                                            **dim_reduction_kwargs)
        data = self._assign_labels(data=data, label=label)
        data["cluster_size"] = self._cluster_size()
        fig, ax = plt.subplots(figsize=figsize)
        kwargs = self._scatterplot_defaults(**kwargs)
        kwargs["s"] =
        if not discrete:
            im = ax.scatter(self.data[f"{method}1"],
                            self.data[f"{method}2"],
                            c=self.data[label],
                            cmap=palette,
                            **kwargs)
            ax.set_xlabel(f"{method}1")
            ax.set_ylabel(f"{method}2")
            fig.colorbar(im, ax=ax, **colourbar_kwargs)
            return ax
        colours = plt.get_cmap(palette)
        for i, (l, df) in enumerate(self.data.groupby(label)):
            ax.scatter(df[f"{method}1"], df[f"{method}2"], c=colours[i], label=l, **kwargs)
        ax.set_xlabel(f"{method}1")
        ax.set_ylabel(f"{method}2")
        ax.legend(**legend_kwargs)
        return ax

    def _cluster_size(self):
        return self.data.groupby("sample_id")["cluster_id"].value_counts()

    def _sample_size(self):
        return self.data.sample_id.value_counts()

    def _assign_labels(self,
                       data: pd.DataFrame,
                       label: str):
        for (cid, sid), df in data.groupby(["sample_id", "cluster_id"]):
            x = self.data[(self.data["cluster_id"] == cid) & (self.data["sample_id"] == sid)]
            assert len(x[label].unique()) == 1, "Chosen label is not unique within clusters"
            data.loc[df.index, label] = x[label].values[0]
        return data

    def cluster_graph(self):
        pass

    def _plotting_labels(self,
                         label: str,
                         populations: list or None) -> list:
        """
        Generates a list of values to be used for colouring data points in plot

        Parameters
        ----------
        label : str
            column name for values to use for colour
        populations : list, optional
            if label = 'population_label', a list of populations can be specified, the resulting
            list of labels is filtered to contain only populations in this list
        Returns
        -------
        List
            list of labels
        """
        if label == 'population_label':
            if populations is None:
                return self.data[label].values
            return list(map(lambda x: x if x in populations else 'None', self.data['population_label'].values))
        else:
            return list(self.data[label].values)

    def scatter_plot(self,
                     label: str,
                     features: list,
                     discrete: bool,
                     populations: list or None = None,
                     n_components: int = 2,
                     dim_reduction_method: str = 'UMAP',
                     mask: pd.DataFrame or None = None,
                     scale_factor: int = 100,
                     figsize: tuple = (12, 8),
                     dim_reduction_kwargs: dict or None = None,
                     matplotlib_kwargs: dict or None = None) -> plt.Axes:
        """
        Generate a 2D/3D scatter plot (dimensions depends on the number of components chosen for dimension
        reduction. Each data point is labelled according to the option provided to the label arguments. If a value
        is given to both primary and secondary label, the secondary label colours the background and the primary label
        colours the foreground of each datapoint.

        Parameters
        ----------
        label : str
            option for the primary label, must be a valid column name in Explorer attribute 'data'
            (check valid column names using Explorer.data.columns)
        features : list
            list of column names used as feature space for dimensionality reduction
        discrete : bool
            Are the labels for this plot discrete or continuous? If True, labels will be treated as
            discrete, otherwise labels will be coloured using a gradient and a colourbar will be provided.
        populations : list, optional
            if label has value of 'population_label', only populations in this
            list will be included (events with no population associated will be labelled 'None')
        n_components : int, (default=2)
            number of components to produce from dimensionality reduction, valid values are 2 or 3
        dim_reduction_method :str, (default = 'UMAP')
            method to use for dimensionality reduction, valid values are 'UMAP' or 'PHATE'
        mask : Pandas.DataFrame, optional
            a valid Pandas DataFrame mask to subset data prior to plotting
        scale_factor : int, (default=100)
            Scale factor defines the size of datapoints;
            size = meta_scale_factor * proportion of events in cluster relative to root population
        figsize: tuple (default=(12,8)
            Figure size
        dim_reduction_kwargs : dict, optional
            additional keyword arguments to pass to dimensionality reduction algorithm
        matplotlib_kwargs : dict, optional
            additional keyword arguments to pass to matplotlib call
        Returns
        -------
        matplotlib.axes
        """

        assert n_components in [2, 3], 'n_components must have a value of 2 or 3'
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        matplotlib_kwargs = matplotlib_kwargs or {}
        # Dimensionality reduction
        self.dimenionality_reduction(method=dim_reduction_method,
                                     features=features,
                                     n_components=n_components,
                                     **dim_reduction_kwargs)
        embedding_cols = [f'{dim_reduction_method}{i + 1}' for i in range(n_components)]

        # Label and plotting
        assert label in self.data.columns, f'{label} is not a valid entry, valid labels include: ' \
                                           f'{self.data.columns.tolist()}'
        plabel = self._plotting_labels(label, populations)
        data = self.data.copy()
        if mask is not None:
            data = data[mask]
            plabel = np.array(plabel)[data.index.values]

        size = 10
        if label == "cluster_id" or label == "meta_label":
            assert "cluster_size" in data.columns, "'cluster_size' missing. Generate Explorer object from " \
                                                   "Clustering object by calling 'explore'"
            size = data["cluster_size"] * scale_factor
        if n_components == 2:
            return scprep.plot.scatter2d(data[embedding_cols],
                                         c=plabel,
                                         ticks=False,
                                         label_prefix=dim_reduction_method,
                                         s=size,
                                         discrete=discrete,
                                         legend_loc="lower left",
                                         legend_anchor=(1.04, 0),
                                         legend_title=label,
                                         figsize=figsize,
                                         **matplotlib_kwargs)
        return scprep.plot.scatter3d(data[embedding_cols],
                                     c=plabel,
                                     ticks=False,
                                     size=size,
                                     label_prefix=dim_reduction_method,
                                     discrete=discrete,
                                     legend_loc="lower left",
                                     legend_anchor=(1.04, 0),
                                     legend_title=label,
                                     figsize=figsize,
                                     **matplotlib_kwargs)

    def heatmap(self,
                heatmap_var: str,
                features: list,
                clustermap: bool = False,
                mask: pd.DataFrame or None = None,
                normalise: bool = True,
                summary_func: callable = np.median,
                figsize: tuple or None = (10, 10),
                title: str or None = None,
                col_cluster: bool = False,
                **kwargs):
        """
        Generate a heatmap of marker expression for either clusters or gated populations
        (indicated with 'heatmap_var' argument)

        Parameters
        ----------
        heatmap_var : str
            variable to use, either 'global clusters' or 'gated populations'
        features : list
            list of column names to use for generating heatmap
        clustermap : bool, (default=False)
            if True, rows (clusters/populations) are grouped by single linkage clustering
        mask : Pandas.DataFrame, optional
            a valid Pandas DataFrame mask to subset data prior to plotting (optional)
        normalise : bool, (default=True)
            if True, data is normalised prior to plotting (normalised using Sklean's MinMaxScaler function)
        summary_func: callable (default=Numpy.mean)
            function used to values for display in heatmap
        figsize : tuple, (default=(10,5))
            tuple defining figure size passed to matplotlib call
        title: str, optional
            figure title
        col_cluster: bool, (default=False)
            If True and clustermap is True, columns AND rows are clustered

        Returns
        -------
        matplotlib.axes
        """
        d = self.data.copy()
        if mask is not None:
            d = d[mask]
        d = d[features + [heatmap_var]]
        d[features] = d[features].apply(pd.to_numeric)
        if normalise:
            d[features] = MinMaxScaler().fit_transform(d[features])
        d = d.groupby(by=heatmap_var)[features].apply(summary_func)
        if clustermap:
            ax = sns.clustermap(d, col_cluster=col_cluster, cmap='viridis', figsize=figsize, **kwargs)
            return ax
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(d, linewidth=0.5, ax=ax, cmap='viridis', **kwargs)
        if title is not None:
            ax.set_title(title)
        return ax

    def plot_representation(self,
                            x_variable: str,
                            y_variable: str,
                            discrete: bool = True,
                            mask: pd.DataFrame or None = None,
                            figsize: tuple = (6, 6),
                            **kwargs):
        """
        Present a breakdown of how a variable is represented in relation to a
        cluster/population/meta cluster

        Parameters
        ----------
        x_variable : str
            either cluster, population or meta cluster
        y_variable : str
            variable for comparison (give a value of 'subject' for subject represenation
        discrete : bool, (default=True)
            if True, the variable is assumed to be discrete
        mask : Pandas.DataFrame, optional
            a valid Pandas DataFrame mask to subset data prior to plotting (optional)
        figsize : tuple, (default=(6,6))
            tuple defining figure size passed to matplotlib call

        Returns
        -------
        matplotlib.axes
        """
        d = self.data.copy()
        if mask is not None:
            d = self.data[mask]
        assert x_variable in ["cluster_id", "meta_label", "population_label"], f'x_variable must be one of' \
                                                                               f'"cluster_id", "meta_label", ' \
                                                                               f'"population_label"'
        if y_variable == 'subject':
            x = d[[x_variable, 'subject_id']].groupby(x_variable)['subject_id'].nunique() / len(
                d.subject_id.unique()) * 100
            fig, ax = plt.subplots(figsize=figsize)
            x.sort_values().plot(kind='bar', ax=ax, **kwargs)
            ax.set_ylabel('subject representation (%)')
            ax.set_xlabel(x_variable)
            return ax
        if y_variable in d.columns and discrete:
            x = (d[[x_variable, y_variable]].groupby(x_variable)[y_variable]
                 .value_counts(normalize=True)
                 .rename('percentage')
                 .mul(100)
                 .reset_index()
                 .sort_values(y_variable))
            fig, ax = plt.subplots(figsize=figsize)
            p = sns.barplot(x=x_variable, y=f'percentage', hue=y_variable, data=x, ax=ax, **kwargs)
            _ = plt.setp(p.get_xticklabels(), rotation=90)
            ax.set_ylabel(f'% cells ({y_variable})')
            ax.set_xlabel(x_variable)
            return ax
        if y_variable in d.columns and not discrete:
            fig, ax = plt.subplots(figsize=figsize)
            d = d.sample(10000)
            ax = sns.swarmplot(x=x_variable, y=y_variable, data=d, ax=ax, s=3, **kwargs)
            return ax

    @staticmethod
    def _no_such_pop_cluster(d, _id):
        assert d.shape[0] == 0, f'unable to subset data on {_id}'

    def plot_2d(self,
                primary_id: dict,
                x: str,
                y: str,
                secondary_id: dict or None = None,
                xlim: tuple or None = None,
                ylim: tuple or None = None,
                ax: plt.Axes or None = None) -> plt.Axes:
        """
        Plots two-dimensions as either a scatter plot or a 2D histogram (defaults to 2D histogram if number of
        data-points exceeds 1000). This can be used for close inspection of populations in contrast to their clustering
        assignments. Particularly useful for debugging or inspecting anomalies.

        Parameters
        ----------
        primary_id : dict
            variable for selection of primary dataset to plot, e.g. if primary ID is a population name,
            then only data-points belonging to that gated population will be displayed. Alternatively if it is the name of
            a cluster or meta-cluster, then only data-points assigned to that cluster will be displayed.
        secondary_id : dict, optional
            variable for selection of secondary dataset, plotted over the top of the primary dataset for comparison
        x : str
            Name of variable to plot on the x-axis
        y : str
            Name of variable to plot on the y-axis
        xlim : tuple, optional
            limit the x-axis to a given range (optional)
        ylim : tuple, optional
            limit the y-axis to a given range (optional)
        ax: Matplotlib.Axes (optional)

        Returns
        -------
        matplotlib.axes
        """

        # Checks and balances
        def check_id(id_dict, data):
            e = 'Primary/Secondary ID should be a dictionary with keys: "column_name" and "value"'
            assert all([x_ in id_dict.keys() for x_ in ['column_name', 'value']]), e
            assert id_dict['column_name'] in data.columns, f'{id_dict["column_name"]} is not a valid column name'
            assert id_dict['value'] in data[id_dict['column_name']].values, f'No such value {id_dict["value"]} in ' \
                                                                            f'{id_dict["column_name"]}'

        ax = ax or plt.subplots(figsize=(5, 5))[1]
        check_id(primary_id, self.data)
        if secondary_id is not None:
            check_id(secondary_id, self.data)
        assert all([c in self.data.columns for c in [x, y]]), 'Invalid axis; must be an existing column in dataframe'

        # Build plot
        d = self.data[self.data[primary_id['column_name']] == primary_id['value']]
        d2 = None
        if secondary_id is not None:
            d2 = self.data[self.data[secondary_id['column_name']] == secondary_id['value']]

        if d.shape[0] < 1000:
            ax.scatter(d[x], d[y], marker='o', s=1, c='b', alpha=0.8, label=primary_id['value'])
        else:
            ax.hist2d(d[x], d[y], bins=500, norm=LogNorm(), label=primary_id['value'])
        if d2 is not None:
            ax.scatter(d2[x], d2[y], marker='o', s=1, c='r', alpha=0.8, label=secondary_id['value'])

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend(fontsize=11, bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.,
                  markerscale=6)
        return ax
