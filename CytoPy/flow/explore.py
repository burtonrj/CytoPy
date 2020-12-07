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
from ..flow.descriptives import box_swarm_plot, stat_test
from ..feedback import vprint, progress_bar
from .dim_reduction import dimensionality_reduction
from mongoengine.base.datastructures import EmbeddedDocumentList
from warnings import warn
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
    if not discrete:
        if palette not in SEQ_COLOURS:
            warn("Palette invalid for discrete labelling, defaulting to 'inferno'")
            return "inferno"
    return palette


def scatterplot(data: pd.DataFrame,
                method: str,
                label: str,
                discrete: bool,
                size: str or None = None,
                scale_factor: int = 15,
                figsize: tuple = (10, 12),
                palette: str = "tab20",
                colourbar_kwargs: dict or None = None,
                legend_kwargs: dict or None = None,
                **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    colourbar_kwargs = colourbar_kwargs or {}
    legend_kwargs = legend_kwargs or {"bbox_to_anchor": (1.15, 1.)}
    palette = _set_palette(discrete=discrete, palette=palette)
    if not discrete:
        im = ax.scatter(data[f"{method}1"],
                        data[f"{method}2"],
                        c=data[label],
                        cmap=palette,
                        **kwargs)
        ax.set_xlabel(f"{method}1")
        ax.set_ylabel(f"{method}2")
        fig.colorbar(im, ax=ax, **colourbar_kwargs)
        return ax
    colours = cycle(plt.get_cmap(palette).colors)
    for l, df in data.groupby(label):
        if size is not None:
            kwargs["s"] = df[size].values * scale_factor
        ax.scatter(df[f"{method}1"], df[f"{method}2"], color=next(colours), label=l, **kwargs)
    ax.set_xlabel(f"{method}1")
    ax.set_ylabel(f"{method}2")
    ax.legend(**legend_kwargs)
    return ax


def _scatterplot_defaults(**kwargs):
    updated_kwargs = {k: v for k, v in kwargs.items()}
    defaults = {"edgecolor": "black",
                "alpha": 0.75,
                "linewidth": 2,
                "s": 5}
    for k, v in defaults.items():
        if k not in updated_kwargs.keys():
            updated_kwargs[k] = v
    return updated_kwargs


def _assert_unique_label(x):
    assert len(x) == 1, "Chosen label is not unique within clusters"
    return x[0]


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
        self.meta_vars = []
        if data is not None:
            self.data = data
            self.meta_vars = [i for i in META_VARS if i in data.columns]
        self.verbose = verbose
        self.print = vprint(verbose)

    def load_from_file(self,
                       path: str,
                       **kwargs):
        """
        Load either primary data or summarised data saved to disk (specified by 'key'
        Parameters
        ----------
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
                  save: bool = True) -> None or pd.DataFrame:
        """
        Update contained dataframe according to a given mask.

        Parameters
        ----------
        mask : Pandas.DataFrame
            Valid pandas dataframe mask
        save: bool (default=True)
        Returns
        -------
        None or Pandas.DataFrame
        """
        if not save:
            return self.data[mask]
        self.data = self.data[mask]
        return None

    @data_loaded
    def _summarise(self,
                   grp_keys: list,
                   features: list,
                   summary_method: str = "median",
                   mask: pd.DataFrame or None = None):
        data = self.data
        if mask is not None:
            data = self.mask_data(mask=mask, save=False)
        if summary_method == "mean":
            return data.groupby(by=grp_keys)[features].mean().reset_index()
        if summary_method == "median":
            return data.groupby(by=grp_keys)[features].median().reset_index()
        raise ValueError("Summary method should be 'median' or 'mean'")

    def summarise_clusters(self,
                           features: list,
                           identifier: str = "sample_id",
                           summary_method: str = "median",
                           mask: pd.DataFrame or None = None):
        assert identifier in ["sample_id", "subject_id"], "identifier should be 'sample_id' or 'subject_id'"
        return self._summarise(grp_keys=[identifier, "cluster_id"],
                               features=features,
                               summary_method=summary_method,
                               mask=mask)

    def summarise_metaclusters(self,
                               features: list,
                               summary_method: str = "median",
                               mask: pd.DataFrame or None = None):
        return self._summarise(grp_keys=["meta_label"],
                               features=features,
                               summary_method=summary_method,
                               mask=mask)

    def summarise_populations(self,
                              features: list,
                              identifier: str = "sample_id",
                              summary_method: str = "median",
                              mask: pd.DataFrame or None = None):
        assert identifier in ["sample_id", "subject_id"], "identifier should be 'sample_id' or 'subject_id'"
        return self._summarise(grp_keys=[identifier, "population_label"],
                               features=features,
                               summary_method=summary_method,
                               mask=mask)

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
        data: pd.DataFrame
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
        assert n_components in [2, 3], 'n_components must have a value of 2 or 3'
        assert label in self.data.columns, f'{label} is not valid, must be an existing column in linked dataframe'

        embeddings = [f"{method}{i + 1}" for i in range(n_components)]
        if not all([x in self.data.columns for x in embeddings]):
            dim_reduction_kwargs = dim_reduction_kwargs or {}
            self.dimenionality_reduction(method=method,
                                         n_components=n_components,
                                         features=features,
                                         **dim_reduction_kwargs)
        kwargs = _scatterplot_defaults(**kwargs)
        return scatterplot(data=self.data,
                           method=method,
                           label=label,
                           discrete=discrete,
                           figsize=figsize,
                           palette=palette,
                           colourbar_kwargs=colourbar_kwargs,
                           legend_kwargs=legend_kwargs,
                           **kwargs)

    def cluster_plot(self,
                     label: str,
                     features: list,
                     discrete: bool,
                     n_components: int = 2,
                     method: str = "UMAP",
                     dim_reduction_kwargs: dict or None = None,
                     scale_factor: int = 15,
                     figsize: tuple = (12, 8),
                     palette: str = "tab20",
                     colourbar_kwargs: dict or None = None,
                     legend_kwargs: dict or None = None,
                     **kwargs):
        assert n_components in [2, 3], 'n_components must have a value of 2 or 3'
        assert label in self.data.columns, f'{label} is not valid, must be an existing column in linked dataframe'
        dim_reduction_kwargs = dim_reduction_kwargs or {}
        data = self.summarise_clusters(features=features, identifier="sample_id", summary_method="median")
        data = self.dimenionality_reduction(method=method,
                                            data=data,
                                            n_components=n_components,
                                            features=features,
                                            **dim_reduction_kwargs)
        self._assign_labels(data=data, label=label)
        self.cluster_size(data)
        kwargs = kwargs or {}
        kwargs = _scatterplot_defaults(**kwargs)
        return scatterplot(data=data,
                           method=method,
                           label=label,
                           discrete=discrete,
                           figsize=figsize,
                           palette=palette,
                           colourbar_kwargs=colourbar_kwargs,
                           legend_kwargs=legend_kwargs,
                           size="cluster_size",
                           scale_factor=scale_factor,
                           **kwargs)

    def cluster_size(self,
                     data: pd.DataFrame):
        lookup = self._cluster_size_lookup()
        cluster_n = data[["sample_id", "cluster_id"]].apply(lambda x: lookup.loc[x["sample_id"], x["cluster_id"]],
                                                            axis=1)
        lookup = self._sample_size_lookup()
        sample_n = data["sample_id"].apply(lambda x: lookup.loc[x])
        data["sample_n"], data["cluster_n"] = sample_n, cluster_n
        data["cluster_size"] = cluster_n / sample_n * 100

    def _cluster_size_lookup(self):
        return self.data.groupby("sample_id")["cluster_id"].value_counts()

    def _sample_size_lookup(self):
        return self.data.sample_id.value_counts()

    def _assign_labels(self, data: pd.DataFrame, label: str):
        lookup = self.data.groupby(["sample_id", "cluster_id"])[label].unique().apply(_assert_unique_label)
        data[label] = data[["sample_id", "cluster_id"]].apply(lambda x: lookup.loc[x[0], x[1]], axis=1)

    def cluster_graph(self):
        pass

    def clustered_heatmap(self,
                          heatmap_var: str,
                          features: list,
                          mask: pd.DataFrame or None = None,
                          summary_method: str = "median",
                          **kwargs):
        """
        Generate a heatmap of marker expression for either clusters or gated populations
        (indicated with 'heatmap_var' argument)

        Parameters
        ----------
        heatmap_var : str
            variable to use, either "clusters", "meta clusters", or "populations"
        features : list
            list of column names to use for generating heatmap
        mask : Pandas.DataFrame, optional
            a valid Pandas DataFrame mask to subset data prior to plotting (optional)
        summary_method: str
        Returns
        -------
        matplotlib.axes
        """
        if heatmap_var == "clusters":
            data = self.summarise_clusters(identifier="sample_id",
                                           features=features,
                                           mask=mask,
                                           summary_method=summary_method).set_index("cluster_id")
        elif heatmap_var == "meta clusters":
            data = self.summarise_metaclusters(features=features,
                                               summary_method=summary_method,
                                               mask=mask).set_index("meta_label")
        elif heatmap_var == "populations":
            data = self.summarise_populations(features=features,
                                              identifier="sample_id",
                                              summary_method=summary_method,
                                              mask=mask).set_index("population_label")
        else:
            raise ValueError("heatmap var should be one of: 'clusters', 'meta clusters' or 'populations'")
        data[features] = data[features].apply(pd.to_numeric)
        kwargs = kwargs or {"col_cluster": True,
                            "figsize": (10, 15),
                            "standard_scale": 1,
                            "cmap": "viridis"}
        return sns.clustermap(data, **kwargs)

    def boxplot_and_heatmap(self,
                            group: str,
                            features: list,
                            summary_method: str = "median",
                            mask: pd.DataFrame or None = None,
                            figsize: tuple = (20, 8),
                            stats: bool = True,
                            boxplot_kwargs: dict or None = None,
                            swarmplot_kwargs: dict or None = None,
                            boxplot_palette: str or None = None,
                            clustermap_kwargs: dict or None = None):
        assert "meta_label" in self.meta_vars, "boxplot_and_heatmap requires meta-clustering been performed"
        assert group in self.meta_vars, f"Invalid meta variable (group), must be one of: {self.meta_vars}"
        # Clustermap
        plot_df = self.summarise_metaclusters(features=features,
                                              summary_method=summary_method,
                                              mask=mask)
        clustermap_kwargs = clustermap_kwargs or {"standard_scale": 1,
                                                  "cmap": "viridis"}
        g = sns.clustermap(data=plot_df.set_index("meta_label"),
                           figsize=figsize,
                           **clustermap_kwargs)
        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")
        g.gs.update(left=0.05, right=0.45)
        gs2 = GridSpec(1, 1, left=0.6)
        ax2 = g.fig.add_subplot(gs2[0])
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
        ax2.set_xlabel("")
        # BoxSwarm plot
        plot_df = self.summarise_clusters(features=features,
                                          identifier="sample_id",
                                          summary_method=summary_method,
                                          mask=mask)
        self.cluster_size(plot_df)
        self._assign_labels(plot_df, "meta_label")
        self._assign_labels(plot_df, group)
        plot_df_c = plot_df.groupby(["sample_id", "meta_label", group])["cluster_n"].sum().reset_index()
        plot_df_s = plot_df.groupby(["sample_id"])["sample_n"].sum().reset_index()
        plot_df = plot_df_c.merge(plot_df_s, on="sample_id")
        plot_df["cluster_size"] = plot_df["cluster_n"]/plot_df["sample_n"]
        box_swarm_plot(plot_df=plot_df,
                       x="meta_label",
                       y="cluster_size",
                       hue=group,
                       ax=ax2,
                       palette=boxplot_palette,
                       boxplot_kwargs=boxplot_kwargs,
                       swarmplot_kwargs=swarmplot_kwargs)
        if stats:
            stats = stat_test(plot_df,
                              group1="meta_label",
                              dep_var_name="cluster_size",
                              group2=group)
            return g, stats,
        return g

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
        d = self.data
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

    def _check_2d_plot_content(self, content: list):
        err = "'content' should be a list of dictionaries, each with the keys 'source' and 'value' where source " \
              "is a valid meta-variable e.g. cluster_id and 'value' the data to be plotted"
        assert all(isinstance(x, dict) for x in content), err
        assert all([all([i in x.keys() for i in ["source", "value"]]) for x in content]), err
        for c in content:
            assert c["source"] in self.meta_vars, f"{c['source']} is an invalid meta-variable"
            assert c['value'] in self.data[c["source"]].values, \
                f"{c['value']} not found for meta-variable {c['source']}"

    def plot_2d(self,
                x: str,
                y: str,
                content: list,
                background_type: str = "population_label",
                background: str or None = None,
                xlim: tuple or None = None,
                ylim: tuple or None = None,
                ax: plt.Axes or None = None,
                bins: str = "sqrt",
                kde: bool = False) -> plt.Axes:
        """

        Parameters
        ----------
        x
        y
        content
        background_type
        background
        xlim
        ylim
        ax
        bins
        kde

        Returns
        -------

        """
        self._check_2d_plot_content(content=content)
        if ax is None:
            ax = plt.subplots(figsize=(5, 5))[1]
        assert all([c in self.data.columns for c in [x, y]]), 'Invalid axis; must be an existing column in dataframe'
        x = ["population_label", "meta_label", "cluster_id"]
        err = f"Invalid background_type, should be one of {x}"
        assert background_type in x, err
        data = self.data
        if background is not None:
            data = data[data[background_type] == background]
            assert data.shape[0] > 3, "Less than 3 datapoint for the chosen background"
            if data.shape[0] < 1000:
                ax.scatter(data[x],
                           data[y],
                           marker='o',
                           s=1,
                           c='black',
                           alpha=0.8)
            else:
                bins = [np.histogram_bin_edges(data[x].values, bins=bins),
                        np.histogram_bin_edges(data[y].values, bins=bins)]
                ax.hist2d(data[x],
                          data[y],
                          bins=bins,
                          norm=LogNorm(),
                          cmap="jet")
        colours = cycle(plt.get_cmap("tab20"))
        for c in content:
            colour = next(colours)
            df = data[data[c["source"]] == c["value"]]
            ax.scatter(df[x],
                       df[y],
                       marker='o',
                       s=1,
                       c=colour,
                       alpha=0.8,
                       label=df[f"{c['source']}: {c['value']}"])
            if kde:
                sns.kdeplot(df[x],
                            df[y],
                            color=colour)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.legend(fontsize=11, bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.,
                  markerscale=6)
        return ax
