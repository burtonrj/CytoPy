from mongoengine.base.datastructures import EmbeddedDocumentList
from ...data.fcs_experiments import FCSExperiment
from ...data.subject import Subject, MetaDataDictionary, gram_status, bugs, hmbpp_ribo, biology
from ...data.fcs import Cluster, Population, ClusteringDefinition
from ..transforms import scaler
from ..gating.actions import Gating
from ..feedback import progress_bar
from ..dim_reduction import dimensionality_reduction
from .flowsom import FlowSOM
from .consensus import ConsensusCluster
from anytree import Node
from matplotlib.colors import LogNorm
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering, KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import phenograph
import scprep
np.random.seed(42)


def filter_dict(d: dict, keys: list) -> dict:
    """
    Given a dictionary, filter the contents and return a new dictionary containing the given keys

    Parameters
    ----------
    d : dict
        dictionary to filter
    keys : list
        list of keys to include in new dictionary

    Returns
    -------
    dict
        filtered dictionary

    """
    nd = {k: d[k] for k in d.keys() if k in keys}
    return nd


def meta_dict_lookup(key: str) -> str or None:
    """
    Lookup a value in the Meta Data dictionary; this collection contains a description of meta data variables and can
    be used for guidance for the user, see data.patient.MetaDataDictionary

    Parameters
    ----------
    key : str
        value to lookup

    Returns
    -------
    str or None
        Description of given variable or None if variable does not exist

    """
    lookup = MetaDataDictionary.objects(key=key)
    if not lookup:
        return None
    return lookup[0].desc


def _fetch_clustering_class(params: dict) -> (callable, dict):
    """
    Provides a Sklearn clustering object for Consensus Clustering.

    Parameters
    ----------
    params : dict
        dictionary of parameters

    Returns
    -------
    callable, dict
        Sklearn clustering object and updated parameters dictionary

    """
    clustering = AgglomerativeClustering
    e = 'No clustering class (or invalid) provided for meta-clustering, defaulting to agglomerative ' \
        'clustering; value should either be "agglomerative" or "kmeans"'
    if 'cluster_class' not in params.keys():
        raise ValueError(e)
    elif params['cluster_class'] not in ['agglomerative', 'kmeans']:
        raise ValueError(e)
    elif params['cluster_class'] == 'kmeans':
        clustering = KMeans
        params.pop('cluster_class')
    params.pop('cluster_class')
    return clustering, params


class Explorer:
    """
    The Explorer class is used to visualise the results of a clustering analysis and explore the results in
    contrast to patient meta-data. The user should provide a Pandas DataFrame derived from one of the clustering
    objects SingleClustering, GlobalClustering, or MetaClustering. Alternatively the user can provide a path name for
    a previously saved Explorer dataframe. At a minimum this dataframe (which contains single cell data or summary
    single cell data if MetaClustering) should have a column called 'pt_id' which contains the patient ID for each cell.

    Parameters
    ----------
    data : Pandas.DataFrame
        a Pandas DataFrame generated from a Clustering object
    path : str, optional
        Path to existing Pandas.DataFrame as a csv file

    """
    def __init__(self, data: pd.DataFrame or None = None, path: str or None = None):
        if data is not None:
            self.data = data
        else:
            assert path is not None, 'No dataframe provided, please provide a string value to path to load a csv file'
            self.data = pd.read_csv(path)
        assert ('pt_id' in self.data.columns), 'Error: please ensure that dataframe is populated with the patient ID ' \
                                               'prior to object construction'

    def drop_data(self, mask: pd.DataFrame) -> None:
        """Update contained dataframe according to a given mask.

        Parameters
        ----------
        mask : Pandas.DataFrame
            Valid pandas dataframe mask

        Returns
        -------
        None
        """
        self.data = self.data[mask]

    def info(self) -> None:
        """Print a description of the DataFrame contained within

        Returns
        -------
        None
        """
        print('---- Reference column info ----')
        for x in self.data.columns:
            if x == 'pt_id':
                print('pt_id: unique patient identifier')
            elif x == 'population_label':
                print('population_label: name of gated population single cell is associated to')
            elif x == 'cluster_id':
                print('cluster_id: name of cluster single cell is associated to, in the case of meta-clustering, '
                      'each row will have a unique sample_id and cluster_id combination')
            elif x == 'meta_cluster_id':
                print('meta_cluster_id: name of meta-cluster that each row (corresponding to a cluster) is associated '
                      'to')
            elif x == 'sample_id':
                print('sample_id: for meta-clustering, each row corresponds to a unique cluster within a sample, this '
                      'column pertains to the unique identifier for the origin sample')
            else:
                desc = meta_dict_lookup(x)
                if not desc:
                    print(f'{x}: no description available')
                else:
                    print(f'{x}: {desc}')
        print('-----------------------------')
        return self.data.info()

    def save(self, path: str) -> None:
        """
        Save the contained dataframe to a new csv file

        Parameters
        ----------
        path : str
            output path for csv file

        Returns
        -------
        None

        """
        self.data.to_csv(path, index=False)

    def load_meta(self, variable: str) -> None:
        """
        Load meta data for each patient. Must be provided with a variable that is a field with a single value
        NOT an embedded document. A column will be generated in the Pandas DataFrame stored in the attribute 'data'
        that pertains to the variable given and the value will correspond to that of the patients.

        Parameters
        ----------
        variable : str
            field name to populate data with

        Returns
        -------
        None

        """
        self.data[variable] = None
        for pt_id in progress_bar(self.data.pt_id.unique()):
            if pt_id is None:
                continue
            p = Subject.objects(subject_id=pt_id).get()
            try:
                assert type(p[variable]) != EmbeddedDocumentList, 'Chosen variable is an embedded document.'
                self.data.loc[self.data.pt_id == pt_id, variable] = p[variable]
            except KeyError:
                print(f'{pt_id} is missing meta-variable {variable}')
                self.data.loc[self.data.pt_id == pt_id, variable] = None

    def load_infectious_data(self, multi_org: str = 'list'):
        """
        Load the bug data from each patient and populate 'data' accordingly. As default variables will be created as
        follows:
        * organism_name = If 'multi_org' equals 'list' then multiple organisms will be stored as a comma separated list
        without duplicates, whereas if the value is 'mixed' then multiple organisms will result in a value of 'mixed'.
        * organism_type = value of either 'gram positive', 'gram negative', 'virus', 'mixed' or 'fungal'
        * hmbpp = True or False based on HMBPP status (Note: it only takes one positive organism for this value to be
        True)
        * ribo = True or False based on Ribo status (Note: it only takes one positive organism for this value to be
        True)

        Parameters
        ----------
        multi_org: str (Default value = 'list')

        Returns
        -------
        None

        """
        self.data['organism_name'] = 'Unknown'
        self.data['gram_status'] = 'Unknown'
        self.data['organism_name_short'] = 'Unknown'
        self.data['hmbpp'] = 'Unknown'
        self.data['ribo'] = 'Unknown'

        for pt_id in progress_bar(self.data.pt_id.unique()):
            if pt_id is None:
                continue
            p = Subject.objects(subject_id=pt_id).get()
            self.data.loc[self.data.pt_id == pt_id, 'organism_name'] = bugs(subject=p, multi_org=multi_org)
            self.data.loc[self.data.pt_id == pt_id, 'organism_name_short'] = bugs(subject=p, multi_org=multi_org,
                                                                                  short_name=True)
            self.data.loc[self.data.pt_id == pt_id, 'hmbpp'] = hmbpp_ribo(subject=p, field='hmbpp_status')
            self.data.loc[self.data.pt_id == pt_id, 'ribo'] = hmbpp_ribo(subject=p, field='ribo_status')
            self.data.loc[self.data.pt_id == pt_id, 'gram_status'] = gram_status(subject=p)

    def load_biology_data(self, test_name: str, summary_method: str = 'average'):
        """
        Load the pathology results of a given test from each patient and populate 'data' accordingly. As multiple
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
        self.data[test_name] = self.data['pt_id'].apply(lambda x: biology(x, test_name, summary_method))

    def dimenionality_reduction(self, method: str, features: list,
                                n_components: int = 2, overwrite: bool = False,
                                **kwargs) -> None:
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
        None

        """
        embedding_cols = [f'{method}_{i}' for i in range(n_components)]
        if all([x in self.data.columns for x in embedding_cols]) and not overwrite:
            print(f'Embeddings for {method} already exist, change arg "overwrite" to True to overwrite existing')
            return
        self.data = dimensionality_reduction(self.data, features, method,
                                             n_components, return_embeddings_only=False,
                                             **kwargs)

    def _plotting_labels(self, label: str, populations: list or None) -> list:
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

    def scatter_plot(self, label: str, features: list, discrete: bool,
                     populations: list or None = None, n_components: int = 2,
                     dim_reduction_method: str = 'UMAP',
                     mask: pd.DataFrame or None = None, figsize: tuple = (12, 8),
                     meta: bool = False, meta_scale_factor: int = 100,
                     dim_reduction_kwargs: dict or None = None,
                     matplotlib_kwargs: dict or None = None) -> plt.Axes:
        """
        Generate a 2D/3D scatter plot (dimensions depends on the number of components chosen for dimensionality
        reduction. Each data point is labelled according to the option provided to the label arguments. If a value
        is given to both primary and secondary label, the secondary label colours the background and the primary label
        colours the foreground of each datapoint.

        Parameters
        ----------
        label : str
            option for the primary label, must be one of the following:
            * A valid column name in Explorer attribute 'data' (check valid column names using Explorer.data.columns)
            * 'global clusters' - requires that phenograph_clustering method has been called prior to plotting. Each data
            point will be coloured according to cluster association.
            * 'gated populations' - each data point is coloured according to population identified by prior gating
        features : list
            list of column names used as feature space for dimensionality reduction
        discrete : bool
            Are the labels for this plot discrete or continuous? If True, labels will be treated as
            discrete, otherwise labels will be coloured using a gradient and a colourbar will be provided.
        populations : list, optional
            if primary/secondary label has value of 'gated populations', only populations in this
            list will be included (events with no population associated will be labelled 'None')
        n_components : int, (default=2)
            number of components to produce from dimensionality reduction, valid values are 2 or 3
        dim_reduction_method :str, (default = 'UMAP')
            method to use for dimensionality reduction, valid values are 'UMAP' or 'PHATE'
        mask : Pandas.DataFrame, optional
            a valid Pandas DataFrame mask to subset data prior to plotting
        figsize : tuple, (default=(12,8))
            tuple of figure size to pass to matplotlib call
        meta : bool, (default=False)
            if True, data is assumed to come from meta-clustering
        meta_scale_factor : int, (default=100)
            if meta is True, scale factor defines the size of datapoints;
            size = meta_scale_factor * proportion of events in cluster relative to root population
        dim_reduction_kwargs : dict, optional
            additional keyword arguments to pass to dimensionality reduction algorithm
        matplotlib_kwargs : dict, optional
            additional keyword arguments to pass to matplotlib call

        Returns
        -------
        matplotlib.axes

        """
        assert n_components in [2, 3], 'n_components must have a value of 2 or 3'
        # Dimensionality reduction
        if dim_reduction_kwargs is None:
            dim_reduction_kwargs = dict()
        if matplotlib_kwargs is None:
            matplotlib_kwargs = dict()
        self.dimenionality_reduction(method=dim_reduction_method,
                                     features=features,
                                     n_components=n_components,
                                     **dim_reduction_kwargs)
        embedding_cols = [f'{dim_reduction_method}_{i}' for i in range(n_components)]
        # Label and plotting
        assert label in self.data.columns, f'{label} is not a valid entry, valid labels include: ' \
                                           f'{self.data.columns.tolist()}'
        plabel = self._plotting_labels(label, populations)
        data = self.data
        if mask is not None:
            data = data[mask]
            plabel = np.array(plabel)[data.index.values]

        def check_meta():
            """ """
            assert all([x in self.data.columns for x in ['cluster_size', 'meta_cluster_id']]), \
                "DataFrame should contain columns 'cluster_size' and 'meta_cluster_id', was the explorer object " \
                "populated from MetaClustering?"

        if n_components == 2:
            if meta:
                check_meta()
                return scprep.plot.scatter2d(data[embedding_cols], c=plabel, ticks=False,
                                             label_prefix=dim_reduction_method, s=data['cluster_size']*meta_scale_factor,
                                             discrete=discrete, legend_loc="lower left",
                                             legend_anchor=(1.04, 0), legend_title=label,
                                             figsize=figsize, **matplotlib_kwargs)

            return scprep.plot.scatter2d(data[embedding_cols], c=plabel, ticks=False,
                                         label_prefix=dim_reduction_method,
                                         discrete=discrete, legend_loc="lower left",
                                         legend_anchor=(1.04, 0), legend_title=label,
                                         figsize=figsize, **matplotlib_kwargs)
        else:
            if meta:
                check_meta()
                return scprep.plot.scatter3d(data[embedding_cols], c=plabel, ticks=False,
                                             label_prefix=dim_reduction_method, s=data['cluster_size']*meta_scale_factor,
                                             discrete=discrete, legend_loc="lower left",
                                             legend_anchor=(1.04, 0), legend_title=label,
                                             figsize=figsize, **matplotlib_kwargs)
            return scprep.plot.scatter3d(data[embedding_cols], c=plabel, ticks=False,
                                         label_prefix=dim_reduction_method,
                                         discrete=discrete, legend_loc="lower left",
                                         legend_anchor=(1.04, 0), legend_title=label,
                                         figsize=figsize, **matplotlib_kwargs)

    def heatmap(self, heatmap_var: str, features: list,
                clustermap: bool = False, mask: pd.DataFrame or None = None,
                normalise: bool = True, figsize: tuple = (10, 5), title: str or None = None,
                col_cluster: bool = False, **kwargs):
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
        d = self.data.copy(deep=True)
        if mask is not None:
            d = d[mask]
        d = d[features + [heatmap_var]]
        d[features] = d[features].apply(pd.to_numeric)
        if normalise:
            d[features] = preprocessing.MinMaxScaler().fit_transform(d[features])
        d = d.groupby(by=heatmap_var).mean()
        if clustermap:
            ax = sns.clustermap(d, col_cluster=col_cluster, cmap='viridis', figsize=figsize, **kwargs)
            return ax
        fig, ax = plt.subplots(figsize=(16, 10))
        ax = sns.heatmap(d, linewidth=0.5, ax=ax, cmap='viridis', **kwargs)
        if title is not None:
            ax.set_title(title)
        return ax

    def plot_representation(self, x_variable: str, y_variable, discrete: bool = True,
                            mask: pd.DataFrame or None = None, figsize: tuple = (6, 6),
                            **kwargs):
        """
        Present a breakdown of how a variable is represented in relation to a cluster/population/meta cluster

        Parameters
        ----------
        x_variable : str
            either cluster, population or meta cluster
        y_variable : str
            variable for comparison (give a value of 'patient' for patient represenation
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
        x_var_options = {'cluster': 'cluster_id', 'population': 'population_label', 'meta cluster': 'meta_cluster_id'}
        assert x_variable in x_var_options.keys(), f'x_variable must be one of {x_var_options.keys()}'
        x_variable = x_var_options[x_variable]
        if y_variable == 'patient':
            x = d[[x_variable, 'pt_id']].groupby(x_variable)['pt_id'].nunique() / len(d.pt_id.unique()) * 100
            fig, ax = plt.subplots(figsize=figsize)
            x.sort_values().plot(kind='bar', ax=ax, **kwargs)
            ax.set_ylabel('Patient representation (%)')
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
        assert d.geom[0] == 0, f'unable to subset data on {_id}'

    def plot_2d(self, primary_id: dict, x: str, y: str, secondary_id: dict or None = None,
                xlim: tuple or None = None, ylim: tuple or None = None) -> plt.Axes:
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
        check_id(primary_id, self.data)
        if secondary_id is not None:
            check_id(secondary_id, self.data)
        assert all([c in self.data.columns for c in [x, y]]), 'Invalid axis; must be an existing column in dataframe'

        # Build plot
        fig, ax = plt.subplots(figsize=(5, 5))
        d = self.data[self.data[primary_id['column_name']] == primary_id['value']]
        d2 = None
        if secondary_id is not None:
            d2 = self.data[self.data[secondary_id['column_name']] == secondary_id['value']]

        if d.geom[0] < 1000:
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


class Clustering:
    """
    Parent class for Clustering. Provides access to universal cluster method. Clustering objects generate a new
    pandas dataframe where either single cell data (or a summary of single cell data) are assigned to clusters,
    this assignment is stored in a new column named 'cluster_id' (or 'meta_cluster_id'; see MetaClustering)

    Parameters
    ----------
    ce: ClusteringDefinition
        A clustering definition saved to the underlying database and reused across multiple projects
    data: Pandas.DataFrame
        a Pandas DataFrame that is populated with clustering information can be passed onto the
        Explorer class for visualisation

    """
    def __init__(self, clustering_definition: ClusteringDefinition):
        self.ce = clustering_definition
        self.data = pd.DataFrame()
        self.graph = None
        self.q = None

    def _has_data(self):
        """Internal method. Check that self.data has been populated. If not raise an Assertion Error."""
        err_msg = 'Error: no sample is currently associated to this object, before proceeding first ' \
                  'associate a sample and its data using the `load_data` method'
        assert self.data.shape[0] > 0, err_msg

    def _check_null(self) -> list:
        """
        Internal method. Check for null values in the underlying dataframe. Returns a list of column names for columns
        with no missing values.

        Returns
        -------
        List
            List of valid columns

        """
        f = self.ce.features
        null_cols = self.data[f].isnull().sum()[self.data[f].isnull().sum() > 0].index.values
        if null_cols.size != 0:
            print('Warning: the following columns contain null values and will be excluded from clustering '
                  f'analysis: {null_cols}')
        return [x for x in self.ce.features if x not in null_cols]

    def _population_labels(self, data: pd.DataFrame, root_node: Node) -> pd.DataFrame:
        """
        Internal function. Called when loading data. Populates DataFrame column named 'population_label' with the
        name of the node associated with each event most downstream of the root population.

        Parameters
        ----------
        data : Pandas.DataFrame
            Pandas DataFrame of events corresponding to root population from single patient
        root_node : Node
            anytree Node object of root population

        Returns
        -------
        Pandas.DataFrame
            Pandas DataFrame with 'population_label' column

        """

        def recursive_label(d, n):
            mask = d.index.isin(n.index)
            d.loc[mask, 'population_label'] = n.name
            if len(n.children) == 0:
                return d
            for c in n.children:
                recursive_label(d, c)
            return d

        data = data.copy()
        data['population_label'] = self.ce.root_population
        data = recursive_label(data, root_node)
        return data

    def cluster(self) -> np.array:
        """
        Perform clustering analysis as specified by the ClusteringDefinition. Valid methods currently include PhenoGraph
        and FlowSOM. Returns clustering assignments.

        Returns
        -------
        Numpy.array
            Numpy array of clustering assignments.

        """
        self._has_data()
        if self.ce.method == 'PhenoGraph':
            params = {k: v for k, v in self.ce.parameters}
            features = self._check_null()
            communities, graph, q = phenograph.cluster(self.data[features], **params)
            if self.ce.cluster_prefix is not None:
                communities = np.array(list(map(lambda x: f'{self.ce.cluster_prefix}_{x}', communities)))
            self.graph = graph
            self.q = q
            return communities
        elif self.ce.method == 'FlowSOM':
            params = {k: v for k, v in self.ce.parameters}
            features = self._check_null()
            init_params = filter_dict(params, ['neighborhood_function', 'normalisation'])
            train_params = filter_dict(params, ['som_dim', 'sigma', 'learning_rate', 'batch_size',
                                                'seed', 'weight_init'])
            meta_params = filter_dict(params, ['cluster_class', 'min_n', 'max_n', 'iter_n', 'resample_proportion'])
            clustering, meta_params = _fetch_clustering_class(meta_params)
            som = FlowSOM(data=self.data, features=features, **init_params)
            som.train(**train_params)
            som.meta_cluster(cluster_class=clustering, **meta_params)
            self.graph = som.flatten_weights
            return som.predict()
        else:
            raise ValueError('Valid clustering methods are: PhenoGraph and FlowSOM')


class SingleClustering(Clustering):
    """
    Perform clustering on a single fcs file group. User provides a clustering definition
    (see data.clustering.ClusteringExperiment) and then can use 'load_data' method to load fcs
    data from an FCS experiment and associated sample.
    Call 'cluster' method to performing clustering using the method described in the clustering definition.
    To save the clustering result, use the 'save_clusters' method. The resulting clusters will be saved to the
    'root population' (given in the clustering definition) of the sample FCS file group.
    If the user calls 'load_data' on a sample with existing clusters in the database, for the clustering definition
    given, the object will be populated accordingly. If you wish to overwrite these clusters, class 'delete_clusters'
    followed by 'save_clusters' with the 'overwrite' argument set to True.

    Parameters
    ----------
    data : Pandas.DataFrame
        a Pandas DataFrame of FCS single cell data that clustering will be performed upon
    ce : ClusteringDefinition
        associated clustering definition
    clusters : dict
        dictionary object of identified clusters

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clusters = dict()
        self.data = None
        self.experiment = None
        self.sample_id = None

    def delete_clusters(self):
        """
        All clusters will be deleted from the object.
        """
        self.clusters = dict()

    def load_data(self, experiment: FCSExperiment,
                  sample_id: str,
                  include_population_label: bool = True,
                  scale: str or None = None):
        """
        Given some FCS Experiment and the name of a sample associated to that experiment, populate the 'data'
        parameter with single cell data from the given sample and FCS experiment.

        Parameters
        ----------
        experiment : FCSExperiment

        sample_id : str
            sample identifier
        include_population_label : bool, (default = True)
            if True, data frame populated with population labels from gates
        scale : str, optional
            specify how to scale the data

        Returns
        -------
        None

        """
        self.experiment = experiment
        self.sample_id = sample_id
        fg = experiment.pull_sample(sample_id)
        root_p = fg.get_population(self.ce.root_population)
        if self.ce.clustering_uid in root_p.list_clustering_experiments():
            self._load_clusters(root_p)
        sample = Gating(experiment, sample_id, include_controls=False)
        transform = True
        if not self.ce.transform_method:
            transform = False
        self.data = sample.get_population_df(self.ce.root_population,
                                             transform=transform,
                                             transform_method=self.ce.transform_method)
        if scale is not None:
            self.data[self.ce.features] = scaler(self.data[self.ce.features], scale_method=scale)[0]
        if not self.ce.features:
            self.data = self.data[self.ce.features]
        pt = Subject.objects(files__contains=sample.mongo_id)
        self.data['pt_id'] = None
        if pt:
            self.data['pt_id'] = pt[0].subject_id
        if include_population_label:
            self.data = self._population_labels(self.data, sample.populations[self.ce.root_population])

    def _load_clusters(self, root_p: Population):
        """
        Internal method. Load existing clusters (associated to given cluster UID)
        associated to the root population of chosen sample.

        Parameters
        ----------
        root_p : Population

        Returns
        -------
        None

        """
        clusters = root_p.get_many_clusters(self.ce.clustering_uid)
        for c in clusters:
            self.clusters[c.cluster_id] = dict(n_events=c.n_events,
                                               prop_of_root=c.prop_of_root,
                                               index=c.load_index(),
                                               meta_cluster_id=c.meta_cluster_id)

    def _add_cluster(self, name: str, index_mask: np.array):
        """
        Internal method. Add new cluster to internal collection of clusters.

        Parameters
        ----------
        name : str
            Name of the cluster
        index_mask : Numpy.array
            Indexes corresponding to events in cluster

        Returns
        -------
        None

        """
        index = self.data.index[index_mask]
        self.clusters[name] = dict(index=index,
                                   n_events=len(index),
                                   prop_of_root=len(index)/self.data.geom[0])

    def save_clusters(self, overwrite: bool = False):
        """
        Clusters will be saved to the root population of associated sample.

        Parameters
        ----------
        overwrite : bool, (Default value = False)
            if True, existing clusters in database entry will be overwritten.

        Returns
        -------
        None

        """
        # Is the given UID unique?
        self._has_data()
        fg = self.experiment.pull_sample(self.sample_id)
        root_p = fg.get_population(self.ce.root_population)
        if self.ce.clustering_uid in root_p.list_clustering_experiments():
            if overwrite:
                root_p.delete_clusters(self.ce.clustering_uid)
            else:
                raise ValueError(f'Error: a clustering experiment with UID {self.ce.clustering_uid} '
                                 f'has already been associated to the root population {self.ce.root_population}')
        for name, cluster_data in self.clusters.items():
            c = Cluster(cluster_id=name,
                        n_events=cluster_data['n_events'],
                        prop_of_root=cluster_data['prop_of_root'],
                        cluster_experiment=self.ce)
            c.save_index(cluster_data['index'])
            root_p.clustering.append(c)
        # Save the clusters
        fg.save()

    def get_cluster_dataframe(self, cluster_id: str, meta: bool = False) -> pd.DataFrame or None:
        """
        Return a Pandas DataFrame of single cell data for a single cluster.

        Parameters
        ----------
        cluster_id : str
            name of cluster to retrieve
        meta : bool, (Default value = False)
            if True, fetch cluster by meta_cluster_id (default = False)
            :return DataFrame of events in cluster (Data returned untransformed)

        Returns
        -------
        Pandas.DataFrame or None

        """
        if not meta:
            assert cluster_id in self.clusters.keys(), f'Invalid cluster_id; {cluster_id} not recognised'
            return self.data.loc[self.clusters[cluster_id].get('index')]
        clusters_idx = [c.get('index') for c in self.clusters.values() if c.get('meta_cluster_id') == cluster_id]
        if clusters_idx:
            return self.data.loc[np.unique(np.concatenate(clusters_idx, 0), axis=0)]
        return None

    def cluster(self):
        """
        Perform clustering as described by the clustering definition.
        Results saved to internal attribute 'clusters'.
        """
        cluster_assignments = super().cluster()
        for x in np.unique(cluster_assignments):
            mask = np.where(cluster_assignments == x)[0]
            self._add_cluster(x, mask)

    def explorer(self) -> Explorer:
        """
        Returns an Explorer object with the single cell data and clusters, for plotting and exploring meta-data.
        See flow.clustering.main.Explorer for details.

        Returns
        -------
        Explorer

        """
        self._has_data()
        data = self.data.copy()
        data['cluster_id'] = None
        for c_name, c_data in self.clusters.items():
            data.loc[c_data['index'], 'cluster_id'] = c_name
        return Explorer(data=data)


class GlobalClustering(Clustering):
    """
    Sample data from all (or some) samples in an FCS Experiment and perform clustering on the concatenated dataset.
    This is useful when you want to look at some global change in immunological topology and batch effects are minimal.

    """
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

    def load_data(self, experiment: FCSExperiment,
                  samples: list or str = 'all',
                  sample_n: int or None = 1000):
        """
        Load fcs file data, including any associated gates or clusters

        Parameters
        ----------
        experiment : FCSExperiment
            FCSExperiment to load samples from
        samples : list or str, (Default value = 'all')
            list of sample IDs (if value = 'all', then all samples from FCS Experiment are used)
        sample_n : int, optional, (Default value = 1000)
            if an integer value is provided, each file will be downsampled to the indicated
            amount (optional)

        Returns
        -------
        None

        """
        print(f'------------ Loading flow data: {experiment.experiment_id} ------------')
        for sid in progress_bar(samples):
            # Pull root population from file and transform
            g = Gating(experiment, sid, include_controls=False)
            if self.ce.transform_method is not None:
                fdata = g.get_population_df(population_name=self.ce.root_population,
                                            transform=True,
                                            transform_method=self.ce.transform_method)
            else:
                fdata = g.get_population_df(population_name=self.ce.root_population)

            assert fdata is not None, f'Population {self.ce.root_population} does not exist for {sid}'

            # Downsample
            if sample_n is not None:
                if sample_n < fdata.geom[0]:
                    fdata = fdata.sample(n=sample_n)

            # Label each single cell (row) with
            fdata = self._population_labels(fdata, g.populations[self.ce.root_population])
            fdata = fdata.reset_index()
            fdata = fdata.rename({'index': 'original_index'}, axis=1)
            pt = Subject.objects(files__contains=g.mongo_id)
            if pt:
                fdata['pt_id'] = pt[0].subject_id
            else:
                print(f'File group {g.id} in experiment {experiment.experiment_id} is not associated to any patient')
                fdata['pt_id'] = None
            self.data = pd.concat([self.data, fdata])
        print('------------ Completed! ------------')

    def cluster(self):
        """
        Perform clustering as specified in the Clustering Definition.

        """
        self.data['cluster_id'] = super().cluster()

    def explorer(self) -> Explorer:
        """
        Returns an Explorer object with the single cell data and clusters, for plotting and exploring meta-data.
        See flow.clustering.main.Explorer for details.

        Returns
        -------
        Explorer

        """
        return Explorer(data=self.data)


class MetaClustering(Clustering):
    """
    Perform meta-clustering; this is when, given a list of samples that have already been clustered, find groupings
    (cluster of the clusters) that describe the commonality between individual clustering.
    Performing clustering on individual samples has the benefit in that clustering will not be disrupted by batch effect
    (technical variation between patients) but we must find a way of contrasting the clustering between patients. This
    is what this class is for. The approach taken is that each clusters centroid is calculated (the median of each
    feature) and these centroids form a new multi-dimensional data point which are then subsequently clustered.
    Clustering is performed either by a single run of PhenoGraph clustering, or by Consensus Clustering.
    For stable clusters it is recommended to use Consensus Clustering.

    Parameters
    ----------
    experiment : FCSExperiment
        This is the FCS Experiment that samples will be taken from
    samples : str or list, (default='all')
        list of sample names to extract clusters from associated experiment
    scale : str, optional, (default='norm')
        scaling function to apply to data prior to clustering
    load_existing_meta: bool, (default=False)
        If True, existing meta-clusters (with the meta-clustering UID in the clustering definition) will be loaded
        into the object

    """
    def __init__(self, experiment: FCSExperiment,
                 samples: str or list = 'all',
                 scale: str or None = 'norm',
                 load_existing_meta: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.experiment = experiment
        self.scale = scale
        if type(samples) == str:
            assert samples == 'all', 'Invalid input, samples must be a list of existing samples or a value of "all" for ' \
                                     'all samples'
            samples = experiment.list_samples()
        self.data = self.load_clusters(samples)
        if load_existing_meta:
            print('Attempting to load existing meta-clusters')
            self.load_existing_clusters()
            print('meta_cluster_id populated with existing cluster results')
        print('------------ Completed! ------------')

    def load_existing_clusters(self):
        """
        Load existing meta-cluster ID for each cluster (note: a cluster can only ever have one meta-cluster ID).
        Meta-clusters are saved to DataFrame in column 'meta_cluster_id'.
        """
        def load(sample_id, cluster_id):
            fg = self.experiment.pull_sample(sample_id)
            root = fg.get_population(self.ce.root_population)
            clusters = root.get_many_clusters(self.ce.meta_clustering_uid_target)
            cluster = [c for c in clusters if c.cluster_id == cluster_id][0]
            assert cluster.meta_cluster_id, f'Meta cluster missing from {sample_id}, repeat clustering'
            return cluster.meta_cluster_id
        self.data['meta_cluster_id'] = self.data.apply(lambda row: load(row['sample_id'], row['cluster_id']), axis=1)

    def load_clusters(self, samples: list) -> pd.DataFrame:
        """
        Load the clusters from each sample and populate a new dataframe. Each row of the dataframe corresponds to
        a unique cluster from an individual patient sample, with the values of each feature being the median (centroid
        of this cluster).

        Parameters
        ----------
        samples : list
            List of samples to fetch from the experiment

        Returns
        -------
        Pandas.DataFrame

        """
        print('--------- Meta Clustering: Loading data ---------')
        print('Each sample will be fetched from the database and a summary matrix created. Each row of this summary '
              'matrix will be a vector describing the centroid (the median of each channel/marker) of each cluster. ')
        columns = self.ce.features + ['sample_id', 'cluster_id']
        clusters = pd.DataFrame(columns=columns)
        target_clustering_def = ClusteringDefinition.objects(clustering_uid=self.ce.meta_clustering_uid_target)
        assert target_clustering_def, f'No such clustering definition {self.ce.meta_clustering_uid_target} to target'
        for s in progress_bar(samples):
            clustering = SingleClustering(clustering_definition=target_clustering_def[0])
            try:
                clustering.load_data(experiment=self.experiment,
                                     sample_id=s,
                                     scale=self.scale,
                                     include_population_label=True)
            except KeyError as e:
                print(f'failed to load data for {s}: {e}')
                continue
            pt_id = clustering.data['pt_id'].values[0]
            if len(clustering.clusters.keys()) == 0:
                print(f'No clusters found for clustering UID '
                      f'{target_clustering_def[0].clustering_uid} and sample {s}')
            for c_name in clustering.clusters.keys():
                c_data = clustering.get_cluster_dataframe(c_name)
                n = c_data.shape[0]
                relative_n = n/clustering.data.geom[0]
                pop_label = c_data['population_label'].mode()[0]
                c_data = c_data.median()
                c_data['population_label'] = pop_label
                c_data['cluster_id'], c_data['cluster_size'], c_data['cluster_n'], \
                    c_data['pt_id'], c_data['sample_id'] = c_name, relative_n, n, pt_id, s
                clusters = clusters.append(c_data, ignore_index=True)
        return clusters

    def cluster(self):
        """
        Perform clustering as defined by clustering definition. Clustering results saved to DataFrame
        in 'meta_cluster_id' column.

        """
        if self.ce.method == 'ConsensusClustering':
            params = {k: v for k, v in self.ce.parameters}
            features = self._check_null()
            init_params = filter_dict(params, ['cluster_class', 'smallest_cluster_n', 'largest_cluster_n',
                                               'n_resamples', 'resample_proportion'])
            clustering, init_params = _fetch_clustering_class(init_params)
            consensus_clust = ConsensusCluster(cluster=clustering, **init_params)
            consensus_clust.fit(self.data[features].values)
            self.data['meta_cluster_id'] = consensus_clust.predict_data(self.data[features])
        elif self.ce.method == 'PhenoGraph':
            self.data['meta_cluster_id'] = super().cluster()
        else:
            print('Invalid clustering method, must be: "PhenoGraph" or "ConsensusClustering"; '
                  'consensus clustering is recommended for cluster stability')

    def inspect_heatmap(self, **kwargs):
        """
        Generate a clustered heatmap of clustering results. Rows are coloured according to population labels (as generated
        by gating)

        Parameters
        ----------
        kwargs :
            Additional keyword arguments to pass to call to Seaborn.clustermap

        Returns
        -------
        matplotlib.axes

        """
        data = self.data.copy()
        data = data.set_index('pt_id')
        palette = sns.husl_palette(len(data.population_label.unique()), s=.45)
        palette_mappings = dict(zip(data.population_label.unique(), palette))
        pop_label_colours = pd.Series(data.index.get_label_values('population_labels'),
                                      index=data.index).map(palette_mappings)
        return sns.clustermap(data[self.ce.features], row_colors=pop_label_colours, **kwargs)

    def save(self):
        """
        Save results of meta-clustering to underlying database

        """
        def _save(x):
            fg = self.experiment.pull_sample(x.sample_id)
            pop = fg.get_population(self.ce.root_population)
            cluster = [c for c in pop.clustering if c.cluster_id == x.cluster_id][0]
            cluster.meta_cluster_id = x.meta_cluster_id
            pop.update_cluster(x.cluster_id, cluster)
            fg.update_population(pop.population_name, pop)

        assert 'meta_cluster_id' in self.data.columns, 'Must run meta-clustering prior to calling save'
        self.data.apply(_save, axis=1)
        self.experiment.meta_cluster_ids = list(self.data.meta_cluster_id.values)
        self.experiment.save()

    def label_cluster(self, meta_cluster_id: str, new_id: str, prefix: str or None = 'cluster'):
        """
        Given an existing meta cluster ID and a new replacement ID, update meta clustering IDs

        Parameters
        ----------
        meta_cluster_id : str
            meta cluster ID to be replaced
        new_id : str
            new ID to use
        prefix : str or None, (Default value = 'cluster')
            optional prefix to append to new ID (default = "cluster")

        Returns
        -------
        None

        """
        assert meta_cluster_id in self.data.meta_cluster_id.values, 'Invalid meta cluster ID provided'
        if prefix is not None:
            new_id = f'{prefix}_{new_id}'
        self.data.loc[self.data.meta_cluster_id == meta_cluster_id, 'meta_cluster_id'] = new_id

    def explorer(self) -> Explorer:
        """
        Returns an Explorer object with the single cell data and clusters, for plotting and exploring meta-data.
        See flow.clustering.main.Explorer for details.

        Returns
        -------
        Explorer

        """
        return Explorer(data=self.data)

