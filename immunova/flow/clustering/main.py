from mongoengine.base.datastructures import EmbeddedDocumentList
from immunova.data.fcs_experiments import FCSExperiment
from immunova.data.patient import Patient, Bug, MetaDataDictionary
from immunova.data.fcs import Cluster, Population
from immunova.data.clustering import ClusteringDefinition
from immunova.flow.dim_reduction import dimensionality_reduction
from immunova.flow.gating.actions import Gating
from immunova.flow.utilities import progress_bar
from anytree import Node
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import phenograph
import scprep


def meta_dict_lookup(key: str):
    lookup = MetaDataDictionary.objects(key=key)
    if not lookup:
        return None
    return lookup[0].desc


class Explorer:
    def __init__(self, data: pd.DataFrame or None = None, path: str or None = None):
        if data is not None:
            self.data = data
        elif path is None:
            raise ValueError('No dataframe provided, please provide a string value to path to load a csv file')
        else:
            self.data = pd.read_csv(path)
        if 'pt_id' not in self.data.columns:
            raise ValueError('Error: please ensure that dataframe is populated with the patient ID prior to '
                             'object construction')

    def info(self):
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

    def save(self, path: str):
        self.data.to_csv(path, index=False)

    def load_meta(self, variable: str):
        """
        Load meta data for each patient. Must be provided with a variable that is a field with a single value
        NOT an embedded document. A column will be generated in the Pandas DataFrame stored in the attribute 'data'
        that pertains to the variable given and the value will correspond to that of the patients.
        :param variable: field name to populate data with
        """
        self.data[variable] = None
        for pt_id in progress_bar(self.data.pt_id.unique()):
            if pt_id is None:
                continue
            p = Patient.objects(patient_id=pt_id).get()
            if type(p[variable]) == EmbeddedDocumentList:
                raise TypeError('Chosen variable is an embedded document.')
            self.data.loc[self.data.pt_id == pt_id, variable] = p[variable]

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
        """
        self.data['organism_name'] = 'Unknown'
        self.data['organism_type'] = 'Unknown'
        self.data['hmbpp'] = 'Unknown'
        self.data['ribo'] = 'Unknown'

        for pt_id in progress_bar(self.data.pt_id.unique()):
            if pt_id is None:
                continue
            p = Patient.objects(patient_id=pt_id).get()
            self.data.loc[self.data.pt_id == pt_id, 'organism_name'] = self.__bugs(patient=p, multi_org=multi_org)
            self.data.loc[self.data.pt_id == pt_id, 'organism_type'] = self.__org_type(patient=p)
            self.data.loc[self.data.pt_id == pt_id, 'hmbpp'] = self.__hmbpp_ribo(patient=p, field='hmbpp_status')
            self.data.loc[self.data.pt_id == pt_id, 'ribo'] = self.__hmbpp_ribo(patient=p, field='ribo_status')

    @staticmethod
    def __bugs(patient: Patient, multi_org: str) -> str:
        """
        Internal function. Fetch the name of isolated organisms for each patient.
        :param patient: Patient model object
        :param multi_org: If 'multi_org' equals 'list' then multiple organisms will be stored as a comma separated list
        without duplicates, whereas if the value is 'mixed' then multiple organisms will result in a value of 'mixed'.
        :return: string of isolated organisms comma seperated, or 'mixed' if multi_org == 'mixed' and multiple organisms
        listed for patient
        """
        if not patient.infection_data:
            return 'Unknown'
        orgs = [b.org_name for b in patient.infection_data if b.org_name]
        if not orgs:
            return 'Unknown'
        if len(orgs) == 1:
            return orgs[0]
        if multi_org == 'list':
            return ','.join(orgs)
        return 'mixed'

    @staticmethod
    def __org_type(patient: Patient) -> str:
        """
        Parse all infectious isolates for each patient and return the organism type isolated, one of either:
        'gram positive', 'gram negative', 'virus', 'mixed' or 'fungal'
        :param patient: Patient model object
        :return: common organism type isolated for patient
        """

        def bug_type(b: Bug):
            if not b.organism_type:
                return 'Unknown'
            if b.organism_type == 'bacteria':
                return b.gram_status
            return b.organism_type

        bugs = list(set(map(bug_type, patient.infection_data)))
        if len(bugs) == 0:
            return 'Unknown'
        if len(bugs) == 1:
            return bugs[0]
        return 'mixed'

    @staticmethod
    def __hmbpp_ribo(patient: Patient, field: str) -> str:
        """
        Given a value of either 'hmbpp' or 'ribo' for 'field' argument, return True if any Bug has a positive status
        for the given patient ID.
        :param patient: Patient model object
        :param field: field name to search for; expecting either 'hmbpp_status' or 'ribo_status'
        :return: common value of hmbpp_status/ribo_status
        """
        if all([b[field] is None for b in patient.infection_data]):
            return 'Unknown'
        if all([b[field] == 'P+ve' for b in patient.infection_data]):
            return 'P+ve'
        if all([b[field] == 'N-ve' for b in patient.infection_data]):
            return 'N-ve'
        return 'mixed'

    def load_biology_data(self, test_name: str, summary_method: str = 'average'):
        """
        Load the pathology results of a given test from each patient and populate 'data' accordingly. As multiple
        results may exist for one particular test, a summary method should be provided, this should have a value as
        follows:
        * average - the average test result is generated and stored
        * max - the maximum value is stored
        * min - the minimum value is stored
        * median - the median test result is generated and stored
        """
        self.data[test_name] = self.data['pt_id'].apply(lambda x: self.__biology(x, test_name, summary_method))

    @staticmethod
    def __biology(pt_id: str, test_name: str, method: str) -> np.float or None:
        """
        Given some test name, return a summary statistic of all results for a given patient ID
        :param pt_id: patient identifier
        :param test_name: name of test to search for
        :param method: summary statistic to use
        """
        if pt_id is None:
            return None
        tests = Patient.objects(patient_id=pt_id).get().patient_biology
        tests = [t.result for t in tests if t.test == test_name]
        if not tests:
            return None
        if method == 'max':
            return np.max(tests)
        if method == 'min':
            return np.min(tests)
        if method == 'median':
            return np.median(tests)
        return np.average(tests)

    def dimenionality_reduction(self, method: str, features: list,
                                n_components: int = 2, overwrite: bool = False,
                                **kwargs):
        embedding_cols = [f'{method}_{i}' for i in range(n_components)]
        if all([x in self.data.columns for x in embedding_cols]) and not overwrite:
            print(f'Embeddings for {method} already exist, change arg "overwrite" to True to overwrite existing')
            return
        embeddings = dimensionality_reduction(self.data, features, method,
                                              n_components, return_embeddings_only=True,
                                              **kwargs)
        for d, e in zip(embeddings, embedding_cols):
            self.data[e] = d

    def _plotting_labels(self, label: str, populations: list or None):
        if label == 'population_label':
            if populations is None:
                return self.data[label].values
            return list(map(lambda x: x if x in populations else 'None', self.data['population_label'].values))
        else:
            return self.data[label].values

    def scatter_plot(self, primary_label: str, features: list, discrete: bool, secondary_label: str or None = None,
                     populations: list or None = None, n_components: int = 2,
                     dim_reduction_method: str = 'UMAP', **kwargs) -> plt.Axes:
        """
        Generate a 2D/3D scatter plot (dimensions depends on the number of components chosen for dimensionality
        reduction. Each data point is labelled according to the option provided to the label arguments. If a value
        is given to both primary and secondary label, the secondary label colours the background and the primary label
        colours the foreground of each datapoint.
        :param primary_label: option for the primary label, must be one of the following:
        * A valid column name in Explorer attribute 'data' (check valid column names using Explorer.data.columns)
        * 'global clusters' - requires that phenograph_clustering method has been called prior to plotting. Each data
        point will be coloured according to cluster association.
        * 'gated populations' - each data point is coloured according to population identified by prior gating
        :param features: list of column names used as feature space for dimensionality reduction
        :param secondary_label: option for the secondary label, options same as primary_label (optional)
        :param populations: if primary/secondary label has value of 'gated populations', only populations in this
        list will be included (events with no population associated will be labelled 'None')
        :param n_components: number of components to produce from dimensionality reduction, valid values are 2 or 3
        (default = 2)
        :param dim_reduction_method: method to use for dimensionality reduction, valid values are 'UMAP' or 'PHATE'
        (default = 'UMAP')
        :param kwargs: additional keyword arguments to pass to dimensionality reduction algorithm
        :return: matplotlib subplot axes object
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        if n_components not in [2, 3]:
            raise ValueError('n_components must have a value of 2 or 3')

        # Dimensionality reduction
        self.dimenionality_reduction(dim_reduction_method=dim_reduction_method,
                                     features=features, n_components=n_components, **kwargs)
        embedding_cols = [f'{dim_reduction_method}_{i}' for i in range(n_components)]
        # Label and plotting
        for label in [primary_label, secondary_label]:
            if label:
                if label not in self.data.columns:
                    raise ValueError(f'Error: {label} is not a valid entry, valid labels include: '
                                     f'{self.data.columns.tolist()}')

        plabel = self._plotting_labels(primary_label, populations)

        if secondary_label is not None:
            slabel = self._plotting_labels(secondary_label, populations)
            if n_components == 2:
                ax = scprep.plot.scatter2d(self.data[embedding_cols], c=slabel, ticks=False,
                                           label_prefix=dim_reduction_method, ax=ax, s=100,
                                           discrete=discrete, legend_loc="lower left",
                                           legend_anchor=(1.04, 1), legend_title=secondary_label)

            else:
                ax = scprep.plot.scatter3d(self.data[embedding_cols], c=slabel, ticks=False,
                                           label_prefix=dim_reduction_method, ax=ax, s=100,
                                           discrete=discrete, legend_loc="lower left",
                                           legend_anchor=(1.04, 1), legend_title=secondary_label)
        if n_components == 2:
            ax = scprep.plot.scatter2d(self.data[embedding_cols], c=plabel, ticks=False,
                                       label_prefix=dim_reduction_method, ax=ax, s=1,
                                       discrete=discrete, legend_loc="lower left",
                                       legend_anchor=(1.04, 0), legend_title=primary_label)
        else:
            ax = scprep.plot.scatter3d(self.data[embedding_cols], c=plabel, ticks=False,
                                       label_prefix=dim_reduction_method, ax=ax, s=1,
                                       discrete=discrete, legend_loc="lower left",
                                       legend_anchor=(1.04, 0), legend_title=primary_label)
        return ax

    def heatmap(self, heatmap_var: str, features: list, clustermap: bool = False, populations: list or None = None):
        """
        Generate a heatmap of marker expression for either global clusters or gated populations
        (indicated with 'heatmap_var' argument)
        :param heatmap_var: variable to use, either 'global clusters' or 'gated populations'
        :param features: list of column names to use for generating heatmap
        :param clustermap: if True, rows (clusters/populations) are grouped by single linkage clustering
        """
        heatmap_var = self._plotting_labels(heatmap_var, populations)
        d = self.data[features + [heatmap_var]]
        d[features] = d[features].apply(pd.to_numeric)
        d = d.groupby(by=heatmap_var).mean()
        if clustermap:
            ax = sns.clustermap(d, col_cluster=False, cmap='viridis', figsize=(16, 10))
            return ax
        fig, ax = plt.subplots(figsize=(16, 10))
        ax = sns.heatmap(d, linewidth=0.5, ax=ax, cmap='viridis')
        ax.set_title('MFI (averaged over all patients) for PhenoGraph clusters')
        return ax

    def plot_representation(self, x_variable: str, y_variable, discrete: bool = True):
        """
        Present a breakdown of how a variable is represented in relation to a cluster/population/meta cluster
        :param x_variable: either cluster, population or meta cluster
        :param y_variable: variable for comparison (give a value of 'patient' for patient represenation
        :param discrete: if True, the variable is assumed to be discrete
        :return: matplotlib axes object
        """
        x_var_options = {'cluster': 'cluster_id', 'population': 'population_label', 'meta cluster': 'meta_cluster_id'}
        if x_variable not in x_var_options.keys():
            raise ValueError(f'x_variable must be one of {x_var_options.keys()}')
        x_variable = x_var_options[x_variable]
        if y_variable == 'patient':
            x = self.data[[x_variable, 'pt_id']].groupby(x_variable)['pt_id'].nunique() / len(
                self.data.pt_id.unique()) * 100
            fig, ax = plt.subplots(figsize=(12, 7))
            x.sort_values().plot(kind='bar', ax=ax)
            ax.set_ylabel('Patient representation (%)')
            ax.set_xlabel(x_variable)
            return ax
        if y_variable in self.data.columns and discrete:
            x = (self.data[[x_variable, y_variable]].groupby(x_variable)[y_variable]
                 .value_counts(normalize=True)
                 .rename('percentage')
                 .mul(100)
                 .reset_index()
                 .sort_values(y_variable))
            fig, ax = plt.subplots(figsize=(12, 7))
            p = sns.barplot(x=x_variable, y=f'percentage', hue=y_variable, data=x, ax=ax)
            _ = plt.setp(p.get_xticklabels(), rotation=90)
            ax.set_ylabel(f'% cells ({y_variable})')
            ax.set_xlabel(x_variable)
            return ax
        if y_variable in self.data.columns and not discrete:
            fig, ax = plt.subplots(figsize=(15, 5))
            d = self.data.sample(10000)
            ax = sns.swarmplot(x=x_variable, y=y_variable, data=d, ax=ax, s=3)
            return ax


class Clustering:
    def __init__(self, clustering_definition: ClusteringDefinition):
        self.ce = clustering_definition
        self.data = pd.DataFrame()

    def _has_data(self):
        """
        Internal method. Check that self.data has been populated. If not raise an error.
        """
        if self.data is None:
            raise ValueError('Error: no sample is currently associated to this object, before proceeding first '
                             'associate a sample and its data using the `load_data` method')

    def cluster(self):
        self._has_data()
        if self.ce.method == 'PhenoGraph':
            params = {k: v for k, v in self.ce.parameters}
            communities, graph, q = phenograph.cluster(self.data[self.ce.features], **params)
            if self.ce.cluster_prefix is not None:
                communities = np.array(map(lambda x: f'{self.ce.cluster_prefix}_{x}', communities))
            return communities, graph, q
        elif self.ce.method == 'FlowSOM':
            print('FlowSOM is not implemented yet!')
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

    Parameters:
        self.data - a Pandas DataFrame of FCS single cell data that clustering will be performed upon
        self.ce - associated clustering definition
        self.clusters - dictionary object of identified clusters
    """
    def __init__(self, **kwargs):
        """
        Constructor for generating a Clustering object.
        :param clustering_definition: ClusteringDefinition object (see data.clustering.ClusteringDefinition)
        """
        super().__init__(**kwargs)
        if self.ce.meta_clustering:
            raise ValueError('Error: definition is for meta-clustering, not single clustering')
        self.clusters = dict()
        self.data = None
        self.experiment = None
        self.sample_id = None

    def delete_clusters(self):
        """
        All clusters will be deleted from the object.
        """
        self.clusters = dict()

    def load_data(self, experiment: FCSExperiment, sample_id: str):
        """
        Given some FCS Experiment and the name of a sample associated to that experiment, populate the 'data'
        parameter with single cell data from the given sample and FCS experiment.
        :param experiment: FCS experiment object
        :param sample_id: sample identifier
        """
        fg = experiment.pull_sample(sample_id)
        root_p = [p for p in fg.populations if p.population_name == self.ce.root_population]
        if not root_p:
            raise ValueError(f'Error: {self.ce.root_population} does not exist for sample {sample_id}')
        if self.ce.clustering_uid in root_p[0].list_clustering_experiments():
            self._load_clusters(root_p[0])
        sample = Gating(experiment, sample_id, include_controls=False)
        transform = True
        if not self.ce.transform_method:
            transform = False
        self.data = sample.get_population_df(self.ce.root_population,
                                             transform=transform,
                                             transform_method=self.ce.transform_method)
        if not self.ce.features:
            self.data = self.data[self.ce.features]
        pt = Patient.objects(files__contains=sample.mongo_id)
        self.data['pt_id'] = None
        if pt:
            self.data['pt_id'] = pt[0].patient_id

    def _load_clusters(self, root_p: Population):
        """
        Internal method. Load existing clusters (associated to given cluster UID)
        associated to the root population of chosen sample.
        :param root_p: root population Population object
        """
        clusters = root_p.pull_clusters(self.ce.clustering_uid)
        for c in clusters:
            self.clusters[c.cluster_id] = dict(n_events=c.n_events,
                                               prop_of_root=c.prop_of_root,
                                               index=c.load_index())

    def _add_cluster(self, name: str, indexes: np.array):
        """
        Internal method. Add new cluster to internal collection of clusters.
        :param name: Name of the cluster
        :param indexes: Indexes corresponding to events in cluster
        """
        self.clusters[name] = dict(index=indexes,
                                   n_events=len(indexes),
                                   prop_of_root=len(indexes)/self.data.shape[0])

    def save_clusters(self, overwrite: bool = False):
        """
        Clusters will be saved to the root population of associated sample.
        :param overwrite: if True, existing clusters in database entry will be overwritten.
        """
        # Is the given UID unique?
        self._has_data()
        fg = self.experiment.pull_sample(self.sample_id)
        root_p = [p for p in fg.populations if p.population_name == self.ce.root_population][0]
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

    def get_cluster_dataframe(self, cluster_id: str) -> pd.DataFrame:
        """
        Return a Pandas DataFrame of single cell data for a single cluster.
        """
        return self.data[self.data.index.isin(self.clusters[cluster_id]['index'])]

    def cluster(self):
        """
        Perform clustering as described by the clustering definition.
        """
        outcome = super().cluster()
        if self.ce.method == 'PhenoGraph':
            self._phenograph(outcome)
        elif self.ce.method == 'FlowSOM':
            self._flowsom(outcome)

    def _flowsom(self, outcome):
        pass

    def _phenograph(self, outcome: tuple):
        """
        Internal method. Assign PhenoGraph clustering results to clusters.
        """
        communities, graph, q = outcome
        for x in np.unique(communities):
            indices = np.where(communities == x)[0]
            self._add_cluster(x, indices)

    def explorer(self) -> Explorer:
        """
        Returns an Explorer object with the single cell data and clusters, for plotting and exploring meta-data.
        See flow.clustering.main.Explorer for details.
        """
        self._has_data()
        data = self.data.copy()
        data['cluster_id'] = None
        for c_name, c_data in self.clusters.items():
            data.loc[c_data['index'], 'cluster_id'] = c_name
        return Explorer(data=data)


class GlobalCluster(Clustering):
    """
    Sample data from all (or some) samples in an FCS Experiment and perform clustering on the concatenated dataset.
    """
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        if self.ce.meta_clustering:
            raise ValueError('Error: definition is for meta-clustering, not single clustering')

    def load_data(self, experiment: FCSExperiment, samples: list or str = 'all', sample_n: int or None = 1000):
        """
        Load fcs file data, including any associated gates or clusters
        :param experiment: FCSExperiment to load samples from
        :param samples: list of sample IDs (if value = 'all', then all samples from FCS Experiment are used)
        :param sample_n: if an integer value is provided, each file will be downsampled to the indicated
        amount (optional)
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

            if fdata is None:
                raise ValueError(f'Population {self.ce.root_population} does not exist for {sid}')

            # Downsample
            if sample_n is not None:
                if sample_n < fdata.shape[0]:
                    fdata = fdata.sample(n=sample_n)

            # Label each single cell (row) with
            fdata = self.__population_labels(fdata, g.populations[self.ce.root_population])
            fdata = fdata.reset_index()
            fdata = fdata.rename({'index': 'original_index'}, axis=1)
            pt = Patient.objects(files__contains=g.mongo_id)
            if pt:
                fdata['pt_id'] = pt[0].patient_id
            else:
                print(f'File group {g.id} in experiment {experiment.experiment_id} is not associated to any patient')
                fdata['pt_id'] = None
            self.data = pd.concat([self.data, fdata])
        print('------------ Completed! ------------')

    def __population_labels(self, data: pd.DataFrame, root_node: Node) -> pd.DataFrame:
        """
        Internal function. Called when loading data. Populates DataFrame column named 'population_label' with the
        name of the node associated with each event most downstream of the root population.
        :param data: Pandas DataFrame of events corresponding to root population from single patient
        :param root_node: anytree Node object of root population
        :return: Pandas DataFrame with 'population_label' column
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

    def cluster(self):
        outcome = super().cluster()
        if self.ce.method == 'PhenoGraph':
            self.data['cluster_id'] = outcome[0]

    def explorer(self) -> Explorer:
        return Explorer(data=self.data)


class MetaClustering(Clustering):
    def __init__(self, experiment: FCSExperiment, samples: str or list = 'all', **kwargs):
        super().__init__(**kwargs)
        if not self.ce.meta_clustering:
            raise ValueError('Error: definition is not for meta-clustering')
        self.experiment = experiment
        if samples == 'all':
            samples = experiment.list_samples()
        self.data = self.load_clusters(samples)

    def load_clusters(self, samples: list):
        print('--------- Meta Clustering: Loading data ---------')
        print('Each sample will be fetched from the database and a summary matrix created. Each row of this summary '
              'matrix will be a vector describing the centroid (the median of each channel/marker) of each cluster. ')
        columns = self.ce.features + ['sample_id', 'cluster_id']
        clusters = pd.DataFrame(columns=columns)
        for s in progress_bar(samples):
            clustering = SingleClustering(clustering_definition=self.ce)
            clustering.load_data(experiment=self.experiment, sample_id=s)
            pt_id = clustering.data['pt_id'].values[0]
            if len(clustering.clusters.keys()) == 0:
                print(f'No clusters found for clustering UID {self.ce.clustering_uid} and sample {s}')
            for c_name in clustering.clusters.keys():
                c_data = clusters.get_cluster_dataframe(c_name)
                n = c_data.shape[0]
                c_data = c_data.median()
                c_data['cluster_id'], c_data['cluster_size'], c_data['pt_id'], c_data['sample_id'] = c_name, n, pt_id, s
                clusters = clusters.append(c_data, ignore_index=True)
        print('------------------------------------------------')
        return clusters

    def cluster(self, **kwargs):
        outcome = super().cluster()
        if self.ce.method == 'PhenoGraph':
            self.data['meta_cluster_id'] = outcome[0]

    def explorer(self) -> Explorer:
        return Explorer(data=self.data)

