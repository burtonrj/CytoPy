from ..data.fcs_experiments import FCSExperiment
from ..data.patient import Patient, gram_status, bugs, biology, hmbpp_ribo, org_type
from immunova.flow.dim_reduction import dimensionality_reduction
from ..data.fcs import ClusteringDefinition
from ..flow.gating.actions import Gating
from ..flow.clustering.main import SingleClustering
from ..flow.gating.transforms import apply_transform
from ..flow.gating.utilities import kde
from ..flow.utilities import progress_bar
from yellowbrick.features import RadViz
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np
import scprep


class Extract:
    """
    Performs feature extraction and summary for a single sample. Generates a Pandas DataFrame where each column
    is an extracted feature. This data-frame forms the feature space that will be used for biomarker discovery.

    Arguments:
        - sample_id: unique identifier for sample to process
        - experiment: FCSExperiment object within which the sample is contained
        - clustering_definition: ClusteringDefinition object for the clustering results to include in summary
        - exclude_populations: list of population names to be excluded from summary
        - exclude_clusters: list of cluster names to be excluded from summary

    Methods:
        - population_summary: generates new columns for each population as follows:
            * {NAME}_n = number of cells in populations
            * {NAME}_pop = proportion of cells in population relative to parent population
            * {NAME}_pot = proportion of cells in population relative to total cells in sample
        - cluster_summary: generates new columns for each cluster as follows:
            * {NAME}_n = number of cells in cluster
            * {NAME}_pot = proportion of cells in cluster relative to total cells in sample
        - ratios: for each possible pair of populations/clusters, calculate the ratio and generate column names as such:
            * {Population1}:{Population2}
            * {Population}:{Cluster}
            * {Cluster1}:{Cluster2}

    """
    def __init__(self, sample_id: str, experiment: FCSExperiment,
                 clustering_definition: ClusteringDefinition,
                 exclude_populations: list or None = None,
                 exclude_clusters: list or None = None,
                 verbose: bool = True):
        self.gating = Gating(experiment, sample_id)
        self.clustering = SingleClustering(clustering_definition=clustering_definition)
        self.clustering.load_data(experiment=experiment, sample_id=sample_id)
        self.data = pd.DataFrame({'sample_id': [self.gating.id], 'pt_id': self.clustering.data.pt_id.values[0]})
        self.verbose = verbose
        if exclude_populations is not None:
            self.populations = {k: v for k, v in self.gating.populations.items() if k not in exclude_populations}
        else:
            self.populations = self.gating.populations
        if exclude_clusters is not None:
            meta_cluster_ids = experiment.meta_cluster_ids
            self.clusters = {_id: {'n_events': 0,
                                   'prop_of_root': 0.0,
                                   'index': np.array([])}
                             for _id in meta_cluster_ids}
            for k, v in self.clustering.clusters.items():
                mid = v.meta_cluster_id
                self.clusters[mid]['n_events'] = self.clusters[mid].get('n_events', 0) + v.get('index', 0)
                self.clusters[mid]['prop_of_root'] = self.clusters[mid].get('prop_of_root', 0) + v.get('prop_of_root', 0)
                self.clusters[mid]['index'] = np.unique(np.concatenate(self.clusters[mid].get('index', 0),
                                                                       v.get('index', 0), axis=0), axis=0)
        else:
            self.clusters = self.clustering.clusters

    def population_summary(self, include_mfi: bool = True, transform: str = 'logicle') -> None:
        """
        Generates new columns for each population as follows:
            * {NAME}_n = number of cells in populations
            * {NAME}_pop = proportion of cells in population relative to parent population
            * {NAME}_pot = proportion of cells in population relative to total cells in sample
        :return: None
        """
        for pop_name, node in self.populations.items():
            self.data[f'{pop_name}_n'] = len(node.index)
            self.data[f'{pop_name}_pop'] = node.prop_of_parent
            self.data[f'{pop_name}_pot'] = node.prop_of_total
            if include_mfi:
                mfi = self.summarise_mfi(population_name=pop_name, transform=transform)
                for marker, mfi_ in mfi.items():
                    self.data[f'{pop_name}_{marker}_MFI'] = mfi_

    def summarise_mfi(self, population_name: str or None = None, cluster_name: str or None = None,
                      transform: str = 'logicle'):
        if population_name is not None and cluster_name is not None:
            raise ValueError('Provide the name of a population OR the name of a cluster, not both')
        if population_name is not None:
            data = self.gating.get_population_df(population_name, transform=True, transform_method=transform)
        elif cluster_name is not None:
            if self.verbose:
                print('Note: for clusters transform argument is ignored, data transformed according to clustering '
                      'definition')
            data = self.clustering.get_cluster_dataframe(cluster_name, meta=True)
        else:
            raise ValueError('Must provide either name of a cluster or name of a population')

        markers = [c for c in data.columns if all([fs not in c for fs in ['FSC', 'SSC']])]
        mfi = np.exp(np.log(data[markers].prod(axis=0))/data[markers].notna().sum(axis=0))
        return mfi.to_dict()

    def cluster_summary(self, include_mfi: bool = False) -> None:
        """
        Generates new columns for each cluster as follows:
            * {NAME}_n = number of cells in cluster
            * {NAME}_pot = proportion of cells in cluster relative to total cells in sample
        :return: None
        """
        for c_name, c_data in self.clusters.clusters.items():
            self.data[f'{c_name}_n'] = c_data['n_events']
            self.data[f'{c_name}_pot'] = c_data['prop_of_total']
            if include_mfi:
                mfi = self.summarise_mfi(cluster_name=c_name)
                for marker, mfi_ in mfi.items():
                    self.data[f'{c_name}_{marker}_MFI'] = mfi_

    def ratios(self) -> None:
        """
        For each possible pair of populations/clusters, calculate the ratio and generate column names as such:
            * {Population1}:{Population2}
            * {Population}:{Cluster}
            * {Cluster1}:{Cluster2}
        :return: None
        """
        if self.verbose:
            print('-------- Calculating population ratios --------')
        for pname1, pnode1 in self.populations.items():
            for pname2, pnode2 in self.populations.items():
                if pname1 == pname2:
                    continue
                self.data[f'{pname1}:{pname2}'] = len(pnode1.index)/len(pnode2.index)
            for c_name, c_data in self.clusters.clusters.items():
                self.data[f'{pname1}:{c_name}'] = len(pnode1.index)/c_data['n_events']
        if self.verbose:
            print('-------- Calculating cluster ratios --------')
        for cname1, cdata1 in self.clusters.items():
            for cname2, cdata2 in self.clusters.items():
                if cname1 == cname2:
                    continue
                self.data[f'{cname1}:{cname2}'] = cdata1['n_events'] / cdata2['n_events']

    @staticmethod
    def _relative_fold_change(x: np.array, y: np.array, center='median') -> np.float:
        """
        Given two populations, x and y, calculate the center for each (the median by default) and then return the fold change in x relative to y
        :param x: first population
        :param y: second population
        :param center: how to calculate the center value, options = ['median', 'mean'] (default = 'median')
        :return: fold change in x relative to y
        """
        if center == 'median':
            return np.median(x)-np.median(y)/np.median(x)
        return np.mean(x)-np.mean(y)/np.mean(x)

    def fmo_stats_1d(self, transform: str = 'logicle'):
        for fmo_id in self.gating.fmo.keys():
            if self.verbose:
                print(f'-------- FMO Summary: {fmo_id} --------')
            for pop_id in progress_bar(self.populations.keys()):
                fmo_data = apply_transform(self.gating.get_fmo_data(pop_id, fmo_id), transform_method=transform)
                whole_data = apply_transform(self.gating.get_population_df(pop_id), transform_method=transform)
                ks_stat, p = stats.ks_2samp(fmo_data[fmo_id].values, whole_data[fmo_id].values)
                self.data[f'{pop_id}_{fmo_id}_ks_statistic'] = ks_stat
                self.data[f'{pop_id}_{fmo_id}_ks_statistic_pval'] = p
                self.data[f'{pop_id}_{fmo_id}_fold_change_MFI'] = self._relative_fold_change(whole_data[fmo_id].values,
                                                                                             fmo_data[fmo_id].values)
                w_probs, _ = kde(data=whole_data[fmo_id], x=fmo_id, kde_bw=0.01)
                f_probs, _ = kde(data=fmo_data[fmo_id], x=fmo_id, kde_bw=0.01)
                self.data[f'{pop_id}_{fmo_id}_kl_divergence'] = stats.entropy(pk=w_probs, qk=f_probs)
        if self.verbose:
            print('-------- Completed! --------')

    def fmo_stats_multiple(self, fmo: list, population: str, transform: str = 'logicle'):
        if not self.gating.fmo.keys():
            if self.verbose:
                print(f'{self.gating.id} has no associated FMOs, aborting')
                return
        assert all([x in self.gating.fmo.keys() for x in fmo]), f'Some/all requested FMOs missing, ' \
                                                                f'valid FMOs include {self.gating.fmo.keys()}'
        assert population in self.gating.populations.keys(), f'Invalid population name, ' \
                                                             f'valid populations include: ' \
                                                             f'{self.gating.populations.keys()}'

        whole_data = apply_transform(self.gating.get_population_df(population),
                                     transform_method=transform)
        fmo_data = {name: apply_transform(self.gating.get_fmo_data(population, name),
                                          transform_method=transform)
                    for name in fmo}
        centroids = {name: data[fmo].median() for name, data in fmo_data.items()}
        euclidean_dist = {name: np.linalg.norm(whole_data[fmo].median - c) for name, c in centroids.items()}
        for name in euclidean_dist.keys():
            self.data[f'{name}_euclidean_dist'] = euclidean_dist[name]
        self.data[f'{fmo}_avg_euclidean_dist'] = np.mean(euclidean_dist.values())


class BuildFeatureSpace:
    def __init__(self, exclude_samples: list or None = None, exclude_patients: list or None = None,
                 population_transform: str = 'logicle', verbose: bool = False):
        self.exclude_samples = exclude_samples
        self.exclude_patients = exclude_patients
        self.population_transform = population_transform
        self.verbose = verbose
        self.data = pd.DataFrame()

    def add_experiment(self, experiment: FCSExperiment, meta_clustering_definition: ClusteringDefinition,
                       exclude_populations: list or None = None,
                       exclude_clusters: list or None = None,
                       include_mfi: bool = False, include_fmos: bool = False,
                       multi_dim_fmo_comparisons: list or None = None):
        target_ce = ClusteringDefinition.objects(clustering_uid=meta_clustering_definition.meta_clustering_uid_target)
        samples = experiment.list_samples()
        if self.exclude_samples is not None:
            samples = [s for s in samples if s not in self.exclude_samples]
        data = pd.DataFrame()
        for s in samples:
            if self.verbose:
                print(f'------------- {s} -------------')
            extraction = Extract(s, experiment, target_ce, exclude_populations, exclude_clusters, self.verbose)
            extraction.cluster_summary(include_mfi)
            extraction.population_summary(include_mfi, transform=self.population_transform)
            extraction.ratios()
            if include_fmos:
                extraction.fmo_stats_1d(self.population_transform)
                if multi_dim_fmo_comparisons is not None:
                    for fmo in multi_dim_fmo_comparisons:
                        assert all([x in fmo.keys() for x in ['comparisons', 'population']]), \
                            'multi_dim_fmo_comparisons should be a list of dictionary objects, each with a key ' \
                            '"comparisons", being a list of FMOs to compare, and "population", ' \
                            'the name of the population for which the comparison is relevant'
                        extraction.fmo_stats_multiple(fmo=fmo.get('comparisons'), population=fmo.get('population'))
            data = pd.concat([data, extraction.data])
        self.data = self.data.merge(data)

    def new_ratio(self, column_one: str, column_two: str):
        for c in [column_one, column_two]:
            assert c in self.data.columns, f'Invalid column name: {c}'
        self.data[f'{column_one}:{column_two}'] = self.data[column_one]/self.data[column_two]

    def label_dataset(self, variable: str, embedding: str or None = None):
        if embedding is not None:
            assert embedding in ['infection', 'biology', 'drug'], "Embedding should be one of " \
                                                                  "['infection', 'biology', 'drug']"
        labels = list()
        for p in self.data.pt_id.values:
            pt = Patient.objects(patient_id=p).get()
            if embedding:
                if variable not in dir(pt):
                    print(f'{variable} not in properties for {p}; is the variable embedded in infection_data, biology, '
                          f'or drug data? Label will be null for now.')
                    labels.append(None)
                else:
                    labels.append(pt[variable])
            elif embedding == 'infection':
                if variable == 'gram_status':
                    labels.append(gram_status(pt))
                elif variable == 'bugs':
                    labels.append(bugs(pt, multi_org='mixed', short_name=True))
                elif variable == 'org_type':
                    labels.append(org_type(pt))
                elif variable == 'hmbpp':
                    labels.append(hmbpp_ribo(pt, 'hmbpp'))
                elif variable == 'ribo':
                    labels.append(hmbpp_ribo(pt, 'ribo'))
            else:
                raise ValueError('Immunova currently only supports labelling with non-embedded variables or '
                                 'infectious data.')
        self.data['label'] = labels

    def missing_data(self):
        missing = pd.DataFrame(self.data.isna().sum() / len(self.data),
                               columns=['Missing (%)']).reset_index()
        sns.set(style="whitegrid")
        sns.set_color_codes("pastel")
        ax = sns.barplot(x="Missing (%)", y="index", data=missing, color="b")
        ax.set_ylabe('')
        return ax

    def _is_data(self):
        assert self.data.shape[0] > 0, 'DataFrame empty, have you populated the feature space yet?'

    def _get_numeric(self):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        return self.data.select_dtypes(include=numerics)

    def multicollinearity_matrix(self, exclude_columns: list or None, method='pearson', plot: bool = True,
                                 cmap='vlag', linewidths=.75, figsize: tuple = (10, 10), **kwargs):
        self._is_data()
        feature_space = self._get_numeric()
        if exclude_columns:
            feature_space = feature_space[[c for c in feature_space.columns if c not in exclude_columns]]
        corr_matrix = feature_space.corr(method=method)
        if plot:
            return sns.clustermap(corr_matrix, cmap=cmap, linewidths=linewidths, figsize=figsize, **kwargs)
        return corr_matrix

    def anova_one_way(self):
        self._is_data()
        from sklearn.feature_selection import f_classif
        feature_space = self._get_numeric()
        x = feature_space[[c for c in feature_space.columns if c != 'label']].values
        y = feature_space['label'].values
        return f_classif(x, y)

    def plot_variance(self, n: int = 100):
        self._is_data()
        feature_space = self._get_numeric()
        var = pd.DataFrame(feature_space.var().sort_values(ascending=False)[0:n], columns=['Variance']).reset_index()
        sns.set(style="whitegrid")
        sns.set_color_codes("pastel")
        ax = sns.barplot(x="Variance", y="index", data=var, color="b")
        ax.set_ylabe('')
        return ax

    def violin_plot(self, features: list, **kwargs):
        self._is_data()
        feature_space = self._get_numeric()
        x = feature_space[features]
        x['label'] = feature_space['label']
        x = x.melt(id_vars='label', var_name='Feature', value_name='Value')
        g = sns.FacetGrid(x, col='Features')
        g = g.map(sns.violinplot, 'label', 'Value', **kwargs)
        return g

    def l1_SVM(self, features: list, reg_search_space: tuple = (-2, 0, 50)):
        from sklearn.svm import LinearSVC
        from sklearn.utils import check_random_state
        check_random_state(42)
        feature_space = self._get_numeric()
        x = feature_space[features].values
        y = feature_space['label'].values
        self._is_data()

        cs = np.logspace(*reg_search_space)
        coefs = []
        for c in cs:
            clf = LinearSVC(C=c, penalty='l1', loss='squared_hinge', dual=False, tol=1e-3)
            clf.fit(feature_space, y)
            coefs.append(list(clf.coef_[0]))
        coefs = np.array(coefs)
        plt.figure(figsize=(10, 5))
        for i, col in enumerate(range(len(features))):
            plt.plot(cs, coefs[:, col], label=features[i])
        plt.xscale('log')
        plt.title('L1 penalty - Linear SVM')
        plt.xlabel('C')
        plt.ylabel('Coefficient value')
        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plt.show()

    def volcano_plot(self):
        pass

    def radial_visualiser(self, features: list, **kwargs):
        self._is_data()
        feature_space = self._get_numeric()
        x = feature_space[features]
        y = feature_space['labels']
        viz = RadViz(classes=y.unique(), **kwargs)
        viz.fit(x, y)
        viz.transform(x)
        viz.show()

    def dim_reduction_scatterplot(self, method: str = 'UMAP', n_components: int = 2, figsize: tuple = (10, 10),
                                  features: list or None = None, dim_reduction_kwargs: dict or None = None,
                                  matplotlib_kwargs: dict or None = None):
        self._is_data()
        assert n_components in [2, 3], 'n_components must have a value of 2 or 3'
        feature_space = self._get_numeric()
        if features is not None:
            feature_space = feature_space[features]
        if dim_reduction_kwargs is None:
            dim_reduction_kwargs = dict()
        if matplotlib_kwargs is None:
            matplotlib_kwargs = dict()
        embeddings = dimensionality_reduction(feature_space, features, method,
                                              n_components, return_embeddings_only=False,
                                              **dim_reduction_kwargs)

        if n_components == 2:
            return scprep.plot.scatter2d(embeddings, c=self.data['label'].values, ticks=False,
                                         label_prefix=method, discrete=True, legend_loc="lower left",
                                         legend_anchor=(1.04, 0), legend_title='Label',
                                         figsize=figsize, **matplotlib_kwargs)
        else:
            return scprep.plot.scatter3d(embeddings, c=self.data['label'].values, ticks=False,
                                         label_prefix=method, discrete=True, legend_loc="lower left",
                                         legend_anchor=(1.04, 0), legend_title='Label',
                                         figsize=figsize, **matplotlib_kwargs)

