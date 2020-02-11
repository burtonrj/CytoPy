from ..data.fcs_experiments import FCSExperiment
from ..data.subject import Subject, gram_status, bugs, hmbpp_ribo, org_type
from cytopy.flow.dim_reduction import dimensionality_reduction
from ..data.fcs import ClusteringDefinition
from ..flow.gating.actions import Gating
from ..flow.clustering.main import SingleClustering
from cytopy.flow.transforms import apply_transform
from ..flow.utilities import kde_multivariant
from ..flow.utilities import progress_bar
from sklearn.feature_selection import f_classif
from sklearn.svm import LinearSVC
from sklearn.utils import check_random_state
from multiprocessing import Pool, cpu_count
from itertools import combinations, product
from functools import partial
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
        - summarise_mfi: given a population or cluster name, subset data to contain just this population/cluster
        and return the mean fluorescence intensity for each fluorochrome as a dictionary where the key is the
        fluorochrome and the value is the MFI
        - fmo_stats_1d: given a target population, for each FMO that exists for the associated sample, calculate the
        following descriptives:
            * Kolmogorov–Smirnov test statistic and p-value: quantifies the distance between the EDF of the sample
            data and the CDF of the FMO; this provides a statistical comparison between the populations and indicates
            if the removal of the fluorochrome significantly changes the distribution
            * Fold change in MFI: the fold change in MFI in the FMO data relative to the sample data
            * KL Divergence: the


    """
    def __init__(self, sample_id: str, experiment: FCSExperiment,
                 clustering_definition: ClusteringDefinition,
                 include_populations: list or None = None,
                 include_clusters: list or None = None,
                 verbose: bool = True, sml_profiles: dict or None = None,
                 include_controls: bool = False,
                 transform: str or None = 'logicle'):
        # Other attributes
        self.include_controls = include_controls
        self.verbose = verbose
        self.sml_profiles = sml_profiles
        self.transform = transform
        # Load data
        self.gating = Gating(experiment, sample_id, include_controls=include_controls)
        clustering = SingleClustering(clustering_definition=clustering_definition)
        clustering.load_data(experiment=experiment, sample_id=sample_id)
        self.raw_data = self.gating.data.copy()
        if transform is not None:
            self.raw_data = apply_transform(data=self.raw_data, transform_method=transform)
        self.fluorochromes = [c for c in self.raw_data.columns if all([fs not in c for fs in ['FSC', 'SSC', 'Time']])]
        # Init data-frame
        self.data = pd.DataFrame({'pt_id': [clustering.data.pt_id.values[0]]})
        # Collect gated populations and clusters
        if include_populations is None:
            include_populations = self.gating.populations.keys()
        self.gated_populations = {k: v for k, v in self.gating.populations.items() if k in include_populations}
        if include_clusters is None:
            include_clusters = experiment.meta_cluster_ids
        meta_cluster_ids = experiment.meta_cluster_ids
        self.clusters = {_id: {'n_events': 0,
                               'prop_of_root': 0.0,
                               'index': np.array([])}
                         for _id in meta_cluster_ids if _id in include_clusters}
        for k, v in clustering.clusters.items():
            mid = v.get('meta_cluster_id')
            self.clusters[mid]['n_events'] = self.clusters[mid].get('n_events', 0) + len(v.get('index', []))
            self.clusters[mid]['prop_of_root'] = self.clusters[mid].get('prop_of_root', 0) + v.get('prop_of_root', 0)
            self.clusters[mid]['index'] = np.unique(np.concatenate((self.clusters[mid].get('index', []),
                                                                    v.get('index', [])), axis=0), axis=0)

    def population_summary(self, include_mfi: bool = True) -> None:
        """
        Generates new columns for each population as follows:
            * {NAME}_n = number of cells in populations
            * {NAME}_pop = proportion of cells in population relative to parent population
            * {NAME}_pot = proportion of cells in population relative to total cells in sample
        :return: None
        """
        if self.verbose:
            print('...summarise populations')
        for pop_name, node in self.gated_populations.items():
            self.data[f'{pop_name}_n'] = len(node.index)
            self.data[f'{pop_name}_pop'] = node.prop_of_parent
            self.data[f'{pop_name}_pot'] = node.prop_of_total
            if include_mfi:
                mfi = self.summarise_mfi(population_name=pop_name)
                if not mfi:
                    continue
                for marker, mfi_ in mfi.items():
                    self.data[f'{pop_name}_{marker}_MFI'] = mfi_

    def summarise_mfi(self, population_name: str or None = None, cluster_name: str or None = None):
        """
        Return a dictionary of MFI values for each fluorochrome for the given population/cluster.
        Expects either a population name or a cluster name, not both. Raises ValueError is None provided for both, or
        a value is given for both arguments.
        :param population_name: name of population to summarise
        :param cluster_name: name of cluster to summarise
        :return: Dictionary of MFI values
        """
        if population_name is not None and cluster_name is not None:
            raise ValueError('Provide the name of a population OR the name of a cluster, not both')
        if population_name is not None:
            return self.raw_data.loc[self.gated_populations[population_name].index, self.fluorochromes].mean().to_dict()
        if cluster_name is not None:
            return self.raw_data.loc[self.clusters[cluster_name]['index'], self.fluorochromes].mean().to_dict()
        raise ValueError('Must provide either name of a cluster or name of a population')

    def cluster_summary(self, include_mfi: bool = False) -> None:
        """
        Generates new columns for each cluster as follows:
            * {NAME}_n = number of cells in cluster
            * {NAME}_pot = proportion of cells in cluster relative to total cells in sample
        :return: None
        """
        if self.verbose:
            print('...summarise clusters')
        for c_name, c_data in self.clusters.items():
            self.data[f'{c_name}_n'] = c_data['n_events']
            self.data[f'{c_name}_pot'] = c_data['prop_of_root']
            if include_mfi:
                mfi = self.summarise_mfi(cluster_name=c_name)
                if not mfi:
                    continue
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
            print('...calculating population ratios')
        combos = list(combinations(list(self.gated_populations.keys()), 2))
        for p1, p2 in combos:
            pn1, pn2 = self.gated_populations.get(p1), self.gated_populations.get(p2)
            if len(pn1.index) == 0 or len(pn2.index) == 0:
                continue
            self.data[f'{p1}:{p2}'] = len(pn1.index) / len(pn2.index)
        combos = list(product(self.gated_populations.keys(), self.clusters.keys()))
        for p, c in combos:
            pn, cd = self.gated_populations.get(p), self.clusters.get(c)
            if len(pn.index) == 0 or len(cd.get('index')) == 0:
                continue
            self.data[f'{p}:{c}'] = len(pn.index) / cd.get('n_events')

        if self.verbose:
            print('...calculating cluster ratios')
        combos = list(combinations(list(self.clusters.keys()), 2))
        for c1, c2 in combos:
            cd1, cd2 = self.clusters.get(c1), self.clusters.get(c2)
            if cd1.get('n_events') == 0 or cd2.get('n_events') == 0:
                continue
            self.data[f'{c1}:{c2}'] = cd1['n_events'] / cd2['n_events']

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

    def fmo_stats_1d(self, populations: list):
        assert self.include_controls, 'include_controls currently set to False, reinitialise object with argument ' \
                                      'set to True to include FMO data'
        for fmo_id in self.gating.fmo.keys():
            if self.verbose:
                print(f'...FMO Summary: {fmo_id}')
            if self.verbose:
                print('...performing KDE calculations')

            for pop_id in progress_bar(populations):
                fmo_data = apply_transform(self.gating.get_fmo_data(pop_id, fmo_id, self.sml_profiles),
                                           transform_method=self.transform)
                whole_data = self.raw_data.loc[self.gated_populations.get(pop_id).index, :]
                ks_stat, p = stats.ks_2samp(fmo_data[fmo_id].values, whole_data[fmo_id].values)
                self.data[f'{pop_id}_{fmo_id}_ks_statistic'] = ks_stat
                self.data[f'{pop_id}_{fmo_id}_ks_statistic_pval'] = p
                self.data[f'{pop_id}_{fmo_id}_fold_change_MFI'] = self._relative_fold_change(whole_data[fmo_id].values,
                                                                                             fmo_data[fmo_id].values)
                if whole_data.shape[0] > 1000:
                    whole_data = whole_data.sample(1000)
                if fmo_data.shape[0] > 1000:
                    fmo_data = whole_data.sample(1000)
                w_probs = kde_multivariant(x=whole_data[fmo_id].values.reshape(-1, 1))
                f_probs = kde_multivariant(x=fmo_data[fmo_id].values.reshape(-1, 1))
                self.data[f'{pop_id}_{fmo_id}_kl_divergence'] = stats.entropy(pk=w_probs, qk=f_probs)
        if self.verbose:
            print('...Completed!')

    def fmo_stats_multiple(self, fmo: list, population: str, transform: str = 'logicle'):
        assert self.include_controls, 'include_controls currently set to False, reinitialise object with argument ' \
                                      'set to True to include FMO data'
        if not self.gating.fmo.keys():
            if self.verbose:
                print(f'{self.gating.id} has no associated FMOs, aborting')
                return
        assert all([x in self.gating.fmo.keys() for x in fmo]), f'Some/all requested FMOs missing, ' \
                                                                f'valid FMOs include {self.gating.fmo.keys()}'
        assert population in self.gating.populations.keys(), f'Invalid population name, ' \
                                                             f'valid populations include: ' \
                                                             f'{self.gating.populations.keys()}'
        if self.verbose:
            print('...calculating summary of multi-dimensional FMO topology')
        whole_data = self.raw_data.loc[self.gated_populations.get(population).index, :]
        fmo_data = {name: apply_transform(self.gating.get_fmo_data(population, name, self.sml_profiles),
                                          transform_method=transform)
                    for name in fmo}
        centroids = {name: data[fmo].median() for name, data in fmo_data.items()}
        euclidean_dist = {name: np.linalg.norm(whole_data[fmo].median - c) for name, c in centroids.items()}
        for name in euclidean_dist.keys():
            self.data[f'{name}_euclidean_dist'] = euclidean_dist[name]
        self.data[f'{fmo}_avg_euclidean_dist'] = np.mean(euclidean_dist.values())
        if self.verbose:
            print('...Completed!')


def _fetch_experiment_data(sample_id: str, experiment: FCSExperiment, target_ce: ClusteringDefinition,
                           include_populations: list or None = None, include_clusters: list or None = None,
                           include_mfi: bool = False, fmo1d_populations: dict or None = None,
                           multi_dim_fmo_comparisons: dict or None = None, transform: str = 'logicle',
                           sml_profiles: dict or None = None):
    data = pd.DataFrame()
    include_controls = False
    if fmo1d_populations is not None or multi_dim_fmo_comparisons is not None:
        include_controls = True
    extraction = Extract(sample_id=sample_id,
                         experiment=experiment,
                         clustering_definition=target_ce,
                         include_populations=include_populations,
                         include_clusters=include_clusters, verbose=False,
                         sml_profiles=sml_profiles,
                         include_controls=include_controls,
                         transform=transform)
    extraction.cluster_summary(include_mfi)
    extraction.population_summary(include_mfi)
    extraction.ratios()
    if fmo1d_populations is not None:
        extraction.fmo_stats_1d(fmo1d_populations)
    if multi_dim_fmo_comparisons is not None:
        for fmo in multi_dim_fmo_comparisons:
            assert all([x in fmo.keys() for x in ['comparisons', 'population']]), \
                'multi_dim_fmo_comparisons should be a list of dictionary objects, each with a key ' \
                '"comparisons", being a list of FMOs to compare, and "population", ' \
                'the name of the population for which the comparison is relevant'
            extraction.fmo_stats_multiple(fmo=fmo.get('comparisons'), population=fmo.get('population'))
    data = pd.concat([data, extraction.data])
    return data


class BuildFeatureSpace:
    def __init__(self, exclude_samples: list or None = None, exclude_patients: list or None = None,
                 population_transform: str = 'logicle', include_mfi: bool = False, path: str or None = None):
        self.exclude_samples = exclude_samples
        self.exclude_patients = exclude_patients
        self.population_transform = population_transform
        self.include_mfi = include_mfi
        self.data = None
        if path is not None:
            self.data = pd.read_csv(path)

    def add_experiment(self, experiment: FCSExperiment,
                       meta_clustering_definition: ClusteringDefinition,
                       include_populations: list or None = None,
                       include_clusters: list or None = None,
                       fmo1d_populations: list or None = None,
                       multi_dim_fmo_comparisons: list or None = None,
                       prefix: str or None = None,
                       sml_profiles: dict or None = None):

        target_ce = ClusteringDefinition.objects(clustering_uid=meta_clustering_definition.meta_clustering_uid_target).get()
        samples = experiment.list_samples()
        if self.exclude_samples is not None:
            samples = [s for s in samples if s not in self.exclude_samples]
        include_controls = False
        if fmo1d_populations is not None or multi_dim_fmo_comparisons is not None:
            include_controls = True
        fetch = partial(_fetch_experiment_data, experiment=experiment,
                        target_ce=target_ce, include_populations=include_populations,
                        include_clusters=include_clusters, include_mfi=self.include_mfi,
                        fmo1d_populations=fmo1d_populations, multi_dim_fmo_comparisons=multi_dim_fmo_comparisons,
                        transform=self.population_transform, sml_profiles=sml_profiles)
        if not include_controls:
            pool = Pool(cpu_count())
            data = pd.concat(progress_bar(pool.imap(fetch, samples), total=len(samples)))
            pool.close()
            pool.join()
        else:
            print('Warning: unable to use high-level multi-processing when summarising FMO data, therefore this '
                  'process could take some time')
            data = list()
            for s in progress_bar(samples):
                data.append(fetch(s))
            data = pd.concat(data)
        if prefix is not None:
            names = {c: f'{prefix}_{c}' for c in data.columns if c != 'pt_id'}
            data = data.rename(columns=names)
        if self.data is None:
            self.data = data
        else:
            self.data = self.data.merge(data, how='outer', on='pt_id')

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
            pt = Subject.objects(patient_id=p).get()
            if embedding is None:
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

    def is_missing(self, threshold):
        missing = self.data.isna().sum()[self.data.isna().sum() / self.data.shape[0] > threshold].index.values
        print(f'A total of {len(missing)} columns have missing data exceeding the given threshold')
        return list(missing)

    def plot_missing(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(self.data.isna(), ax=ax, cbar=False)
        ax.set_xticks([])
        fig.show()

    def drop_missing(self, threshold):
        missing = self.is_missing(threshold)
        print('Dropping columns...')
        self.data = self.data[[c for c in self.data.columns if c not in missing]]
        print('Done!')

    def _is_data(self):
        assert self.data.shape[0] > 0, 'DataFrame empty, have you populated the feature space yet?'

    def _get_numeric(self):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        return self.data.select_dtypes(include=numerics)

    def multicollinearity_matrix(self, exclude_columns: list or None = None, method='pearson', plot: bool = True,
                                 cmap='vlag', figsize: tuple = (10, 10), **kwargs):
        self._is_data()
        feature_space = self._get_numeric()
        if exclude_columns:
            feature_space = feature_space[[c for c in feature_space.columns if c not in exclude_columns]]
        corr_matrix = feature_space.corr(method=method)
        if plot:
            return sns.clustermap(corr_matrix, cmap=cmap, figsize=figsize, **kwargs)
        return corr_matrix

    def drop_high_multicollinearity(self, corr_threshold: float = 0.5, method: str = 'pearson'):
        self._is_data()
        feature_space = self._get_numeric()
        corr_matrix = feature_space.corr(method=method)
        drop_columns = set()
        for c in corr_matrix.columns:
            high_mc = corr_matrix[c][corr_matrix[c] > corr_threshold]
            keep = np.argmax(feature_space[high_mc.index].var())
            drop = set(v for v in high_mc if v != keep)
            drop_columns = drop_columns.union(drop)
        self.data = self.data[[c for c in self.data.columns if c not in drop_columns]]

    def anova_one_way(self):
        self._is_data()
        assert 'label' in self.data.columns, 'Dataset is not labelled!'
        features = [c for c in self.data.columns if c not in ['label', 'pt_id']]
        data = self._handle_null_label()
        x = data[features].values
        y = data['label'].values
        f, pval = f_classif(x, y)
        return pd.DataFrame({'Feature': features, 'F-value': f, 'p-value': pval})

    def plot_variance(self, n: int = 100, figsize=(20, 8), **kwargs):
        var = self.rank_variance(n=n)
        var['Variance'] = np.log(var['Variance'])
        sns.set(style="whitegrid")
        sns.set_color_codes("pastel")
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.barplot(x="Variance", y="index", data=var, color="b", ax=ax, **kwargs)
        ax.set_ylabel('')
        ax.set_xlabel('log(Variance)')
        return ax

    def rank_variance(self, n: int = 100):
        self._is_data()
        feature_space = self._get_numeric()
        var = pd.DataFrame(feature_space.var().sort_values(ascending=False)[0:n], columns=['Variance']).reset_index()
        return var

    def violin_plot(self, feature: str, figsize=(5, 5), palette='vlag', point_size=5, point_linewidth=0.5,
                    point_color='.5', y_label=None, x_label=None, style='white', font_scale=1.2):
        sns.set(font_scale=font_scale)
        sns.set_style(style)
        self._is_data()
        assert 'label' in self.data.columns, 'Dataset is not labelled!'
        data = self._handle_null_label()
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x=feature, y='label', data=data, palette=palette, ax=ax)
        sns.swarmplot(x=feature, y='label', data=data, size=point_size,
                      color=point_color, linewidth=point_linewidth, ax=ax)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if x_label is not None:
            ax.set_xlabel(x_label)
        fig.show()

    def swarmplot(self, features: list, figsize=(6, 6), xlabel=None, ylabel=None,
                  style='white', font_scale=1.2, **kwargs):
        sns.set(font_scale=font_scale)
        sns.set(style=style)
        self._is_data()
        assert 'label' in self.data.columns, 'Dataset is not labelled!'
        features = features + ['label']
        data = pd.melt(self.data[features], "label", var_name="Feature")
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.swarmplot(x="Feature", y="value", hue="label", data=data, ax=ax, **kwargs)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('')
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        ax.set_ylabel('')
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        fig.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        fig.show()

    def l1_SVM(self, features: list, reg_search_space: tuple = (-2, 0, 50)):
        check_random_state(42)
        x = self.data[features].values
        y = self.data['label'].values
        self._is_data()

        cs = np.logspace(*reg_search_space)
        coefs = []
        for c in cs:
            clf = LinearSVC(C=c, penalty='l1', loss='squared_hinge', dual=False, tol=1e-3)
            clf.fit(x, y)
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

    def radial_visualiser(self, features: list, **kwargs):
        self._is_data()
        feature_space = self._get_numeric()
        x = feature_space[features]
        y = feature_space['labels']
        viz = RadViz(classes=y.unique(), **kwargs)
        viz.fit(x, y)
        viz.transform(x)
        viz.show()

    def _handle_null_label(self, data: pd.DataFrame or None = None):
        if data is None:
            data = self.data.copy()
        missing_labels = list(data.loc[data['label'].isnull()]['pt_id'].values)
        if missing_labels:
            print(f'The following patients have missing labels and will be omitted: {missing_labels}')
        return data.loc[~data['label'].isnull()]

    def dim_reduction_scatterplot(self, method: str = 'UMAP', n_components: int = 2, figsize: tuple = (10, 10),
                                  dim_reduction_kwargs: dict or None = None, features: list or None = None,
                                  matplotlib_kwargs: dict or None = None):
        self._is_data()
        assert 'label' in self.data.columns, 'Dataset is not labelled!'
        assert n_components in [2, 3], 'n_components must have a value of 2 or 3'
        data = self._handle_null_label()
        if dim_reduction_kwargs is None:
            dim_reduction_kwargs = dict()
        if matplotlib_kwargs is None:
            matplotlib_kwargs = dict()
        if features is None:
            features = [c for c in data.columns if c not in ['pt_id', 'label']]
        embeddings = dimensionality_reduction(data, features, method,
                                              n_components, return_embeddings_only=True,
                                              **dim_reduction_kwargs)

        if n_components == 2:
            return scprep.plot.scatter2d(embeddings, c=data['label'].values, ticks=False,
                                         label_prefix=method, discrete=True, legend_loc="lower left",
                                         legend_anchor=(1.04, 0), legend_title='Label',
                                         figsize=figsize, **matplotlib_kwargs)
        else:
            return scprep.plot.scatter3d(embeddings, c=data['label'].values, ticks=False,
                                         label_prefix=method, discrete=True, legend_loc="lower left",
                                         legend_anchor=(1.04, 0), legend_title='Label',
                                         figsize=figsize, **matplotlib_kwargs)

