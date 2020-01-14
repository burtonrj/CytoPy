from ..data.fcs_experiments import FCSExperiment
from ..data.fcs import ClusteringDefinition
from ..flow.gating.actions import Gating
from ..flow.clustering.main import SingleClustering
from ..flow.gating.transforms import apply_transform
from ..flow.gating.utilities import kde
from ..flow.utilities import progress_bar
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import pandas as pd
import numpy as np


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
            self.clusters = {k: v for k, v in self.clustering.clusters.items() if k not in exclude_clusters}
        else:
            self.clusters = self.clustering.clusters

    def population_summary(self) -> None:
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

    def cluster_summary(self) -> None:
        """
        Generates new columns for each cluster as follows:
            * {NAME}_n = number of cells in cluster
            * {NAME}_pot = proportion of cells in cluster relative to total cells in sample
        :return: None
        """
        for c_name, c_data in self.clusters.clusters.items():
            self.data[f'{c_name}_n'] = c_data['n_events']
            self.data[f'{c_name}_pot'] = c_data['prop_of_total']

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
        for pname1, pnode1 in progress_bar(self.populations.items()):
            for pname2, pnode2 in self.populations.items():
                if pname1 == pname2:
                    continue
                self.data[f'{pname1}:{pname2}'] = len(pnode1.index)/len(pnode2.index)
            for c_name, c_data in self.clusters.clusters.items():
                self.data[f'{pname1}:{c_name}'] = len(pnode1.index)/c_data['n_events']
        if self.verbose:
            print('-------- Calculating cluster ratios --------')
        for cname1, cdata1 in progress_bar(self.clusters.clusters.items()):
            for cname2, cdata2 in self.clusters.clusters.items():
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

    def fmo_stats_1d(self):
        for fmo_id in self.gating.fmo.keys():
            if self.verbose:
                print(f'-------- FMO Summary: {fmo_id} --------')
            for pop_id in progress_bar(self.populations.keys()):
                fmo_data = apply_transform(self.gating.get_fmo_data(pop_id, fmo_id))
                whole_data = apply_transform(self.gating.get_population_df(pop_id))
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

    def fmo_stats_multiple(self, fmo: list, population: str):
        if not self.gating.fmo.keys():
            if self.verbose:
                print(f'{self.gating.id} has no associated FMOs, aborting')
                return
        assert all([x in self.gating.fmo.keys() for x in fmo]), f'Some/all requested FMOs missing, valid FMOs include {self.gating.fmo.keys()}'
        assert population in self.gating.populations.keys(), f'Invalid population name, valid populations include: {self.gating.populations.keys()}'

        whole_data = apply_transform(self.gating.get_population_df(population))
        fmo_data = {name: apply_transform(self.gating.get_fmo_data(population, name)) for name in fmo}
        centroids = {name: data[fmo].median() for name, data in fmo_data.items()}
        euclidean_dist = {name: np.linalg.norm(whole_data[fmo].median - c) for name, c in centroids.items()}
        for name in euclidean_dist.keys():
            self.data[f'{name}_euclidean_dist'] = euclidean_dist[name]
        self.data[f'{fmo}_avg_euclidean_dist'] = np.mean(euclidean_dist.values())


class ExtractFromExperiment:
    def __init__(self):
        pass

    def metacluster_summaries(self):
        pass


class ExtractFromPatient:
    def __init__(self):
        pass

    def ratios(self):
        pass


class BuildFeatureSpace:
    def __init__(self):
        pass

    def insert_patient(self):
        pass
