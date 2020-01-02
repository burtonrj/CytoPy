from immunova.flow.gating.actions import Gating
from immunova.flow.clustering.main import SingleClustering
from immunova.flow.gating.transforms import apply_transform
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import pandas as pd
import numpy as np


class Extract:
    def __init__(self, gating: Gating, clusters: SingleClustering, exclude_populations: list or None = None):
        self.gating = gating
        self.clusters = clusters
        self.data = pd.DataFrame({'sample_id': [gating.id], 'pt_id': clusters.data.pt_id.values[0]})
        if exclude_populations is None:
            self.populations = {k: v for k, v in self.gating.populations.items() if k not in exclude_populations}
        else:
            self.populations = self.gating.populations

    def population_summaries(self):
        for pop_name, node in self.populations.items():
            self.data[f'{pop_name}_n'] = len(node.index)
            self.data[f'{pop_name}_pop'] = node.prop_of_parent
            self.data[f'{pop_name}_pot'] = node.prop_of_total

    def cluster_summaries(self):
        for c_name, c_data in self.clusters.clusters.items():
            self.data[f'{c_name}_n'] = c_data['n_events']
            self.data[f'{c_name}_pot'] = c_data['prop_of_total']

    def ratios(self):
        for pname1, pnode1 in self.populations.items():
            for pname2, pnode2 in self.populations.items():
                if pname1 == pname2:
                    continue
                self.data[f'{pname1}:{pname2}'] = len(pnode1.index)/len(pnode2.index)
            for c_name, c_data in self.clusters.clusters.items():
                self.data[f'{pname1}:{c_name}'] = len(pnode1.index)/c_data['n_events']
        for cname1, cdata1 in self.clusters.clusters.items():
            for cname2, cdata2 in self.clusters.clusters.items():
                if cname1 == cname2:
                    continue
                self.data[f'{cname1}:{cname2}'] = cdata1['n_events'] / cdata2['n_events']

    def _relative_fold_increase(self, x, y, centre='median'):
        x = MinMaxScaler().fit_transform(x.reshape(-1, 1))
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1))
        if centre == 'median':
            return np.median(x)/np.median(y)
        return np.mean(x)/np.mean(y)

    def fmo_stats(self):
        for fmo_id in self.gating.fmo.keys():
            for pop_id in self.populations.keys():
                fmo_data = apply_transform(self.gating.get_fmo_data(pop_id, fmo_id))
                whole_data = apply_transform(self.gating.get_population_df(pop_id))
                ks_stat, p = stats.ks_2samp(fmo_data[fmo_id].values, whole_data[fmo_id].values)
                self.data[f'{pop_id}_{fmo_id}_ks_statistic'] = ks_stat
                self.data[f'{pop_id}_{fmo_id}_ks_statistic_pval'] = p
                self.data[f'{pop_id}_{fmo_id}_fold_change_MFI'] = ks_stat
                self.data[f'{pop_id}_{fmo_id}_relative_entropy'] = ''

    def centroid_euclidean_separation(self):
        pass

    def relative_entropy(self):
        pass


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
