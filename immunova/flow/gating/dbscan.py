from immunova.flow.gating.defaults import Geom, GateOutput
from immunova.flow.gating.utilities import density_dependent_downsample
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import collections


def dbscan_gate(data, x, y, min_pop_size, distance_nn, expected_populations, core_only, sample: float = 0.2,
                sampling_method='uniform', density_sampling_params=None, nn=10):
    """
    Cluster based gating using the DBSCAN algorithm
    :param data: pandas dataframe representing compensated and transformed flow cytometry data
    :param x:
    :param y:
    :param min_pop_size: minimum population size for a population cluster
    :param distance_nn: nearest neighbour distance (smaller value will create tighter clusters)
    :param expected_populations: list of dictionaries defining expected populations. Expected key value pairs are;
    id - name of the population, and target - medoid of the expected population
    :param core_only: if True, only core samples in density clusters will be included
    :param sample: sample size to perform clustering on (due to computational complexity, N > 50000 is not
    recommended)
    :param sampling_method:
    :param density_sampling_params:
    :param nn: number of neighbours to use for upsampling with K-nearest neighbours (see documentation for more info)
    :return: dictionary of gating outputs (see documentation for internal standards)
    """
    output = GateOutput()
    data = data.copy()
    if sampling_method == 'uniform':
        s = data.sample(frac=sample)
    elif sampling_method == 'density':
        try:
            s = density_dependent_downsample(data, features=[x, y], **density_sampling_params)
        except TypeError or KeyError as e:
            output.error = 1
            output.error_msg = f'Error: invalid params for density dependent downsampling; {e}'
            return output
    else:
        s = data[[x, y]]
    db = DBSCAN(eps=distance_nn, min_samples=min_pop_size, algorithm='ball_tree', n_jobs=-1).fit(s[[x, y]])
    db_labels = db.labels_

    if core_only:
        non_core_mask = np.ones(len(db_labels), np.bool)
        non_core_mask[db.core_sample_indices_] = 0
        np.put(db_labels, non_core_mask, -1)

    if len(set(db_labels)) == 1:
        output.warnings.append('Failed to identify any distinct populations')

    if len(set([x for x in db_labels if x != -1])) != len(expected_populations):
        output.warnings.append(f'Expected {len(expected_populations)} populations, '
                               f'identified {len(np.where(db_labels != -1)[0])}')

    # Assign remaining events
    knn = KNeighborsClassifier(n_neighbors=nn, weights='distance', n_jobs=-1)
    knn.fit(s[[x, y]], db_labels)

    if sampling_method:
        data['labels'] = knn.predict(data[[x, y]])
    else:
        data['labels'] = db_labels

    populations = collections.defaultdict(list)
    for p in expected_populations:
        label = knn.predict(np.reshape(p['target'], (1, -1)))
        populations[label[0]].append(p['id'])

    for l, p_id in populations.items():
        if len(p_id) > 1:
            output.error_msg = f'Populations f{p_id} assigned to the same cluster {l}'
            output.error = 1
            return output
        if l == -1:
            output.warnings.append(f'Population {p_id} assigned to noise (i.e. population not found)')
    populations[-1] = ['noise']

    def rename_label(x):
        if x in populations.keys():
            return populations[x][0]
        return 'noise'
    data['labels'] = data['labels'].apply(rename_label)

    for p in populations.keys():
        output.add_child(name=p, idx=data[data['labels'] == p].index.values, geom=None)
    return output
