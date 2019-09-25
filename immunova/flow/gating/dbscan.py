from immunova.flow.gating.defaults import GateOutput, Geom
from immunova.flow.gating.utilities import density_dependent_downsample
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
import collections


def dbscan_gate(data, x, y, min_pop_size, distance_nn, child_populations, core_only, sample: float = 0.2,
                sampling_method='uniform', density_sampling_params=None, nn=10):
    """
    Cluster based gating using the DBSCAN algorithm
    :param data: pandas dataframe representing compensated and transformed flow cytometry data
    :param x:
    :param y:
    :param min_pop_size: minimum population size for a population cluster
    :param distance_nn: nearest neighbour distance (smaller value will create tighter clusters)
    :param child_populations:
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
    if data.shape[0] == 0:
        output.warnings.append('No events in parent population!')
        for c, _ in child_populations.items:
            output.add_child(name=c, idx=[], geom=Geom(shape='cluster', x=x, y=y))
    if sampling_method == 'uniform':
        s = data.sample(frac=sample)
    elif sampling_method == 'density':
        try:
            if not density_sampling_params:
                density_sampling_params = dict()
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

    if len(set([x for x in db_labels if x != -1])) != len(child_populations):
        output.warnings.append(f'Expected {len(child_populations)} populations, '
                               f'identified {len(np.where(db_labels != -1)[0])}')

    # Assign remaining events
    knn = KNeighborsClassifier(n_neighbors=nn, weights='distance', n_jobs=-1)
    knn.fit(s[[x, y]], db_labels)

    if sampling_method:
        data['labels'] = knn.predict(data[[x, y]])
    else:
        data['labels'] = db_labels
    # Predict what cluster the mediod of expected populations falls into
    populations = collections.defaultdict(list)
    for name, c in child_populations.items():
        target = c['target']
        label = knn.predict(np.reshape(target, (1, -1)))
        populations[label[0]].append(name)
    # Check for duplicate assignment of expected population or assignment to noise
    for l, p_id in populations.items():
        if len(p_id) > 1:
            output.warnings.append(f'Populations f{p_id} assigned to the same clusters {l}')
        if l == -1:
            output.warnings.append(f'Population {p_id} assigned to noise (i.e. population not found)')
    populations[-1] = ['noise']

    def rename_label(x):
        if x in populations.keys():
            return populations[x][0]
        return 'noise'
    data['labels'] = data['labels'].apply(rename_label)
    for p in list(child_populations.keys()):
        g = Geom(shape='cluster', x=x, y=y)
        output.add_child(name=p, idx=data[data['labels'] == p].index.values, geom=g.as_dict())
    return output
