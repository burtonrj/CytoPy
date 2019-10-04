from immunova.flow.gating.defaults import GateOutput
from immunova.flow.gating.utilities import density_dependent_downsample
from immunova.flow.gating.dbscan import process_populations
import collections
import hdbscan


def __cluster_sample(data, frac, min_pop_size, method, out_obj, density_params=None):
    data = data.copy()
    if method == 'uniform':
        s = data.sample(frac=frac)
    elif method == 'density':
        try:
            if not density_params:
                density_params = dict()
            s = density_dependent_downsample(data, features=list(data.columns), **density_params)
        except TypeError or KeyError as e:
            out_obj.error = 1
            out_obj.error_msg = f'Error: invalid params for density dependent downsampling; {e}'
            return None, out_obj
    else:
        out_obj.error = 1
        out_obj.error_msg = 'Error: invalid sampling method, must be one of: `uniform`, `density`'
        return None, out_obj
    model = hdbscan.HDBSCAN(core_dist_n_jobs=-1, min_cluster_size=min_pop_size, prediction_data=True)
    model.fit(s)
    labels_probs = hdbscan.approximate_predict(model, data)
    return model, labels_probs


def hdbscan_gate(data, x, y, min_pop_size,
                 expected_populations, inclusion_threshold=None,
                 sample: float or None = None,
                 sampling_method='uniform',
                 density_sampling_params=None):
    output = GateOutput()
    data = data[[x, y]]
    if sample:
        model, labels_probs = __cluster_sample(data, sample, min_pop_size, sampling_method,
                                         output, density_sampling_params)
        if model is None:
            return output
        data['label'] = labels_probs[0]
        data['label_strength'] = labels_probs[1]
    else:
        model = hdbscan.HDBSCAN(core_dist_n_jobs=-1, min_cluster_size=min_pop_size, prediction_data=True)
        model.fit(data)
        data['label'] = model.labels_
        data['label_strength'] = model.probabilities_

    if inclusion_threshold is not None:
        mask = data['label_strength'] < inclusion_threshold
        data.loc[mask, 'label'] = -1
    population_assignments = collections.defaultdict(list)
    for p in expected_populations:
        label, _ = hdbscan.approximate_predict(model, [p['target']])
        population_assignments[label[0]].append(p['id'])
    output = process_populations(data, x, y, population_assignments,
                                 [p['id'] for p in expected_populations], output)
    return output

