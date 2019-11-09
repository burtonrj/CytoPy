from immunova.flow.gating.base import GateError
from immunova.flow.gating.transforms import apply_transform
from immunova.flow.gating.actions import Gating
from tqdm import tqdm
import numpy as np

def calculate_reference_sample(experiment):
    print('Warning: this process can take some time as comparisons are made between all samples in the experiment.')
    samples = experiment.list_samples()
    n = len(samples)
    norms = np.zeros(shape=[n, n])
    ref_ind = None
    for i in tqdm(range(0, n)):
        data_i = experiment.pull_sample_data(sample_id=samples[i], data_type='raw',
                                             output_format='matrix')
        data_i = data_i[[x for x in data_i.columns if x != 'Time']]
        data_i = apply_transform(data_i, transform_method='log_transform')
        if data_i is None:
            print(f'Error: failed to fetch data for {samples[i]}. Skipping.')
            continue

        cov_i = np.cov(data_i, rowvar=False)
        for j in range(0, n):
            data_j = experiment.pull_sample_data(sample_id=samples[j], data_type='raw',
                                                 output_format='matrix')
            data_j = data_j[[x for x in data_j.columns if x != 'Time']]
            data_j = apply_transform(data_j, transform_method='log_transform')
            cov_j = np.cov(data_j, rowvar=False)
            cov_diff = cov_i - cov_j
            norms[i, j] = np.linalg.norm(cov_diff, ord='fro')
            norms[j, i] = norms[i, j]
            avg = np.mean(norms, axis=1)
            ref_ind = np.argmin(avg)[0]
    if ref_ind is not None:
        return samples[ref_ind]
    else:
        raise DeepGateError('Error: unable to calculate sample with minimum average distance. You must choose'
                            ' manually.')

class DeepGateError(Exception):
    pass

class DeepGating:
    def __init__(self, experiment, reference_sample, samples='all', target_populations,
                 activation_f='softplus', transform='log'):
        self.experiment = experiment
        self.activation_f = activation_f
        self.transform = transform
        self.load_reference(reference_sample)

        if self.samples == 'all':
            self.samples = self.experiment.list_samples()
        else:
            self.samples = samples

    def load_reference(self, reference_sample):
        ref = Gating(self.experiment, reference_sample)
        if len(ref.populations) < 2:
            raise DeepGateError(f'Error: reference sample {reference_sample} does not contain any gated populations, '
                                f'please ensure that the reference sample has been gated prior to training.')
        for population, node in ref.populations.items():





