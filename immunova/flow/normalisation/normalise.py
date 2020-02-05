from immunova.data.fcs_experiments import FCSExperiment
from immunova.data.fcs import Normalisation
from immunova.data.patient import Patient
from immunova.flow.gating.actions import Gating
from immunova.flow.normalisation.MMDResNet import MMDNet
from immunova.flow.supervised.ref import calculate_reference_sample
from immunova.flow.utilities import progress_bar, kde_multivariant, hellinger_dot, ordered_load_transform
from immunova.flow.dim_reduction import dimensionality_reduction
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.stats import entropy as kl
from scipy.spatial.distance import jensenshannon as jsd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
np.random.seed(42)


def jsd_divergence(x, y):
    div = jsd(x, y)
    assert div is not None, 'JSD is null'
    if div in [np.inf, -np.inf]:
        return 1
    return div


def kl_divergence(x, y):
    div = kl(x, y)
    assert div is not None, 'KL divergence is Null'
    return div


def indexed_kde(named_x: tuple, kde_f: callable):
    q = kde_f(named_x[1])
    return named_x[0], q


class CalibrationError(Exception):
    pass


class Normalise:
    """
    Class for normalising a flow cytometry file using a reference target file
    """
    def __init__(self, experiment: FCSExperiment, source_id: str, root_population: str,
                 features: list, reference_sample: str or None = None, transform: str = 'logicle',
                 **mmdresnet_kwargs):
        """
        Constructor for Normalise object
        :param experiment: FCSExperiment object
        :param source_id: sample ID for the file to normalise
        :param reference_sample: sample ID to use as target distribution (leave as 'None' if unknown and use the
        `calculate_reference_sample` method to find an optimal reference sample)
        :param transform: transformation to apply to raw FCS data (default = 'logicle')
        :param mmdresnet_kwargs: keyword arguments for MMD-ResNet
        """
        self.experiment = experiment
        self.source_id = source_id
        self.root_population = root_population
        self.transform = transform
        self.features = [c for c in features if c.lower() != 'time']
        self.calibrator = MMDNet(data_dim=len(self.features), **mmdresnet_kwargs)
        self.reference_sample = reference_sample or None

        if source_id not in self.experiment.list_samples():
            raise CalibrationError(f'Error: invalid target sample {source_id}; '
                                   f'must be one of {self.experiment.list_samples()}')
        else:
            self.source = self.__load_and_transform(sample_id=source_id)

    def load_model(self, model_path: str) -> None:
        """
        Load an existing MMD-ResNet model from .h5 file
        :param model_path: path to model .h5 file
        :return: None
        """
        self.calibrator.load_model(path=model_path)

    def calculate_reference_sample(self) -> None:
        """
        Calculate the optimal reference sample. This is performed as described in Li et al paper
        (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860171/) on DeepCyTOF: for every 2 samples i, j compute
        the Frobenius norm of the difference between their covariance matrics and then select the sample
         with the smallest average distance to all other samples. Optimal sample assigned to self.reference_sample.
        :return: None
        """
        self.reference_sample = calculate_reference_sample(self.experiment)
        print(f'{self.reference_sample} chosen as optimal reference sample.')

    def __load_and_transform(self, sample_id) -> pd.DataFrame:
        """
        Given a sample ID, retrieve the sample data and apply transformation
        :param sample_id: ID corresponding to sample for retrieval
        :return: transformed data as a list of dictionary objects:
        {id: file id, typ: type of file (either 'complete' or 'control'), data: Pandas DataFrame}
        """
        gating = Gating(experiment=self.experiment, sample_id=sample_id)
        data = gating.get_population_df(self.root_population,
                                        transform=True,
                                        transform_method=self.transform,
                                        transform_features=self.features)
        if data is None:
            raise CalibrationError(f'Error: unable to load data for population {self.root_population}')
        return data[self.features]

    def __put_norm_data(self, file_id: str, data: pd.DataFrame):
        """
        Given a file ID and a Pandas DataFrame, fetch the corresponding File document and insert the normalised data.
        :param file_id: ID for file for insert
        :param data: Pandas DataFrame of normalised and transformed data
        :return:
        """
        source_fg = self.experiment.pull_sample(self.source_id)
        file = [f for f in source_fg.files if f.file_id == file_id][0]
        norm = Normalisation()
        norm.put(data.values, root_population=self.root_population, method='MMD-ResNet')
        file.norm = norm
        source_fg.save()

    def normalise_and_save(self) -> None:
        """
        Apply normalisation to source sample and save result to the database.
        :return:
        """
        if self.calibrator.model is None:
            print('Error: normalisation model has not yet been calibrated')
            return None
        print(f'Saving normalised data for {self.source_id} population {self.root_population}')
        data = self.calibrator.model.predict(self.source)
        data = pd.DataFrame(data, columns=self.source.columns)
        self.__put_norm_data(self.source_id, data)
        print('Save complete!')

    def calibrate(self, initial_lr=1e-3, lr_decay=0.97, evaluate=False, save=False) -> None:
        """
        Train the MMD-ResNet to minimise the Maximum Mean Discrepancy between our target and source sample.
        :param initial_lr: initial learning rate (default = 1e-3)
        :param lr_decay: decay rate for learning rate (default = 0.97)
        :param evaluate: If True, the performance of the training is evaluated and a PCA plot of aligned distributions
        is generated (default = False).
        :param save: If True, normalisation is applied to source sample and saved to database.
        :return: None
        """
        if self.reference_sample is None:
            print('Error: must provide a reference sample for training. This can be provided during initialisation, '
                  'by assigning a valid value to self.reference_sample, or by calling `calculate_reference_sample`.')
            return
        if self.reference_sample not in self.experiment.list_samples():
            print(f'Error: invalid reference sample {self.reference_sample}; must be one of '
                  f'{self.experiment.list_samples()}')
            return
        # Load and transform data
        target = self.__load_and_transform(self.reference_sample)
        print('Warning: calibration can take some time and is dependent on the sample size')
        self.calibrator.fit(self.source, target, initial_lr, lr_decay, evaluate=evaluate)
        print('Calibration complete!')
        if save:
            self.normalise_and_save()


class EvaluateBatchEffects:
    def __init__(self, experiment: FCSExperiment, root_population: str,
                 transform: str = 'logicle', scale: str or None = None,
                 sample_n: str or None = 10000, exclude: list or None = None):
        self.experiment = experiment
        self.sample_ids = experiment.list_samples()
        self.transform = transform
        self.scale = scale
        self.sample_n = sample_n
        self.root_population = root_population
        self.data = self.load_data(experiment, exclude)
        self.kde_cache = dict()

    def load_data(self, experiment: FCSExperiment, exclude: list or None = None) -> dict:
        """
        Load new dataset from given FCS Experiment
        :param experiment:
        :param exclude:
        :return:
        """
        self.experiment = experiment
        self.sample_ids = experiment.list_samples()
        self.kde_cache = dict()
        if exclude is not None:
            self.sample_ids = [s for s in self.sample_ids if s not in exclude]
        lt = partial(ordered_load_transform,
                     experiment=experiment,
                     root_population=self.root_population,
                     transform=self.transform,
                     scale=self.scale,
                     sample_n=self.sample_n)
        pool = Pool(cpu_count())
        samples_df = pool.map(lt, self.sample_ids)
        samples = dict()
        for sample_id, df in samples_df:
            if df is not None:
                samples[sample_id] = df

        pool.close()
        pool.join()
        return samples

    def marker_variance(self, reference_id: str,
                        markers: list, comparison_samples: list,
                        figsize: tuple = (10, 10)):
        """
        For a given reference sample and a list of markers of interest, create a grid of KDE plots with the reference
        sample given in red and comparison samples given in blue.
        :param reference_id: this sample will appear in red on KDE plots
        :param markers: list of valid marker names to plot in KDE grid
        :param comparison_samples: list of valid sample names in the current experiment
        :param figsize: size of resulting figure
        :return: Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        nrows = math.ceil(len(markers)/3)
        assert reference_id in self.sample_ids, 'Invalid reference ID for experiment currently loaded'
        assert all([x in self.sample_ids for x in comparison_samples]), 'Invalid sample IDs for experiment currently ' \
                                                                        'loaded'
        reference = self.data[reference_id]
        print('Plotting...')
        i = 0
        for marker in progress_bar(markers):
            i += 1
            if marker not in reference.columns:
                print(f'{marker} absent from reference sample, skipping')
            ax = fig.add_subplot(nrows, 3, i)
            ax = sns.kdeplot(reference[marker], shade=True, color="r", ax=ax)
            ax.set_title(f'Total variance in {marker}')
            for d in comparison_samples:
                d = self.data[d]
                if marker not in d.columns:
                    continue
                ax = sns.kdeplot(d[marker], color='b', shade=False, alpha=0.5, ax=ax)
                ax.get_legend().remove()
        fig.show()

    def dim_reduction_grid(self, reference_id, comparison_samples: list, features: list, figsize: tuple = (10, 10),
                           method: str = 'PCA', kde: bool = False):
        """
        Generate a grid of embeddings using a valid dimensionality reduction technique, in each plot a reference sample
        is shown in blue and a comparison sample in red. The reference sample is conserved across all plots.
        :param reference_id:
        :param comparison_samples:
        :param features:
        :param figsize:
        :param method:
        :param kde:
        :return:
        """
        fig = plt.figure(figsize=figsize)
        nrows = math.ceil(len(comparison_samples)/3)
        assert reference_id in self.sample_ids, 'Invalid reference ID for experiment currently loaded'
        assert all([x in self.sample_ids for x in comparison_samples]), 'Invalid sample IDs for experiment currently ' \
                                                                        'loaded'
        print('Plotting...')
        reference = self.data[reference_id]
        reference['label'] = 'Target'
        assert all([f in reference.columns for f in features]), f'Invalid features, must be in: {reference.columns}'
        reference, reducer = dimensionality_reduction(reference,
                                                      features=features,
                                                      method=method,
                                                      n_components=2,
                                                      return_reducer=True)
        i = 0
        for s in progress_bar(comparison_samples):
            i += 1
            df = self.data[s]
            df['label'] = 'Comparison'
            ax = fig.add_subplot(nrows, 3, i)
            if not all([f in df.columns for f in features]):
                print(f'Features missing from {s}, skipping')
                continue
            embeddings = reducer.transform(df[features])
            x = f'{method}_0'
            y = f'{method}_1'
            ax.scatter(reference[x], reference[y], c='blue', s=4, alpha=0.2)
            if kde:
                sns.kdeplot(reference[x], reference[y], c='blue', n_levels=100, ax=ax, shade=False)
            ax.scatter(embeddings[:, 0], embeddings[:, 1], c='red', s=4, alpha=0.1)
            if kde:
                sns.kdeplot(embeddings[:, 0], embeddings[:, 1], c='red', n_levels=100, ax=ax, shade=False)
        fig.show()

    def divergence_barplot(self, target_id: str, comparisons: list, root_population: str,
                           sample_n: int = 10000, figsize: tuple = (8, 8),
                           transform: str = 'logicle', scale: str or None = None,
                           kde_kernel: str = 'gaussian', divergence_method: str = 'hellinger',
                           verbose: bool = False,
                           **kwargs):
        divergence = self.calc_divergence(target_id=target_id,
                                          kde_kernel=kde_kernel,
                                          divergence_method=divergence_method,
                                          verbose=verbose,
                                          comparisons=comparisons)
        # Plotting
        fig, ax = plt.subplots(figsize=figsize)
        if verbose:
            print('Plotting...')
        hd_ = {'sample_id': list(), 'hellinger_distance': list()}
        for n, h in divergence:
            hd_['sample_id'].append(n)
            hd_[f'{divergence_method} distance'].append(h)
        hd_ = pd.DataFrame(hd_).sort_values(by=f'{divergence_method} distance', ascending=True)
        sns.set_color_codes("pastel")
        ax = sns.barplot(y='sample_id', x=f'{divergence_method} distance',
                         data=hd_, color='b', ax=ax, **kwargs)
        ax.set_xlabel(f'{divergence_method} distance')
        ax.set_ylabel('Sample ID')
        fig.show()

    def divergence_matrix(self, exclude: list or None = None, figsize: tuple = (12, 12),
                          kde_kernel: str = 'gaussian', divergence_method: str = 'jsd',
                          **kwargs):
        """
        Generate a clustered heatmap of pairwise statistical distance comparisons. This can be used to find
        samples of high similarity and conversely demonstrates samples that greatly differ.
        :param exclude: list of sample IDs to be omitted from plot
        :param figsize: size of resulting Seaborn clusterplot figure
        :param kde_kernel: name of kernel to use for density estimation (default = 'gaussian')
        :param divergence_method: name of statistical distance metric to use; valid choices are:
            *jsd: squared Jensen-Shannon Divergence (default)
            *kl: Kullback–Leibler divergence (relative entropy); warning, asymmetrical
            *hellinger: squared Hellinger Divergence
        :param kwargs: additional keyword arguments to be passed to Seaborn ClusterPlot
        (seaborn.pydata.org/generated/seaborn.clustermap.html#seaborn.clustermap)
        :return: Seaborn ClusterGrid instance
        """
        samples = self.sample_ids
        if exclude is not None:
            samples = [s for s in samples if s not in exclude]
        divergence_df = pd.DataFrame()

        for s in progress_bar(samples):
            divergence = self.calc_divergence(target_id=s,
                                              kde_kernel=kde_kernel,
                                              divergence_method=divergence_method,
                                              verbose=False,
                                              comparisons=samples)
            hd_ = defaultdict(list)
            for n, h in divergence:
                hd_[n].append(h)
            hd_ = pd.DataFrame(hd_)
            hd_['sample_id'] = s
            divergence_df = pd.concat([divergence_df, hd_])

        if divergence_method == 'jsd':
            center = 0.5
        else:
            center = 0
        return sns.clustermap(divergence_df.set_index('sample_id'),
                              center=center, cmap="vlag", linewidths=.75,
                              figsize=figsize, **kwargs)

    def calc_divergence(self, target_id: str, comparisons: list, kde_kernel: str = 'gaussian',
                        divergence_method: str = 'jsd', verbose: bool = False) -> np.array:
        """
        Calculate the statistical distance between the probability density function of a target sample and one or many
        comparison samples.
        :param target_id: sample ID for PDF p
        :param root_population: name of population to retrieve samples from
        :param comparisons: list of sample IDs that will form PDF q
        :param sample_n: number of cells to sample from each (optional but recommended; default = 10000)
        :param transform: method used to transform the data prior to processing (default = 'logicle')
        :param scale: scaling function to apply to data post-transformation (optional; default = None)
        :param kde_kernel: name of kernel to use for density estimation (default = 'gaussian')
        :param divergence_method: name of statistical distance metric to use; valid choices are:
            *jsd: squared Jensen-Shannon Divergence (default)
            *kl: Kullback–Leibler divergence (relative entropy); warning, asymmetrical
            *hellinger: squared Hellinger distance
        :param verbose: If True, function will return regular feedback (default = False)
        :return: List of tuples in format (SAMPLE_NAME, DIVERGENCE)
        """
        assert divergence_method in ['jsd', 'kl', 'hellinger'], 'Invalid divergence metric must be one of ' \
                                                                '[jsd, kl, hellinger]'
        if verbose:
            if divergence_method == 'kl':
                print('Warning: Kullback-Leiber Divergence chosen as statistical distance metric, KL divergence '
                      'is an asymmetrical function and as such it should not be used for generating a divergence matrix')

        # Calculate PDF for target
        if verbose:
            print('Calculating PDF for target...')
        if target_id not in self.kde_cache.keys():
            target = self.data[target_id].values
            self.kde_cache[target_id] = kde_multivariant(target, bandwidth='cross_val', kernel=kde_kernel)

        # Calc PDF for other samples and calculate F-divergence
        if verbose:
            print(f'Calculate PDF for all other samples and calculate F-divergence metric: {divergence_method}...')

        # Fetch data frame for all other samples if kde not previously computed
        samples_df = [(name, s.values) for name, s in self.data.items()
                      if name in comparisons and name not in self.kde_cache.keys()]
        kde_f = partial(kde_multivariant, bandwidth='cross_val', kernel=kde_kernel)
        kde_indexed_f = partial(indexed_kde, kde_f=kde_f)
        pool = Pool(cpu_count())
        q_ = pool.map(kde_indexed_f, samples_df)
        pool.close()
        pool.join()
        for name, q in q_:
            self.kde_cache[name] = q
        if divergence_method == 'jsd':
            return [(name, jsd_divergence(self.kde_cache[target_id], q)) for name, q in self.kde_cache.items()]
        if divergence_method == 'kl':
            return [(name, kl_divergence(self.kde_cache[target_id], q)) for name, q in self.kde_cache.items()]
        return [(name, hellinger_dot(self.kde_cache[target_id], q)) for name, q in self.kde_cache.items()]
