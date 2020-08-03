from ..data.experiments import Experiment
from ..feedback import progress_bar, vprint
from .dim_reduction import dimensionality_reduction
from .transforms import scaler
from .gating_tools import load_population
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Dict
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math


def calculate_ref_sample(experiment: Experiment,
                         exclude_samples: list or None = None,
                         sample_n: int = 1000,
                         verbose: bool = True):
    """
    Given an FCS Experiment with multiple FCS files, calculate the optimal reference file.

    This is performed as described in Li et al paper (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860171/) on
    DeepCyTOF: for every 2 samples i, j compute the Frobenius norm of the difference between their covariance matrics
    and then select the sample with the smallest average distance to all other samples.

    This is an optimised version of supervised.ref.calculate_red_sample that leverages the multi-processing library
    to speed up operations

    Parameters
    ----------
    experiment: FCSExperiment
        Experiment to find reference sample for
    exclude_samples: list, optional
        If given, any samples in list will be excluded
    sample_n: int, (default=1000)
        Data is downsampled prior to running algorithm, this specifies how many observations to sample from each
    verbose: bool, (default=True)
        Feedback
    Returns
    -------
    str
        Sample ID of reference sample
    """
    vprint = print if verbose else lambda *a, **k: None
    if exclude_samples is None:
        exclude_samples = []
    vprint('-------- Calculating Reference Sample (Multi-processing) --------')
    # Calculate common features
    vprint('...match feature space between samples')
    features = find_common_features(experiment)
    # List samples
    all_samples = [x for x in experiment.list_samples() if x not in exclude_samples]
    vprint('...pulling data')
    # Fetch data
    pool = Pool(cpu_count())
    f = partial(pull_data_hashtable, experiment=experiment, features=features, sample_n=sample_n)
    all_data_ = pool.map(f, all_samples)
    vprint('...calculate covariance matrix for each sample')
    # Calculate covar for each
    all_data = dict()
    for d in all_data_:
        all_data.update(d)
    del all_data_
    all_data = {k: np.cov(v, rowvar=False) for k, v in all_data.items()}
    vprint('...search for sample with smallest average euclidean distance to all other samples')
    # Make comparisons
    n = len(all_samples)
    norms = np.zeros(shape=[n, n])
    ref_ind = None
    for i in range(0, n):
        cov_i = all_data[all_samples[i]]
        for j in range(0, n):
            cov_j = all_data[all_samples[j]]
            cov_diff = cov_i - cov_j
            norms[i, j] = np.linalg.norm(cov_diff, ord='fro')
            norms[j, i] = norms[i, j]
            avg = np.mean(norms, axis=1)
            ref_ind = np.argmin(avg)
    pool.close()
    pool.join()
    return all_samples[int(ref_ind)]


def _indexed_dimensionality_reduction(indexed_data: (str, pd.DataFrame),
                                      **kwargs):
    return indexed_data[0], dimensionality_reduction(indexed_data[1].values, **kwargs)


def _transform(indexed_data: (str, np.array),
               reducers: dict):
    return reducers.get(indexed_data[0]).transform(indexed_data[1])

def _jsd_n_comparison(embeddings: Dict[dict]):

    # Wrangle into a dataframe
    all_data = pd.DataFrame()
    for sample_id, sample_embeddings in embeddings.items():
        sample_data = pd.DataFrame()
        for n, embedding_values in sample_embeddings.items():
            df = pd.DataFrame(embedding_values, columns=["component_1", "component_2"])
            df["n"] = n
            sample_data = pd.concat([sample_data, df])
        sample_data["sample_id"] = sample_id
        all_data = pd.concat([all_data, sample_data])


def _visual_n_comparison(embeddings: Dict[dict]):
    pass


class EvaluateBatchEffects:
    def __init__(self, experiment: Experiment,
                 root_population: str,
                 samples: list or None = None,
                 reference_sample: str or None = None,
                 transform: str = 'logicle',
                 verbose: bool = True,
                 njobs: int = -1):
        self.experiment = experiment
        self.sample_ids = samples or experiment.list_samples()
        self.transform = transform
        self.root_population = root_population
        self.verbose = verbose
        self.print = vprint(verbose)
        self.reference_id = reference_sample
        self.kde_cache = dict()
        self.data = dict()
        self.njobs = njobs
        if self.njobs < 0:
            self.njobs = cpu_count()

    def load_and_sample(self,
                        sample_n: int = 10000):
        _load = partial(load_population,
                        experiment=self.experiment,
                        population=self.root_population,
                        sample_n=sample_n,
                        transform=self.transform,
                        indexed=True)
        with Pool(self.njobs) as pool:
            data = list(progress_bar(pool.imap(_load, self.sample_ids),
                                     verbose=self.verbose,
                                     total=len(self.sample_ids)))
        self.data = {x[0]: x[1] for x in data}

    def select_optimal_reference(self):
        pass

    def select_optimal_sample_n(self,
                                method: str = "jsd",
                                sample_range: list or None = None,
                                scale: str or None = "standard",
                                dimensionality_reduction_method: str = "UMAP",
                                scaler_kwargs: dict or None = None,
                                **kwargs):
        if scaler_kwargs is None:
            scaler_kwargs = {}
        assert method in ["jsd", "visual"], "Method should be either 'jsd' or 'visual'"
        sample_range = sample_range or np.arange(1000, 11000, 1000)
        self.print("=============================================")
        self.print("Finding optimal sample N")
        self.print("---------------------------------------------")
        self.print("Warning: this process is computationally intensive. Although CytoPy will leverage "
                   "all available cores to reduce the time cost, this might take some time to complete.")
        largest_sample = sample_range[-1:]
        sample_range = sample_range[:-1]

        self.print(f"Fitting chosen reducer {dimensionality_reduction_method} using largest chosen sample size: "
                   f"{largest_sample}...")
        self.print("...collecting data...")
        self.load_and_sample(sample_n=largest_sample)
        if scale:
            self.print("...scaling data...")
            self.scale_data(scale=scale, **scaler_kwargs)
        _dim_reduction = partial(_indexed_dimensionality_reduction,
                                 features=list(self.data.values())[0].columns.values,
                                 method=dimensionality_reduction_method,
                                 n_components=2,
                                 return_embeddings_only=True,
                                 return_reducer=True,
                                 **kwargs)
        self.print("...fitting data...")
        with Pool(self.njobs) as pool:
            indexed_data = [(k, v) for k, v in self.data.items()]
            indexed_embeddings = list(progress_bar(pool.imap(_dim_reduction, indexed_data),
                                                   verbose=self.verbose,
                                                   total=len(indexed_data)))
        reducers = {x[0]: x[2] for x in indexed_embeddings}
        embeddings = dict()
        embeddings[largest_sample] = {x[0]: x[1] for x in indexed_embeddings}
        self.print("---------------------------------------------")
        self.print("Applying transform to remaining sample n's...")
        for n in sample_range:
            self.print(f"Sample n = {n}...")
            self.print("...collecting data...")
            self.load_and_sample(sample_n=largest_sample)
            if scale:
                self.print("...scaling data...")
                self.scale_data(scale=scale, **scaler_kwargs)
            self.print("...transforming data...")
            _apply_transform = partial(_transform,
                                       reducers=reducers)
            with Pool(self.njobs) as pool:
                indexed_data = [(k, v) for k, v in self.data.items()]
                indexed_embeddings = list(progress_bar(pool.imap(_apply_transform, indexed_data),
                                                       verbose=self.verbose,
                                                       total=len(indexed_data)))
            embeddings[n] = {x[0]: x[1] for x in indexed_embeddings}
        if method == "jsd":
            return _jsd_n_comparison(embeddings=embeddings)
        return _visual_n_comparison(embeddings=embeddings)

    def scale_data(self,
                   scale: str or None = "standard",
                   **kwargs):
        self.data = {k: pd.DataFrame(scaler(data=v.values,
                                            scale_method=scale,
                                            return_scaler=False,
                                            **kwargs),
                                     columns=v.columns) for k, v in self.data.items()}

    def marker_variance(self,
                        comparison_samples: list,
                        markers: list or None = None,
                        figsize: tuple = (10, 10),
                        xlim: tuple or None = None,
                        **kwargs):
        fig = plt.figure(figsize=figsize)
        assert len(self.data) > 0, "No data currently loaded. Call 'load_and_sample'"
        assert self.reference_id in self.data.keys(), 'Invalid reference ID for experiment currently loaded'
        assert all([x in self.sample_ids for x in comparison_samples]), \
            f'One or more invalid sample IDs; valid IDs include: {self.data.keys()}'
        if markers is None:
            markers = self.data.get(self.reference_id).columns.tolist()

        i = 0
        nrows = math.ceil(len(markers) / 3)
        fig.suptitle(f'Per-channel KDE, Reference: {self.reference_id}', y=1.05)
        for marker in progress_bar(markers, verbose=self.verbose):
            i += 1
            ax = fig.add_subplot(nrows, 3, i)
            ax = sns.kdeplot(self.data.get(self.reference_id)[marker], shade=True, color="b", ax=ax, **kwargs)
            ax.set_title(f'Total variance in {marker}')
            if xlim:
                ax.set_xlim(xlim)
            for comparison_sample_id in comparison_samples:
                if marker not in self.data.get(comparison_sample_id).columns:
                    warn(f"{marker} missing from {comparison_sample_id}, this marker will be ignored")
                else:
                    ax = sns.kdeplot(self.data.get(comparison_sample_id)[marker],
                                     color='r', shade=False, alpha=0.5, ax=ax)
                    ax.get_legend().remove()
            ax.set(aspect="auto")
        fig.tight_layout()
        return fig

    def dimensionality_reduction(self):
        pass
