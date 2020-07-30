from ..data.experiments import Experiment
from ..feedback import progress_bar, vprint
from .dim_reduction import dimensionality_reduction
from .gating_tools import Gating
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


class EvaluateBatchEffects:
    def __init__(self, experiment: Experiment,
                 root_population: str,
                 samples: list or None = None,
                 reference_sample: str or None = None,
                 transform: str = 'logicle',
                 verbose: bool = True):
        self.experiment = experiment
        self.sample_ids = samples or experiment.list_samples()
        self.transform = transform
        self.root_population = root_population
        self.verbose = verbose
        self.print = vprint(verbose)
        self.reference_id = reference_sample
        self.kde_cache = dict()
        self.data = dict()

    def load_and_sample(self,
                        sample_n: int = 10000):
        for sample_id in progress_bar(self.sample_ids, verbose=self.verbose):
            gating = Gating(experiment=self.experiment,
                            sample_id=sample_id,
                            include_controls=False,
                            verbose=False)
            data = gating.get_population_df(population_name=self.root_population,
                                            transform=self.transform,
                                            transform_features="all")
            if data.shape[0] < sample_n:
                warn(f"{sample_id} has less than {sample_n} events (n={data.shape[0]}). Using all available data.")
                self.data[sample_id] = data
            else:
                self.data[sample_id] = data.sample(n=sample_n)

    def select_optimal_reference(self):
        pass

    def select_optimal_sample_n(self,
                                method: str = "jsd",
                                sample_range: list or None = None,
                                dimensionality_reduction_method: str = "UMAP"):
        assert method in ["jsd", "visual"], "Method should be either 'jsd' or 'visual'"
        sample_range = sample_range or np.arange(1000, 10000, 1000)
        pass

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