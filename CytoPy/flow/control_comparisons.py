from ..feedback import progress_bar
from ..data.experiment import Experiment
from pingouin import normality, ttest, wilcoxon, compute_effsize
from scipy.stats.mstats import gmean
from warnings import warn
import pandas as pd


def _melt(data: pd.DataFrame):
    return data.melt(var_name="Features", value_name="FI")


def _diff_geomean(x, y):
    return abs(gmean(x) - gmean(y))


class ControlComparison:
    def __init__(self,
                 experiment: Experiment,
                 sample_id: str,
                 population: str,
                 ctrl_id: str,
                 transform: str = "logicle"):
        self.transform = transform
        self.ctrl_id = ctrl_id
        self.fg = experiment.get_sample(sample_id)
        self._pop = self.fg.get_population(population)
        self.primary = self.fg.load_population_df(population=population,
                                                  transform=transform)
        self.ctrl = self.fg.load_ctrl_population_df(ctrl=ctrl_id,
                                                    population=population,
                                                    transform=transform)

    @property
    def population(self):
        return self._pop

    @population.setter
    def population(self, population: str):
        self._pop = self.fg.get_population(population)
        self.primary = self.fg.load_population_df(population=population,
                                                  transform=self.transform)
        self.ctrl = self.fg.load_ctrl_population_df(ctrl=self.ctrl_id,
                                                    population=population,
                                                    transform=self.transform)

    def _x(self, x: str or None = None):
        x = x or self.ctrl_id
        assert x in self.primary.columns, f"{x} not a valid feature; expected one of {self.primary.columns}"
        return x

    def normality(self, x: str or None = None, **kwargs):
        x = self._x(x)
        primary = normality(self.primary[x])
        ctrl = normality(self.ctrl[x])
        primary["id"] = "primary staining"
        ctrl["id"] = f"{self.ctrl_id} ctrl staining"
        return pd.concat([primary, ctrl])

    def plot(self, x: str or None = None):
        x = self._x(x)

    def __call__(self,
                 x: str,
                 stat_test: str or callable or None = None,
                 effect_size: str = "geometric_mean"):
        """

        Parameters
        ----------
        stat_test : str or callable or None
        """
        x = self._x(x)
        if stat_test is None:
            norm = self.normality(x=x)
            stat_test = "wilcoxon"
            if all([x is True for x in norm["normal"].values]):
                stat_test = "paired ttest"
        if isinstance(stat_test, str):
            assert stat_test in ["paired ttest", "wilcoxon"], "Invalid stat_test, should be None, callable " \
                                                              "or 'paried ttest' or 'wilcoxon'"
            if stat_test == "paired ttest":
                stats = ttest(self.primary[x].values,
                              self.ctrl[x].values,
                              paired=True,
                              tail="two-sided",
                              correction="auto")
            else:
                stats = wilcoxon(self.primary[x].values,
                                 self.ctrl[x].values,
                                 tail="two-sided")
        else:
            stats = stat_test(self.primary[x].values,
                              self.ctrl[x].values)
        if effect_size == "geometric_mean":
            stats[effect_size] = _diff_geomean(self.primary[x].values,
                                               self.ctrl[x].values)
        else:
            stats[effect_size] = compute_effsize(self.primary[x].values,
                                                 self.ctrl[x].values,
                                                 paired=True,
                                                 eftype=effect_size)
        stats["population"] = self.population.population_name
        stats["ctrl_id"] = self.ctrl_id
        stats["sample_id"] = self.fg.primary_id
        return stats


def compute_ctrl_populations(experiment: Experiment,
                             ctrl_ids: list,
                             populations: list,
                             sample_ids: list or None = None,
                             scoring: str = "balanced_accuracy",
                             verbose: bool = True,
                             downsample: int or float = 0.1,
                             sml_population_mappings: dict or None = None,
                             **kwargs):
    kwargs = kwargs or {}
    sample_ids = sample_ids or list(experiment.list_samples())
    for sid in progress_bar(sample_ids, verbose=verbose):
        fg = experiment.get_sample(sid)
        for pop in populations:
            for ctrl_id in ctrl_ids:
                if pop in fg.list_populations() and ctrl_id in fg.controls:
                    if ctrl_id not in fg.get_population(population_name=pop).ctrl_index.keys():
                        continue
                    try:
                        fg.estimate_ctrl_population(ctrl=ctrl_id,
                                                    population=pop,
                                                    verbose=verbose,
                                                    scoring=scoring,
                                                    downsample=downsample,
                                                    population_mappings=sml_population_mappings,
                                                    **kwargs)
                    except AssertionError as e:
                        warn(f"Failed to identify {pop} in {ctrl_id} for {fg.primary_id}; {str(e)}")
                        continue
        fg.save()
        print("\n")


def experiment_control_comparisons(experiment: Experiment,
                                   ctrl_ids: list,
                                   populations: list,
                                   transform: str = "logicle",
                                   var_names: list or None = None,
                                   sample_ids: list or None = None,
                                   stat_test: str or None = None,
                                   effect_size="geometric_mean"):
    var_names = var_names or ctrl_ids
    sample_ids = sample_ids or list(experiment.list_samples())
    results = list()
    for sid in progress_bar(sample_ids, verbose=True):
        for pop in populations:
            for i, ctrl_id in enumerate(ctrl_ids):
                cc = ControlComparison(experiment=experiment,
                                       sample_id=sid,
                                       population=pop,
                                       ctrl_id=ctrl_id,
                                       transform=transform)
                results.append(cc(x=var_names[i],
                                  stat_test=stat_test,
                                  effect_size=effect_size))
    return pd.concat(results)