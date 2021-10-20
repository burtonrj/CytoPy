import logging

import numpy as np
import pandas as pd
import pingouin as pg
from mlxtend.evaluate import permutation_test

logger = logging.getLogger(__name__)


def _wrangle_x_y(x: np.array, y: np.array) -> pd.DataFrame:
    xd = pd.DataFrame({"X": x})
    xd["Group"] = "x"
    yd = pd.DataFrame({"X": y})
    yd["Group"] = "y"
    return pd.concat([xd, yd]).reset_index(drop=True)


def two_groups(
    x: np.array, y: np.array, paired: bool = False, try_parametric: bool = True, **permutation_test_kwargs
) -> pd.DataFrame:
    data = _wrangle_x_y(x, y)
    normal = pg.normality(data=data, dv="X", group="Group").iloc[0]["normal"]
    equal_var = pg.homoscedasticity(data=data, dv="X", group="Group").iloc[0]["equal_var"]
    equal_sample_size = len(x) == len(y)
    if normal and try_parametric:
        if equal_var:
            welch = not equal_sample_size
            stats = pg.ttest(x, y, paired=paired, correction=welch)
            stats["Test"] = "Welch T-test" if welch else "T-test"
        else:
            stats = pg.ttest(x, y, paired=paired, correction=True)
            stats["Test"] = "Welch T-test"
    else:
        if equal_var:
            if paired:
                stats = pg.wilcoxon(x, y)
                stats["Test"] = "Wilcoxon signed-rank test"
            else:
                stats = pg.mwu(x, y)
                stats["Test"] = "Mann-Whitney U test"
        else:
            permutation_test_kwargs = permutation_test_kwargs or {}
            permutation_test_kwargs["method"] = "approximate"
            permutation_test_kwargs["seed"] = 42
            eff_size = pg.compute_effsize(x, y, eftype="CLES")
            p = permutation_test(x, y, paired=paired, **permutation_test_kwargs)
            stats = pd.DataFrame({"Test": ["Permutation test"], "p-val": [p], "CLES": eff_size})
    stats["normal"] = normal
    stats["equal_var"] = equal_var
    stats["equal_sample_size"] = equal_sample_size
    stats = stats.reset_index(drop=True)
    stats = stats[["Test", "p-val"] + [x for x in stats.columns if x not in ["Test", "p-val"]]]
    return stats


def multiple_groups(
    data: pd.DataFrame, dv: str, group: str, try_parametric: bool = True, between_groups: bool = True
) -> pd.DataFrame:
    normal = pg.normality(data=data, dv=dv, group=group).iloc[0]["normal"]
    equal_var = pg.homoscedasticity(data=data, dv=dv, group=group).iloc[0]["equal_var"]
    if between_groups:
        if normal and try_parametric:
            if equal_var:
                stats = pg.anova(data=data, dv=dv, between=group)
                stats["Test"] = "Classic ANOVA"
            else:
                stats = pg.welch_anova(data=data, dv=dv, between=group)
                stats["Test"] = "Welch ANOVA"
        else:
            stats = pg.kruskal(data=data, dv=dv, between=group)
            stats["Test"] = "Kruskal-Wallis"
    else:
        if normal and try_parametric:
            stats = pg.rm_anova(data=data, dv=dv, within=group)
            stats["Test"] = "Repeated measures ANOVA"
        else:
            if data["dv"].nunique() == 2:
                stats = pg.cochran(data=data, dv=dv, within=group)
                stats["Test"] = "Cochran test"
            else:
                stats = pg.friedman(data=data, dv=dv, within=group)
                stats["Test"] = "Friedman test"
    stats["normal"] = normal
    stats["equal_var"] = equal_var
    return stats
