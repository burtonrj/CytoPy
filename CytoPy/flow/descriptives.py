#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This descriptive module contains utility functions for descriptive statistics and basic plotting.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from pingouin import normality, kruskal, welch_anova, ttest, mwu
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="white", font_scale=1.3)
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})


def appropriate_stat(data: pd.DataFrame,
                     dv: str,
                     group: str):
    """
    Given a Pandas DataFrame, inspect the dependent variable (factored by the group) for
    normality and equal variance. Will apply Welch Anova if data is normal and number
    of unique values in group is greater than 2, otherwise will apply Welch T-test. If
    the data is not normally distributed, will apply kruskal wallis test if number
    of unique values in group is greater than 2, otherwise will apply Mann-Whitney U tests.

    All tests implemented with the Pingouin library, returning DataFrame of results.
    (See Pingouin for individual tests)

    Parameters
    ----------
    data: Pandas.DataFrame
    dv: str
        Dependent variable (should be a column name)
    group: str
        Grouping variable (should be a column name)

    Returns
    -------
    Pandas.DataFrame
    """
    if normality(data=data,
                 dv=dv,
                 group=group).iloc[0]["normal"]:
        if len(data[group].unique()) > 2:
            return welch_anova(data=data, dv=dv, between=group)
        xy = [data[data[group] == i][dv].values for i in data[group].unique()]
        return ttest(*xy, paired=False, tail="two-sided", correction="auto")
    if len(data[group].unique()) > 2:
        return kruskal(data=data, dv=dv, between=group)
    xy = [data[data[group] == i][dv].values for i in data[group].unique()]
    return mwu(*xy, tail="two-sided")


def stat_test(*data,
              group1="Population",
              dep_var_name="% of T cells",
              group2: str = "Patient type",
              data_labels: list or None = None,
              id_vars: list or None = None,
              filter_: list or None = None,
              wide_format: bool = True):
    # Set defaults
    id_vars = id_vars or ["sample_id", group2]
    if data_labels is not None:
        assert len(data) == len(data_labels), "length of data does not match length of data labels"
    data_labels = data_labels or [f'data{i + 1}' for i in range(len(data))]

    # Iterate over each dataframe
    stats = list()
    for df, label in zip(data, data_labels):
        # Convert to long
        long_df = df
        if wide_format:
            long_df = df.melt(var_name=group1,
                              value_name=dep_var_name,
                              id_vars=id_vars)
        if filter_ is not None:
            long_df = long_df[long_df[group1].isin(filter_)]
        # Iterate over each subgroup e.g. population
        subgroups = list()
        for grp_id, grp in long_df.groupby(group1):
            try:
                if len(grp[group2].unique()) < 2:
                    continue
                grp_stats = appropriate_stat(data=grp.dropna(), dv=dep_var_name, group=group2)
                grp_stats['subgroup'] = grp_id
                grp_stats["x"] = group2
                grp_stats["y"] = dep_var_name
                subgroups.append(grp_stats)
            except AssertionError as e:
                warn(f"Failed to generate stats for {grp_id}; {str(e)}")
        subgroups = pd.concat(subgroups)
        subgroups["data"] = label
        stats.append(subgroups)
    return pd.concat(stats)
