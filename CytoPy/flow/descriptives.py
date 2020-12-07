from pingouin import normality, kruskal, welch_anova, ttest, mwu
from warnings import warn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="white", font_scale=1.3)
sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})


def box_swarm_plot(plot_df: pd.DataFrame,
                   x: str,
                   y: str,
                   hue: str,
                   ax: plt.Axes,
                   palette: str,
                   boxplot_kwargs: dict or None = None,
                   swarmplot_kwargs: dict or None = None):
    boxplot_kwargs = boxplot_kwargs or {}
    swarmplot_kwargs = swarmplot_kwargs or {}
    sns.boxplot(data=plot_df,
                x=x,
                y=y,
                hue=hue,
                ax=ax,
                showfliers=False,
                boxprops=dict(alpha=.3),
                palette=palette,
                **boxplot_kwargs)
    sns.swarmplot(data=plot_df,
                  x=x,
                  y=y,
                  hue=hue,
                  ax=ax,
                  dodge=True,
                  palette=palette,
                  **swarmplot_kwargs)


def multi_box_swarm_plot(*data,
                         group1="Population",
                         dep_var_name="% of T cells",
                         id_vars=None,
                         filter_=None,
                         group2="Patient type",
                         figsize=(15, 6),
                         titles=None,
                         ylim=None,
                         palette: str or None = None,
                         boxplot_kwargs: dict or None = None,
                         swarmplot_kwargs: dict or None = None):
    id_vars = id_vars or ["sample_id", group2]
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i, df in enumerate(data):
        plot_df = df.melt(var_name=group1,
                          value_name=dep_var_name,
                          id_vars=id_vars)
        if filter_ is not None:
            plot_df = plot_df[plot_df[group1].isin(filter_)]
        box_swarm_plot(plot_df=plot_df,
                       x=group1,
                       y=dep_var_name,
                       hue=group2,
                       palette=palette,
                       ax=axes[i],
                       boxplot_kwargs=boxplot_kwargs,
                       swarmplot_kwargs=swarmplot_kwargs)
        if titles is not None:
            axes[i].set_title(titles[i])
        if ylim is not None:
            axes[i].set_ylim(ylim)
        if i == 0:
            handles, labels = axes[i].get_legend_handles_labels()
            axes[i].legend(handles[0:2], labels[0:2], bbox_to_anchor=(1, 1.3))
        else:
            axes[i].legend().remove()
    fig.tight_layout()
    return fig, axes


def _appropriate_stat(data: pd.DataFrame,
                      dv: str,
                      group: str):
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

    # Interate over each dataframe
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
                grp_stats = _appropriate_stat(data=grp.dropna(), dv=dep_var_name, group=group2)
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
