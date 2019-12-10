from immunova.flow.gating.utilities import density_dependent_downsample
from immunova.flow.dim_reduction import dimensionality_reduction
import matplotlib.pyplot as plt
import seaborn as sns
import random


class PlottingError(Exception):
    pass


def random_colours(n):
    colours = list()
    while len(colours) < n:
        r = lambda: random.randint(0,255)
        c = '#%02X%02X%02X' % (r(),r(),r())
        if c not in colours:
            colours.append(c)
    return colours


def sample_data(data, n, frac=None, method='uniform'):
    if frac is not None:
        if method == 'uniform':
            return data.sample(frac=frac, random_state=42)
        if method == 'density':
            return density_dependent_downsample(data=data, features=data.columns, frac=frac)
    if method == 'uniform':
        return data.sample(n=n, random_state=42)
    if method == 'density':
        return density_dependent_downsample(data=data, features=data.columns, sample_n=n)
    raise PlottingError('Error: invalid sampling method, must be either `uniform` or `density`')


def dim_reduction_plot(data, method, features, title, sample_n=100000, sample_method='uniform', n_components=2,
                       save_path=None):
    if n_components not in [2, 3]:
        raise PlottingError('Error: number of components must be either 2 or 3')
    sample = sample_data(data, sample_n, method=sample_method)
    data = dimensionality_reduction(sample, features, method, n_components)
    fig, ax = plt.subplots(figsize=(10, 10))
    if n_components == 2:
        ax.scatter(data[f'{method}_0'], data[f'{method}_1'], s=1, alpha=0.5)
        ax.set_xlabel(f'{method}_0')
        ax.set_ylabel(f'{method}_1')
    else:
        ax.scatter(data[f'{method}_0'], data[f'{method}_1'], data[f'{method}_2'], s=1, alpha=0.5)
        ax.set_xlabel(f'{method}_0')
        ax.set_ylabel(f'{method}_1')
        ax.set_zlabel(f'{method}_3')
    ax.set_title(title)
    if save_path is not None:
        fig.savefig(f'{save_path}/{title}.jpg', res=300, bbox_inches='tight')


def label_dataset_clusters(data, clusters):
    data = data.copy()
    data['clusters'] = -1
    for name, c in clusters.items():
        data['clusters'] = data['clusters'].mask(data.index.isin(c['index']), name)
    return data


def __coloured_scatter(data, method, n_components, title, label_prefix):
    unique_clusters = data['clusters'].unique()
    colours = {name: colour for name, colour in zip(unique_clusters, random_colours(len(unique_clusters)))}

    fig, ax = plt.subplots(figsize=(10, 10))
    if n_components == 2:
        for cluster, colour in colours.items():
            d = data[data['clusters'] == cluster]
            ax.scatter(d[f'{method}_0'], d[f'{method}_1'], s=1, alpha=0.5, c=[colour], label=f'Cluster {cluster}')
        ax.set_xlabel(f'{method}_0')
        ax.set_ylabel(f'{method}_1')
    else:
        for cluster, colour in colours.items():
            d = data[data['clusters'] == cluster]
            ax.scatter(d[f'{method}_0'], d[f'{method}_1'], d[f'{method}_3'],
                       s=1, alpha=0.5, c=[colour], label=f'{label_prefix} {cluster}')
        ax.set_xlabel(f'{method}_0')
        ax.set_ylabel(f'{method}_1')
        ax.set_zlabel(f'{method}_3')

    ax.legend(loc="lower left", scatterpoints=1, fontsize=10, bbox_to_anchor=(1.02, 0))
    ax.set_title(title)
    return fig, ax


def plot_clusters(data, clusters, method, title, sample_n=100000, sample_method='uniform', n_components=2,
                  save_path=None):
    if n_components not in [2, 3]:
        raise PlottingError('Error: number of components must be either 2 or 3')
    # Label dataset and perform dim reduction
    data = label_dataset_clusters(data, clusters)
    features = [x for x in data.columns if x != 'clusters']
    if sample_n is not None:
        data = sample_data(data, sample_n, method=sample_method)
    data = dimensionality_reduction(data, features, method, n_components)
    # Plotting
    fig, ax = __coloured_scatter(data, method, n_components, title, label_prefix='Cluster:')
    if save_path is not None:
        fig.savefig(f'{save_path}/{title}_scatter.jpg', res=300, bbox_inches='tight')
    fig.show()


def check_overlapping_populations(gating, root_population, labels):
    valid = True
    for pop in labels:
        dependencies = gating.find_dependencies(pop)
        if root_population in dependencies:
            print(f'Error: {root_population} downstream from {pop}, please amend before plotting')
            valid = False
        for pop_i in [x for x in dependencies if x != pop]:
            if pop_i in dependencies:
                print(f'Error: {pop_i} downstream from {pop}, please amend before plotting')
            valid = False
    return valid


def dim_reduction_plot_gates(gating, root_population, features, labels, method, title, sample_n=100000,
                             sample_method='uniform', n_components=2, save_path=None, plot_missed=None):
    if check_overlapping_populations(gating, root_population, labels):
        raise PlottingError('Error: one or more errors encountered when checking population labels')
    data = gating.get_population_df(root_population, transform=True, transform_method='logicle')
    if not data:
        raise PlottingError(f'Error: unable to load population {root_population}')
    sample = sample_data(data, sample_n, method=method)
    data = dimensionality_reduction(sample, features, method, n_components)
    data = label_dataset_clusters(data, {name: node for name, node in gating.populations.items() if name in labels})
    fig, ax = __coloured_scatter(data, method, n_components, title, label_prefix='Population:')

    if plot_missed == 'scatter':
        d = data[data['clusters'] == -1]
        if n_components == 2:
            ax.scatter(d[f'{method}_0'], d[f'{method}_1'], c='black', s=5, alpha=0.35, label='Missed')
        else:
            ax.scatter(d[f'{method}_0'], d[f'{method}_1'], d[f'{method}_2'], c='black', s=5, alpha=0.35, label='Missed')
        ax.legend(loc="lower left", scatterpoints=1, fontsize=10, bbox_to_anchor=(1.02, 0))

    if plot_missed == 'kde':
        if n_components == 3:
            raise PlottingError('Error: KDE for missed samples currently only supports 2D plots')
        x = data[data['clusters'] == -1][f'{method}_0'].values
        y = data[data['clusters'] == -1][f'{method}_1'].values
        sns.kdeplot(x, y, n_levels=30, cmap="Purples_d", ax=ax)
    if save_path is not None:
        fig.savefig(f'{save_path}/{title}.jpg', res=300, bbox_inches='tight')
    fig.show()


def heatmap(data, clusters, title, save_path=None):
    data = label_dataset_clusters(data, clusters)
    mfi_summary = data.groupby(by='clusters').mean()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = sns.heatmap(mfi_summary, robust=True, square=True, ax=ax)
    ax.set_title(title)
    if save_path is not None:
        fig.savefig(f'{save_path}/{title}_heatmap.jpg', res=300, bbox_inches='tight')
    fig.show()


def clustermap(data, clusters, title, save_path=None):
    data = label_dataset_clusters(data, clusters)
    mfi_summary = data.groupby(by='clusters').mean()
    g = sns.clustermap(mfi_summary, col_cluster=False, square=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    if save_path is not None:
        g.savefig(f'{save_path}/{title}_heatmap.jpg', res=300, bbox_inches='tight')
    fig.show()


