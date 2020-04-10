from data.fcs import FileGroup
from flow.transforms import apply_transform
from scipy.spatial import ConvexHull
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from matplotlib import patches
from anytree import Node
from itertools import cycle
import pandas as pd
import numpy as np
import random


def transform_axes(data: pd.DataFrame, axes_vars: dict, transforms: dict) -> pd.DataFrame:
    """
    Transform axes by either logicle, log, asinh, or hyperlog transformation

    Parameters
    ----------
    data : Pandas.DataFrame
        pandas dataframe containing data to plot
    axes_vars : dict
        dictionary object, key corresponds to one of 3 possible axes (x, y or z) and value
        the variable to plot
    transforms : dict
        dictionary object, key corresponds to one of 3 possible axes (x, y or z) and value
        the transform to be applied

    Returns
    -------
    Pandas.DataFrame
        Transformed DataFrame
    """
    def check_vars(x):
        assert x in data.columns, f'Error: {x} is not a valid variable, must be one of: {data.columns}'
    data = data.copy()
    map(check_vars, axes_vars.values())
    for ax, var in axes_vars.items():
        if transforms.get(ax, None) is None:
            continue
        data = apply_transform(data=data, transform_method=transforms[ax], features_to_transform=[axes_vars[ax]])
    return data[axes_vars.values()]


def plot_axis_lims(x: str, y: str, xlim: tuple, ylim: tuple) -> tuple and tuple:
    """Establish axis limits

    Parameters
    ----------
    x : str
        name of x axis
    y : str
        name of y axis
    xlim : tuple
        custom x-axis limit
    ylim : tuple
        custom y-axis limit

    Returns
    -------
    tuple, tuple
        x-axis limits, y-axis limits

    """

    def update_lim(a, lim):
        if not lim:
            if any([a.find(c) != -1 for c in ['FSC', 'SSC']]):
                return 0, 250000
        return lim
    return update_lim(x, xlim), update_lim(y, ylim)


class Plot:
    """
    Class for producing static FACs plots. Must be associated to a Gating object.

    Parameters
    ----------
    gating_object: Gating
        Gating object to retrieve data from for plotting
    default_axis: str
        default value to attribute to y-axis
    """
    def __init__(self, gating_object, default_axis):
        self.gating = gating_object
        self.default_axis = default_axis
        colours = ['black', 'gray', 'brown', 'orange', 'red',
                   'coral', 'peru', 'olive', 'magenta', 'crimson',
                   'orchid']
        random.shuffle(colours)
        self.colours = cycle(colours)

    @staticmethod
    def _plot_asthetics(ax: matplotlib.pyplot.axes, x: str, y: str,
                        xlim: tuple or None, ylim: tuple or None, title: str) -> matplotlib.pyplot.axes:
        """
        Customise axes aesthetics

        Parameters
        -----------
        ax: matplotlib.pyplot.axes

        x: str
            name of x-axis
        y: str
            name of y-axis
        xlim: tuple, optional
            custom x-axis limit
        ylim: tuple, optional
            custom y-axis limit
        title: str. optional
            plot title

        Returns
        --------
        matplotlib.pyplot.axes
            Updated axes object
        """
        if xlim:
            ax.set_xlim(xlim[0], xlim[1])
        if ylim:
            ax.set_ylim(ylim[0], ylim[1])

        ax.set_ylabel(y)
        ax.set_xlabel(x)
        ax.set_title(title)
        return ax

    def plot_gate(self, gate_name: str,
                  xlim: tuple or None = None,
                  ylim: tuple or None = None,
                  transforms: dict or None = None,
                  figsize: tuple = (5, 5)) -> None:
        """
        Produce a static plot of a gate

        Parameters
        ----------
        gate_name : str
            Name of gate to plot
        xlim : tuple, optional
            custom x-axis limit (default = None)
        ylim : tuple, optional
            custom y-axis limit (default = None)
        transforms : dict, optional
            dictionary object, key corresponds to one of 3 possible axes (x, y or z) and value
            the variable to plot (If None, defaults to logicle transform for every axis)
        figsize: tuple, (default=(5,5))
            Figure size passed to matplotlib.pyplot.subplots call

        Returns
        -------
        None
        """
        assert gate_name in self.gating.gates.keys(), f'Error: could not find {gate_name} in attached gate object'
        gate = self.gating.gates[gate_name]
        kwargs = {k: v for k, v in gate.kwargs}

        y = self.default_axis
        if 'y' in kwargs.keys():
            y = kwargs['y']
        axes_vars = {'x': kwargs['x'], 'y': y}

        if transforms is None:
            xt = kwargs.get('transform_x', None)
            yt = kwargs.get('transform_y', None)
            transforms = dict(x=xt, y=yt)
            for a in ['x', 'y']:
                if any([x in axes_vars[a] for x in ['FSC', 'SSC']]):
                    transforms[a] = None

        geoms = {c: self.gating.populations[c].geom for c in gate.children
                 if self.gating.populations[c].geom is not None}
        data = transform_axes(data=self.gating.get_population_df(gate.parent, transform=False),
                              axes_vars=axes_vars,
                              transforms=transforms)
        xlim, ylim = plot_axis_lims(x=axes_vars['x'], y=y, xlim=xlim, ylim=ylim)
        num_axes = 1
        fig, axes = plt.subplots(ncols=num_axes, figsize=figsize)
        self._geom_plot(data=data, fig=fig, axes=axes,
                        geoms=geoms, axes_vars=axes_vars,
                        xlim=xlim, ylim=ylim, name=gate_name)

    def _geom_plot(self, data: pd.DataFrame, fig: matplotlib.pyplot.figure,
                   axes: np.array or matplotlib.pyplot.axes, geoms: dict,
                   axes_vars: dict, xlim: tuple, ylim: tuple, name: str) -> None:
        """
        Wrapper function for producing a plot of a gate that generates a geometric object

        This wrapper has been created because future versions of CytoPy might allow for creation
        of multiple axis in figure for comparison of gates within one figure. Also might expand to
        allow for control driven gates.

        Parameters
        -----------
        data: Pandas.DataFrame
            population data
        geoms: dict
            dictionary object; keys correspond to child population names and values their geometric gate definition
        axes_vars: dict
            dictionary object, key corresponds to one of 3 possible axes (x, y or z) and value
            the variable to plot
        xlim: tuple
            custom x-axis limit
        ylim: tuple
            custom y-axis limit
        fig: matplotlib.pyplot.figure

        axes: matplotlib.pyplot.axes or array of axes objects

        name: str
            Name of gate to be included in plot title

        Returns
        --------
        None
        """
        self._build_geom_plot(data=data, x=axes_vars['x'], y=axes_vars['y'], geoms=geoms,
                              ax=axes, xlim=xlim, ylim=ylim, title=name)
        fig.tight_layout()
        fig.show()

    def _build_geom_plot(self, data: pd.DataFrame, x: str, y: str or None, geoms: dict, ax: matplotlib.pyplot.axes,
                         xlim: tuple or None, ylim: tuple or None, title: str) -> matplotlib.pyplot.axes or None:
        """
        Produce a plot of a gate that generates a geometric object

        Parameters
        -----------
        data: Pandas.DataFrame
            pandas dataframe of events data to plot
        xlim: tuple, optional
            custom x-axis limit (default = None)
        ylim: tuple, optional
            custom y-axis limit (default = None)
        ax: matplotlib.pyplot.axes

        title: str
            plot title

        Returns
        --------
        matplotlib.pyplot.axes
            Updated axes object
        """

        if any([x['shape'] == 'sml' for _, x in geoms.items()]):
            print(f'Error: {title} is a supervised machine learning gate. This type of gating does not produce'
                  f'2D geometries but instead classifies cells based using high dimensional feature space. To observe'
                  f'a population classified by this method in 2D, use the `plot_sml` method')
            return None

        if data.shape[0] < 1000:
            ax.scatter(x=data[x], y=data[y], s=3)
            ax = self._plot_asthetics(ax, x, y, xlim, ylim, title)
        else:
            ax = self._2dhist(ax, data, x, y)
            ax = self._plot_asthetics(ax, x, y, xlim, ylim, title)

        # Draw geom
        for (child_name, geom), cc in zip(geoms.items(), self.colours):
            colour = '#EB1313'
            if geom is None or geom == dict():
                print(f'Population {child_name} has no associated gate, skipping...')
                continue
            if geom['shape'] == 'threshold':
                ax.axvline(geom['threshold'], c=colour, lw=2.5)
            if geom['shape'] == '2d_threshold':
                ax.axvline(geom['threshold_x'], c=colour, lw=2.5)
                ax.axhline(geom['threshold_y'], c=colour, lw=2.5)
            if geom['shape'] == 'ellipse':
                ellipse = patches.Ellipse(xy=geom['centroid'], width=geom['width'], height=geom['height'],
                                          angle=geom['angle'], fill=False, edgecolor=colour, lw=2.5)
                ax.add_patch(ellipse)
            if geom['shape'] == 'rect':
                rect = patches.Rectangle(xy=(geom['x_min'], geom['y_min']),
                                         width=((geom['x_max']) - (geom['x_min'])),
                                         height=(geom['y_max'] - geom['y_min']),
                                         fill=False, edgecolor=colour, lw=2.5)
                ax.add_patch(rect)
            if geom['shape'] == 'poly':
                x = geom['cords']['x']
                y = geom['cords']['y']
                ax.plot(x, y, '-k', c=colour, label=child_name, lw=2.5)

    @staticmethod
    def _2dhist(ax: matplotlib.pyplot.axes, data: pd.DataFrame, x: str, y: str) -> matplotlib.pyplot.axes:
        """
        Generate a standard 2D histogram

        Parameters
        -----------
        ax: matplotlib.pyplot.axes

        data: dict
            dictionary of data to plot (primary and controls)
        x: str
            name of x-axis dimension
        y: str
            name of y-axis dimension

        Returns
        --------
        matplotlib.pyplot.axes
            Updated matplotlib axes object
        """
        if data.shape[0] <= 100:
            bins = 50
        elif data.shape[0] > 1000:
            bins = 500
        else:
            bins = int(data.shape[0] * 0.5)
        ax.hist2d(data[x], data[y], bins=bins, norm=LogNorm(), cmap=plt.cm.jet)
        return ax

    def plot_population(self,
                        population_name: str,
                        x: str,
                        y: str,
                        xlim: tuple = None,
                        ylim: tuple = None,
                        transforms: dict or None = None,
                        sample: float or None = None,
                        figsize: tuple = (5, 5),
                        ctrl_id: str or None = None):
        """
        Generate a static plot of a population

        Parameters
        ----------
        population_name : str
            name of population to plot
        x : str
            name of x-axis dimension
        y : str
            name of y-axis dimension
        xlim : tuple, optional
            tuple of x-axis limits
        ylim : tuple, optional
            tuple of y-axis limits
        transforms : dict, optional
            dictionary object, key corresponds to one of 3 possible axes (x, y or z) and value
            the variable to plot (If None, defaults to logicle transform for every axis)
        sample : float, optional
            if a float value is provided, given proportion of data is sampled prior to plotting (optional)
        figsize: tuple, (default=(5,5))
            Figure size passed to matplotlib.pyplot.subplots call
        ctrl_id: str, optional
            If given, the equivalent population within the associated control data is plotted

        Returns
        -------
        None
        """
        fig, ax = plt.subplots(figsize=figsize)
        data = self.gating.get_population_df(population_name,
                                             transform=False,
                                             ctrl_id=ctrl_id).copy()
        assert data is not None, 'Invalid population'

        if transforms is None:
            print('No transforms provided, defaulting to logicle')
            transforms = dict(x='logicle', y='logicle')
        data = transform_axes(data=data, axes_vars={'x': x, 'y': y}, transforms=transforms)
        if sample is not None:
            data = data.sample(frac=sample)

        xlim, ylim = plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)
        if data.shape[0] < 1000:
            ax.scatter(x=data[x], y=data[y], s=3)
            ax = self._plot_asthetics(ax, x, y, xlim, ylim, title=population_name)
        else:
            self._2dhist(ax, data, x, y)
            ax = self._plot_asthetics(ax, x, y, xlim, ylim, title=population_name)
        fig.show()

    def compare_control(self,
                        population: str,
                        ctrl_id: list,
                        x: str,
                        y: str,
                        xlim: tuple = None,
                        ylim: tuple = None,
                        transforms: dict or None = None,
                        figsize: tuple = (8, 8)):
        """
        Under the assumption that control gating has been performed on the underlying object,
        plot the target population (overlaid on its parent population) and on the same plot, show
        the predicted population for controls

        Parameters
        ----------
        population: str
            Population of interest
        ctrl_id: list
            One or more controls to plot for comparison (must be valid control IDs)
        x: str
            X-axis variable
        y: str
            Y-axis variable
        xlim: tuple, optional
            x-axis limits
        ylim: tuple, optional
            y-axis limits
        transforms: dict, optional
            How to transform axis variables (defaults to logicle transform if not given)
        figsize: tuple
            Figure size
        Returns
        -------
        Matplotlib.axes

        """
        if transforms is None:
            print('No transforms provided, defaulting to logicle')
            transforms = dict(x='logicle', y='logicle')
        axes_vars = {'x': x, 'y': y}
        base_pop = transform_axes(data=self.gating.get_population_df(population, transform=False),
                                  axes_vars=axes_vars,
                                  transforms=transforms)
        parent = transform_axes(data=self.gating.get_population_df(self.gating.populations[population].parent.name,
                                                                   transform=False),
                                axes_vars=axes_vars,
                                transforms=transforms)
        ctrls = [transform_axes(data=self.gating.get_population_df(population,
                                                                   transform=False,
                                                                   ctrl_id=c),
                                axes_vars=axes_vars,
                                transforms=transforms) for c in ctrl_id]

        fig, ax = plt.subplots(figsize=figsize)
        xlim, ylim = plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)
        ax = self._plot_asthetics(ax, x, y, xlim, ylim, title=f'Control comparison; {population}; {ctrl_id}')
        ax = self._2dhist(ax=ax, data=parent, x=x, y=y)
        pop_hull = {pid: {'data': d[[x, y]].values, 'hull': ConvexHull(d[[x, y]].values)}
                    for pid, d in zip([population] + ctrl_id, [base_pop] + ctrls)}
        legend_handles = list()
        for pid, data_hull in pop_hull.items():
            colour = next(self.colours)
            d = data_hull.get('data')
            for simplex in data_hull.get('hull').simplices:
                ax.plot(d[simplex, 0], d[simplex, 1], '-k', c=colour)
            legend_handles.append(mpatches.Patch(color=colour, label=pid))
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))
        return ax

    def backgate(self,
                 base_population: str,
                 x: str,
                 y: str,
                 pgeoms: list or None = None,
                 poverlay: list or None = None,
                 cgeoms: list or None = None,
                 coverlay: list or None = None,
                 cluster_root_population: str or None = None,
                 meta_clusters: bool = True,
                 xlim: tuple or None = None,
                 ylim: tuple or None = None,
                 transforms: dict or None = None,
                 title: str or None = None,
                 figsize: tuple = (8, 8)) -> None:
        """
        This function allows for plotting of multiple populations on the backdrop of some population upstream of
        the given populations. Each population is highlighted by either a polygon gate (for population names listed
        in 'geoms') or a scatter plot (for population names listed in 'overlay')

        Parameters
        ----------
        base_population : str
            upstream population to form the backdrop of plot
        x : str
            name of the x-axis variable
        y : str
            name of the y-axis variable
        pgeoms : list, optional
            list of populations to display as a polygon 'gate'
        poverlay : list, optional
            list of populations to display as a scatter plot
        cgeoms : list, optional
            list of clusters to display as a polygon 'gate'
        coverlay : list, optional
            list of clusters to display as a scatter plot
        cluster_root_population: str, optional
            root population to retrieve clusters from (only required if values given for cgeoms and/or coverlay)
        xlim : tuple, optional
            x-axis limits
        ylim : tuple, optional
            y-axis limit
        transforms : dict, optional
            dictionary of transformations to be applied to axis {'x' or 'y': transform method}
        title : str, optional
            title for plot
        figsize : tuple, (default=(8, 8))
            tuple of figure size to pass to matplotlib call

        Returns
        -------
        None
        """
        def _default(x):
            if x is None:
                return []
            return x
        pgeoms, poverlay, cgeoms, coverlay = _default(pgeoms), _default(poverlay), _default(cgeoms), _default(coverlay)
        # Check populations exist
        fg = FileGroup.objects(id=self.gating.mongo_id).get()
        if cgeoms or coverlay:
            assert cluster_root_population, 'If plotting clusters, must provide root population'
            all_clusters = list(fg.get_population(cluster_root_population).list_clusters(meta=meta_clusters))
        else:
            all_clusters = []
        for p in [base_population] + pgeoms + poverlay:
            assert p in self.gating.populations.keys(), f'Error: could not find {p}, valid populations include: ' \
                                                        f'{self.gating.populations.keys()}'
        for c in cgeoms + coverlay:
            assert c in all_clusters, f'Error: could not find {c}, valid clusters include: {all_clusters}'
        # Check root population is upstream
        for p in pgeoms + poverlay:
            dependencies = self.gating.find_dependencies(p)
            assert base_population not in dependencies, f'Error: population {p} is upstream from ' \
                                                        f'the chosen root population {base_population}'
        # Establish axis vars and transforms
        axes_vars = {'x': x, 'y': y}
        if transforms is None:
            transforms = dict(x='logicle', y='logicle')
            for a in ['x', 'y']:
                if any([x in axes_vars[a] for x in ['FSC', 'SSC']]):
                    transforms[a] = None
        populations = pgeoms + poverlay

        # Get root data
        root_data = transform_axes(self.gating.get_population_df(base_population),
                                   transforms=transforms, axes_vars=axes_vars)
        # Get population data
        pop_data = {p: root_data.loc[self.gating.populations[p].index][[x, y]].values
                    for p in populations}
        # Establish convex hull and centroids for populations
        pop_hull = {p: ConvexHull(d) for p, d in pop_data.items() if p in pgeoms}

        # Get cluster data
        c_hull = dict()
        cgeoms = [(c, self.gating._cluster_idx(c, clustering_root=cluster_root_population, meta=meta_clusters))
                  for c in cgeoms]
        coverlay = [(c, self.gating._cluster_idx(c, clustering_root=cluster_root_population, meta=meta_clusters))
                    for c in coverlay]
        if cgeoms:
            c_hull = {c: ConvexHull(root_data.loc[idx][[x, y]].values) for c, idx in cgeoms}

        # Build plotting constructs
        if title is None:
            title = f'{base_population}: {populations}'
        fig, ax = plt.subplots(figsize=figsize)
        ax = self._2dhist(ax=ax, data=root_data, x=x, y=y)
        ax = self._plot_asthetics(ax, x, y, xlim, ylim, title)
        colours = ['black', 'gray', 'brown', 'red', 'orange',
                   'coral', 'peru', 'olive', 'magenta', 'crimson',
                   'orchid']
        random.shuffle(colours)
        colours = cycle(colours)
        for p in pop_data.keys():
            c = next(colours)
            if p in pgeoms:
                for simplex in pop_hull[p].simplices:
                    ax.plot(pop_data[p][simplex, 0], pop_data[p][simplex, 1], '-k', c=c)
            else:
                ax.scatter(x=pop_data[p][:, 0], y=pop_data[p][:, 1], s=15, c=[c], alpha=0.8, label=p,
                           linewidth=0.4, edgecolors='black')
        for c, idx in coverlay:
            col = next(colours)
            cd = root_data.loc[idx][[x, y]].values
            ax.scatter(x=cd[:, 0], y=cd[:, 1], s=15, alpha=0.8, label=c, c=[col], linewidth=0.4, edgecolors='black')
        for c, idx in cgeoms:
            col = next(colours)
            cd = root_data.loc[idx][[x, y]].values
            for simplex in c_hull[c].simplices:
                ax.plot(cd[simplex, 0], cd[simplex, 1], '-k', c=col)
        ax.legend()

    def _get_data_transform(self, node: Node, geom: dict) -> pd.DataFrame:
        """
        Internal method. Fetch and transform data for grid plot

        Parameters
        ----------
        node : None
            population Node
        geom : dict
            gate geom

        Returns
        -------
        Pandas.DataFrame
            Transformed DataFrame
        """
        x, y = geom['x'], geom['y'] or self.default_axis
        data = self.gating.get_population_df(node.name)[[x, y]]
        transform_x = geom.get('transform_x')
        transform_y = geom.get('transform_y')
        if not transform_x and all([k in x for k in ['FSC', 'SSC']]):
            transform_x = 'logicle'
        if not transform_y and all([k in y for k in ['FSC', 'SSC']]):
            transform_y = 'logicle'
        data = transform_axes(data, transforms={'x': transform_x, 'y': transform_y}, axes_vars={'x': x, 'y': y})
        return data[(data[x] > data[x].quantile(0.01)) & (data[y] < data[y].quantile(0.99))]








