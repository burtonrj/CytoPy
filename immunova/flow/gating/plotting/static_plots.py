from immunova.data.gating import Gate
from immunova.flow.gating.utilities import centroid
from immunova.flow.gating.transforms import apply_transform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches
from itertools import cycle
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np
import random


class Plot:
    """
    Class for producing facs plots
    """
    def __init__(self, gating_object):
        """
        Constructor for plotting class
        :param gating_object: Gating object to generate plots from
        """
        self.gating = gating_object
        self.colours = ['#FF0000', '#8B0000', '#FFA500', '#BDB76B', '#7CFC00', '#32CD32',
                        '#008000', '#FF1493', '#2F4F4F', '#000000']
        random.shuffle(self.colours)
        self.colours = ['#EB1313'] + self.colours

    @staticmethod
    def __transform_gate(data: pd.DataFrame, gate: Gate):
        data = data.copy()
        kwargs = {k: v for k, v in gate.kwargs}
        if 'transform_x' in kwargs.keys():
            if kwargs['transform_x'] is not None:
                data = apply_transform(data, [kwargs['x']], transform_method=kwargs['transform_x'])
        else:
            # Default = Logicle transform
            data = apply_transform(data, [kwargs['x']], transform_method='logicle')
        if 'transform_y' in kwargs.keys():
            if kwargs['transform_y'] is not None:
                data = apply_transform(data, [kwargs['y']], transform_method=kwargs['transform_y'])
        else:
            # Default = Logicle transform
            data = apply_transform(data, [kwargs['y']], transform_method='logicle')
        return data

    def __get_gate_data(self, gate: Gate) -> dict:
        """
        Given a gate, return all associated data that gate acts upon
        :param gate: Gate document
        :return: Dictionary of pandas dataframes
        """
        kwargs = {k: v for k, v in gate.kwargs}
        data = dict(primary=self.gating.get_population_df(gate.parent))
        for x in ['fmo_x', 'fmo_y']:
            if x in kwargs.keys():
                data[x] = self.gating.get_fmo_data(gate.parent)
        return data

    @staticmethod
    def __plot_axis_lims(x: str, y: str, xlim: tuple or None = None, ylim: tuple or None = None) -> tuple and tuple:
        """
        Establish axis limits
        :param x: name of x axis
        :param y: name of y axis
        :param xlim: custom x-axis limit (default = None)
        :param ylim: custom y-axis limit (default = None)
        :return: x-axis limits, y-axis limits
        """
        if not xlim:
            if any([x.find(c) != -1 for c in ['FSC', 'SSC']]):
                xlim = (0, 250000)
            else:
                xlim = (0, 1)
        if not ylim:
            if any([y.find(c) != -1 for c in ['FSC', 'SSC']]):
                ylim = (0, 250000)
            else:
                ylim = (0, 1)
        return xlim, ylim

    @staticmethod
    def __plot_asthetics(ax: matplotlib.pyplot.axes, x: str, y: str,
                         xlim: tuple, ylim: tuple, title: str) -> matplotlib.pyplot.axes:
        """
        Customise axes asthetics
        :param ax: matplotlib axes object
        :param x: name of x-axis
        :param y: name of y-axis
        :param xlim: custom x-axis limit (default = None)
        :param ylim: custom y-axis limit (default = None)
        :param title: plot title
        :return: Updated axes object
        """
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_ylabel(y)
        ax.set_xlabel(x)
        ax.set_title(title)
        return ax

    def plot_gate(self, gate_name: str, xlim: tuple or None = None, ylim: tuple or None = None) -> None:
        """
        Produce a static plot of a gate
        :param gate_name: Name of gate to plot
        :param xlim: custom x-axis limit (default = None)
        :param ylim: custom y-axis limit (default = None)
        :return: None
        """
        if gate_name not in self.gating.gates.keys():
            print(f'Error: could not find {gate_name} in attached gate object')
        gate = self.gating.gates[gate_name]
        data = self.__get_gate_data(gate)
        num_axes = len(data.keys())
        fig, axes = plt.subplots(ncols=num_axes)
        self.__geom_plot(data, fig, axes, gate, xlim, ylim)

    def __build_geom_plot(self, data: pd.DataFrame, gate: Gate, ax: matplotlib.pyplot.axes,
                          xlim: tuple, ylim: tuple, title: str) -> matplotlib.pyplot.axes or None:
        """
        Produce a plot of a gate that generates a geometric object
        :param data: pandas dataframe of events data to plot
        :param gate: Gate object to plot
        :param xlim: custom x-axis limit (default = None)
        :param ylim: custom y-axis limit (default = None)
        :param ax: matplotlib axes object
        :param title: plot title
        :return: Updated axes object
        """
        kwargs = {k: v for k, v in gate.kwargs}
        x, y = kwargs['x'], kwargs['y'] or 'FSC-A'
        xlim, ylim = self.__plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)
        geoms = {c: self.gating.populations[c].geom for c in gate.children}
        if geoms['shape'] == 'sml':
            print(f'Error: {gate.gate_name} is a supervised machine learning gate. This type of gating does not produce'
                  f'2D geometries but instead classifies cells based using high dimensional feature space. To observe'
                  f'a population classified by this method in 2D, use the `plot_sml` method')
            return None
        if data.shape[0] < 1000:
            ax.scatter(x=data[x], y=data[y], s=3)
            ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title)
        else:
            ax = self.__2dhist(ax, data, x, y)
            ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title)

        colours = cycle(self.colours)
        # Draw geom
        for (child_name, geom), colour in zip(geoms.items(), colours):
            colour = '#EB1313'
            if geom is None or geom == dict():
                print(f'Population {child_name} has no associated gate, skipping...')
                continue
            if geom['shape'] == 'threshold':
                ax.axvline(geom['threshold'], c=colour)
            if geom['shape'] == '2d_threshold':
                ax.axvline(geom['threshold_x'], c=colour)
                ax.axhline(geom['threshold_y'], c=colour)
            if geom['shape'] == 'ellipse':
                ellipse = patches.Ellipse(xy=geom['centroid'], width=geom['width'], height=geom['height'],
                                          angle=geom['angle'], fill=False, edgecolor=colour)
                ax.add_patch(ellipse)
            if geom['shape'] == 'rect':
                rect = patches.Rectangle(xy=(geom['x_min'], geom['y_min']),
                                         width=((geom['x_max']) - (geom['x_min'])),
                                         height=(geom['y_max'] - geom['y_min']),
                                         fill=False, edgecolor=colour)
                ax.add_patch(rect)
            if geom['shape'] == 'poly':
                x = geom['cords']['x']
                y = geom['cords']['y']
                ax.plot(x, y, '-k', c=colour, label=child_name)
                ax.legend()

    def __geom_plot(self, data: dict, fig: matplotlib.pyplot.figure,
                    axes: np.array or matplotlib.pyplot.axes, gate: Gate,
                    xlim: tuple, ylim: tuple) -> None:
        """
        Wrapper function for producing a plot of a gate that generates a geometric object
        :param data: dictionary of data to plot (primary and controls)
        :param gate: Gate object to plot
        :param xlim: custom x-axis limit (default = None)
        :param ylim: custom y-axis limit (default = None)
        :param fig: matplotlib figure object
        :param axes: matplotlib axes object or array of axes objects
        :return: None
        """
        if hasattr(axes, '__iter__'):
            for ax, (name, d) in zip(axes, data.items()):
                d = self.__transform_gate(d, gate)
                self.__build_geom_plot(d, gate, ax, xlim, ylim, f'{gate.gate_name}_{name}')
        else:
            d = self.__transform_gate(data['primary'], gate)
            self.__build_geom_plot(d, gate, axes, xlim, ylim, f'{gate.gate_name}')
        fig.show()

    @staticmethod
    def __2dhist(ax: matplotlib.pyplot.axes, data: pd.DataFrame, x: str, y: str) -> matplotlib.pyplot.axes:
        """
        Generate a standard 2D histogram
        :param ax: matplotlib axes object
        :param data: dictionary of data to plot (primary and controls)
        :param x: name of x-axis dimension
        :param y: name of y-axis dimension
        :return: Updated matplotlib axes object
        """
        if data.shape[0] <= 100:
            bins = 50
        elif data.shape[0] > 1000:
            bins = 500
        else:
            bins = int(data.shape[0] * 0.5)
        ax.hist2d(data[x], data[y], bins=bins, norm=LogNorm())
        return ax

    def plot_population(self, population_name: str, x: str, y: str, xlim: tuple = None, ylim: tuple = None,
                        transform_x: str or None = None, transform_y: str or None = None):
        """
        Generate a static plot of a population
        :param population_name: name of population to plot
        :param x: name of x-axis dimension
        :param y: name of y-axis dimension
        :param xlim: tuple of x-axis limits
        :param ylim: tuple of y-axis limits
        :return: None
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        if population_name in self.gating.populations.keys():
            data = self.gating.get_population_df(population_name).copy()
        else:
            print(f'Invalid population name, must be one of {self.gating.populations.keys()}')
            return None
        if transform_x is not None:
            data = apply_transform(data, [x], transform_x)
        if transform_y is not None:
            data = apply_transform(data, [y], transform_y)
        xlim, ylim = self.__plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)
        if data.shape[0] < 1000:
            ax.scatter(x=data[x], y=data[y], s=3)
            ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title=population_name)
        else:
            self.__2dhist(ax, data, x, y)
            ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title=population_name)
        fig.show()
