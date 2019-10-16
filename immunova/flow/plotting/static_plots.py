from immunova.data.gating import Gate
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches
from itertools import cycle
from scipy.spatial import ConvexHull
import pandas as pd
import numpy as np


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
        if 'Clustering' in gate.class_:
            self.__cluster_plot(gate, xlim, ylim)
            return
        data = self.__get_gate_data(gate)
        num_axes = len(data.keys())
        fig, axes = plt.subplots(ncols=num_axes)
        self.__geom_plot(data, fig, axes, gate, xlim, ylim)

    def __cluster_plot(self, gate: Gate, xlim: tuple, ylim: tuple) -> None:
        """
        Produce a plot of clusters generated from a cluster gate
        :param gate: Gate object to plot
        :param xlim: custom x-axis limit (default = None)
        :param ylim: custom y-axis limit (default = None)
        :return: None
        """
        # Axes information
        kwargs = {k: v for k, v in gate.kwargs}
        x, y = kwargs['x'], kwargs['y'] or 'FSC-A'
        xlim, ylim = self.__plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)
        fig, ax = plt.subplots(figsize=(5, 5))
        d = self.gating.get_population_df(gate.parent)
        ax = self.__2dhist(ax, d, x, y)
        colours = cycle(['green', 'blue', 'red', 'magenta', 'cyan'])
        for child, colour in zip(gate.children, colours):
            d = self.gating.get_population_df(child)
            if d is None:
                continue
            d = d[[x, y]].values
            centroid = self.__centroid(d)
            ax.scatter(x=centroid[0], y=centroid[1], c=colour, s=8, label=child)
            hull = ConvexHull(d)
            for simplex in hull.simplices:
                ax.plot(d[simplex, 0], d[simplex, 1], 'k-', c='red')
        self.__plot_asthetics(ax, x, y, xlim, ylim, title=gate.gate_name)
        ax.legend()
        fig.show()

    @staticmethod
    def __centroid(data: np.array):
        length = data.shape[0]
        sum_x = np.sum(data[:, 0])
        sum_y = np.sum(data[:, 1])
        return sum_x / length, sum_y / length

    def __build_geom_plot(self, data: pd.DataFrame, gate: Gate, ax: matplotlib.pyplot.axes,
                          xlim: tuple, ylim: tuple, title: str) -> matplotlib.pyplot.axes:
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
        geom = self.gating.populations[gate.children[0]].geom
        if data.shape[0] < 1000:
            ax.scatter(x=data[x], y=data[y], s=3)
            ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title)
        else:
            ax = self.__2dhist(ax, data, x, y)
            ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title)
        # Draw geom
        if geom.shape == 'threshold_1d':
            ax.axvline(geom['threshold'], c='r')
        if geom.shape == 'threshold_2d':
            ax.axvline(geom['threshold_x'], c='r')
            ax.axhline(geom['threshold_y'], c='r')
        if geom.shape == 'ellipse':
            ellipse = patches.Ellipse(xy=geom['centroid'], width=geom['width'], height=geom['height'],
                                      angle=geom['angle'], fill=False, edgecolor='r')
            ax.add_patch(ellipse)
        if geom.shape == 'rect':
            rect = patches.Rectangle(xy=(geom['x_min'], geom['y_min']),
                                     width=((geom['x_max']) - (geom['x_min'])),
                                     height=(geom['y_max'] - geom['y_min']),
                                     fill=False, edgecolor='r')
            ax.add_patch(rect)

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
                self.__build_geom_plot(d, gate, ax, xlim, ylim, f'{gate.gate_name}_{name}')
        else:
            self.__build_geom_plot(data['primary'], gate, axes, xlim, ylim, f'{gate.gate_name}')
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

    def plot_population(self, population_name: str, x: str, y: str, xlim: tuple = None, ylim: tuple = None):
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
            data = self.gating.get_population_df(population_name)
        else:
            print(f'Invalid population name, must be one of {self.gating.populations.keys()}')
            return None
        xlim, ylim = self.__plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)
        if data.shape[0] < 1000:
            ax.scatter(x=data[x], y=data[y], s=3)
            ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title=population_name)
        else:
            self.__2dhist(ax, data, x, y)
            ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title=population_name)
        fig.show()
