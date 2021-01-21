#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Central to the analysis of Cytometry data is visualisation. For exploratory
analysis and 'gating' this normally comes in the form of bi-axial plots of
different cell surface markers or intracellular stains. This module contains
the CreatePlot class which houses the functionality for all one and two
dimensional plotting of cytometry data. This class interacts with Population
objects to present the data in multiple ways. This can be as standard 2D histograms
as is common in software like FlowJo, but also allows for plotting of Population
geometries (the shapes that define the gates that generated a Population) or
overlaying downstream populations for 'back-gating' purposes.

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


from ..data.gate import Gate, ThresholdGate, PolygonGate, EllipseGate, Population
from ..data.geometry import ThresholdGeom, PolygonGeom
from ..flow.transform import apply_transform
from warnings import warn
from typing import List, Generator, Dict
from scipy.spatial import ConvexHull
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches
from itertools import cycle
import seaborn as sns
import pandas as pd
import numpy as np

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


class CreatePlot:
    """
    Generate 1D or 2d histograms of cell populations as identified by cytometry. Supports plotting of individual
    populations, single or multiple gates, "backgating" (plotting child populations overlaid on parent) and
    overlaying populations from control samples on their equivalent in the primary sample.

    Attributes
    -----------
    transform_x: str (default = "logicle")
        How to transform the x-axis. Method 'plot_gate' overwrites this value with the value associated with
        the gate
    transform_y: str (default = "logicle")
        How to transform the y-axis. Method 'plot_gate' overwrites this value with the value associated with
        the gate
    xlabel: str, optional
        x-axis label
    ylabel: str, optional
        y-axis label
    xlim: (float, float), optional
        x-axis limits; if not given defaults to the range of 0.001 quantile to 0.999 quantile
    ylim: (float, float), optional
        y-axis limits; if not given defaults to the range of 0.001 quantile to 0.999 quantile
    ax: matplotlib.pyplot.axes, optional
        If not given, an axes object will be generated with the given figsize. Access the figure
        object through 'fig' attribute
    figsize: (int, int)
        Ignored if an Axes object is given
    bins: int or str (default="scotts")
        How many bins to use for 2D histogram. Can also provide a value of "scotts", "sturges", "rice" or "sqrt" to
        use a given method to estimate suitable bin size
    cmap: str, (default="jet")
        Colormap for 2D histogram
    style: str, optional (default="white")
        Plotting style (passed to seaborn.set_style)
    autoscale: bool (default=True)
        Allow matplotlib to calculate optimal view
        (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.autoscale.html)
    font_scale: float, optional (default=1.2)
        Font scale (passed to seaborn.set_context)
    bw: str or float, (default="scott")
        Bandwidth for 1D KDE (see seaborn.kdeplot)
    axis_ticks: bool (default=True)
        Show axis ticks with axis labels
    """

    def __init__(self,
                 transform_x: str or None = "logicle",
                 transform_y: str or None = "logicle",
                 xlabel: str or None = None,
                 ylabel: str or None = None,
                 xlim: (float, float) or None = None,
                 ylim: (float, float) or None = None,
                 title: str or None = None,
                 ax: matplotlib.pyplot.axes or None = None,
                 figsize: (int, int) = (5, 5),
                 bins: int or str = "sqrt",
                 cmap: str = "jet",
                 style: str or None = "white",
                 font_scale: float or None = 1.2,
                 bw: str or float = "scott",
                 autoscale: bool = True,
                 axis_ticks: bool = True):
        self.transforms = {'x': transform_x, 'y': transform_y}
        self.labels = {'x': xlabel, 'y': ylabel}
        self.autoscale = autoscale
        self.lims = {'x': xlim or [None, None], 'y': ylim or [None, None]}
        self.title = title
        self.bw = bw
        if type(bins) == str:
            valid_bin_str = ["scott", "sturges", "rice", "sqrt", "stone", "doane", "fd", "auto"]
            assert bins in valid_bin_str, f"bins should be an integer or one of {valid_bin_str}"
        self.bins = bins
        self.fig, self._ax = None, ax
        if self._ax is None:
            self.fig, self._ax = plt.subplots(figsize=figsize)
        self.cmap = plt.get_cmap(cmap)
        if axis_ticks:
            sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
        if style is not None:
            sns.set_style(style)
        if font_scale is not None:
            sns.set_context(font_scale=font_scale)
        plt.xticks(rotation=90)
        self._ax.xaxis.labelpad = 20
        self._ax.yaxis.labelpad = 20

    def _hist1d(self,
                data: pd.DataFrame,
                x: str,
                **kwargs):
        """
        Generate a 1D KDE plot using Seaborn. Resulting axis assigned to self.ax

        Parameters
        ----------
        data: Pandas.DataFrame
            Data to plot
        x: str
            Name of the channel to plot
        kwargs:
            Additional keyword arguments passed to seaborn.kdeplot

        Returns
        -------
        None
        """
        sns.kdeplot(data=data[x], bw_method=self.bw, ax=self._ax, **kwargs)

    def _hist2d(self,
                data: pd.DataFrame,
                x: str,
                y: str,
                **kwargs) -> None:
        """
        Generate a 2D histogram using Matplotlib
        (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist2d.html)

        Parameters
        ----------
        data: Pandas.DataFrame
            Data to plot
        x: str
            X-axis channel name
        y: str
            Y-axis channel name
        kwargs:
            Additional keyword arguments to pass to matplotlib.pyplot.axes.hist2d

        Returns
        -------
        None
        """
        bins = [np.histogram_bin_edges(data[x].values, bins=self.bins),
                np.histogram_bin_edges(data[y].values, bins=self.bins)]
        self._ax.hist2d(data[x], data[y], bins=bins, norm=LogNorm(), cmap=self.cmap, **kwargs)

    def _set_axis_limits(self,
                         data: pd.DataFrame,
                         x: str,
                         y: str or None):
        """
        Set the axis limits. Mutates self._ax

        Parameters
        ----------
        data: Pandas.DataFrame
            Data being plotting (used to estimate quantiles9
        x: str
            X-axis channel
        y: str or None
            Y-axis channel (optional)

        Returns
        -------
        None
        """
        if self.autoscale:
            self._ax.autoscale(enable=True)
        else:
            x_min = self.lims.get("x")[0] or data[x].quantile(q=0.001)
            x_max = self.lims.get("x")[1] or data[x].quantile(q=0.999)
            self._ax.set_xlim((x_min, x_max))
            if y is not None:
                y_min = self.lims.get("y")[0] or data[y].quantile(q=0.001)
                y_max = self.lims.get("y")[1] or data[y].quantile(q=0.999)
                self._ax.set_ylim((y_min, y_max))

    def _set_aesthetics(self,
                        x: str,
                        y: str or None):
        """
        Set plot aesthetics: title and axis labels. Mutates self._ax

        Parameters
        ----------
        x: str
            X-axis channel (for default x-axis label)
        y: str
            Y-axis channel (for default y-axis label)

        Returns
        -------
        None
        """
        if self.title:
            self._ax.set_title(self.title)
        if self.labels.get("x"):
            self._ax.set_xlabel(self.labels.get("x"))
        else:
            self._ax.set_xlabel(x)
        if self.labels.get("y") and y is not None:
            self._ax.set_ylabel(self.labels.get("y"))
        elif y is not None:
            self._ax.set_ylabel(y)

    def _transform_axis(self,
                        data: pd.DataFrame,
                        x: str,
                        y: str or None):
        """
        Transform plotting data according to objects transforms attribute

        Parameters
        ----------
        data: Pandas.DataFrame
            Data to plot
        x: str
            Name of X-axis channel
        y: str or None
            Name of Y-axis channel

        Returns
        -------
        Pandas.DataFrame
            Transformed data
        """
        data = data.copy()
        transforms = {column: self.transforms.get(axis) for column, axis in zip([x, y], ["x", "y"])
                      if self.transforms.get(axis) is not None and column is not None}
        if len(list(transforms.keys())) > 0:
            data = apply_transform(data,
                                   features_to_transform=transforms)
        return data

    def plot(self,
             data: pd.DataFrame,
             x: str,
             y: str or None = None,
             **kwargs):
        """
        Plot a single population as either a 2D histogram or 1D KDE

        Parameters
        ----------
        data: Pandas.DataFrame
            Population dataframe
        x: str
            Channel to plot on the x-axis
        y: str, optional
            Channel to plot on the y-axis
        kwargs:
            Keyword arguments to be passed to matplotlib.pyplot.axes.hist2d or seaborn.kdeplot (depending on whether
            a y-axis variable is given)

        Returns
        -------
        Matplotlib.pyplot.axes
            Axis object
        """
        data = self._transform_axis(data=data, x=x, y=y)
        if y is None:
            self._hist1d(data=data, x=x, **kwargs)
        else:
            self._hist2d(data=data, x=x, y=y, **kwargs)
        self._set_axis_limits(data=data, x=x, y=y)
        self._set_aesthetics(x=x, y=y)
        return self._ax

    def plot_gate_children(self,
                           gate: Gate or ThresholdGate or EllipseGate or PolygonGate,
                           parent: pd.DataFrame,
                           lw: float = 2.5,
                           y: str or None = None,
                           transform_x: str or None = None,
                           transform_y: str or None = None,
                           plot_kwargs: dict or None = None,
                           legend_kwargs: dict or None = None):
        """
       Plot a Gate object. This will plot the geometric shapes generated from a single Gate, overlaid on the
       parent population given as Pandas.DataFrame. It should be noted, this will plot the geometric definitions
       of a gates children, i.e. the expected populations. If you have generated new populations from new data
       using a Gate you should plot with the 'plot_population_geoms' method

       Parameters
       ----------
       gate: Gate or ThresholdGate or EllipseGate or PolygonGate
       parent: Pandas.DataFrame
           Parent DataFrame
       lw: float (default = 2.5)
           Linewidth for shapes to plot
       plot_kwargs:
           Additional keyword arguments to pass to plot_population (generates the plot of parent population)
       legend_kwargs:
           Additional keyword arguments to pass to axis legend. Defaults:
           * bbox_to_anchor = (0.5, 1.05)
           * loc = "upper center"
           * ncol = 3
           * fancybox = True
           * shadow = False
       y: str (optional)
           Overrides the plotting configurations for the gate if y is missing
           and allows user to plot a two-dimensional instead of one dimensional plot.
           Only value for ThresholdGate.
       transform_x: str (optional)
           Overrides the transformation to the x-axis variable
       transform_y: str (optional)
           Overrides the transformation to the x-axis variable

       Returns
       -------
       Matplotlib.pyplot.axes
           Axis object
       """
        plot_kwargs = plot_kwargs or {}
        legend_kwargs = legend_kwargs or dict()
        gate_colours = cycle(["#c92c2c",
                              "#2df74e",
                              "#e0d572",
                              "#000000",
                              "#64b9c4",
                              "#9e3657"])
        self.transforms = {"x": gate.transformations.get("x", None) or transform_x,
                           "y": gate.transformations.get("y", None) or transform_y}
        self._ax = self.plot(data=parent,
                             x=gate.x,
                             y=gate.y or y,
                             **plot_kwargs)
        # If threshold, add threshold lines to plot and return axes
        if isinstance(gate, ThresholdGate):
            return self._plot_threshold(definitions={c.definition: c.name for c in gate.children},
                                        geoms=[c.geom for c in gate.children],
                                        lw=lw)
        # Otherwise, we assume polygon shape
        return self._plot_polygon(geoms=[c.geom for c in gate.children],
                                  labels=[c.name for c in gate.children],
                                  colours=gate_colours,
                                  lw=lw,
                                  legend_kwargs=legend_kwargs)

    def plot_population_geoms(self,
                              parent: pd.DataFrame,
                              children: List[Population],
                              lw: float = 2.5,
                              y: str or None = None,
                              transform_x: str or None = None,
                              transform_y: str or None = None,
                              plot_kwargs: dict or None = None,
                              legend_kwargs: dict or None = None):
        """
        This will plot the geometric shapes from the list of child populations generated from a single Gate,
        overlaid on the parent population upon which the Gate has been applied. The parent data should be provided
        as a Pandas DataFrame of single cell data and the Geoms of the resulting Populations in the list
        'children'.

        Parameters
        ----------
        parent: Pandas.DataFrame
            Parent DataFrame
        children: list
            List of Population objects that derive from the parent. Population geometries will
            be overlaid on the parent population.
        lw: float (default = 2.5)
            Linewidth for shapes to plot
        plot_kwargs:
            Additional keyword arguments to pass to plot_population (generates the plot of parent population)
        legend_kwargs:
            Additional keyword arguments to pass to axis legend. Defaults:
                * bbox_to_anchor = (0.5, 1.05)
                * loc = "upper center"
                * ncol = 3
                * fancybox = True
                * shadow = False
        y: str (optional)
            Overrides the plotting configurations for the gate if y is missing
            and allows user to plot a two-dimensional instead of one dimensional plot.
            Only value for ThresholdGate.
        transform_x: str (optional)
            Overrides the transformation to the x-axis variable
        transform_y: str (optional)
            Overrides the transformation to the x-axis variable

        Returns
        -------
        Matplotlib.pyplot.axes
            Axis object
        """
        gate_colours = cycle(["#c92c2c",
                              "#2df74e",
                              "#e0d572",
                              "#000000",
                              "#64b9c4",
                              "#9e3657"])
        assert len(set(str(type(x.geom) for x in children))), "Children geometries must all be of the same type"
        if y is not None:
            assert isinstance(children[0].geom, ThresholdGeom), "Can only override y-axis variable for Threshold " \
                                                                "geometries"
        plot_kwargs = plot_kwargs or {}
        legend_kwargs = legend_kwargs or dict()
        # Plot the parent population
        self.transforms = {"x": children[0].geom.transform_x or transform_x,
                           "y": children[0].geom.transform_y or transform_y}
        self._ax = self.plot(data=parent,
                             x=children[0].geom.x,
                             y=children[0].geom.y or y,
                             **plot_kwargs)
        # If threshold, add threshold lines to plot and return axes
        if isinstance(children[0].geom, ThresholdGeom):
            return self._plot_threshold(definitions={c.definition: c.population_name for c in children},
                                        geoms=[c.geom for c in children],
                                        lw=lw)
        # Otherwise, we assume polygon shape
        return self._plot_polygon(geoms=[c.geom for c in children],
                                  labels=[c.population_name for c in children],
                                  colours=gate_colours,
                                  lw=lw,
                                  legend_kwargs=legend_kwargs)

    def _plot_threshold(self,
                        definitions: Dict[str, str],
                        geoms: List[ThresholdGeom],
                        lw: float):
        """
        Plot Child populations from ThresholdGate

        Parameters
        ----------
        definitions: dict
            Dictionary of {definition: population/child name}
        geoms: list
            List of ThresholdGeom objects
        lw: float
            Line width for thresholds

        Returns
        -------
        Matplotlib.pyplot.axes
            Axis object
        """
        x = geoms[0].x_threshold
        y = geoms[0].y_threshold
        self._add_threshold(x=x,
                            y=y,
                            labels=definitions,
                            lw=lw)
        return self._ax

    def _plot_polygon(self,
                      geoms: List[PolygonGeom],
                      labels: List[str],
                      colours: list or Generator,
                      lw: float,
                      legend_kwargs: dict):
        """
        Plot the Child populations of a Polygon or Elliptical gate.

        Parameters
        ----------
        geoms: list
            List of Population objects
        colours: list or generator
            Colour(s) of polygon outline(s)
        lw: float
            Line width of polygon outline(s)
        legend_kwargs: dict
            Additional keyword arguments to pass to axis legend. Defaults:
            * bbox_to_anchor = (0.5, 1.05)
            * loc = "upper center"
            * ncol = 3
            * fancybox = True
            * shadow = False
        Returns
        -------
        Matplotlib.pyplot.axes
            Axis object
        """
        for i, g in enumerate(geoms):
            colour = next(colours)
            self._add_polygon(x_values=g.x_values,
                              y_values=g.y_values,
                              colour=colour,
                              label=labels[i],
                              lw=lw)
        self._set_legend(**legend_kwargs)
        return self._ax

    def _add_polygon(self,
                     x_values: list,
                     y_values: list,
                     colour: str,
                     label: str,
                     lw: float):
        """
        Add a polygon shape to the axes object

        Parameters
        ----------
        x_values: list or array
            x-axis coordinates
        y_values: list or array
            y-axis coordinates
        colour: str
            colour of the polygon line
        label: str
            label to associate to polygon for legend
        lw: float
            linewidth of polygon line

        Returns
        -------
        None
        """
        self._ax.plot(x_values, y_values, '-k', c=colour, label=label, lw=lw)

    def _add_ellipse(self,
                     center: (float, float),
                     width: float,
                     height: float,
                     angle: float,
                     colour: str,
                     lw: float,
                     label: str):
        """
        Add an ellipse shape to the axes object

        Parameters
        ----------
        center: (float, float)
            center point of ellipse
        width: float
            width of ellipse
        height: float
            heignt of ellipse
        angle: float
            angle of ellipse
        colour: str
            colour of ellipse edge
        lw: float
            linewidth of ellipse
        label: str
            label for axis legend

        Returns
        -------
        None
        """
        ellipse = patches.Ellipse(xy=center,
                                  width=width,
                                  height=height,
                                  angle=angle,
                                  fill=False,
                                  edgecolor=colour,
                                  lw=lw,
                                  label=label)
        self._ax.add_patch(ellipse)

    def _add_threshold(self,
                       x: float,
                       y: float or None,
                       lw: float,
                       labels: dict or None = None):
        """
        Add a 1D or 2D threshold (red horizontal axis line and optionally a red vertical axis line)

        Parameters
        ----------
        x: float
            Axis position for horizontal threshold
        y: float, optional
            Axis position for vertical threshold
        labels: dict or None
            label for axis legend
        lw: float
            linewidth

        Returns
        -------
        None
        """
        x_range = self._ax.get_xlim()
        y_range = self._ax.get_ylim()
        self._ax.axvline(x, lw=lw, c="#c92c2c")
        if y is not None:
            self._ax.axhline(y, lw=lw, c="#c92c2c")
            # Label regions for two axis
            if labels is not None:
                xy = [(x + ((x_range[1] - x_range[0]) * .2),
                       y + ((y_range[1] - y_range[0]) * .2)),
                      (x - ((x_range[1] - x_range[0]) * .2),
                       y - ((y_range[1] - y_range[0]) * .2)),
                      (x + ((x_range[1] - x_range[0]) * .2),
                       y - ((y_range[1] - y_range[0]) * .2)),
                      (x - ((x_range[1] - x_range[0]) * .2),
                       y + ((y_range[1] - y_range[0]) * .2))]
                for d, xy_ in zip(["++", "--", "+-", "-+"], xy):
                    label = [v for l, v in labels.items() if d in l]
                    if len(label) == 0:
                        continue
                    label = label[0]
                    if label in ["++", "--", "+-", "-+"]:
                        label = f"{label[0]} {label[1]}"
                    self._ax.annotate(text=label,
                                      xy=xy_,
                                      fontsize="small",
                                      c="black",
                                      backgroundcolor="white",
                                      bbox=dict(facecolor='white', edgecolor='black', pad=5.0))
        else:
            # Label regions for one axis
            if labels is not None:
                self._ax.annotate(text=labels.get("+"),
                                  xy=(x + ((x_range[1] - x_range[0]) * .2),
                                      y_range[1] * .75),
                                  fontsize="medium",
                                  c="black",
                                  backgroundcolor="white",
                                  bbox=dict(facecolor='white', edgecolor='black', pad=5.0))
                self._ax.annotate(text=labels.get("-"),
                                  xy=(x - ((x_range[1] - x_range[0]) * .2),
                                      y_range[1] * .75),
                                  fontsize="medium",
                                  c="black",
                                  backgroundcolor="white",
                                  bbox=dict(facecolor='white', edgecolor='black', pad=5.0))

    def backgate(self,
                 parent: pd.DataFrame,
                 children: Dict[str, pd.DataFrame],
                 x: str,
                 y: str or None = None,
                 colours: str or None = "pastel",
                 alpha: float = .75,
                 size: float = 5,
                 method: str or dict = "scatter",
                 shade: bool = True,
                 plot_kwargs: dict or None = None,
                 overlay_kwargs: dict or None = None,
                 legend_kwargs: dict or None = None):
        """
        Plot one or more populations as an overlay atop a given parent population.
        Method should be either "scatter" (default) or "kde", which controls how
        overlaid populations will appear.

        Parameters
        ----------
        parent: Pandas.DataFrame
            Parent single cell data
        children: dict
            Dictionary of Pandas DataFrames, where the key corresponds to the
            population name and the value the Pandas DataFrame of single cell events
        x: str
            X-axis variable
        y: str (optional)
            Y-axis variable
        colours: str (defaults to Seaborn pastel palette)
            Name of the palette to use for colouring overlaid populations
        alpha: float (default=0.75)
            If method is 'scatter', controls transparency of markers
        size: float (default=5)
            If method is 'scatter', controls size of markers
        method: str or dict (default="scatter)
            Method should be either "scatter" (default), "polygon", or "kde", which controls how
            overlaid populations will appear. If a dictionary is provided, then it should
            have keys matching that of 'children' and values being the methods to use
            for each overlay.
        shade: bool (default=True)
            If method is 'kde', specifies whether to shade in the contours
        plot_kwargs: dict (optional)
            Keyword arguments passed to CreatePlot.plot
        overlay_kwargs: dict (optional)
            Keyword arguments passed to plt.scatter or seaborn.kdeplot cals
        legend_kwargs: dict (optional)
            Additional keyword arguments to pass to axis legend. Defaults:
            * bbox_to_anchor = (0.5, 1.05)
            * loc = "upper center"
            * ncol = 3
            * fancybox = True
            * shadow = False
        Returns
        -------
        Matplotlib.axes
        """
        colours = cycle(sns.color_palette(colours))
        plot_kwargs = plot_kwargs or {}
        overlay_kwargs = overlay_kwargs or {}
        legend_kwargs = legend_kwargs or {}
        self._ax = self.plot(data=parent,
                             x=x,
                             y=y,
                             **plot_kwargs)
        self._set_axis_limits(data=parent, x=x, y=y)

        for c, (child_name, df) in zip(colours, children.items()):
            method_ = method
            if isinstance(method, dict):
                method_ = method.get(child_name, "scatter")
            if method_ == "kde":
                self._overlay(data=df,
                              x=x,
                              y=y,
                              method=method,
                              color=c,
                              shade=shade,
                              label=child_name,
                              **overlay_kwargs)
            elif method_ == "scatter":
                self._overlay(data=df,
                              x=x,
                              y=y,
                              method=method,
                              color=c,
                              s=size,
                              alpha=alpha,
                              label=child_name,
                              **overlay_kwargs)
            else:
                self._overlay(data=df,
                              x=x,
                              y=y,
                              method=method,
                              color=c,
                              label=child_name,
                              **overlay_kwargs)
        self._set_legend(shape_n=len(children), **legend_kwargs)
        return self._ax

    def _overlay(self,
                 data: pd.DataFrame,
                 x: str,
                 y: str or None,
                 method: str,
                 label: str,
                 **kwargs):
        """
        Plot a single population dataframe on self._ax using the given method

        Parameters
        ----------
        data: Pandas.DataFrame
        x: str
        y: str, optional
        method: str
        label: str
        kwargs:
            Additional keyword arguments passed to self._ax.scatter or sns.kdeplot

        Returns
        -------
        None
        """
        assert method in ["scatter", "kde", "polygon"], "Overlay method should be 'scatter' or 'kde'"
        if y is None and method == "scatter":
            warn("1-dimensional plot, defaulting to KDE overlay")
            method = "kde"
        data = self._transform_axis(data=data,
                                    x=x,
                                    y=y)
        if method == "scatter":
            self._ax.scatter(x=data[x],
                             y=data[y],
                             label=label,
                             **kwargs)
        elif method == "polygon":
            d = data[[x, y]].values
            hull = ConvexHull(d)
            for simplex in hull.simplices:
                self._ax.plot(d[simplex, 0],
                              d[simplex, 1],
                              '-',
                              **kwargs)
        else:
            if y is None:
                sns.kdeplot(data=data[x],
                            ax=self._ax,
                            label=label,
                            **kwargs)
            else:
                sns.kdeplot(data=data[x],
                            data2=data[y],
                            ax=self._ax,
                            label=label,
                            **kwargs)

    def overlay_plot(self,
                     data1: pd.DataFrame,
                     data2: pd.DataFrame,
                     x: str,
                     y: str or None = None,
                     colour: str = "#db4b6a",
                     alpha: float = .75,
                     size: float = 5,
                     method: str = "scatter",
                     shade: bool = True,
                     plot_kwargs: dict or None = None,
                     overlay_kwargs: dict or None = None):
        """
        Plot data2 overlaid a 2D histogram or 1D KDE plot of data1.

        Parameters
        ----------
        data1: Pandas.DataFrame
        data2: Pandas.DataFrame
        x: str
        y: str, optional
        colour: str (default="#db4b6a")
        alpha: float (default=0.75)
        size: float (default=5.)
        method: str (default="scatter)
        shade: bool (default=True)
        plot_kwargs: dict
        overlay_kwargs: dict

        Returns
        -------
        None
        """
        plot_kwargs = plot_kwargs or {}
        overlay_kwargs = overlay_kwargs or {}
        self._ax = self.plot(data=data1,
                             x=x,
                             y=y,
                             **plot_kwargs)
        if method == "kde":
            self._overlay(data=data2,
                          x=x,
                          y=y,
                          method=method,
                          color=colour,
                          shade=shade,
                          **overlay_kwargs)
            return self._ax
        self._overlay(data=data2,
                      x=x,
                      y=y,
                      method=method,
                      color=colour,
                      size=size,
                      alpha=alpha,
                      **overlay_kwargs)
        return self._ax

    def _set_legend(self,
                    **kwargs):
        """
        Generate a legend for self._ax

        Parameters
        ----------
        kwargs:
            Keyword arguments for Matplotlib.Axes.legend

        Returns
        -------
        None
        """
        anchor = kwargs.get("bbox_to_anchor", (1.1, 0.95))
        loc = kwargs.get("loc", 2)
        ncol = kwargs.get("ncol", 3)
        fancy = kwargs.get("fancybox", True)
        shadow = kwargs.get("shadow", False)
        self._ax.legend(loc=loc,
                        bbox_to_anchor=anchor,
                        ncol=ncol,
                        fancybox=fancy,
                        shadow=shadow)
