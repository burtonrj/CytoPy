from ..data.gates import Gate
from ..data.populations import PopulationGeometry, Population
from ..flow.transforms import apply_transform
from warnings import warn
from typing import List
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches
from itertools import cycle
import seaborn as sns
import pandas as pd
import numpy as np


class CreatePlot:
    """
    Generate 1D or 2d histograms of cell populations as identified by cytometry. Supports plotting of individual
    populations, single or multiple gates, "backgating" (plotting child populations overlaid on parent) and
    overlaying populations from control samples on their equivalent in the primary sample. All plotting is performed
    by accessing a Gating object.

    Parameters
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
    font_scale: float, optional (default=1.2)
        Font scale (passed to seaborn.set_context)
    bw: str or float, (default="scott")
        Bandwidth for 1D KDE (see seaborn.kdeplot)
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
        self.tranforms = {'x': transform_x, 'y': transform_y}
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
        Generate a 1D KDE plot

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
        sns.kdeplot(data=data[x], bw=self.bw, ax=self._ax, **kwargs)

    def _hist2d(self,
                data: pd.DataFrame,
                x: str,
                y: str,
                **kwargs) -> None:
        """
        Generate a 2D histogram

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
        Set the axis limits

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
        Set plot aesthetics: title and axis labels

        Parameters
        ----------
        x: str
            X-axis channel (for default x-axis label)
        y: str
            Y-axis channel (for default y-axis label)

        Returns
        -------

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
        transforms = {x: self.tranforms.get(axis) for axis in ["x", "y"]
                      if self.tranforms.get(axis) is not None}
        if len(transforms) > 0:
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

    def plot_population_geom(self,
                             parent: pd.DataFrame,
                             geom: PopulationGeometry,
                             lw: float = 2.5,
                             line_colour: str = "#c92c2c",
                             population_name: str or None = None,
                             plot_kwargs: dict or None = None,
                             legend_kwargs: dict or None = None):
        plot_kwargs = plot_kwargs or {}
        population_name = population_name or ""
        self.tranforms = {"x": geom.transform_x,
                          "y": geom.transform_y}
        self._ax = self.plot(data=parent,
                             x=geom.x,
                             y=geom.y,
                             **plot_kwargs)
        if geom.x_threshold:
            x = geom.x_threshold
            y = geom.y_threshold
            self._add_threshold(x=x,
                                y=y,
                                lw=lw)
            return self._ax
        if geom.x_values is not None and geom.y_values is not None:
            self._add_polygon(x_values=geom.x_values,
                              y_values=geom.y_values,
                              colour=line_colour,
                              label=population_name,
                              lw=lw)
        elif all(geom[x] is not None for x in ["center", "width", "height", "angle"]):
            self._add_ellipse(center=geom.center,
                              width=geom.width,
                              height=geom.height,
                              angle=geom.angle,
                              colour=line_colour,
                              lw=lw,
                              label=population_name)
            raise ValueError(f"Geom shape is invalid")
        self._set_legend(shape_n=1, **legend_kwargs)
        return self._ax

    def plot_gate(self,
                  gate: Gate,
                  parent: pd.DataFrame,
                  children: List[Population],
                  lw: float = 2.5,
                  plot_kwargs: dict or None = None,
                  legend_kwargs: dict or None = None):
        """
        Plot a Gate object. This will plot the geometric shapes generated from a single Gate, overlaid on the
        parent population upon which the Gate has been applied in the context of the current Gating object and
        it's associated cytometry data.

        Parameters
        ----------
        children
        parent
        gate
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
        assert len(children) == len(gate.children), f"Number of given geometries does not match expected " \
                                                    f"number of children from gate {gate.gate_name}"
        for child in gate.children:
            err = f"{child.population_name} missing from given geometries"
            assert child.population_name in [c.population_name for c in children], err
        plot_kwargs = plot_kwargs or {}
        legend_kwargs = legend_kwargs or dict()
        # Plot the parent population
        self.tranforms = {"x": gate.preprocessing.transform_x or "linear",
                          "y": gate.preprocessing.transform_y or "linear"}
        self._ax = self.plot(data=parent,
                             x=gate.x,
                             y=gate.y,
                             **plot_kwargs)
        # If threshold, add threshold lines to plot and return axes
        if gate.shape == "threshold":
            x = children[0].geom.x_threshold
            y = children[0].geom.y_threshold
            self._add_threshold(x=x,
                                y=y,
                                labels={c.definition: c.population_name for c in children},
                                lw=lw)
            return self._ax
        # Otherwise, we assume some other shape
        count = 0
        for child in children:
            count += 1
            colour = next(gate_colours)
            if gate.shape == "polygon":
                x_values, y_values = child.geom.shape.exterior.xy
                self._add_polygon(x_values=x_values,
                                  y_values=y_values,
                                  colour=colour,
                                  label=child.population_name,
                                  lw=lw)
            elif gate.shape == "ellipse":
                self._add_ellipse(center=child.geom.center,
                                  width=child.geom.width,
                                  height=child.geom.height,
                                  angle=child.geom.angle,
                                  colour=colour,
                                  lw=lw,
                                  label=child.population_name)
            else:
                raise ValueError(f"Gate shape not recognised: {gate.shape}")
            if len(gate.children) == 2 and gate.binary:
                # Gate produces exactly two populations: within gate and outside of gate
                break
        self._set_legend(shape_n=count, **legend_kwargs)
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
                    label = [v for l, v in labels.items() if d in l][0]
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
                 children: dict,
                 x: str,
                 y: str or None = None,
                 colour: str = "#db4b6a",
                 alpha: float = .75,
                 size: float = 5,
                 method: str = "scatter",
                 shade: bool = True,
                 plot_kwargs: dict or None = None,
                 overlay_kwargs: dict or None = None,
                 **legend_kwargs):
        if plot_kwargs is None:
            plot_kwargs = {}
        if overlay_kwargs is None:
            overlay_kwargs = {}
        self._ax = self.plot(data=parent,
                             x=x,
                             y=y,
                             **plot_kwargs)
        for child_name, df in children.items():
            if method == "kde":
                self._overlay(data=df,
                              x=x,
                              y=y,
                              method=method,
                              color=colour,
                              shade=shade,
                              **overlay_kwargs)
            else:
                self._overlay(data=df,
                              x=x,
                              y=y,
                              method=method,
                              color=colour,
                              size=size,
                              alpha=alpha,
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
        assert method in ["scatter", "kde"], "Overlay method should be 'scatter' or 'kde'"
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

    def overlay_control(self,
                        data: pd.DataFrame,
                        ctrl: pd.DataFrame,
                        x: str,
                        y: str or None = None,
                        colour: str = "#db4b6a",
                        alpha: float = .75,
                        size: float = 5,
                        method: str = "scatter",
                        shade: bool = True,
                        plot_kwargs: dict or None = None,
                        overlay_kwargs: dict or None = None):

        if plot_kwargs is None:
            plot_kwargs = {}
        if overlay_kwargs is None:
            overlay_kwargs = {}
        self._ax = self.plot(data=data,
                             x=x,
                             y=y,
                             **plot_kwargs)
        if method == "kde":
            self._overlay(data=ctrl,
                          x=x,
                          y=y,
                          method=method,
                          color=colour,
                          shade=shade,
                          **overlay_kwargs)
            return self._ax
        self._overlay(data=ctrl,
                      x=x,
                      y=y,
                      method=method,
                      color=colour,
                      size=size,
                      alpha=alpha,
                      **overlay_kwargs)
        return self._ax

    def _set_legend(self,
                    shape_n: int,
                    **kwargs):
        anchor = kwargs.get("bbox_to_anchor", (1.1, 0.95))
        loc = kwargs.get("loc", 2)
        ncol = kwargs.get("ncol", 3)
        fancy = kwargs.get("fancybox", True)
        shadow = kwargs.get("shadow", False)
        if shape_n == 1:
            if self._ax.get_legend() is not None:
                self._ax.get_legend().remove()
        else:
            self._ax.legend(loc=loc,
                            bbox_to_anchor=anchor,
                            ncol=ncol,
                            fancybox=fancy,
                            shadow=shadow)
