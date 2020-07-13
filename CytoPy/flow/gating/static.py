from .base import Gate, GateError
from .utilities import rectangular_filter
from ..utilities import inside_ellipse


class Static(Gate):
    """
    Gating with static geometric objects
    
    Parameters
    -----------
    kwargs: 
        Gate constructor arguments (see cytopy.flow.gating.base)
    """
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def rect_gate(self,
                  x_min: int or float,
                  x_max: int or float,
                  y_min: int or float,
                  y_max: int or float):
        """
        Gate with a static rectangular gate
        
        Parameters
        -----------
        x_min: int or float
            left x coordinate
        x_max: int or float
            right x coordinate
        y_min:  int or float
            bottom y coordinate
        y_max:  int or float
            top y coordinate

        Returns
        --------
        ChildPopulationCollection
            Updated child populations
        """
        if self.y is None:
            raise GateError('For a rectangular filter gate a value for `y` must be given')
        pos_pop = rectangular_filter(self.data, self.x, self.y,
                                     {'xmin': x_min, 'xmax': x_max, 'ymin': y_min, 'ymax': y_max})
        neg_pop = self.data[~self.data.index.isin(pos_pop.index.values)]
        neg = self.child_populations.fetch_by_definition('-')
        pos = self.child_populations.fetch_by_definition('+')
        for x, d in zip([pos, neg], ['+', '-']):
            self.child_populations.populations[x].update_geom(shape='rect', x=self.x, y=self.y,
                                                              x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                                                              definition=d, transform_x=self.transform_x,
                                                              transform_y=self.transform_y)
        self.child_populations.populations[pos].update_index(idx=pos_pop.index.values)
        self.child_populations.populations[neg].update_index(idx=neg_pop.index.values)
        return self.child_populations

    def threshold_2d(self,
                     threshold_x: float,
                     threshold_y: float):
        """
        Static two-dimensional threshold gate

        Parameters
        ----------
        threshold_x: float
            Threshold in x-axis plane
        threshold_y: float
            Threshold in y-axis plane

        Returns
        -------
        ChildPopulationCollection
            Updated child populations
        """
        if self.empty_parent:
            return self.child_populations
        if self.y is None:
            raise GateError('For a 2D threshold gate gate a value for `y` must be given')
        method = f'X: manual threshold, Y: manual threshold'
        self.child_update_2d(threshold_x, threshold_y, method)
        return self.child_populations

    def ellipse_gate(self,
                     centroid: tuple,
                     width: int or float,
                     height: int or float,
                     angle: int or float):
        """
        Static elliptical gate

        Parameters
        ----------
        centroid: tuple
            Center of ellipse
        width: int or float
            Width of ellipse
        height: int or float
            Height of ellipse
        angle: int or float
            Angle of ellipse

        Returns
        -------
        ChildPopulationCollection
            Updated child populations
        """
        if self.y is None:
            raise GateError('For a ellipse filter gate a value for `y` must be given')
        pos_mask = inside_ellipse(self.data[[self.x, self.y]].values, centroid, width, height, angle)
        pos_pop = self.data[pos_mask]
        neg_pop = self.data[~self.data.index.isin(pos_pop.index.values)]
        neg = self.child_populations.fetch_by_definition('-')
        pos = self.child_populations.fetch_by_definition('+')
        for x, d in zip([pos, neg], ['+', '-']):
            self.child_populations.populations[x].update_geom(shape='ellipse', x=self.x, y=self.y,
                                                              centroid=centroid, width=width, height=height,
                                                              angle=angle, definition=d, transform_x=self.transform_x,
                                                              transform_y=self.transform_y)
        self.child_populations.populations[pos].update_index(idx=pos_pop.index.values)
        self.child_populations.populations[neg].update_index(idx=neg_pop.index.values)
        return self.child_populations

    def border_gate(self,
                    bottom_cutoff: float = 0.01,
                    top_cutoff: float = 0.99):
        """
        Generates a static boarder gate; a rectangular gate whom's height and width are upper and lower
        quantiles of the underlying data

        Parameters
        ----------
        bottom_cutoff: float, (default=0.01)
            Lower quantile
        top_cutoff: float, (default=0.99)
            Upper quanitle

        Returns
        -------
        ChildPopulationCollection
            Updated child populations
        """
        if self.y is None:
            raise GateError('For a border filter gate a value for `y` must be given')
        pos_pop = self.data.copy()
        lt_x, lt_y = pos_pop[self.x].quantile(bottom_cutoff), pos_pop[self.y].quantile(bottom_cutoff)
        tt_x, tt_y = pos_pop[self.x].quantile(top_cutoff), pos_pop[self.y].quantile(top_cutoff)
        return self.rect_gate(x_min=lt_x, x_max=tt_x, y_min=lt_y, y_max=tt_y)