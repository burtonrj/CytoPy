from immunova.flow.gating.base import Gate, GateError
from immunova.flow.gating.utilities import rectangular_filter, inside_ellipse


class Static(Gate):
    def __init__(self, **kwargs):
        """
        Gating with static geometric objects
        :param kwargs: Gate constructor arguments (see immunova.flow.gating.base)
        """
        super().__init__(**kwargs)

    def rect_gate(self, x_min: int or float, x_max: int or float, y_min: int or float, y_max: int or float):
        """
        Gate with a static rectangular gate
        :param x_min: left x coordinate
        :param x_max: right x coordinate
        :param y_min: bottom y coordinate
        :param y_max: top y coordinate
        :return: Updated child populations
        """
        if self.y is None:
            raise GateError('For a rectangular filter gate a value for `y` must be given')
        pos_pop = rectangular_filter(self.data, self.x, self.y,
                                     {'xmin': x_min, 'xmax': x_max, 'ymin': y_min, 'ymax': y_max})
        neg_pop = self.data[~self.data.index.isin(pos_pop.index.values)]
        neg = self.child_populations.fetch_by_definition('-')
        pos = self.child_populations.fetch_by_definition('+')
        for x in [pos, neg]:
            self.child_populations.populations[x].update_geom(shape='rect', x=self.x, y=self.y,
                                                              x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        self.child_populations.populations[pos].update_index(idx=pos_pop.index.values, merge_options='overwrite')
        self.child_populations.populations[neg].update_index(idx=neg_pop.index.values, merge_options='overwrite')
        return self.child_populations

    def threshold_2d(self, threshold_x: float, threshold_y: float):
        if self.empty_parent:
            return self.child_populations
        if self.y is None:
            raise GateError('For a 2D threshold gate gate a value for `y` must be given')
        method = f'X: manual threshold, Y: manual threshold'
        self.child_update_2d(threshold_x, threshold_y, method)
        return self.child_populations

    def ellipse_gate(self, centroid: tuple, width: int or float, height: int or float, angle: int or float):
        if self.y is None:
            raise GateError('For a ellipse filter gate a value for `y` must be given')
        pos_mask = inside_ellipse(self.data[[self.x, self.y]].values, centroid, width, height, angle)
        pos_pop = self.data[pos_mask]
        neg_pop = self.data[~self.data.index.isin(pos_pop.index.values)]
        neg = self.child_populations.fetch_by_definition('-')
        pos = self.child_populations.fetch_by_definition('+')
        for x in [pos, neg]:
            self.child_populations.populations[x].update_geom(shape='ellipse', x=self.x, y=self.y,
                                                              centroid=centroid, width=width, height=height,
                                                              angle=angle)
        self.child_populations.populations[pos].update_index(idx=pos_pop.index.values, merge_options='overwrite')
        self.child_populations.populations[neg].update_index(idx=neg_pop.index.values, merge_options='overwrite')
        return self.child_populations

    def border_gate(self, bottom_cutoff: float = 0.01, top_cutoff: float = 0.99):
        if self.y is None:
            raise GateError('For a border filter gate a value for `y` must be given')
        pos_pop = self.data.copy()
        lt_x, lt_y = pos_pop[self.x].quantile(bottom_cutoff), pos_pop[self.y].quantile(bottom_cutoff)
        tt_x, tt_y = pos_pop[self.x].quantile(top_cutoff), pos_pop[self.y].quantile(top_cutoff)

        pos_pop = pos_pop[(pos_pop[self.x] > lt_x) & (pos_pop[self.y] > lt_y)]
        pos_pop = pos_pop[(pos_pop[self.x] < tt_x) & (pos_pop[self.y] < tt_y)]
        neg_pop = self.data[~self.data.index.isin(pos_pop.index.values)]
        neg = self.child_populations.fetch_by_definition('-')
        pos = self.child_populations.fetch_by_definition('+')
        for x in [pos, neg]:
            self.child_populations.populations[x].update_geom(shape='rect', x=self.x, y=self.y,
                                                              x_min=lt_x, x_max=tt_x, y_min=lt_y, y_max=tt_y)
        self.child_populations.populations[pos].update_index(idx=pos_pop.index.values, merge_options='overwrite')
        self.child_populations.populations[neg].update_index(idx=neg_pop.index.values, merge_options='overwrite')
        return self.child_populations