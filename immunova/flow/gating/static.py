from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.gating.base import Gate, GateError
from immunova.flow.gating.utilities import rectangular_filter, inside_ellipse
import pandas as pd


class Static(Gate):
    def __init__(self, data: pd.DataFrame, x: str, child_populations: ChildPopulationCollection,
                 y: str or None = None):
        """
        Gating with static geometric objects
        :param data: pandas dataframe of fcs data for gating
        :param x: name of X dimension
        :param y: name of Y dimension (optional)
        :param child_populations: ChildPopulationCollection (see flow.gating.defaults.ChildPopulationCollection)
        """
        super().__init__(data=data, x=x, y=y, child_populations=child_populations,
                         frac=None, downsample_method='uniform',
                         density_downsample_kwargs=None)
        self.y = y

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
            self.child_populations.populations[x].update_geom(shape='geom', x=self.x, y=self.y,
                                                              x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
        self.child_populations.populations[pos].update_index(idx=pos_pop.index.values, merge_options='overwrite')
        self.child_populations.populations[neg].update_index(idx=neg_pop.index.values, merge_options='overwrite')
        return self.child_populations

    def ellipse_gate(self, centroid: tuple, width: int or float, height: int or float, angle: int or float):
        if self.y is None:
            raise GateError('For a rectangular filter gate a value for `y` must be given')
        pos_mask = inside_ellipse(self.data[self.x, self.y].values, centroid, width, height, angle)
        pos_pop = self.data[pos_mask]
        neg_pop = self.data[~self.data.index.isin(pos_pop.index.values)]
        neg = self.child_populations.fetch_by_definition('-')
        pos = self.child_populations.fetch_by_definition('+')
        for x in [pos, neg]:
            self.child_populations.populations[x].update_geom(shape='geom', x=self.x, y=self.y,
                                                              centroid=centroid, width=width, height=height,
                                                              angle=angle)
        self.child_populations.populations[pos].update_index(idx=pos_pop.index.values, merge_options='overwrite')
        self.child_populations.populations[neg].update_index(idx=neg_pop.index.values, merge_options='overwrite')
        return self.child_populations

