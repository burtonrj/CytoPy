from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.gating.transforms import apply_transform
from sklearn.neighbors import KDTree
from functools import partial
import pandas as pd
import numpy as np


class GateError(Exception):
    pass


class Gate:
    """
    Base class for gate definition.

    Attributes:
        - data: pandas dataframe of fcs data for gating
        - x: name of X dimension
        - y: name of Y dimension (optional)
        - child populations: ChildPopulationCollection (see docs)
        - output: GateOutput object for standard gating output
    """
    def __init__(self, data: pd.DataFrame, x: str, child_populations: ChildPopulationCollection, y: str or None = None,
                 frac: float or None = None, downsample_method: str = 'uniform',
                 density_downsample_kwargs: dict or None = None, transform_x: str or None = 'logicle',
                 transform_y: str or None = 'logicle'):
        """
        Constructor for Gate definition
        :param data: pandas dataframe of fcs data for gating
        :param x: name of X dimension
        :param y: name of Y dimension (optional)
        :param child_populations: ChildPopulationCollection (see docs)
        """
        self.data = data.copy()
        self.x = x
        self.y = y
        if transform_x is not None:
            self.data = apply_transform(self.data, features_to_transform=[self.x], transform_method=transform_x)
        if transform_y is not None and self.y is not None:
            self.data = apply_transform(self.data, features_to_transform=[self.y], transform_method=transform_y)
        self.child_populations = child_populations
        self.warnings = list()
        self.empty_parent = self.__empty_parent()
        self.frac = frac
        self.downsample_method = downsample_method
        self.density_downsample_kwargs = density_downsample_kwargs

    def sampling(self, data, threshold):
        """
        For a given dataset perform down-sampling
        :param data: pandas dataframe of events data to downsample
        :param threshold: threshold below which sampling is not necessary
        :return: down-sampled data
        """
        if self.frac is None:
            return None
        if data.shape[0] < threshold:
            return None
        if self.downsample_method == 'uniform':
            return data.sample(frac=self.frac)
        elif self.downsample_method == 'density':
            try:
                assert self.density_downsample_kwargs
                assert type(self.density_downsample_kwargs) == dict
                features = [self.x]
                if self.y is not None:
                    features.append(self.y)
                return self.density_dependent_downsample(frac=self.frac, features=features,
                                                         **self.density_downsample_kwargs)
            except AssertionError:
                print('If apply density dependent down-sampling then a dictionary of keyword arguments is required'
                      ' as input for density_downsample_kwargs')
        else:
            GateError('Invalid input, downsample_method must be either `uniform` or `density`')

    def __empty_parent(self):
        """
        Test if input data (parent population) is empty. If the population is empty the child population data will
        be finalised and any gating actions terminated.
        :return: True if empty, else False.
        """
        if self.data.shape[0] == 0:
            self.warnings.append('No events in parent population!')
            for name in self.child_populations.populations.keys():
                self.child_populations.populations[name].update_geom(shape=None, x=self.x, y=self.y)
            return True
        return False

    def child_update_1d(self, threshold: float, method: str, merge_options: str) -> None:
        """
        Internal method. Given a threshold and method generated from 1 dimensional threshold gating, update the objects child
        population collection.
        :param threshold: threshold value for gate
        :param method: method used for generating threshold
        :param merge_options: must have value of 'overwrite' or 'merge'. Overwrite: existing index values in child
        populations will be overwritten by the results of the gating algorithm. Merge: index values generated from
        the gating algorithm will be merged with index values currently associated to child populations
        :return: None
        """
        neg = self.child_populations.fetch_by_definition('-')
        pos = self.child_populations.fetch_by_definition('+')
        if neg is None or pos is None:
            GateError('Invalid ChildPopulationCollection; must contain definitions for - and + populations')
        pos_pop = self.data[self.data[self.x] > threshold]
        neg_pop = self.data[self.data[self.x] < threshold]
        for x in [pos, neg]:
            self.child_populations.populations[x].update_geom(shape='threshold', x=self.x, y=self.y,
                                                              method=method)
        self.child_populations.populations[pos].update_index(idx=pos_pop.index.values, merge_options=merge_options)
        self.child_populations.populations[neg].update_index(idx=neg_pop.index.values, merge_options=merge_options)

    def child_update_2d(self, x_threshold: float, y_threshold: float, method: str) -> None:
        """
        Internal method. Given thresholds and method generated from 2 dimensional threshold gating,
        update the objects child population collection.
        :param x_threshold: threshold value for gate in x-dimension
        :param y_threshold: threshold value for gate in y-dimension
        :param method: method used for generating threshold
        :return: None
        """
        xp_idx = self.data[self.data[self.x] > x_threshold].index.values
        yp_idx = self.data[self.data[self.y] > y_threshold].index.values
        xn_idx = self.data[self.data[self.x] < x_threshold].index.values
        yn_idx = self.data[self.data[self.y] < y_threshold].index.values

        negneg = self.child_populations.fetch_by_definition('--')
        pospos = self.child_populations.fetch_by_definition('++')
        posneg = self.child_populations.fetch_by_definition('+-')
        negpos = self.child_populations.fetch_by_definition('-+')

        if any([x is None for x in [negneg, negpos, posneg, pospos]]):
            GateError('Invalid ChildPopulationCollection; must contain definitions for --, -+, +-, and ++ populations')

        pos_idx = np.intersect1d(xn_idx, yn_idx)
        self.child_populations.populations[negneg].update_index(idx=pos_idx, merge_options='merge')
        pos_idx = np.intersect1d(xp_idx, yp_idx)
        self.child_populations.populations[pospos].update_index(idx=pos_idx, merge_options='merge')
        pos_idx = np.intersect1d(xn_idx, yp_idx)
        self.child_populations.populations[negpos].update_index(idx=pos_idx, merge_options='merge')
        pos_idx = np.intersect1d(xp_idx, yn_idx)
        self.child_populations.populations[posneg].update_index(idx=pos_idx, merge_options='merge')

        for x in [negneg, negpos, posneg, pospos]:
            self.child_populations.populations[x].update_geom(shape='2d_threshold', x=self.x,
                                                              y=self.y, method=method, threshold_x=x_threshold,
                                                              threshold_y=y_threshold)

    def uniform_downsample(self, frac: float):
        """
        Sample associated events data
        :param frac: fraction of dataset to return as a sample
        :return: sampled pandas dataframe
        """
        return self.data.sample(frac=frac)

    def density_dependent_downsample(self, features: list, frac: float = 0.1, alpha: int = 5,
                                     mmd_sample_n: int = 2000, outlier_dens: float = 1,
                                     target_dens: float = 5):
        """
        Perform density dependent down-sampling to remove risk of under-sampling rare populations;
        adapted from SPADE*

        * Extracting a cellular hierarchy from high-dimensional cytometry data with SPADE
        Peng Qiu-Erin Simonds-Sean Bendall-Kenneth Gibbs-Robert
        Bruggner-Michael Linderman-Karen Sachs-Garry Nolan-Sylvia Plevritis - Nature Biotechnology - 2011

        :param features:
        :param frac:fraction of dataset to return as a sample
        :param alpha: used for estimating distance threshold between cell and nearest neighbour (default = 5 used in
        original paper)
        :param mmd_sample_n: number of cells to sample for generation of KD tree
        :param outlier_dens: used to exclude cells with the lowest local densities; int value as a percentile of the
        lowest local densities e.g. 1 (the default value) means the bottom 1% of cells with lowest local densities
        are regarded as noise
        :param target_dens: determines how many cells will survive the down-sampling process; int value as a
        percentile of the lowest local densities e.g. 5 (the default value) means the density of bottom 5% of cells
        will serve as the density threshold for rare cell populations
        :return: Down-sampled pandas dataframe
        """

        def prob_downsample(local_d, target_d, outlier_d):
            if local_d <= outlier_d:
                return 0
            if outlier_d < local_d <= target_d:
                return 1
            if local_d > target_d:
                return target_d / local_d

        df = self.data.copy()
        mmd_sample = df.sample(mmd_sample_n)
        tree = KDTree(mmd_sample[features], metric='manhattan')
        dist, _ = tree.query(mmd_sample[features], k=2)
        dist = [x[1] for x in dist]
        dist_threshold = dist * alpha
        ld = tree.query_radius(df[features], r=dist_threshold)
        od = np.percentile(ld, q=outlier_dens)
        td = np.percentile(ld, q=target_dens)
        prob_f = partial(prob_downsample, td=td, od=od)
        prob = list(map(lambda x: prob_f(x), ld))
        return df.sample(frac=frac, weights=prob)



