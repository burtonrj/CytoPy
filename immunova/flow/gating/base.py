from immunova.flow.gating.defaults import ChildPopulationCollection
from sklearn.neighbors import KDTree
from functools import partial
import pandas as pd
import numpy as np


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
    def __init__(self, data: pd.DataFrame, x: str, y: str or None, child_populations: ChildPopulationCollection):
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
        self.child_populations = child_populations
        self.error = False
        self.error_msg = None
        self.warnings = list()
        self.empty_parent = self.__empty_parent()

    def __empty_parent(self):
        """
        Test if input data (parent population) is empty. If the population is empty the child population data will
        be finalised and any gating actions terminated.
        :return: True if empty, else False.
        """
        if self.data.shape[0] == 0:
            self.warnings.append('No events in parent population!')
            for name in self.child_populations.populations.keys():
                self.child_populations.populations[name].update_geom(shape='cluster', x=self.x, y=self.y)
            return True
        return False

    def unifrom_downsample(self, frac: float):
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



