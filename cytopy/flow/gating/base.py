from cytopy.flow.gating.defaults import ChildPopulationCollection
from cytopy.flow.transforms import apply_transform
from shapely.geometry.polygon import Polygon
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree
from functools import partial
from collections import defaultdict
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
                 transform_y: str or None = 'logicle', low_memory: bool = False):
        """
        Constructor for Gate definition
        :param data: pandas dataframe of fcs data for gating
        :param x: name of X dimension
        :param y: name of Y dimension (optional)
        :param child_populations: ChildPopulationCollection (see docs)one
        """
        self.data = data.copy()
        self.x = x
        self.y = y
        self.transform_x = transform_x
        self.transform_y = transform_y
        if transform_x is not None:
            self.data = apply_transform(self.data, features_to_transform=[self.x], transform_method=transform_x)
        if transform_y is not None and self.y is not None:
            self.data = apply_transform(self.data, features_to_transform=[self.y], transform_method=transform_y)
        self.child_populations = child_populations
        self.warnings = list()
        self.empty_parent = self.__empty_parent()
        if low_memory:
            if self.data.shape[0] > 20000:
                self.frac = 20000/self.data.shape[0]
            else:
                self.frac = None
        else:
            self.frac = frac
        self.downsample_method = downsample_method
        if self.downsample_method == 'density':
            if density_downsample_kwargs is not None:
                if type(density_downsample_kwargs) != dict:
                    raise GateError('If applying density dependent down-sampling then a dictionary of '
                                    'keyword arguments is required as input for density_downsample_kwargs')
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
            return data
        if self.downsample_method == 'uniform':
            try:
                return data.sample(frac=self.frac)
            except ValueError:
                return data
        elif self.downsample_method == 'density':
            features = [self.x]
            if self.y is not None:
                features.append(self.y)
            if self.density_downsample_kwargs is not None:
                if not type(self.density_downsample_kwargs) == dict:
                    raise GateError('If applying density dependent down-sampling then a dictionary of '
                                    'keyword arguments is required as input for density_downsample_kwargs')
                return self.density_dependent_downsample(frac=self.frac, features=features,
                                                         **self.density_downsample_kwargs)
            return self.density_dependent_downsample(frac=self.frac, features=features)
        else:
            GateError('Invalid input, downsample_method must be either `uniform` or `density`')

    def __empty_parent(self):
        """
        Test if input data (parent population) is empty. If the population is empty the child population data will
        be finalised and any gating actions terminated.
        :return: True if empty, else False.
        """
        if self.data.shape[0] == 0:
            raise GateError('No events in parent population!')

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
        pos_pop = self.data[self.data[self.x].round(decimals=2) >= round(threshold, 2)]
        neg_pop = self.data[self.data[self.x].round(decimals=2) < round(threshold, 2)]
        for x, definition in zip([pos, neg], ['+', '-']):
            self.child_populations.populations[x].update_geom(shape='threshold', x=self.x, y=self.y,
                                                              method=method, threshold=float(threshold),
                                                              definition=definition, transform_x=self.transform_x)
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
        xp_idx = self.data[self.data[self.x].round(decimals=2) >= round(x_threshold, 2)].index.values
        yp_idx = self.data[self.data[self.y].round(decimals=2) >= round(y_threshold, 2)].index.values
        xn_idx = self.data[self.data[self.x].round(decimals=2) < round(x_threshold, 2)].index.values
        yn_idx = self.data[self.data[self.y].round(decimals=2) < round(y_threshold, 2)].index.values

        negneg = self.child_populations.fetch_by_definition('--')
        pospos = self.child_populations.fetch_by_definition('++')
        posneg = self.child_populations.fetch_by_definition('+-')
        negpos = self.child_populations.fetch_by_definition('-+')
        definitions = defaultdict(list)
        for name, definition in zip([negneg, negpos, posneg, pospos], ['--', '-+', '+-', '++']):
            definitions[name].append(definition)

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

        for name, definition in definitions.items():
            if len(definition) == 1:
                definition = definition[0]
            self.child_populations.populations[name].update_geom(shape='2d_threshold', x=self.x,
                                                                 y=self.y, method=method, threshold_x=float(x_threshold),
                                                                 threshold_y=float(y_threshold), definition=definition,
                                                                 transform_x=self.transform_x, transform_y=self.transform_y)

    def uniform_downsample(self, frac: float or None, sample_n: int or None = None, data: pd.DataFrame or None = None):
        """
        Sample associated events data
        :param frac: fraction of dataset to return as a sample
        :return: sampled pandas dataframe
        """
        if data is not None:
            if sample_n is not None:
                return data.sample(n=sample_n)
            return data.sample(frac=frac)
        if sample_n is not None:
            return self.data.sample(n=sample_n)
        return self.data.sample(frac=frac)

    # ToDo move to utilities
    def density_dependent_downsample(self, features: list, frac: float = 0.1, sample_n: int or None = None,
                                     data: pd.DataFrame or None = None, alpha: int = 5, mmd_sample_n: int = 2000,
                                     outlier_dens: float = 1, target_dens: float = 5):
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

        if data is not None:
            df = data.copy()
        else:
            df = self.data.copy()
        mmd_sample = df.sample(mmd_sample_n)
        tree = KDTree(mmd_sample[features], metric='manhattan')
        dist, _ = tree.query(mmd_sample[features], k=2)
        dist = np.median([x[1] for x in dist])
        dist_threshold = dist * alpha
        ld = tree.query_radius(df[features], r=dist_threshold, count_only=True)
        od = np.percentile(ld, q=outlier_dens)
        td = np.percentile(ld, q=target_dens)
        prob_f = partial(prob_downsample, target_d=td, outlier_d=od)
        prob = list(map(lambda x: prob_f(x), ld))
        if sum(prob) == 0:
            print('Error: density dependendent downsampling failed; weights sum to zero. Defaulting to uniform '
                  'samplings')
            if sample_n is not None:
                return df.sample(n=sample_n)
            return df.sample(frac=frac)
        if sample_n is not None:
            return df.sample(n=sample_n, weights=prob)
        return df.sample(frac=frac, weights=prob)

    def generate_chunks(self, chunksize):
        """
        Generate a list of dataframes (chunks) from original data of a target chunksize
        :param chunksize: target size of chunks (might be smaller or larger than intended value depending on the size
        of the data)
        :return: list of pandas dataframes, one for each chunk
        """
        chunks = list()
        d = np.ceil(self.data.shape[0] / chunksize)
        chunksize = int(np.ceil(self.data.shape[0] / d))

        if self.downsample_method == 'uniform':
            sampling_func = partial(self.uniform_downsample, frac=None, sample_n=chunksize)
        else:
            if self.density_downsample_kwargs is not None:
                kwargs = dict(sample_n=chunksize, features=[self.x, self.y], **self.density_downsample_kwargs)
                sampling_func = partial(self.density_dependent_downsample, **kwargs)
            else:
                sampling_func = partial(self.density_dependent_downsample, sample_n=chunksize,
                                        features=[self.x, self.y])
        data = self.data.copy()
        for x in range(0, int(d)):
            if data.shape[0] <= chunksize:
                data['chunk_idx'] = x
                chunks.append(data)
                break
            sample = sampling_func(data=data)
            sample['chunk_idx'] = x
            data = data[~data.index.isin(sample.index)]
            chunks.append(sample)
        return chunks

    def generate_polygons(self, data: pd.DataFrame or None = None) -> dict:
        """
        Generate a dictionary of polygon coordinates and shapely Polygon from clustered data
        objects
        :return: dictionary of polygon coordinates and dictionary of Polygon shapely objects
        """
        if data is None:
            df = self.data.copy()
        else:
            df = data.copy()
        if 'labels' not in df.columns:
            GateError('Method self.__generate_polygons called before cluster assignment')
        polygon_cords = {label: [] for label in df['labels'].unique() if label != -1}
        for label in polygon_cords.keys():
            if label == -1:
                continue
            d = df[df['labels'] == label][[self.x, self.y]].values
            hull = ConvexHull(d)
            polygon_cords[label] = [(d[v, 0], d[v, 1]) for v in hull.vertices]
        polygon_shapes = {label: Polygon(x) for label, x in polygon_cords.items()}
        return polygon_shapes


