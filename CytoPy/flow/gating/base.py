from .defaults import ChildPopulationCollection
from ..transforms import apply_transform
from ..utilities import density_dependent_downsample
from shapely.geometry.polygon import Polygon
from scipy.spatial import ConvexHull
from functools import partial
import pandas as pd
import numpy as np


class GateError(Exception):
    pass


class Gate:
    """
    Base class for gate definition.

    Parameters
    ----------
    data: Pandas.DataFrame
        FCS data for gating
    x: str
        name of X dimension
    y: str, optional
        name of Y dimension (optional)
    child populations: ChildPopulationCollection
        Collection of populations that gate generates
    frac: float, optional
        Fraction of data to sample prior to gating (if None, sampling not performed)
    downsample_method: str, (default='uniform')
        If frac is not None, method used for downsampling (must be either 'uniform' or 'density')
    density_downsample_kwargs: dict, optional
        Keyword arguments to pass to flow.utilitis.density_dependent_downsample if downsample_method is 'density'
    transform_x: str or None, (default='logicle')
        Method used to transform x-axis
    transform_y: str or None, (default='logicle')
        Method used to transform y-axis
    low_memory: bool, (default=False)
        If True, frac is adjusted according to the size of the DataFrame
    """
    def __init__(self,
                 data: pd.DataFrame,
                 x: str,
                 child_populations: ChildPopulationCollection,
                 y: str or None = None,
                 frac: float or None = None,
                 downsample_method: str = 'uniform',
                 density_downsample_kwargs: dict or None = None,
                 transform_x: str or None = 'logicle',
                 transform_y: str or None = 'logicle',
                 low_memory: bool = False):
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
        self.empty_parent = self._empty_parent()
        self.frac = frac
        if low_memory:
            if self.data.shape[0] > 20000:
                self.frac = 20000/self.data.shape[0]
        self.downsample_method = downsample_method
        if self.downsample_method == 'density':
            if density_downsample_kwargs is not None:
                if type(density_downsample_kwargs) != dict:
                    raise GateError('If applying density dependent down-sampling then a dictionary of '
                                    'keyword arguments is required as input for density_downsample_kwargs')
        self.density_downsample_kwargs = density_downsample_kwargs

    def sampling(self,
                 data: pd.DataFrame,
                 threshold: float) -> pd.DataFrame or None:
        """
        For a given dataset perform down-sampling

        Parameters
        -----------
        data: Pandas.DataFrame
            Events data to downsample
        threshold: float
            threshold below which sampling is not necessary

        Returns
        --------
        Pandas.DataFrame or None
            Down-sampled data
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
                return density_dependent_downsample(data=data, frac=self.frac, features=features,
                                                    **self.density_downsample_kwargs)
            return density_dependent_downsample(data=data, frac=self.frac, features=features)
        else:
            GateError('Invalid input, down-sample_method must be either `uniform` or `density`')

    def _empty_parent(self):
        """
        Test if input data (parent population) is empty. If the population is empty the child population data will
        be finalised and any gating actions terminated.

        Returns
        --------
        bool
            True if empty, else False.
        """
        if self.data.shape[0] == 0:
            raise ValueError('No events in parent population!')

    def child_update_1d(self, threshold: float,
                        method: str) -> None:
        """
        Internal method. Given a threshold and method generated from 1 dimensional threshold gating, update the objects child
        population collection.

        Parameters
        -----------
        threshold: float
            threshold value for gate
        method: str
            method used for generating threshold

        Returns
        --------
        None
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
        self.child_populations.populations[pos].update_index(idx=pos_pop.index.values)
        self.child_populations.populations[neg].update_index(idx=neg_pop.index.values)

    def child_update_2d(self,
                        x_threshold: float,
                        y_threshold: float,
                        method: str) -> None:
        """
        Internal method. Given thresholds and method generated from 2 dimensional threshold gating,
        update the objects child population collection.

        Parameters
        -----------
        x_threshold: float
            threshold value for gate in x-dimension
        y_threshold: float
            threshold value for gate in y-dimension
        method: str
            method used for generating threshold

        Returns
        --------
        None
        """
        # Validate 2D threshold gate
        populations = [self.child_populations.fetch_by_definition(d) for d in ['--', '++', '+-', '-+']]
        if any([x is None for x in populations]):
            GateError('Invalid ChildPopulationCollection; must contain definitions for --, -+, +-, and ++ populations')

        # Get index according to threshold
        xp_idx = self.data[self.data[self.x].round(decimals=2) >= round(x_threshold, 2)].index.values
        yp_idx = self.data[self.data[self.y].round(decimals=2) >= round(y_threshold, 2)].index.values
        xn_idx = self.data[self.data[self.x].round(decimals=2) < round(x_threshold, 2)].index.values
        yn_idx = self.data[self.data[self.y].round(decimals=2) < round(y_threshold, 2)].index.values

        negneg_idx = np.intersect1d(xn_idx, yn_idx)
        pospos_idx = np.intersect1d(xp_idx, yp_idx)
        posneg_idx = np.intersect1d(xp_idx, yn_idx)
        negpos_idx = np.intersect1d(xn_idx, yp_idx)

        # Create a list of population definitions, with keys for the definition, population it is associated to
        # and the relative index
        populations_idx = list()
        i = zip(populations, [negneg_idx, pospos_idx, posneg_idx, negpos_idx], ['--', '++', '+-', '-+'])
        for pop_name, index, d in i:
            populations_idx.append(dict(pop_name=pop_name, index=index, definition=d))

        # Now merge the dictionaries on the population name
        merged_poulation_idx = dict()
        for pop in set(d.get('pop_name') for d in populations_idx):
            matching_populations = [d for d in populations_idx if d.get('pop_name') == pop]
            definition = [d.get('definition') for d in matching_populations]
            index = np.unique(np.concatenate([d.get('index') for d in matching_populations]))
            merged_poulation_idx[pop] = dict(definition=definition, index=index)

        # Update index and geom for child populations
        for pop_name in merged_poulation_idx.keys():
            self.child_populations.populations[pop_name].update_index(idx=merged_poulation_idx[pop_name].get('index'))
            definition = merged_poulation_idx[pop_name].get('definition')
            if len(definition) == 1:
                definition = definition[0]
            self.child_populations.populations[pop_name].update_geom(shape='2d_threshold',
                                                                     x=self.x,
                                                                     y=self.y,
                                                                     method=method,
                                                                     threshold_x=float(x_threshold),
                                                                     threshold_y=float(y_threshold),
                                                                     definition=definition,
                                                                     transform_x=self.transform_x,
                                                                     transform_y=self.transform_y)

    def uniform_downsample(self,
                           sample_size: int or float = 0.1,
                           data: pd.DataFrame or None = None) -> pd.DataFrame:
        """
        Sample associated events data

        Parameters
        -----------
        sample_size: int or float, (default=0.1)
            fraction or number of events to sample from dataset
        data: Pandas.DataFrame, optional
            Optional, if given overrides call to self.data

        Returns
        --------
        Pandas.DataFrame
            sampled pandas dataframe
        """
        if data is None:
            data = self.data
        if type(sample_size) is int:
            return data.sample(n=sample_size)
        return data.sample(frac=sample_size)

    def generate_chunks(self,
                        chunksize: int) -> list:
        """
        Generate a list of dataframes (chunks) from original data of a target chunksize

        Parameters
        -----------
        chunksize: int
            target size of chunks (might be smaller or larger than intended value depending on the size
            of the data)

        Returns
        --------
        List
            List of pandas dataframes, one for each chunk
        """
        chunks = list()
        d = np.ceil(self.data.shape[0] / chunksize)
        chunksize = int(np.ceil(self.data.shape[0] / d))

        if self.downsample_method == 'uniform':
            sampling_func = partial(self.uniform_downsample, sample_size=chunksize)
        else:
            if self.density_downsample_kwargs is not None:
                kwargs = dict(sample_n=chunksize, features=[self.x, self.y], **self.density_downsample_kwargs)
                sampling_func = partial(density_dependent_downsample, **kwargs)
            else:
                sampling_func = partial(density_dependent_downsample, sample_n=chunksize,
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

    def generate_polygons(self,
                          data: pd.DataFrame or None = None) -> dict:
        """
        Generate a dictionary of polygon coordinates and shapely Polygon from clustered data
        objects

        Parameters
        -----------
        data: Panda.DataFrame, optional
            Optional, if DataFrame given, overwrites call to self.data

        Returns
        --------
        dict
            Dictionary of polygon coordinates and dictionary of Polygon shapely objects
        """
        if data is None:
            df = self.data.copy()
        else:
            df = data.copy()
        if 'labels' not in df.columns:
            GateError('Method self.__generate_polygons called before cluster assignment')
        polygon_cords = {label: [] for label in df['labels'].unique() if label != -1}
        for label in polygon_cords.keys():
            d = df[df['labels'] == label][[self.x, self.y]].values
            hull = ConvexHull(d)
            polygon_cords[label] = [(d[v, 0], d[v, 1]) for v in hull.vertices]
        polygon_shapes = {label: Polygon(x) for label, x in polygon_cords.items()}
        return polygon_shapes


