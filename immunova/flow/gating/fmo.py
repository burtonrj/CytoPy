from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.gating.density import DensityThreshold, GateError
from scipy.stats import norm
import pandas as pd


class FMOGate(DensityThreshold):
    def __init__(self, fmo_x: pd.DataFrame, fmo_y: pd.DataFrame or None = None, fmo_z: pd.DataFrame or None = None,
                 z_score_threshold: float = 2, **kwargs):
        """
        FMO guided density threshold gating
        :param fmo_x: pandas dataframe of fcs data for x-dimensional FMO
        :param fmo_y: pandas dataframe of fcs data for y-dimensional FMO (optional)
        :param z_score_threshold: when multiple populations are identified in the whole panel sample the FMO gate
        is used a guide for gating. A normal distribution is fitted to the data, with the mean set as the threshold
        calculated on the whole panel sample and an std of 1. Using this distribution a z-score is calculated for the
        FMO threshold. If the z score exceeds z_score_threshold a warning is logged and the fmo is ignored.
        :param kwargs: DensityThreshold constructor arguments (see immunova.gating.density)
        """
        super().__init__(**kwargs)
        self.z_score_t = z_score_threshold
        self.fmo_x = fmo_x.copy()
        self.sample = self.sampling(self.data, 5000)
        self.sample_fmo_x = self.sampling(self.fmo_x, 5000)
        if fmo_y is not None:
            self.fmo_y = fmo_y.copy()
            self.sample_fmo_y = self.sampling(self.fmo_y, 5000)
        else:
            self.fmo_y = None
            self.sample_fmo_y = None

    def fmo_1d(self, merge_options: str = 'overwrite') -> ChildPopulationCollection:
        """
        Perform FMO gating in 1 dimensional space
        :param merge_options: must have value of 'overwrite' or 'merge'. Overwrite: existing index values in child
        populations will be overwritten by the results of the gating algorithm. Merge: index values generated from
        the gating algorithm will be merged with index values currently associated to child populations
        :return: Updated child population collection
        """
        if self.empty_parent:
            return self.child_populations

        # Calculate threshold
        if self.sample is not None:
            data = self.sample
        else:
            data = self.data
        if self.sample_fmo_x is not None:
            fmo = self.sample_fmo_x
        else:
            fmo = self.fmo_x
        threshold, method = self.__1d(data, fmo, self.x)

        self.child_update_1d(threshold, method, merge_options)
        return self.child_populations

    def __1d(self, whole: pd.DataFrame, fmo: pd.DataFrame, feature: str) -> float and str:
        """
        Internal method. Calculate FMO guided threshold gate in 1 dimensional space.
        :param whole: pandas dataframe for events data in whole panel sample
        :param fmo: pandas dataframe for events data in fmo sample
        :param feature: name of the feature to perform gating on
        :return: threshold, method used to obtain threshold
        """
        if fmo.shape[0] == 0:
            raise GateError('No events in parent population in FMO!')
            # Calculate threshold for whole panel (primary sample)
        whole_threshold, whole_method = self.__calc_threshold(whole, feature)
        fmo_threshold, fmo_method = self.__calc_threshold(fmo, feature)
        if whole_method in ['Quantile', 'Standard deviation']:
            return fmo_threshold, fmo_method
        elif whole_method == 'Local minima between pair of highest peaks':
            p = norm.cdf(x=fmo_threshold, loc=whole_threshold, scale=0.1)
            z_score = norm.ppf(p)
            if abs(z_score) >= self.z_score_t:
                self.warnings.append("""FMO threshold z-score >2 (see documentation); the threshold 
                as determined by the FMO is a significant distance from the region of minimum density between the 
                two highest peaks see in the whole panel, therefore the FMO has been ignored. 
                Manual review of gating is advised.""")
                return whole_threshold, whole_method
            else:
                # Take an average of fmo and whole panel threshold
                threshold = (whole_threshold + fmo_threshold)/2
                return threshold, 'FMO guided minimum density threshold'
        else:
            GateError('Unrecognised method returned from __calc_threshold')

    def fmo_2d(self) -> ChildPopulationCollection:
        """
        Perform FMO gating in 2-dimensional space
        :return: Updated child populations
        """
        # If parent is empty just return the child populations with empty index array
        if self.empty_parent:
            return self.child_populations
        if not self.y:
            raise GateError('For a 2D threshold gate a value for `y` is required')
        if self.fmo_y is None:
            raise GateError('For a 2D threshold gate a value for `fmo_y` is required')

        # Calculate threshold
        if self.sample is not None:
            data = self.sample
        else:
            data = self.data

        if self.sample_fmo_x is not None:
            fmo = self.sample_fmo_x
        else:
            fmo = self.fmo_x
        x_threshold, x_method = self.__1d(data, fmo, self.x)

        if self.sample_fmo_y is not None:
            fmo = self.sample_fmo_y
        else:
            fmo = self.fmo_y
        y_threshold, y_method = self.__1d(data, fmo, self.y)
        method = f'X: {x_method}, Y: {y_method}'
        self.child_update_2d(x_threshold, y_threshold, method)
        return self.child_populations
