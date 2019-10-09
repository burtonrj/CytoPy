from immunova.flow.gating.utilities import inside_ellipse, rectangular_filter
from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.gating.base import Gate, GateError
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy import linalg, stats
import pandas as pd
import numpy as np
import math


class MixtureModel(Gate):
    def __init__(self, data: pd.DataFrame, x: str, y: str or None, child_populations: ChildPopulationCollection,
                 frac: float or None = None, downsample_method: str = 'uniform',
                 density_downsample_kwargs: dict or None = None, target: tuple = None,
                 k: int = None, method: str = 'gmm', conf: float = 0.95, rect_filter: dict or None = None,
                 covar: str = 'full'):
        """
        Gating using mixture models
        :param data: pandas dataframe of fcs data for gating
        :param x: name of X dimension
        :param y: name of Y dimension (optional)
        :param child_populations: ChildPopulationCollection (see docs)
        :param frac: fraction of dataset to sample for kde calculation (optional)
        :param downsample_method: method used for down-sampling data (ignored if frac is None)
        :param density_downsample_kwargs: keyword arguments passed to density_dependent_downsampling
        (see flow.gating.base.density_dependent_downsampling) ignored if downsample_method != 'density' or frac is None.
        :param target: centroid of target population (a rough estimate is fine)
        :param k: estimated number of populations in data
        :param method: mixture model method to use; can be either 'gmm' (gaussian) or 'bayesian'
        :param conf: confidence interval for generating elliptical gate (small = tighter gate)
        :param rect_filter: rectangular filter to apply to data prior to gating (optional)
        :param covar: string describing the type of covariance parameters to use (see sklearn documentation for details)
        """
        super().__init__(data=data, x=x, y=y, child_populations=child_populations, frac=frac,
                         downsample_method=downsample_method, density_downsample_kwargs=density_downsample_kwargs)
        self.sample = self.sampling(self.data, 5000)
        self.target = target
        self.k = k
        self.method = method
        self.conf = conf
        self.rect_filter = rect_filter
        self.covar = covar

    def gate(self):
        """
        Apply gate
        :return: Updated child populations
        """
        data = self.data[[self.x, self.y]]

        # Filter if necessary
        if self.rect_filter:
            data = rectangular_filter(data, self.x, self.y, self.rect_filter)

        # Define model
        k = self.k
        if k is None:
            k = 3
        if self.method == 'gmm':
            model = GaussianMixture(n_components=k, covariance_type=self.covar, random_state=42).fit(data)
        elif self.method == 'bayesian':
            model = BayesianGaussianMixture(n_components=k, covariance_type=self.covar, random_state=42).fit(data)
        else:
            raise GateError('Invalid method, must be one of: gmm, bayesian')

        # Select optimal component
        if self.target:
            # Choose component closest to target
            tp_medoid = min(model.means_, key=lambda m: math.hypot(m[0] - self.target[0], m[1] - self.target[1]))
            tp_idx = [list(x) for x in model.means_].index(list(tp_medoid))
            if math.ceil(math.hypot(tp_medoid[0] - self.target[0], tp_medoid[1] - self.target[1])) >= 3:
                self.warnings.append('WARNING: actual population is at least a 3 fold distance from the target. '
                                     'Is this really the population of interest?')
        else:
            # If target isn't specified then select the most populous component
            y_hat = model.predict(data)
            tp_idx = stats.mode(y_hat)[0][0]
        mask, geom = self.create_ellipse(data, model, tp_idx)
        pos_pop = data[mask]
        neg_pop = data[~data.index.isin(pos_pop.index.values)]
        neg = self.child_populations.fetch_by_definition('-')
        pos = self.child_populations.fetch_by_definition('+')
        for x in [pos, neg]:
            self.child_populations.populations[x].update_geom(shape='mixture model', x=self.x, y=self.y, **geom)
        self.child_populations.populations[pos].update_index(idx=pos_pop.index.values, merge_options='overwrite')
        self.child_populations.populations[neg].update_index(idx=neg_pop.index.values, merge_options='overwrite')
        return self.child_populations

    @staticmethod
    def optimise_scaller(tp_medoid: tuple or np.array, eigen_val: float,
                         eigen_vec: np.array, data: pd.DataFrame, channels: list):
        """
        Optimise the scaller value to multiply the eigen vector by determining the scale of the
        confidence ellipse returned by the mixture model gating function - finds the point where
        population size plateau's (see documentation for details)
        :param tp_medoid: target population medoid
        :param eigen_val: eigen value given by the mixture model
        :param eigen_vec: eigen vector given by the mixture model
        :param data: data being modelled
        :param channels: name of columns of interest in data
        :return: optimal scaller and ellipse index mask for resulting scaller
        """
        pop_size = []
        masks = []
        optimal_eigen_val = []
        for scaler in range(1, 11, 1):
            ev = scaler * np.sqrt(eigen_val)
            u = eigen_vec[0] / linalg.norm(eigen_vec[0])
            angle = 180. * np.arctan(u[1] / u[0]) / np.pi
            optimal_eigen_val.append(ev)
            m = inside_ellipse(data[channels].values, tuple(tp_medoid), ev[0], ev[1], 180.+angle)
            masks.append(m)
            pop_size.append(sum(m))
        xx = list(range(1, 11, 1))
        dy = np.diff(pop_size)/np.diff(xx)
        ddy = np.diff(dy)
        return optimal_eigen_val[ddy.argmax()], masks[ddy.argmax()]

    def create_ellipse(self, data, model, tp_idx):
        """
        Given a mixture model (scikit-learn object) and a desired confidence interval, generate mask for events
        that fall inside the 'confidence' ellipse and a 'geom' object
        :param data: parent population upon which the gate has been applied
        :param model: scikit-learn object defining mixture model
        :param tp_idx: index of component that corresponds to positive (target) population
        :return: numpy array 'mask' of events that fall within the confidence ellipse and a parameters of ellipse
        geom
        """
        eigen_val, eigen_vec = linalg.eigh(model.covariances_[tp_idx])
        tp_medoid = tuple(model.means_[tp_idx])
        chi2 = stats.chi2.ppf(self.conf, 2)
        eigen_val = 2. * np.sqrt(eigen_val) * np.sqrt(chi2)
        u = eigen_vec[0] / linalg.norm(eigen_vec[0])
        angle = 180. * np.arctan(u[1] / u[0]) / np.pi
        mask = inside_ellipse(data.values, tp_medoid, eigen_val[0], eigen_val[1], 180. + angle)
        return mask, dict(mean=tp_medoid, width=eigen_val[0], height=eigen_val[1],
                          angle=(180. + angle))

