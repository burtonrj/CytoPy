from .utilities import inside_ellipse, rectangular_filter
from .base import Gate, GateError
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy import linalg, stats
import pandas as pd
import numpy as np
import math


class MixtureModel(Gate):
    """
    Gating using Scikit-Learn implementations of mixture models (https://scikit-learn.org/stable/modules/mixture.html).
    Algorithm fits one or more components to underlying distributions, selecting the component most inline with
    the expected target population. An elliptical gate is generated that mirrors the probability surface of the fitted 
    model.
    
    Parameters
    ----------- 
    target: tuple, optional
        centroid of target population (a rough estimate is fine). If None, the component that describing the most
        events will be chosen (i.e. the larget population will be captured by the elliptical gate)
    k: int
        estimated number of populations in data (i.e. the number of components)
    method: str, (default='gmm')
        mixture model method to use; can be either 'gmm' (gaussian) or 'bayesian'
    conf: float, (default=0.95)
        confidence interval for generating elliptical gate (small = tighter gate)
    rect_filter: dict, optional
        rectangular filter to apply to data prior to gating (see flow.gating.utilities.rectangular_filter)
    covar: str, (default='full')
        string describing the type of covariance parameters to use (see sklearn documentation for details)
    kwargs:
        Gate constructor arguments (see flow.gating.base)
    """
    def __init__(self,
                 target: tuple = None,
                 k: int = None,
                 algo: str = 'gmm',
                 conf: float = 0.95,
                 rect_filter: dict or None = None,
                 covar: str = 'full',
                 **kwargs):
        super().__init__(**kwargs)
        self.sample = self.sampling(self.data, 5000)
        self.target = target
        self.k = k
        self.method = algo
        self.conf = conf
        self.rect_filter = rect_filter
        self.covar = covar

    def gate(self):
        """
        Calculate mixture model, generating gate and resulting populations.

        Returns
        -------
        ChildPopulationCollection
            Updated child population collection
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
        else:
            # If target isn't specified then select the most populous component
            y_hat = model.predict(data)
            tp_idx = stats.mode(y_hat)[0][0]
        mask, geom = self.create_ellipse(data, model, tp_idx)
        pos_pop = data[mask]
        neg_pop = data[~data.index.isin(pos_pop.index.values)]
        neg = self.child_populations.fetch_by_definition('-')
        pos = self.child_populations.fetch_by_definition('+')
        for x, definition in zip([pos, neg], ['+', '-']):
            self.child_populations.populations[x].update_geom(shape='ellipse', x=self.x, y=self.y,
                                                              definition=definition, transform_x=self.transform_x,
                                                              transform_y=self.transform_y, **geom)
        self.child_populations.populations[pos].update_index(idx=pos_pop.index.values)
        self.child_populations.populations[neg].update_index(idx=neg_pop.index.values)
        return self.child_populations

    def create_ellipse(self, data: pd.DataFrame,
                       model: GaussianMixture or BayesianGaussianMixture,
                       tp_idx: np.array) -> np.array and dict:
        """
        Given a mixture model (scikit-learn object) and a desired confidence interval, generate mask for events
        that fall inside the 'confidence' ellipse and a 'geom' object

         data: Pandas.DataFrame
            parent population upon which the gate has been applied
         model: GaussianMixture or BayesianGaussianMixture
            scikit-learn object defining mixture model
         tp_idx: Numpy.array
            index of component that corresponds to positive (target) population

        Returns
        -------
        Numpy.array and dict
            numpy array 'mask' of events that fall within the confidence ellipse and a parameters of ellipse
            geom
        """
        eigen_val, eigen_vec = linalg.eigh(model.covariances_[tp_idx])
        tp_medoid = tuple(model.means_[tp_idx])
        chi2 = stats.chi2.ppf(self.conf, 2)
        eigen_val = 2. * np.sqrt(eigen_val) * np.sqrt(chi2)
        u = eigen_vec[0] / linalg.norm(eigen_vec[0])
        angle = 180. * np.arctan(u[1] / u[0]) / np.pi
        mask = inside_ellipse(data.values, tp_medoid, eigen_val[0], eigen_val[1], 180. + angle)
        return mask, dict(centroid=tp_medoid, width=eigen_val[0], height=eigen_val[1],
                          angle=(180. + angle))

