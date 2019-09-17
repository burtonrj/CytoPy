import numpy as np
from scipy import linalg, stats
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from immunova.flow.gating.utilities import boolean_gate, inside_ellipse, rectangular_filter
from immunova.flow.gating.defaults import GateOutput, Geom
import pandas as pd
import math


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


def create_ellipse(data, x, y, model, conf, tp_idx):
    """
    Given a mixture model (scikit-learn object) and a desired confidence interval, generate mask for events
    that fall inside the 'confidence' ellipse and a 'geom' object
    :param data: parent population upon which the gate has been applied
    :param x:
    :param y:
    :param model: scikit-learn object defining mixture model
    :param conf: critical value for confidence interval (defines the 'tightness' of resulting elliptical gate)
    :param tp_idx: index of component that corresponds to positive (target) population
    :return: numpy array 'mask' of events that fall within the confidence ellipse and a Geom object
    """
    eigen_val, eigen_vec = linalg.eigh(model.covariances_[tp_idx])
    tp_medoid = tuple(model.means_[tp_idx])

    chi2 = stats.chi2.ppf(conf, 2)
    eigen_val = 2. * np.sqrt(eigen_val) * np.sqrt(chi2)
    u = eigen_vec[0] / linalg.norm(eigen_vec[0])
    angle = 180. * np.arctan(u[1] / u[0]) / np.pi
    mask = inside_ellipse(data.values, tp_medoid, eigen_val[0], eigen_val[1], 180. + angle)
    geom = Geom(shape='ellipse', x=x, y=y)
    geom.update(dict(mean=tp_medoid, width=eigen_val[0], height=eigen_val[1],
                angle=180. + angle))
    return mask, geom


def mm_gate(data: pd.DataFrame, x: str, y: str, child_name: str, target: tuple = None, k: int = None,
             method: str = 'gmm', bool_gate: bool = False, conf: float = 0.95, rect_filter: dict or None = None,
             **kwargs) -> GateOutput:
    """

    :param child_name:
    :param data: parent population upon which the gate is applied
    :param x: name of the channel/marker for X dimension
    :param y: name of the channel/marker for Y dimension
    :param target: expected medoid of output population
    :param k: expected number of populations in dataset
    :param method: method to use for mixture model; must be either 'gmm' (default) or 'bayesian' (see scikit-learn
    user guide for information on differences)
    :param bool_gate: if False, the positive population is returned (>= threshold) else the negative population
    :param conf: critical value for confidence interval (defines the 'tightness' of resulting elliptical gate)
    :param rect_filter: rectangular filter applied prior to mixture model gate; dictionary following conventions of
    static.rect_gate
    :param kwargs: additional keyword arguments for mixture model functions (see scikit-learn)
    :return: Output object
    """
    output = GateOutput()
    if 'covar' not in kwargs.keys():
        covar = 'full'
    else:
        covar = kwargs.pop('covar')
    X = data[[x, y]]

    # Filter if necessary
    if rect_filter:
        X = rectangular_filter(X, x, y, rect_filter)

    # Define model
    if method == 'gmm':
        if not k:
            k = 2
        model = GaussianMixture(n_components=k, covariance_type=covar, random_state=42, **kwargs).fit(X)
    elif method == 'bayesian':
        if not k:
            k = 5
        model = BayesianGaussianMixture(n_components=k, covariance_type=covar, random_state=42, **kwargs).fit(X)
    else:
        output.error = 1
        output.error_msg = 'Invalid method, must be one of: gmm, bayesian'
        return output
    # Select optimal component
    if target:
        tp_medoid = min(model.means_, key=lambda m: math.hypot(m[0] - target[0], m[1] - target[1]))
        tp_idx = [list(x) for x in model.means_].index(list(tp_medoid))
        if math.ceil(math.hypot(tp_medoid[0] - target[0], tp_medoid[1] - target[1])) >= 3:
            output.warnings.append('WARNING: actual population is at least a 3 fold distance from the target. '
                                   'Is this really the population of interest?')
    else:
        Y_ = model.predict(X)
        tp_idx = stats.mode(Y_)[0][0]
    mask, geom = create_ellipse(X, x, y, model, conf, tp_idx)
    pos_pop = data[mask]
    pos_pop = boolean_gate(data, pos_pop, bool_gate)
    output.add_child(name=child_name, idx=pos_pop.index.values, geom=geom)
    return output
