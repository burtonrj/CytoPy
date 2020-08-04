from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from scipy import stats
import pandas as pd
import numpy as np


def silvermans(data: np.array):
    return float(0.9*min([np.std(data), stats.iqr(data)/1.34])*(len(data)**(-(1/5))))


def kde_bandwidth_cv(x,
                     bandwidth_search: list or None = None,
                     cv: int = 20):
    """
    Estimate best bandwidth for KDE using cross validation
    Parameters
    -----------
    x:
        data for KDE
    bandwidth_search: tuple, optional
        tuple specifying range of bandwidth values to search (start, end) in cross validation;
        if value is None, 5th and 95th quartile of data is used for lower ad upper limit respectively
    cv: int, (default=20)
        number of folds to use in cross validation (default = 20)
    Returns
    --------
    float
        Optimal bandwidth
    """
    bandwidth_search = bandwidth_search or (np.quantile(x, 0.05), np.quantile(x, 0.95))
    if bandwidth_search[0] == 0:
        bandwidth_search = (0.01, bandwidth_search[1])
    grid = GridSearchCV(KernelDensity(),
                        {'bandwidth': np.linspace(bandwidth_search[0], bandwidth_search[1], 30)},
                        cv=cv)
    grid.fit(x)
    return float(grid.best_estimator_.bandwidth)


def kde(data: pd.DataFrame,
        x: str,
        kde_bw: float or str or None = None,
        bandwidth_search: list or None = None,
        kernel: str = 'gaussian',
        **kwargs) -> np.array:
    """
    Generate a 1D kernel density estimation using the scikit-learn implementation

    Parameters
    -----------
    data: Pandas.DataFrame
        Data for smoothing
    x: str
        column name for density estimation
    kde_bw: float
        bandwidth
    kernel: str, (default='gaussian')
        kernel to use for estimation (see scikit-learn documentation)

    Returns
    --------
    np.array
        Probability density function for array of 1000 x-axis values between min and max of data
    """
    kde_bw = kde_bw or "silvermans"
    if kde_bw == "silvermans":
        kde_bw = silvermans(data[x].values)
    if kde_bw == "cv":
        kde_bw = kde_bandwidth_cv(x=data, bandwidth_search=bandwidth_search)
    assert type(kde_bw) == float, "Invalid bandwidth, must be float or 'silvermans' or 'cv'"
    density = KernelDensity(bandwidth=kde_bw, kernel=kernel, **kwargs)
    d = data[x].values
    density.fit(d[:, None])
    x_d = np.linspace(min(d), max(d), 1000)
    logprob = density.score_samples(x_d[:, None])
    return np.exp(logprob), x_d


def multivariate_kde(data: pd.DataFrame or np.array,
                     features: list or None = None,
                     kde_bw: float or str or None = None,
                     bandwidth_search: list or None = None,
                     kernel: str = 'gaussian',
                     bins: int or None = 1000,
                     **kwargs):
    if type(data) == pd.DataFrame:
        features = features or [i for i, x in enumerate(data.columns) if x in features]
        data = data[features].values
    else:
        if features is not None:
            data = data[features]
    kde_bw = kde_bw or "silvermans"
    if kde_bw == "silvermans":
        kde_bw = silvermans(data)
    if kde_bw == "cv":
        kde_bw = kde_bandwidth_cv(x=data, bandwidth_search=bandwidth_search)
    assert type(kde_bw) == float, "Invalid bandwidth, must be float or 'silvermans' or 'cv'"

    kde = KernelDensity(bandwidth=kde_bw, kernel=kernel, **kwargs)
    kde.fit(data)
    if bins is not None:
        x_grid = np.array([np.linspace(np.amin(data), np.amax(data), bins) for _ in range(data.shape[1])])
        log_pdf = kde.score_samples(x_grid.T)
    else:
        x_grid = None
        log_pdf = kde.score_samples(data)
    return np.exp(log_pdf), x_grid

