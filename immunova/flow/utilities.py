from IPython import get_ipython
from tqdm import tqdm_notebook, tqdm
from sklearn.neighbors import BallTree, KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np


def which_environment():
    """
    Test if module is being executed in the Jupyter environment.
    :return:
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


def progress_bar(x: iter, **kwargs) -> callable:
    """
    Generate a progress bar using the tqdm library. If execution environment is Jupyter, return tqdm_notebook
    otherwise used tqdm.
    :param x: some iterable to pass to tqdm function
    :param kwargs: additional keyword arguments for tqdm
    :return: tqdm or tqdm_notebook, depending on environment
    """
    if which_environment() == 'jupyter':
        return tqdm_notebook(x, **kwargs)
    return tqdm(x, **kwargs)


def faithful_downsampling(data: np.array, h: float):
    """
    An implementation of faithful downsampling as described in:  Zare H, Shooshtari P, Gupta A, Brinkman R.
    Data reduction for spectral clustering to analyze high throughput flow cytometry data. BMC Bioinformatics 2010;11:403
    :param data: numpy array to be downsampled
    :param h: radius for nearest neighbours search
    :return: Downsampled array
    """
    communities = None
    registered = np.zeros(data.shape[0])
    tree = BallTree(data)
    while not all([x == 1 for x in registered]):
        i_ = np.random.choice(np.where(registered == 0)[0])
        registered[i_] = 1
        registering_idx = tree.query_radius(data[i_].reshape(1, -1), r=h)[0]
        registering_idx = [t for t in registering_idx if t != i_]
        registered[registering_idx] = 1
        if communities is None:
            communities = data[registering_idx]
        else:
            communities = np.unique(np.concatenate((communities, data[registering_idx]), 0), axis=0)
    return communities


def hellinger_dot(p, q):
    """
    Hellinger distance between two discrete distributions.
    Original code found here: https://nbviewer.jupyter.org/gist/Teagum/460a508cda99f9874e4ff828e1896862
    :param p: discrete probability distribution, p
    :param q: discrete probability distribution, q
    :return: Hellinger Distance
    """
    z = np.sqrt(p) - np.sqrt(q)
    return np.sqrt(z @ z / 2)


def kde_multivariant(x: np.array, x_grid: np.array, bandwidth: str or float = 'cross_val', bandwidth_search: tuple = (0.01, 0.5), **kwargs):
    assert x.shape[1] == x_grid.shape[1], 'x and x_grid must have equal dimensions'
    if type(bandwidth) == str:
        assert bandwidth == 'cross_val', 'Invalid input for bandwidth, must be either float or "cross_val"'
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': np.linspace(bandwidth_search[0], bandwidth_search[1], 30)},
                            cv=20)
        grid.fit(x)
        bandwidth = grid.best_estimator_.bandwidth
    kde = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde.fit(x)
    log_pdf = kde.score_samples(x_grid)
    return np.exp(log_pdf)
