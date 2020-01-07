from shapely.geometry import Point
from multiprocessing import Pool, cpu_count
from functools import partial
from immunova.flow.gating.base import GateError
from sklearn.neighbors import KernelDensity, KDTree
import pandas as pd
import numpy as np
import inspect


def check_peak(peaks: np.array, probs: np.array, t=0.05) -> np.array:
    """Check peaks against largest peak in list,
    if peak < t*largest peak, then peak is removed
    :param peaks: array of indices for peaks
    :param probs: array of probability values of density estimate
    :param t: height threshold as a percentage of highest peak"""
    if peaks.shape[0] == 1:
        return peaks
    sorted_peaks = np.sort(probs[peaks])[::-1]
    real_peaks = list()
    real_peaks.append(np.where(probs == sorted_peaks[0])[0][0])
    for p in sorted_peaks[1:]:
        if p >= t*sorted_peaks[0]:
            real_peaks.append(np.where(probs == p)[0][0])
    return np.sort(np.array(real_peaks))


def find_local_minima(probs: np.array, xx: np.array, peaks: np.array) -> float:
    """
    Find local minima between the two highest peaks in the density distribution provided
    :param probs: probability for density estimate
    :param xx: x values for corresponding probabilities
    :param peaks: array of indices for identified peaks
    :return: local minima between highest peaks
    """
    sorted_peaks = np.sort(probs[peaks])[::-1]
    if sorted_peaks[0] == sorted_peaks[1]:
        p1_idx, p2_idx = np.where(probs == sorted_peaks[0])[0]
    else:
        p1_idx = np.where(probs == sorted_peaks[0])[0][0]
        p2_idx = np.where(probs == sorted_peaks[1])[0][0]
    if p1_idx < p2_idx:
        between_peaks = probs[p1_idx:p2_idx]
    else:
        between_peaks = probs[p2_idx:p1_idx]
    local_min = min(between_peaks)
    return xx[np.where(probs == local_min)[0][0]]


def kde(data: pd.DataFrame, x: str,
        kde_bw: float, kernel: str='gaussian') -> np.array:
    """
    Generate a kernel density estimation using the scikit-learn implementation
    :param data: data for smoothing
    :param x: column name for density estimation
    :param kde_bw: bandwidth
    :param kernel: kernel to use for estimation (see scikit-learn documentation)
    :return: probability densities for array of 1000 x-axis values between min and max of data
    """
    density = KernelDensity(bandwidth=kde_bw, kernel=kernel)
    d = data[x].values
    density.fit(d[:, None])
    x_d = np.linspace(min(d), max(d), 1000)
    logprob = density.score_samples(x_d[:, None])
    return np.exp(logprob), x_d


def inside_ellipse(data: np.array, center: tuple,
                   width: int or float, height: int or float,
                   angle: int or float) -> object:
    """
    Return mask of two dimensional matrix specifying if a data point (row) falls
    within an ellipse
    :param data - two dimensional matrix (x,y)
    :param center - tuple of x,y coordinate corresponding to center of elipse
    :param width - semi-major axis of eplipse
    :param height - semi-minor axis of elipse
    :param angle - angle of ellipse
    :return numpy array of indices for values inside specified ellipse
    """
    cos_angle = np.cos(np.radians(180.-angle))
    sin_angle = np.sin(np.radians(180.-angle))

    x = data[:, 0]
    y = data[:, 1]

    xc = x - center[0]
    yc = y - center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    rad_cc = (xct ** 2 / (width / 2.)**2) + (yct**2 / (height / 2.)**2)

    in_ellipse = []

    for r in rad_cc:
        if r <= 1.:
            # point in ellipse
            in_ellipse.append(True)
        else:
            # point not in ellipse
            in_ellipse.append(False)
    return in_ellipse


def rectangular_filter(data: pd.DataFrame, x: str, y: str, definition: dict) -> pd.DataFrame or str:
    """
    Given a pandas dataframe of fcs events data and a definition for a rectangular geom,
    filter the pandas dataframe and return only data contained within the rectangular geometric 2D plane
    :param data: pandas dataframe of fcs data to filter
    :param y: name of Y dimension (channel/marker name for column)
    :param x: name of X dimension (channel/marker name for column)
    :param definition: dictionary with keys: ['xmin', 'xmax', 'ymin', 'ymax'] each of integer/float value; see
    static.rect_gate for conventions
    :return: filtered pandas dataframe
    """
    if not all([x in ['xmin', 'xmax', 'ymin', 'ymax'] for x in definition.keys()]):
        raise GateError('Invalid definition for rectangular filter; must be dict with keys: xmin, xmax, ymin, ymax')
    data = data[(data[x] >= definition['xmin']) & (data[x] <= definition['xmax'])]
    data = data[(data[y] >= definition['ymin']) & (data[y] <= definition['ymax'])]
    return data


def centroid(data: np.array):
    x = np.median(data[:, 0])
    y = np.median(data[:, 1])
    return np.array([x, y])


def multi_centroid_calculation(data: pd.DataFrame):
    centroids = list()
    for c in data['labels'].unique():
        d = data[data['labels'] == c].values
        centroid_ = centroid(d)
        centroids.append(dict(chunk_idx=data['chunk_idx'].values[0],
                              cluster=c, x=centroid_[0], y=centroid_[1]))
    return pd.DataFrame(centroids)


def __multiprocess_point_in_poly(df, x, y, poly):
    """
    Return rows in dataframe who's values for x and y are contained in some polygon coordinate shape
    :param df:
    :param x:
    :param y:
    :param poly:
    :return:
    """
    mask = df.apply(lambda r: poly.contains(Point(r[x], r[y])), axis=1)
    return df.loc[mask]


def inside_polygon(df, x, y, poly):
    """
    Return rows in dataframe who's values for x and y are contained in some polygon coordinate shape
    :param df:
    :param x:
    :param y: S
    :param poly:
    :return:
    """
    pool = Pool(cpu_count())
    f = partial(__multiprocess_point_in_poly, x=x, y=y, poly=poly)
    n = int(df.shape[0]/10)
    list_df = [df[i: i+1] for i in range(0, df.shape[0], n)]
    result = pool.map(f, list_df)
    df_positive = pd.concat(result)
    pool.close()
    pool.join()
    return df_positive


def density_dependent_downsample(data: pd.DataFrame, features: list, frac: float = 0.1, sample_n: int or None = None,
                                 alpha: int = 5, mmd_sample_n: int = 2000,
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

    df = data.copy()
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


def get_params(klass, required_only=False, exclude_kwargs=True):
    if required_only:
        required_params = list(map(lambda x: [k for k, v in inspect.signature(x).parameters.items()
                                              if v.default is inspect.Parameter.empty],
                                   [c for c in inspect.getmro(klass)]))
    else:
        required_params = list(map(lambda x: inspect.signature(x).parameters.keys(),
                                   [c for c in inspect.getmro(klass)]))
    required_params = [l for sl in required_params for l in sl]
    if exclude_kwargs:
        return [x for x in required_params if x != 'kwargs']
    return required_params


def check_downstream_overlaps(ref, root_population: str, population_labels: list) -> bool:
    """
    Internal method. Check if a chosen root population is downstream of target populations for classification.
    This is a problem because if the root population is downstream then the model won't have access to the events
    it needs to classify.
    :param ref: Gating object whom's populations you wish to check
    :return: True if overlaps exist, otherwise False
    """
    downstream_overlaps = False
    for pop_i in population_labels:
        dependencies = ref.find_dependencies(pop_i)
        if root_population in dependencies:
            print(f'Error: population {pop_i} is upstream from the chosen root population {root_population}')
            downstream_overlaps = True
        for pop_j in population_labels:
            if pop_j == pop_i:
                continue
            if pop_j in dependencies:
                print(f'Error: population {pop_j} is a dependency of population {pop_i} (i.e. it is downstream '
                      f'from this population). This will result in invalid labelling. If you wish to continue '
                      f'with these population targets, please set multi_label parameter to True')
                downstream_overlaps = True
    return downstream_overlaps
