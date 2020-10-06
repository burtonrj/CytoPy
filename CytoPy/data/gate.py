from .geometry import ThresholdGeom, PolygonGeom
from .populations import Population
from ..flow.transforms import scaler
from typing import List, Dict
from sklearn import cluster, mixture
from scipy.spatial import ConvexHull
from scipy import linalg, stats
from sklearn.cluster import *
from sklearn.mixture import *
from hdbscan import HDBSCAN
import pandas as pd
import numpy as np
import mongoengine
import inspect


class ChildThreshold(mongoengine.EmbeddedDocument):
    """
    Child population of a Threshold gate

    Parameters
    -----------
    name: str
        Population name
    definition: str
        Definition of population e.g "+" or "-" for 1 dimensional gate or "++" etc for 2 dimensional gate
    geom: ThresholdGeom
        Geometric definition for this child population
    """
    name = mongoengine.StringField()
    definition = mongoengine.StringField()
    geom = mongoengine.EmbeddedDocumentField(ThresholdGeom)


class ChildPolygon(mongoengine.EmbeddedDocument):
    """
    Child population of a Polgon or Ellipse gate

    Parameters
    -----------
    name: str
        Population name
    geom: ThresholdGeom
        Geometric definition for this child population
    """
    name = mongoengine.StringField()
    geom = mongoengine.EmbeddedDocumentField(PolygonGeom)


class Gate(mongoengine.Document):
    """
    Base class for a Gate
    """
    gate_name = mongoengine.StringField(required=True)
    parent = mongoengine.StringField(required=True)
    x = mongoengine.StringField(required=True)
    y = mongoengine.StringField(required=False)
    transformations = mongoengine.DictField()
    sampling = mongoengine.DictField()
    dim_reduction = mongoengine.DictField()
    method = mongoengine.StringField(required=True)
    method_kwargs = mongoengine.DictField()

    meta = {
        'db_alias': 'core',
        'collection': 'gates',
        'allow_inheritance': True
    }

    def __init__(self, *args, **values):
        sampling = values.get("sampling", None)
        if sampling is not None:
            err = "Sampling, if given, must contain method for downsampling AND upsampling"
            assert "downsample" in sampling.keys(), err
            assert "upsample" in sampling.keys(), err
        super().__init__(*args, **values)

    def _transform(self,
                   data: pd.DataFrame) -> pd.DataFrame:
        """
        Transfrom dataframe prior to gating

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        Pandas.DataFrame
            Transformed dataframe
        """
        pass

    def _downsample(self,
                    data: pd.DataFrame):
        """
        Perform down-sampling prior to gating. Returns down-sampled dataframe.

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        Pandas.DataFrame
        """
        pass

    def _upsample(self,
                  data: pd.DataFrame,
                  sample: pd.DataFrame):
        """
        Perform up-sampling after gating. Returns up-sampled dataframe.

        Parameters
        ----------
        data: Pandas.DataFrame
            Original data, prior to down-sampling
        sample: Pandas.DataFrame
            Sampled data

        Returns
        -------
        Pandas.DataFrame
        """
        pass

    def _dim_reduction(self,
                       data: pd.DataFrame):
        """
        Perform dimension reduction prior to gating. Returns dataframe
        with appended columns for embeddings

        Parameters
        ----------
        data: Pandas.DataFrame
            Data to reduce

        Returns
        -------
        Pandas.DataFrame
        """
        pass

    def _init_model(self) -> object or None:
        """
        Initialise model used for autonomous gate. If DensityGate or a manual method, this
        will return None.

        Returns
        -------
        object or None
            Either Scikit-Learn clustering object or HDBSCAN or None in the case of
            ThresholdGate or static "manual" method.
        """
        pass


class ThresholdGate(Gate):
    """
    A ThresholdGate is for density based gating that applies one or two-dimensional gates
    to data in the form of straight lines, parallel to the axis that fall in the area of minimum
    density.
    """
    children = mongoengine.EmbeddedDocumentListField(ChildThreshold)

    def add_child(self,
                  child: ChildThreshold) -> None:
        """
        Add a new child for this gate. Checks that definition is valid and overwrites geom with gate information.

        Parameters
        ----------
        child: ChildThreshold

        Returns
        -------
        None
        """
        if self.y is not None:
            assert child.definition in ["++", "+-", "-+", "--"], "Invalid child definition, should be one of: '++', '+-', '-+', or '--'"
        else:
            assert child.definition in ["+", "-"], "Invalid child definition, should be either '+' or '-'"
        child.geom.x = self.x
        child.geom.y = self.y
        child.geom.transform_x, child.geom.transform_y = self.preprocessing.get("transform_x", None), self.preprocessing.get("transform_y", None)
        self.children.append(child)

    def reset_gate(self) -> None:
        """
        Removes existing children and resets all parameters.

        Returns
        -------
        None
        """
        pass

    def label_children(self,
                       labels: dict) -> None:
        """
        Rename children using a dictionary of labels where the key correspond to the existing child name
        and the value is the new desired population name.

        Returns
        -------
        None
        """
        pass

    def _match_to_children(self,
                           new_populations: List[Population]) -> List[Population]:
        """
        Given a list of newly create Populations, match the Populations to the gates children and
        return list of Populations with correct population names.

        Parameters
        ----------
        new_populations: list
            List of newly created Population objects

        Returns
        -------
        list
        """
        pass

    def fit(self,
            data: pd.DataFrame) -> None:
        """
        Fit the gate using a given dataframe. This will generate new children using the calculated
        thresholds. If children already exist will raise an AssertionError and notify user to call
        `fit_predict`.

        Parameters
        ----------
        data: Pandas.DataFrame
            Population data to fit threshold too

        Returns
        -------
        None
        """
        pass

    def fit_predict(self,
                    data: pd.DataFrame) -> List[Population]:
        """
        Fit the gate using a given dataframe and then associate predicted Population objects to
        existing children. If no children exist, an AssertionError will be raised prompting the
        user to call `fit` method.

        Parameters
        ----------
        data: Pandas.DataFrame
            Population data to fit threshold too

        Returns
        -------
        list
            List of predicted Population objects, labelled according to the gates child objects
        """
        pass

    def predict(self,
                data: pd.DataFrame) -> List[Population]:
        """
        Using existing children associated to this gate, the previously calculated thresholds of
        these children will be applied to the given data and then Population objects created and
        labelled to match the children of this gate. NOTE: the data will not be fitted and thresholds
        applied will be STATIC not data driven. For data driven gates call `fit_predict` method.

        Parameters
        ----------
        data: Pandas.DataFrame
            Data to apply static thresholds too

        Returns
        -------
        list
            List of Population objects
        """


class PolygonGate(Gate):
    """
    Polygon gates generate polygon shapes that capture populations of varying shapes. These can
    be generated by any number of clustering algorithms.
    """
    children = mongoengine.EmbeddedDocumentListField(ChildPolygon)

    def add_child(self,
                  child: ChildPolygon) -> None:
        """
        Add a new child for this gate. Checks that child is valid and overwrites geom with gate information.

        Parameters
        ----------
        child: ChildPolygon

        Returns
        -------
        None
        """
        pass

    def reset_gate(self) -> None:
        """
        Removes existing children and resets all parameters.

        Returns
        -------
        None
        """
        pass

    def label_children(self,
                       labels: dict) -> None:
        """
        Rename children using a dictionary of labels where the key correspond to the existing child name
        and the value is the new desired population name.

        Returns
        -------
        None
        """
        pass

    def _match_to_children(self,
                           new_populations: List[Population]) -> List[Population]:
        """
        Given a list of newly create Populations, match the Populations to the gates children and
        return list of Populations with correct population names.

        Parameters
        ----------
        new_populations: list
            List of newly created Population objects

        Returns
        -------
        list
        """
        pass

    def fit(self,
            data: pd.DataFrame) -> None:
        """
        Fit the gate using a given dataframe. This will generate new children using the calculated
        polygons. If children already exist will raise an AssertionError and notify user to call
        `fit_predict`.

        Parameters
        ----------
        data: Pandas.DataFrame
            Population data to fit gate to

        Returns
        -------
        None
        """
        pass

    def fit_predict(self,
                    data: pd.DataFrame) -> List[Population]:
        """
        Fit the gate using a given dataframe and then associate predicted Population objects to
        existing children. If no children exist, an AssertionError will be raised prompting the
        user to call `fit` method.

        Parameters
        ----------
        data: Pandas.DataFrame
            Population data to fit gate to

        Returns
        -------
        list
            List of predicted Population objects, labelled according to the gates child objects
        """
        pass

    def predict(self,
                data: pd.DataFrame) -> List[Population]:
        """
        Using existing children associated to this gate, the previously calculated polygons of
        these children will be applied to the given data and then Population objects created and
        labelled to match the children of this gate. NOTE: the data will not be fitted and polygons
        applied will be STATIC not data driven. For data driven gates call `fit_predict` method.

        Parameters
        ----------
        data: Pandas.DataFrame
            Data to apply static polygons to

        Returns
        -------
        list
            List of Population objects
        """


class EllipseGate(Gate):
    """
    Ellipse gates generate circular or elliptical gates and can be generated from algorithms that are
    centroid based (like K-means) or probabilistic methods that estimate the covariance matrix of one
    or more gaussian components such as mixture models.
    """
    children = mongoengine.EmbeddedDocumentListField(ChildPolygon)

    def __init__(self, *args, **values):
        method = values.get("method", None)
        valid = ["manual", "GaussianMixture", "BayesianGaussianMixture"]
        assert method in valid, f"Elliptical gating method should be one of {valid}"
        super().__init__(*args, **values)

    def add_child(self,
                  child: ChildPolygon) -> None:
        """
        Add a new child for this gate. Checks that child is valid and overwrites geom with gate information.

        Parameters
        ----------
        child: ChildPolygon

        Returns
        -------
        None
        """
        pass

    def reset_gate(self) -> None:
        """
        Removes existing children and resets all parameters.

        Returns
        -------
        None
        """
        pass

    def label_children(self,
                       labels: dict) -> None:
        """
        Rename children using a dictionary of labels where the key correspond to the existing child name
        and the value is the new desired population name.

        Returns
        -------
        None
        """
        pass

    def _match_to_children(self,
                           new_populations: List[Population]) -> List[Population]:
        """
        Given a list of newly create Populations, match the Populations to the gates children and
        return list of Populations with correct population names.

        Parameters
        ----------
        new_populations: list
            List of newly created Population objects

        Returns
        -------
        list
        """
        pass

    def fit(self,
            data: pd.DataFrame) -> None:
        """
        Fit the gate using a given dataframe. This will generate new children using the calculated
        polygons. If children already exist will raise an AssertionError and notify user to call
        `fit_predict`.

        Parameters
        ----------
        data: Pandas.DataFrame
            Population data to fit gate to

        Returns
        -------
        None
        """
        pass

    def fit_predict(self,
                    data: pd.DataFrame) -> List[Population]:
        """
        Fit the gate using a given dataframe and then associate predicted Population objects to
        existing children. If no children exist, an AssertionError will be raised prompting the
        user to call `fit` method.

        Parameters
        ----------
        data: Pandas.DataFrame
            Population data to fit gate to

        Returns
        -------
        list
            List of predicted Population objects, labelled according to the gates child objects
        """
        pass

    def predict(self,
                data: pd.DataFrame) -> List[Population]:
        """
        Using existing children associated to this gate, the previously calculated polygons of
        these children will be applied to the given data and then Population objects created and
        labelled to match the children of this gate. NOTE: the data will not be fitted and polygons
        applied will be STATIC not data driven. For data driven gates call `fit_predict` method.

        Parameters
        ----------
        data: Pandas.DataFrame
            Data to apply static polygons to

        Returns
        -------
        list
            List of Population objects
        """


def create_signature(data: pd.DataFrame,
                     idx: np.array or None = None,
                     summary_method: callable or None = None) -> dict:
    """
    Given a dataframe of FCS events, generate a signature of those events; that is, a summary of the
    dataframes columns using the given summary method.

    Parameters
    ----------
    data: Pandas.DataFrame
    idx: Numpy.array (optional)
        Array of indexes to be included in this operation, if None, the whole dataframe is used
    summary_method: callable (optional)
        Function to use to summarise columns, defaults is Numpy.median
    Returns
    -------
    dict
        Dictionary representation of signature; {column name: summary statistic}
    """
    data = pd.DataFrame(scaler(data=data.values, scale_method="norm", return_scaler=False),
                        columns=data.columns,
                        index=data.index)
    if idx is None:
        idx = data.index.values
    # ToDo this should be more robust
    for x in ["Time", "time"]:
        if x in data.columns:
            data.drop(x, 1, inplace=True)
    summary_method = summary_method or np.median
    signature = data.loc[idx].apply(summary_method)
    return {x[0]: x[1] for x in zip(signature.index, signature.values)}


def threshold_1d(data: pd.DataFrame,
                 x: str,
                 x_threshold: float) -> Dict[str, pd.DataFrame]:
    """
    Apply the given threshold (x_threshold) to the x-axis variable (x) and return the
    resulting dataframes corresponding to the positive and negative populations.
    Returns a dictionary of dataframes: {'-': Pandas.DataFrame, '+': Pandas.DataFrame}

    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
    x_threshold: float

    Returns
    -------
    dict
        Negative population (less than threshold) and positive population (greater than or equal to threshold)
        in a dictionary as so: {'-': Pandas.DataFrame, '+': Pandas.DataFrame}
    """
    pass


def threshold_2d(data: pd.DataFrame,
                 x: str,
                 y: str,
                 x_threshold: float,
                 y_threshold: float) -> Dict[str, pd.DataFrame]:
    """
    Apply the given threshold (x_threshold) to the x-axis variable (x) and the given threshold (y_threshold)
    to the y-axis variable (y), and return the  resulting dataframes as a dictionary:
        '++': Greater than or equal to threshold for both x and y
        '+-': Greater than or equal to threshold for x but less than threshold for y
        '-+': Greater than or equal to threshold for y but less than threshold for x
        '--': Less than threshold for both x and y

    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
    y: str
    x_threshold: float
    y_threshold: float

    Returns
    -------
    dict
    """
    pass


def find_peaks(p: np.array,
               min_peak_threshold: float,
               peak_boundary: float) -> np.array:
    """
    Perform peak finding using the detecta package (see detecta.detect_peaks for details).

    Parameters
    ----------
    p: np.array
        Probability vector as generated from KDE
    min_peak_threshold: float
        Percentage of highest recorded peak below which peaks are ignored. E.g. 0.05 would mean
        any peak less than 5% of the highest peak would be ignored.
    peak_boundary: float
        Bounding window around which only the highest peak is considered. E.g. 0.1 would mean that
        peaks are assessed within a window the size of peak_boundary * length of probability vector and
        only highest peak within window is kept.

    Returns
    -------
    Numpy.array
        Index of peaks
    """
    pass


def smoothed_peak_finding(x: np.array,
                          p: np.array,
                          starting_window_length: int = 11,
                          polyorder: int = 3,
                          min_peak_threshold: float = 0.05,
                          peak_bounary: float = 0.1,
                          **kwargs):
    """
    Given the grid space and probability vector of some PDF calculated using KDE,
    first attempt to smooth the probability vector using a Savitzky-Golay filter
    (see scipy.signal.savgol_filter) and then perform peak finding until the
    number of peaks is less than 3. Window size will be incremented until the
    number of peaks is reduced. If window size exceeds half the length of the
    probability vector, will raise an AssertionError to avoid misrepresentation of
    the data.

    Parameters
    ----------
    x: np.array
        Grid space upon which the PDF has been estimated
    p: np.array
        Probability vector resulting from KDE calculation
    starting_window_length: int (default=11)
        Window length of filter (must be > length of p, < length of p * 0.5, and an odd number)
    polyorder: int (default=3)
        Order of polynomial for filter
    min_peak_threshold: float (default=0.05)
        See CytoPy.data.gate.find_peaks
    peak_bounary: float (default=0.1)
        See CytoPy.data.gate.find_peaks
    kwargs: dict
        Additional keyword arguments to pass to scipy.signal.savgol_filter

    Returns
    -------
    np.array, np.array
        Smooth probability vector and index of peaks
    """
    pass


def find_local_minima(p: np.array,
                      x: np.array,
                      peaks: np.array) -> float:
    """
    Find local minima between the two highest peaks in the density distribution provided

    Parameters
    -----------
    p: Numpy.array
        probability vector as generated from KDE
    x: Numpy.array
        Grid space for probability vector
    peaks: Numpy.array
        array of indices for identified peaks

    Returns
    --------
    float
        local minima between highest peaks
    """
    sorted_peaks = np.sort(p[peaks])[::-1]
    if sorted_peaks[0] == sorted_peaks[1]:
        p1_idx, p2_idx = np.where(p == sorted_peaks[0])[0]
    else:
        p1_idx = np.where(p == sorted_peaks[0])[0][0]
        p2_idx = np.where(p == sorted_peaks[1])[0][0]
    if p1_idx < p2_idx:
        between_peaks = p[p1_idx:p2_idx]
    else:
        between_peaks = p[p2_idx:p1_idx]
    local_min = min(between_peaks)
    return float(x[np.where(p == local_min)[0][0]])


def find_inflection_point(x: np.array,
                          p: np.array,
                          peak_idx: int,
                          incline: bool = False,
                          **kwargs):
    """
    Given some probability vector and grid space that represents a PDF as calculated by KDE,
    and assuming this vector has a single peak of highest density, calculate the inflection point
    at which the peak flattens. Probability vector is first smoothed using Savitzky-Golay filter.

    Parameters
    ----------
    x: np.array
        Grid space for the probability vector
    p: np.array
        Probability vector as calculated by KDE
    peak_idx: int
        Index of the peak
    incline: bool (default=False)
        If true, calculates the inflection point of the incline towards the peak
        as opposed to the decline away from the peak
    kwargs: dict
        Additional keyword argument to pass to scipy.signal.savgol_filter

    Returns
    -------

    """
    pass


def valid_sklearn(klass: str):
    """
    Given the name of a Scikit-Learn class, checks validity. If invalid, raises Assertion error,
    otherwise returns the class name.

    Parameters
    ----------
    klass: str

    Returns
    -------
    str
    """
    valid_clusters = [x[0] for x in inspect.getmembers(cluster, inspect.isclass)
                      if 'sklearn.cluster' in x[1].__module__]
    valid_mixtures = [x[0] for x in inspect.getmembers(mixture, inspect.isclass)
                      if 'sklearn.mixture' in x[1].__module__]
    valid = valid_clusters + valid_mixtures + ["HDBSCAN"]
    err = f"""Invalid class name. Must be one of the following from Scikit-Learn's cluster module: {valid_clusters};
 or from Scikit-Learn's mixture module: {valid_mixtures}; or 'HDBSCAN'"""
    assert klass in valid, err
    return klass


def create_convex_hull(x_values: np.array,
                       y_values: np.array):
    """
    Given the x and y coordinates of a cloud of data points, generate a convex hull,
    returning the x and y coordinates of its vertices.

    Parameters
    ----------
    x_values: Numpy.array
    y_values: Numpy.array

    Returns
    -------
    Numpy.array, Numpy.array
    """
    xy = np.array([[i[0], i[1]] for i in zip(x_values, y_values)])
    hull = ConvexHull(xy)
    x = [int(i) for i in xy[hull.vertices, 0]]
    y = [int(i) for i in xy[hull.vertices, 1]]
    return x, y


def probablistic_ellipse(covariances: np.array,
                         conf: float):
    """
    Given the covariance matrix of a mixture component, calculate a elliptical shape that
    represents a probabilistic confidence interval.

    Parameters
    ----------
    covariances: np.array
        Covariance matrix
    conf: float
        The confidence interval (e.g. 0.95 would give the region of 95% confidence)

    Returns
    -------
    float, float, float
        Width, Height and Angle of ellipse
    """
    eigen_val, eigen_vec = linalg.eigh(covariances)
    chi2 = stats.chi2.ppf(conf, 2)
    eigen_val = 2. * np.sqrt(eigen_val) * np.sqrt(chi2)
    u = eigen_vec[0] / linalg.norm(eigen_vec[0])
    angle = 180. * np.arctan(u[1] / u[0]) / np.pi
    return eigen_val[0], eigen_val[1], (180. + angle)


def inside_ellipse(data: np.array,
                   center: tuple,
                   width: int or float,
                   height: int or float,
                   angle: int or float) -> object:
    """
    Return mask of two dimensional matrix specifying if a data point (row) falls
    within an ellipse

    Parameters
    -----------
    data: Numpy.array
        two dimensional matrix (x,y)
    center: tuple
        x,y coordinate corresponding to center of elipse
    width: int or float
        semi-major axis of eplipse
    height: int or float
        semi-minor axis of elipse
    angle: int or float
        angle of ellipse

    Returns
    --------
    Numpy.array
        numpy array of indices for values inside specified ellipse
    """
    cos_angle = np.cos(np.radians(180. - angle))
    sin_angle = np.sin(np.radians(180. - angle))

    x = data[:, 0]
    y = data[:, 1]

    xc = x - center[0]
    yc = y - center[1]

    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle

    rad_cc = (xct ** 2 / (width / 2.) ** 2) + (yct ** 2 / (height / 2.) ** 2)

    in_ellipse = []

    for r in rad_cc:
        if r <= 1.:
            # point in ellipse
            in_ellipse.append(True)
        else:
            # point not in ellipse
            in_ellipse.append(False)
    return in_ellipse
