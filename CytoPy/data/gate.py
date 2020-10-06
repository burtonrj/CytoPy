from ..flow.transforms import apply_transform
from .geometry import ThresholdGeom, PolygonGeom
from .populations import Population
from ..flow.transforms import scaler
from ..flow.sampling import faithful_downsampling, density_dependent_downsampling, upsample_knn
from ..flow.dim_reduction import dimensionality_reduction
from typing import List, Dict
from scipy.spatial import ConvexHull
from scipy import linalg, stats
from sklearn.cluster import *
from sklearn.mixture import *
from hdbscan import HDBSCAN
import pandas as pd
import numpy as np
import mongoengine
import inspect


class Child(mongoengine.EmbeddedDocument):
    """
    Base class for a gate child population
    """
    name = mongoengine.StringField()
    meta = {"allow_inheritance": True}


class ChildThreshold(Child):
    """
    Child population of a Threshold gate

    Parameters
    -----------
    definition: str
        Definition of population e.g "+" or "-" for 1 dimensional gate or "++" etc for 2 dimensional gate
    geom: ThresholdGeom
        Geometric definition for this child population
    """
    definition = mongoengine.StringField()
    geom = mongoengine.EmbeddedDocumentField(ThresholdGeom)

    def match_definition(self,
                         definition: str):
        """
        Given a definition, return True or False as to whether it matches this ChildThreshold's
        definition. If definition contains multiples separated by a comma, or the ChildThreshold's
        definition contains multiple, first split and then compare. Return True if matches any.

        Parameters
        ----------
        definition: str

        Returns
        -------
        bool
        """
        definition = definition.split(",")
        return any([x in self.definition.split(",") for x in definition])


class ChildPolygon(Child):
    """
    Child population of a Polgon or Ellipse gate

    Parameters
    -----------
    geom: ThresholdGeom
        Geometric definition for this child population
    """
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
    children = mongoengine.EmbeddedDocumentListField(Child)

    meta = {
        'db_alias': 'core',
        'collection': 'gates',
        'allow_inheritance': True
    }

    def __init__(self, *args, **values):
        method = values.get("method", None)
        assert method is not None, "No method given"
        err = f"Module {method} not supported. See docs for supported methods."
        assert method in ["manual", "density"] + list(globals().keys()), err
        super().__init__(*args, **values)
        self.model = None
        if method not in ["manual", "density"]:
            self.model = globals()[method](**self.method_kwargs)

    def _transform(self,
                   data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe prior to gating

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        Pandas.DataFrame
            Transformed dataframe
        """
        transforms = {self.x: self.transformations.get("x", None),
                      self.y: self.transformations.get("y", None)}
        return apply_transform(data=data,
                               features_to_transform=transforms)

    def _downsample(self,
                    data: pd.DataFrame) -> pd.DataFrame or None:
        """
        Perform down-sampling prior to gating. Returns down-sampled dataframe or
        None if sampling method is undefined.

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        Pandas.DataFrame or None
        """
        data = data.copy()
        if self.sampling.get("method", None) == "uniform":
            n = self.sampling.get("n", None) or self.sampling.get("frac", None)
            assert n is not None, "Must provide 'n' or 'frac' for uniform downsampling"
            if isinstance(n, int):
                return data.sample(n=n)
            elif isinstance(n, float):
                return data.sample(frac=0.5)
            else:
                raise ValueError("Sampling parameter 'n' must be an integer or float")
        if self.sampling.get("method", None) == "density":
            kwargs = {k: v for k, v in self.sampling.items() if k != "method"}
            return density_dependent_downsampling(data=data,
                                                  **kwargs)
        if self.sampling.get("method", None) == "faithful":
            h = self.sampling.get("h", 0.01)
            return faithful_downsampling(data=data.values, h=h)
        return None

    def _upsample(self,
                  data: pd.DataFrame,
                  sample: pd.DataFrame,
                  populations: List[Population]):
        """
        Perform up-sampling after gating. Returns list of Population objects
        with index updated to reflect the original data.

        Parameters
        ----------
        data: Pandas.DataFrame
            Original data, prior to down-sampling
        sample: Pandas.DataFrame
            Sampled data
        populations: list
            List of populations with assigned indexes

        Returns
        -------
        list
        """
        sample = sample.copy()
        sample["label"] = None
        for i, p in enumerate(populations):
            sample.loc[sample.index.isin(p.index), "label"] = i
        sample["label"].fillna(-1, inplace=True)
        labels = sample["label"].values
        sample.drop("label", axis=1, inplace=True)
        new_labels = upsample_knn(sample=sample,
                                  original_data=data,
                                  labels=labels,
                                  features=[i for i in [self.x, self.y] if i is not None],
                                  verbose=self.sampling.get("verbose", True),
                                  scoring=self.sampling.get("upsample_scoring", "balanced_accuracy"),
                                  **self.sampling.get("knn_kwargs", {}))
        for i, p in enumerate(populations):
            new_idx = data.index.values[np.where(new_labels == i)]
            if len(new_idx) == 0:
                raise ValueError(f"Up-sampling failed, no events labelled for {p.population_name}")
            p.index = new_idx
        return populations

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
        method = self.dim_reduction.get("method", None)
        if method is None:
            return data
        kwargs = {k: v for k, v in self.dim_reduction.items() if k != "method"}
        data = dimensionality_reduction(data=data,
                                        features=kwargs.get("features", data.columns.tolist()),
                                        method=method,
                                        n_components=2,
                                        return_embeddings_only=False,
                                        return_reducer=False,
                                        **kwargs)
        self.x = f"{method}1"
        self.y = f"{method}2"
        return data

    def reset_gate(self) -> None:
        """
        Removes existing children and resets all parameters.

        Returns
        -------
        None
        """
        self.children = []

    def label_children(self,
                       labels: dict,
                       drop: bool = True) -> None:
        """
        Rename children using a dictionary of labels where the key correspond to the existing child name
        and the value is the new desired population name. If drop is True, then children that are absent
        from the given dictionary will be dropped.

        Parameters
        ----------
        labels: dict
            Mapping for new children name
        drop: bool (default=True)
            If True, children absent from labels will be dropped

        Returns
        -------
        None
        """
        if drop:
            self.children = [c for c in self.children if c.name in labels.keys()]
        for c in self.children:
            c.name = labels.get(c.name)


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
            definition = child.definition.split(",")
            assert all(i in ["++", "+-", "-+", "--"]
                       for i in definition), "Invalid child definition, should be one of: '++', '+-', '-+', or '--'"
        else:
            assert child.definition in ["+", "-"], "Invalid child definition, should be either '+' or '-'"
        child.geom.x = self.x
        child.geom.y = self.y
        child.geom.transform_x, child.geom.transform_y = self.transformations.get("x", None), self.transformations.get("y", None)
        self.children.append(child)

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
        labeled = list()
        for p in new_populations:
            matching_child = [c for c in self.children if c.match_definition(p.definition)]
            assert len(matching_child) == 1, f"Population should match exactly one child, matched: {len(matching_child)}"
            p.population_name = matching_child[0].name
            labeled.append(p)
        return labeled

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
    return {"+": data[data[x] >= x_threshold],
            "-": data[data[x] < x_threshold]}


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
    return {"++": data[(data[x] >= x_threshold) & (data[y] >= y_threshold)],
            "--": data[(data[x] < x_threshold) & (data[y] < y_threshold)],
            "+-": data[(data[x] >= x_threshold) & (data[y] < y_threshold)],
            "-+": data[(data[x] < x_threshold) & (data[y] >= y_threshold)]}


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
