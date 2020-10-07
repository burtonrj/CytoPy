from ..flow.transforms import apply_transform
from .geometry import ThresholdGeom, PolygonGeom
from .populations import Population, merge_multiple_populations
from ..flow.transforms import scaler
from ..flow.sampling import faithful_downsampling, density_dependent_downsampling, upsample_knn
from ..flow.dim_reduction import dimensionality_reduction
from shapely.geometry import Polygon as ShapelyPoly
from collections import Counter
from itertools import chain
from typing import List, Dict
from KDEpy import FFTKDE
from detecta import detect_peaks
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter
from scipy import linalg, stats
from sklearn.cluster import *
from sklearn.mixture import *
from hdbscan import HDBSCAN
import pandas as pd
import numpy as np
import mongoengine


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
        assert method in ["manual", "density", "quantile"] + list(globals().keys()), err
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
        transforms = {self.x: self.transformations.get("x", None)}
        if self.y is not None:
            transforms[self.y] = self.transformations.get("y", None)
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
        raise ValueError("Invalid downsample method, should be one of: 'uniform', 'density' or 'faithful'")

    def _upsample(self,
                  data: pd.DataFrame,
                  sample: pd.DataFrame,
                  populations: List[Population]) -> List[Population]:
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

    def _xy_in_dataframe(self,
                         data: pd.DataFrame):
        assert self.x in data.columns, f"{self.x} missing from given dataframe"
        if self.y:
            assert self.y in data.columns, f"{self.y} missing from given dataframe"

    def reset_gate(self) -> None:
        """
        Removes existing children and resets all parameters.

        Returns
        -------
        None
        """
        self.children = []

    def _duplicate_children(self) -> None:
        """
        Loop through the children and merge any with the same name.

        Returns
        -------
        None
        """
        child_counts = Counter([c.name for c in self.children])
        if set(child_counts.values()) == 1:
            return
        updated_children = []
        for name, count in child_counts.items():
            if count > 2:
                updated_children.append(merge_children([c for c in self.children if c.name == name]))
            else:
                updated_children.append([c for c in self.children if c.name == name][0])
        self.children = updated_children

    def label_children(self,
                       labels: dict,
                       drop: bool = True) -> None:
        """
        Rename children using a dictionary of labels where the key correspond to the existing child name
        and the value is the new desired population name. If the same population name is given to multiple
        children, these children will be merged.
        If drop is True, then children that are absent from the given dictionary will be dropped.

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
        self._duplicate_children()


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
        for c in self.children:
            matching_populations = [p for p in new_populations if c.match_definition(p.definition)]
            if len(matching_populations) == 0:
                continue
            elif len(matching_populations) > 1:
                pop = merge_multiple_populations(matching_populations, new_population_name=c.name)
            else:
                pop = matching_populations[0]
                pop.population_name = c.name
            labeled.append(pop)
        return labeled

    def _quantile_gate(self,
                       data: pd.DataFrame) -> list:
        """
        Fit gate to the given dataframe by simply drawing the threshold at the desired quantile.

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        list
            List of thresholds (one for each dimension)
        """
        q = self.method_kwargs.get("q", None)
        assert q is not None, "Must provide a value for 'q' in method kwargs when using quantile gate"
        if self.y is None:
            return [data[self.x].quantile(q)]
        return [data[self.x].quantile(q), data[self.y].quantile(q)]

    def _process_one_peak(self,
                          d: str,
                          data: pd.DataFrame,
                          x_grid: np.array,
                          p: np.array,
                          peak_idx: int):
        """
        Process the results of a single peak detected. Returns the threshold for
        the given dimension.

        Parameters
        ----------
        d: str
            Name of the dimension (feature) under investigation. Must be a column in data.
        data: Pandas.DataFrame
            Events dataframe
        x_grid: Numpy.array
            x grid upon which probability vector is estimated by KDE
        p: Numpy.array
            probability vector as estimated by KDE

        Returns
        -------
        float
        """
        use_inflection_point = self.method_kwargs.get("use_inflection_point", True)
        if not use_inflection_point:
            q = self.method_kwargs.get("q", None)
            assert q is not None, "Must provide a value for 'q' in method kwargs " \
                                  "for desired quantile if use_inflection_point is False"
            return data[d].quantile(q)
        inflection_point_kwargs = self.method_kwargs.get("inflection_point_kwargs", {})
        return find_inflection_point(x=x_grid,
                                     p=p,
                                     peak_idx=peak_idx,
                                     **inflection_point_kwargs)

    def _fit(self,
             data: pd.DataFrame) -> (list, pd.DataFrame):
        """
        Internal method to fit threshold density gating to a given dataframe. Returns the
        list of thresholds generated and the dataframe the threshold were generated from
        (will be the downsampled dataframe if sampling methods defined).

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        list, Pandas.DataFrame
        """
        if self.method == "manual":
            return self._manual(), data
        thresholds = list()
        self._xy_in_dataframe(data=data)
        data = self._transform(data=data)
        data = self._dim_reduction(data=data)
        dims = [i for i in [self.x, self.y] if i is not None]
        if self.sampling.get("method", None) is not None:
            data = self._downsample(data=data)
        if self.method == "quantile":
            thresholds = self._quantile_gate(data=data)
        else:
            for d in dims:
                x_grid, p = (FFTKDE(kernel=self.method_kwargs.get("kernel", "gaussian"),
                                    bw=self.method_kwargs.get("bw", "silverman"))
                             .fit(data[d].values)
                             .evaluate())
                peaks = find_peaks(p=p,
                                   min_peak_threshold=self.method_kwargs.get("min_peak_threshold", 0.05),
                                   peak_boundary=self.method_kwargs.get("peak_boundary", 0.1))
                assert len(peaks) > 0, "No peaks detected"
                if len(peaks) == 1:
                    thresholds.append(self._process_one_peak(d=d,
                                                             data=data,
                                                             x_grid=x_grid,
                                                             p=p,
                                                             peak_idx=peaks[0]))
                elif len(peaks) == 2:
                    thresholds.append(find_local_minima(p=p, x=x_grid, peaks=peaks))
                else:
                    smoothed_peak_finding_kwargs = self.method_kwargs.get("smoothed_peak_finding_kwargs", {})
                    p, peaks = smoothed_peak_finding(p=p, **smoothed_peak_finding_kwargs)
                    if len(peaks) == 1:
                        thresholds.append(self._process_one_peak(d=d,
                                                                 data=data,
                                                                 x_grid=x_grid,
                                                                 p=p,
                                                                 peak_idx=peaks[0]))
                    else:
                        thresholds.append(find_local_minima(p=p, x=x_grid, peaks=peaks))
        return thresholds, data

    def _manual(self) -> List[float]:
        """
        Wrapper called if manual gating method. Searches the method kwargs and returns static thresholds

        Returns
        -------
        list
        """
        thresholds = [i for i in [self.method_kwargs.get("x_threshold", None),
                                  self.method_kwargs.get("y_threshold", None)] if i is not None]
        assert len(thresholds) > 0, "For manual gating you must provide x_threshold and/or y_threshold"
        assert all([isinstance(i, float) for i in thresholds]), "Thresholds must be floating point values"
        return thresholds

    def fit(self,
            data: pd.DataFrame) -> None or (list, pd.DataFrame):
        """
        Fit the gate using a given dataframe. If children already exist will raise an AssertionError
        and notify user to call `fit_predict`.

        Parameters
        ----------
        data: Pandas.DataFrame
            Population data to fit threshold too

        Returns
        -------
        None
        """
        data = data.copy()
        assert len(self.children) == 0, "Children already defined for this gate. Call 'fit_predict' to " \
                                        "fit to new data and match populations to children, or call " \
                                        "'predict' to apply static thresholds to new data. If you want to " \
                                        "reset the gate and call 'fit' again, first call 'reset_gate'"
        thresholds, _ = self._fit(data=data)
        dims = [i for i in [self.x, self.y] if i is not None]
        if len(dims) == 1:
            for definition in ["+", "-"]:
                self.add_child(ChildThreshold(name=definition,
                                              definition=definition,
                                              geom=ThresholdGeom(x_threshold=thresholds[0])))
        else:
            for definition in ["++", "--", "-+", "+-"]:
                self.add_child(ChildThreshold(name=definition,
                                              definition=definition,
                                              geom=ThresholdGeom(x_threshold=thresholds[0],
                                                                 y_threshold=thresholds[1])))
        return None

    def fit_predict(self,
                    data: pd.DataFrame,
                    parent: str) -> List[Population]:
        """
        Fit the gate using a given dataframe and then associate predicted Population objects to
        existing children. If no children exist, an AssertionError will be raised prompting the
        user to call `fit` method.

        Parameters
        ----------
        data: Pandas.DataFrame
            Population data to fit threshold too
        parent: str
            Name of the parent population that gate is being applied to

        Returns
        -------
        list
            List of predicted Population objects, labelled according to the gates child objects
        """
        data = data.copy()
        thresholds, fitted_data = self._fit(data=data)
        y_threshold = None
        if len(thresholds) == 2:
            y_threshold = thresholds[1]
        results = apply_threshold(data=fitted_data,
                                  x=self.x,
                                  y=self.y,
                                  x_threshold=thresholds[0],
                                  y_threshold=y_threshold)
        pops = self._generate_populations(data=results, parent=parent)
        if self.sampling.get("method", None) is not None:
            pops = self._upsample(data=data, sample=fitted_data, populations=pops)
        return self._match_to_children(new_populations=pops)

    def predict(self,
                data: pd.DataFrame,
                parent: str) -> List[Population]:
        """
        Using existing children associated to this gate, the previously calculated thresholds of
        these children will be applied to the given data and then Population objects created and
        labelled to match the children of this gate. NOTE: the data will not be fitted and thresholds
        applied will be STATIC not data driven. For data driven gates call `fit_predict` method.

        Parameters
        ----------
        data: Pandas.DataFrame
            Data to apply static thresholds too
        parent: str
            Name of the parent population that gate is being applied to

        Returns
        -------
        list
            List of Population objects
        """
        assert len(self.children) > 0, "Must call 'fit' prior to predict"
        pops = list()
        self._xy_in_dataframe(data=data)
        data = self._transform(data=data)
        data = self._dim_reduction(data=data)
        if self.y is not None:
            data = threshold_2d(data=data,
                                x=self.x,
                                y=self.y,
                                x_threshold=self.children[0].geom.x_threshold,
                                y_threshold=self.children[0].geom.y_threshold)
        else:
            data = threshold_1d(data=data, x=self.x, x_threshold=self.children[0].geom.x_threshold)
        return self._generate_populations(data=data, parent=parent)

    def _generate_populations(self,
                              data: dict,
                              parent: str) -> List[Population]:
        """
        Generate populations from a standard dictionary of dataframes that have had thesholds applied.

        Parameters
        ----------
        data: Pandas.DataFrame
        parent: str

        Returns
        -------
        list
            List of Population objects
        """
        pops = list()
        for definition, df in data.items():
            pops.append(Population(population_name=definition,
                                   definition=definition,
                                   parent=parent,
                                   n=df.shape[0],
                                   index=df.index.values,
                                   signature=create_signature(data=df),
                                   geom=ThresholdGeom(x=self.x,
                                                      y=self.y,
                                                      transform_x=self.transformations.get("x", None),
                                                      transform_y=self.transformations.get("y", None),
                                                      x_threshold=self.children[0].geom.x_threshold,
                                                      y_threshold=self.children[0].geom.y_threshold)))
        return pops


class PolygonGate(Gate):
    """
    Polygon gates generate polygon shapes that capture populations of varying shapes. These can
    be generated by any number of clustering algorithms.
    """
    children = mongoengine.EmbeddedDocumentListField(ChildPolygon)

    def _generate_populations(self,
                              data: dict,
                              parent: str) -> List[Population]:
        """
        Generate populations from a standard dictionary of dataframes that have had polygon gate(s) applied.

        Parameters
        ----------
        data: Pandas.DataFrame
        parent: str

        Returns
        -------
        list
            List of Population objects
        """
        pass

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

    def _manual(self) -> ShapelyPoly:
        """
        Wrapper for manual polygon gating. Searches method kwargs for x and y coordinates and returns
        polygon.

        Returns
        -------
        Shapely.geometry.Polygon
        """
        pass

    def _fit(self,
             data: pd.DataFrame) -> List[ShapelyPoly]:
        """
        Internal method for fitting gate to the given data and returning geometric polygons for
        captured populations.

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        list
            List of Shapely polygon's
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

    def _manual(self) -> ShapelyPoly:
        """
        Wrapper for manual elliptical gating. Searches method kwargs for x and y coordinates and returns
        polygon.

        Returns
        -------
        Shapely.geometry.Polygon
        """
        pass

    def _fit(self,
             data: pd.DataFrame) -> List[ShapelyPoly]:
        """
        Internal method for fitting gate to the given data and returning geometric polygons for
        captured populations.

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        list
            List of Shapely polygon's
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


def merge_children(children: list) -> Child or ChildThreshold or ChildPolygon:
    """
    Given a list of Child objects, merge and return single child

    Parameters
    ----------
    children: list

    Returns
    -------
    Child or ChildThreshold or ChildPolygon
    """
    assert len(set([type(x) for x in children])) == 1, \
        f"Children must be of same type; not, {[type(x) for x in children]}"
    assert len(set([c.name for c in children])), "Children should all have the same name"
    if isinstance(children[0], ChildThreshold):
        definition = ",".join([c.definition for c in children])
        return ChildThreshold(name=children[0].name,
                              definition=definition,
                              geom=children[0].geom)
    if isinstance(children[0], ChildPolygon):
        x = np.unique(np.concatenate([np.array(c.geom.x_values) for c in children], axis=0), axis=0).tolist()
        y = np.unique(np.concatenate([np.array(c.geom.y_values) for c in children], axis=0), axis=0).tolist()
        return ChildPolygon(name=children[0].name,
                            geom=PolygonGeom(x=children[0].geom.x,
                                             y=children[0].geom.y,
                                             transform_x=children[0].geom.transform_x,
                                             transform_y=children[0].geom.transform_y,
                                             x_values=x,
                                             y_values=y))
    return children[0]


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


def apply_threshold(data: pd.DataFrame,
                    x: str,
                    x_threshold: float,
                    y: str or None = None,
                    y_threshold: float or None = None) -> Dict[str, pd.DataFrame]:
    """
    Simple wrapper for threshold_1d and threhsold_2d

    Parameters
    ----------
    data: Pandas.DataFrame
    x: str
    x_threshold: float
    y: str (optional)
    y_threshold: float (optional)

    Returns
    -------
    dict
    """
    if y is not None:
        return threshold_2d(data=data,
                            x=x,
                            y=y,
                            x_threshold=x_threshold,
                            y_threshold=y_threshold)
    else:
        return threshold_1d(data=data, x=x, x_threshold=x_threshold)


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
    data = data.copy()
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
    data = data.copy()
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
    peaks = detect_peaks(p,
                         mph=p[np.argmax(p)] * min_peak_threshold,
                         mpd=len(p) * peak_boundary)
    return peaks


def smoothed_peak_finding(p: np.array,
                          starting_window_length: int = 11,
                          polyorder: int = 3,
                          min_peak_threshold: float = 0.05,
                          peak_boundary: float = 0.1,
                          **kwargs) -> (np.array, np.array):
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
    p: np.array
        Probability vector resulting from KDE calculation
    starting_window_length: int (default=11)
        Window length of filter (must be > length of p, < length of p * 0.5, and an odd number)
    polyorder: int (default=3)
        Order of polynomial for filter
    min_peak_threshold: float (default=0.05)
        See CytoPy.data.gate.find_peaks
    peak_boundary: float (default=0.1)
        See CytoPy.data.gate.find_peaks
    kwargs: dict
        Additional keyword arguments to pass to scipy.signal.savgol_filter

    Returns
    -------
    np.array, np.array
        Smooth probability vector and index of peaks
    """
    smoothed = p.copy()
    window = starting_window_length
    while len(find_peaks(smoothed, min_peak_threshold, peak_boundary)) >= 3:
        if window >= len(smoothed) * .5:
            raise ValueError("Stable window size exceeded")
        smoothed = savgol_filter(smoothed, window, polyorder, **kwargs)
        window += 10
    return smoothed, find_peaks(smoothed, min_peak_threshold, peak_boundary)


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
                          window_size: int or None = None,
                          polyorder: int = 3,
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
    window_size: int (optional)
        Window length of filter (must be an odd number). If not given then it is calculated as an
        odd integer nearest to a 10th of the grid length
    polyorder: int (default=3)
        Polynomial order for Savitzky-Golay filter
    kwargs: dict
        Additional keyword argument to pass to scipy.signal.savgol_filter

    Returns
    -------
    float
        Value of x at which the inflection point occurs
    """
    window_size = window_size or int(len(x) * .25)
    if window_size % 2 == 0:
        window_size += 1
    smooth = savgol_filter(p, window_size, polyorder, **kwargs)
    if incline:
        ddy = np.diff(np.diff(smooth[:peak_idx]))
    else:
        ddy = np.diff(np.diff(smooth[peak_idx:]))
    if incline:
        return float(x[np.argmax(ddy)])
    return float(x[peak_idx + np.argmax(ddy)])


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
