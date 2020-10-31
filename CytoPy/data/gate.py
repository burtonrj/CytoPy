#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Gates are traditionally used to subset single cell data in one
or two dimensional space by hand-drawn polygons in a manual and laborious
process. CytoPy attempts to emulate this using autonomous gates, driven
by unsupervised learning algorithms. The gate module contains the
classes that provide the infrastructure to apply these algorithms
to the context of single cell data whilst interacting with the underlying
database that houses our analysis.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from ..flow.transforms import apply_transform
from .geometry import ThresholdGeom, PolygonGeom, inside_polygon, \
    create_convex_hull, create_polygon, polygon_overlap, ellipse_to_polygon, \
    probablistic_ellipse
from .population import Population, merge_multiple_populations, create_signature
from ..flow.sampling import faithful_downsampling, density_dependent_downsampling, upsample_knn
from ..flow.dim_reduction import dimensionality_reduction
from shapely.geometry import Polygon as ShapelyPoly
from shapely.ops import cascaded_union
from sklearn.cluster import *
from sklearn.mixture import *
from hdbscan import HDBSCAN
from warnings import warn
from scipy.spatial.distance import euclidean
from string import ascii_uppercase
from collections import Counter
from typing import List, Dict
from KDEpy import FFTKDE
from detecta import detect_peaks
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import mongoengine

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


class Child(mongoengine.EmbeddedDocument):
    """
    Base class for a gate child population. This is representative of the 'population' of cells
    identified when a gate is first defined and will be used as a template to annotate
    the populations identified in new data.
    """
    name = mongoengine.StringField()
    meta = {"allow_inheritance": True}


class ChildThreshold(Child):
    """
    Child population of a Threshold gate. This is representative of the 'population' of cells
    identified when a gate is first defined and will be used as a template to annotate
    the populations identified in new data.

    Attributes
    -----------
    name: str
        Name of the child
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
    Child population of a Polgon or Ellipse gate. This is representative of the 'population' of cells
    identified when a gate is first defined and will be used as a template to annotate
    the populations identified in new data.

    Attributes
    -----------
    name: str
        Name of the child
    geom: ChildPolygon
        Geometric definition for this child population
    signature: dict
        Average of a population feature space (median of each channel); used to match
        children to newly identified populations for annotating
    """
    geom = mongoengine.EmbeddedDocumentField(PolygonGeom)
    signature = mongoengine.DictField()


class Gate(mongoengine.Document):
    """
    Base class for a Gate. A Gate attempts to separate single cell data in one or
    two-dimensional space using unsupervised learning algorithms. The algorithm is fitted
    to example data to generate "children"; the populations of cells a user expects to
    identify. These children are stored and then when the gate is 'fitted' to new data,
    the resulting populations are matched to the expected children.

    Attributes
    -----------
    gate_name: str (required)
        Name of the gate
    parent: str (required)
        Parent population that this gate is applied to
    x: str (required)
        Name of the x-axis variable forming the one/two dimensional space this gate
        is applied to
    y: str (optional)
        Name of the y-axis variable forming the two dimensional space this gate
        is applied to
    transformations: dict (optional)
        Transform method to be applied to each dimension, should be a dictionary with
        keys corresponding to each variable (e.g. "x" and/or "y") and values the
        transform to apply (e.g. {"x": "logicle"} for logicle transform of x-axis)
    sampling: dict (optional)
         Options for downsampling data prior to application of gate. Should contain a
         key/value pair for desired method e.g ({"method": "uniform"). Available methods
         are: 'uniform', 'density' or 'faithful'. See CytoPy.flow.sampling for details. Additional
         keyword arguments should be provided in the sampling dictionary.
    dim_reduction: dict (optional)
        Experimental feature. Allows for dimension reduction to be performed prior to
        applying gate. Gate will be applied to the resulting embeddings. Provide a dictionary
        with a key "method" and the value as any supported method in CytoPy.flow.dim_reduction.
        Additional keyword arguments should be provided in this dictionary.
    ctrl: str (optional)
        If a value is given here it should be the name of a control specimen commonly associated
        to the samples in an Experiment. When given this signals that the gate is designed to
        be applied to the control data not the primary data.
    method: str (required)
        Name of the underlying algorithm to use. Should have a value of: "manual", "density",
        "quantile" or correspond to the name of an existing class in Scikit-Learn or HDBSCAN.
        If you have a method that follows the Scikit-Learn template but isn't currently present
        in CytoPy and you would like it to be, please contribute to the respository on GitHub
        or contact burtonrj@cardiff.ac.uk
    method_kwargs: dict
        Keyword arguments for initiation of the above method.
    """
    gate_name = mongoengine.StringField(required=True)
    parent = mongoengine.StringField(required=True)
    x = mongoengine.StringField(required=True)
    y = mongoengine.StringField(required=False)
    transformations = mongoengine.DictField()
    sampling = mongoengine.DictField()
    dim_reduction = mongoengine.DictField()
    ctrl = mongoengine.StringField()
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
        if method not in ["manual", "density", "quantile"]:
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
                return data.sample(frac=n)
            else:
                raise ValueError("Sampling parameter 'n' must be an integer or float")
        if self.sampling.get("method", None) == "density":
            kwargs = {k: v for k, v in self.sampling.items()
                      if k not in  ["method", "features"]}
            features = [f for f in [self.x, self.y] if f is not None]
            return density_dependent_downsampling(data=data,
                                                  features=features,
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
        Perform up-sampling after gating using KNN. Returns list of Population objects
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


class ThresholdGate(Gate):
    """
    ThresholdGate inherits from Gate. A Gate attempts to separate single cell data in one or
    two-dimensional space using unsupervised learning algorithms. The algorithm is fitted
    to example data to generate "children"; the populations of cells a user expects to
    identify. These children are stored and then when the gate is 'fitted' to new data,
    the resulting populations are matched to the expected children.

    The ThresholdGate subsets data based on the properties of the estimated probability
    density function of the underlying data. For each axis, kernel density estimation
    (KDEpy.FFTKDE) is used to estimate the PDF and a straight line "threshold" applied
    to the region of minimum density to separate populations.
    This is achieved using a peak finding algorithm and a smoothing procedure, until either:
        * Two predominant "peaks" are found and the threshold is taken as the local minima
        between there peaks
        * A single peak is detected and the threshold is applied as either the quantile
        given in method_kwargs or the inflection point on the descending curve.

    Alternatively the "method" can be "manual" for a static gate to be applied; user should
    provide x_threshold and y_threshold (if two-dimensional) to "method_kwargs", or "method"
    can be "quantile", where the threshold will be drawn at the given quantile, defined by
    "q" in "method_kwargs".

    Additional kwargs to control behaviour of ThresholdGate when method is "density"
    can be given in method_kwargs:
        *  kernel (default="guassian") - kernel used for KDE calculation
        (see KDEpy.FFTKDE for avialable kernels)
        * bw (default="silverman") - bandwidth to use for KDE calculation, can either be
        "siilverman" or "ISJ" or a float value (see KDEpy)
        * min_peak_threshold (default=0.05) - percentage of highest recorded peak below
        which peaks are ignored. E.g. 0.05 would mean any peak less than 5% of the
        highest peak would be ignored.
        * peak_boundary (default=0.1) - bounding window around which only the highest peak
        is considered. E.g. 0.1 would mean that peaks are assessed within a window the
        size of peak_boundary * length of probability vector and only highest peak within
        window is kept.
        inflection_point_kwargs - dictionary; see CytoPy.data.gate.find_inflection_point
        smoothed_peak_finding_kwargs - dictionary; see CytoPy.data.gate.smoothed_peak_finding

    Attributes
    -----------
    gate_name: str (required)
        Name of the gate
    parent: str (required)
        Parent population that this gate is applied to
    x: str (required)
        Name of the x-axis variable forming the one/two dimensional space this gate
        is applied to
    y: str (optional)
        Name of the y-axis variable forming the two dimensional space this gate
        is applied to
    transformations: dict (optional)
        Transform method to be applied to each dimension, should be a dictionary with
        keys corresponding to each variable (e.g. "x" and/or "y") and values the
        transform to apply (e.g. {"x": "logicle"} for logicle transform of x-axis)
    sampling: dict (optional)
         Options for downsampling data prior to application of gate. Should contain a
         key/value pair for desired method e.g ({"method": "uniform"). Available methods
         are: 'uniform', 'density' or 'faithful'. See CytoPy.flow.sampling for details. Additional
         keyword arguments should be provided in the sampling dictionary.
    dim_reduction: dict (optional)
        Experimental feature. Allows for dimension reduction to be performed prior to
        applying gate. Gate will be applied to the resulting embeddings. Provide a dictionary
        with a key "method" and the value as any supported method in CytoPy.flow.dim_reduction.
        Additional keyword arguments should be provided in this dictionary.
    ctrl: str (optional)
        If a value is given here it should be the name of a control specimen commonly associated
        to the samples in an Experiment. When given this signals that the gate is designed to
        be applied to the control data not the primary data.
    method: str (required)
        Name of the underlying algorithm to use. Should have a value of: "manual", "density",
        or "quantile"
    method_kwargs: dict
        Keyword arguments for initiation of the above method.
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

    def _duplicate_children(self) -> None:
        """
        Loop through the children and merge any with the same name.

        Returns
        -------
        None
        """
        child_counts = Counter([c.name for c in self.children])
        if all([i == 1 for i in child_counts.values()]):
            return
        updated_children = []
        for name, count in child_counts.items():
            if count >= 2:
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
        List
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
             data: pd.DataFrame) -> list:
        """
        Internal method to fit threshold density gating to a given dataframe. Returns the
        list of thresholds generated and the dataframe the threshold were generated from
        (will be the downsampled dataframe if sampling methods defined).

        Parameters
        ----------
        data: Pandas.DataFrame

        Returns
        -------
        List
        """
        if self.method == "manual":
            return self._manual()
        thresholds = list()
        self._xy_in_dataframe(data=data)
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
        return thresholds

    def _manual(self) -> list:
        """
        Wrapper called if manual gating method. Searches the method kwargs and returns static thresholds

        Returns
        -------
        List
        """
        thresholds = [i for i in [self.method_kwargs.get("x_threshold", None),
                                  self.method_kwargs.get("y_threshold", None)] if i is not None]
        assert len(thresholds) > 0, "For manual gating you must provide x_threshold and/or y_threshold"
        assert all([isinstance(i, float) for i in thresholds]), "Thresholds must be floating point values"
        return thresholds

    def fit(self,
            data: pd.DataFrame) -> None:
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
        data = self._transform(data=data)
        data = self._dim_reduction(data=data)
        assert len(self.children) == 0, "Children already defined for this gate. Call 'fit_predict' to " \
                                        "fit to new data and match populations to children, or call " \
                                        "'predict' to apply static thresholds to new data. If you want to " \
                                        "reset the gate and call 'fit' again, first call 'reset_gate'"
        thresholds = self._fit(data=data)
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
                    data: pd.DataFrame) -> list:
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
        List
            List of predicted Population objects, labelled according to the gates child objects
        """
        assert len(self.children) > 0, "No children defined for gate, call 'fit' before calling 'fit_predict'"
        data = data.copy()
        data = self._transform(data=data)
        data = self._dim_reduction(data=data)
        thresholds = self._fit(data=data)
        y_threshold = None
        if len(thresholds) == 2:
            y_threshold = thresholds[1]
        results = apply_threshold(data=data,
                                  x=self.x,
                                  y=self.y,
                                  x_threshold=thresholds[0],
                                  y_threshold=y_threshold)
        pops = self._generate_populations(data=results,
                                          x_threshold=thresholds[0],
                                          y_threshold=y_threshold)
        return self._match_to_children(new_populations=pops)

    def predict(self,
                data: pd.DataFrame) -> list:
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
        List
            List of Population objects
        """
        assert len(self.children) > 0, "Must call 'fit' prior to predict"
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
        return self._generate_populations(data=data,
                                          x_threshold=self.children[0].geom.x_threshold,
                                          y_threshold=self.children[0].geom.y_threshold)

    def _generate_populations(self,
                              data: dict,
                              x_threshold: float,
                              y_threshold: float or None) -> list:
        """
        Generate populations from a standard dictionary of dataframes that have had thesholds applied.

        Parameters
        ----------
        data: Pandas.DataFrame
        x_threshold: float
        y_threshold: float (optional)

        Returns
        -------
        List
            List of Population objects
        """
        pops = list()
        for definition, df in data.items():
            pops.append(Population(population_name=definition,
                                   definition=definition,
                                   parent=self.parent,
                                   n=df.shape[0],
                                   index=df.index.values,
                                   signature=create_signature(data=df),
                                   geom=ThresholdGeom(x=self.x,
                                                      y=self.y,
                                                      transform_x=self.transformations.get("x", None),
                                                      transform_y=self.transformations.get("y", None),
                                                      x_threshold=x_threshold,
                                                      y_threshold=y_threshold)))
        return pops


class PolygonGate(Gate):
    """
    PolygonGate inherits from Gate. A Gate attempts to separate single cell data in one or
    two-dimensional space using unsupervised learning algorithms. The algorithm is fitted
    to example data to generate "children"; the populations of cells a user expects to
    identify. These children are stored and then when the gate is 'fitted' to new data,
    the resulting populations are matched to the expected children.

    The PolygonGate subsets data based on the results of an unsupervised learning algorithm
    such a clustering algorithm. PolygonGate supports any clustering algorithm from the
    Scikit-Learn machine learning library. Support is extended to any clustering library
    that follows the Scikit-Learn template, but currently this only includes HDBSCAN.
    Contributions to extend to other libraries are welcome. The name of the class to use
    should be provided in "method" along with keyword arguments for initiating this class
    in "method_kwargs".

    Alternatively the "method" can be "manual" for a static gate to be applied; user should
    provide x_values and y_values (if two-dimensional) to "method_kwargs" as two arrays,
    this will be interpreted as the x and y coordinates of the polygon to fit to the data.

    Attributes
    -----------
    gate_name: str (required)
        Name of the gate
    parent: str (required)
        Parent population that this gate is applied to
    x: str (required)
        Name of the x-axis variable forming the one/two dimensional space this gate
        is applied to
    y: str (required)
        Name of the y-axis variable forming the two dimensional space this gate
        is applied to
    transformations: dict (optional)
        Transform method to be applied to each dimension, should be a dictionary with
        keys corresponding to each variable (e.g. "x" and/or "y") and values the
        transform to apply (e.g. {"x": "logicle"} for logicle transform of x-axis)
    sampling: dict (optional)
         Options for downsampling data prior to application of gate. Should contain a
         key/value pair for desired method e.g ({"method": "uniform"). Available methods
         are: 'uniform', 'density' or 'faithful'. See CytoPy.flow.sampling for details. Additional
         keyword arguments should be provided in the sampling dictionary.
    dim_reduction: dict (optional)
        Experimental feature. Allows for dimension reduction to be performed prior to
        applying gate. Gate will be applied to the resulting embeddings. Provide a dictionary
        with a key "method" and the value as any supported method in CytoPy.flow.dim_reduction.
        Additional keyword arguments should be provided in this dictionary.
    ctrl: str (optional)
        If a value is given here it should be the name of a control specimen commonly associated
        to the samples in an Experiment. When given this signals that the gate is designed to
        be applied to the control data not the primary data.
    method: str (required)
        Name of the underlying algorithm to use. Should have a value of: "manual", or correspond
        to the name of an existing class in Scikit-Learn or HDBSCAN.
        If you have a method that follows the Scikit-Learn template but isn't currently present
        in CytoPy and you would like it to be, please contribute to the respository on GitHub
        or contact burtonrj@cardiff.ac.uk
    method_kwargs: dict
        Keyword arguments for initiation of the above method.
    """
    children = mongoengine.EmbeddedDocumentListField(ChildPolygon)

    def __init__(self, *args, **values):
        super().__init__(*args, **values)
        assert self.y is not None, "Polygon gate expects a y-axis variable"

    def _generate_populations(self,
                              data: pd.DataFrame,
                              polygons: List[ShapelyPoly]) -> List[Population]:
        """
        Given a dataframe and a list of Polygon shapes as generated from the '_fit' method, generate a
        list of Population objects.

        Parameters
        ----------
        data: Pandas.DataFrame
        polygons: list

        Returns
        -------
        List
            List of Population objects
        """
        pops = list()
        for name, poly in zip(ascii_uppercase, polygons):
            pop_df = inside_polygon(df=data, x=self.x, y=self.y, poly=poly)
            geom = PolygonGeom(x=self.x,
                               y=self.y,
                               transform_x=self.transformations.get("x", None),
                               transform_y=self.transformations.get("y", None),
                               x_values=poly.exterior.xy[0],
                               y_values=poly.exterior.xy[1])
            pops.append(Population(population_name=name,
                                   parent=self.parent,
                                   n=pop_df.shape[0],
                                   geom=geom,
                                   index=pop_df.index.values,
                                   signature=create_signature(pop_df)))
        return pops

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
        assert len(set(labels.values())) == len(labels.values()), \
            "Duplicate labels provided. Child merging not available for polygon gates"
        if drop:
            self.children = [c for c in self.children if c.name in labels.keys()]
        for c in self.children:
            c.name = labels.get(c.name)

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
        child.geom.x = self.x
        child.geom.y = self.y
        child.geom.transform_x = self.transformations.get("x", None)
        child.geom.transform_y = self.transformations.get("y", None)
        assert isinstance(child.geom.x_values, list), "ChildPolygon x_values should be of type list"
        assert isinstance(child.geom.y_values, list), "ChildPolygon y_values should be of type list"
        self.children.append(child)

    def _match_to_children(self,
                           new_populations: List[Population]) -> List[Population]:
        """
        Given a list of newly create Populations, match the Populations to the gates children and
        return list of Populations with correct population names. Populations are matched to children
        based on a comparison of their signatures and the % overlap between their geometries.
        Similarity score = fraction area overlap * absolute euclidean distance between signatures.

        Parameters
        -----------
        new_populations: list
            List of newly created Population objects

        Returns
        -------
        List
        """
        matched_populations = list()
        for child in self.children:
            scores = [_child_similarity_score(child=child, population=pop)
                      for pop in new_populations]
            matching_population = new_populations[int(np.argmax(scores))]
            matching_population.population_name = child.name
            matched_populations.append(matching_population)
        return matched_populations

    def _manual(self) -> ShapelyPoly:
        """
        Wrapper for manual polygon gating. Searches method kwargs for x and y coordinates and returns
        polygon.

        Returns
        -------
        Shapely.geometry.Polygon
        """
        x_values, y_values = self.method_kwargs.get("x_values", None), self.method_kwargs.get("y_values", None)
        assert x_values is not None and y_values is not None, "For manual polygon gate must provide x_values and " \
                                                              "y_values"
        return create_polygon(x_values, y_values)

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
        List
            List of Shapely polygon's
        """
        if self.method == "manual":
            return [self._manual()]
        self._xy_in_dataframe(data=data)
        if self.sampling.get("method", None) is not None:
            data = self._downsample(data=data)
        labels = self.model.fit_predict(data[[self.x, self.y]])
        return [create_polygon(*create_convex_hull(x_values=data.iloc[np.where(labels == i)][self.x].values,
                                                   y_values=data.iloc[np.where(labels == i)][self.y].values))
                for i in np.unique(labels)]

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
        assert len(self.children) == 0, "Gate is already defined, call 'reset_gate' to clear children"
        data = self._transform(data=data)
        data = self._dim_reduction(data=data)
        polygons = self._fit(data=data)
        for name, poly in zip(ascii_uppercase, polygons):
            poly_df = inside_polygon(df=data,
                                     x=self.x,
                                     y=self.y,
                                     poly=poly)
            self.add_child(ChildPolygon(name=name,
                                        signature=create_signature(data=poly_df),
                                        geom=PolygonGeom(x_values=poly.exterior.xy[0].tolist(),
                                                         y_values=poly.exterior.xy[1].tolist())))

    def fit_predict(self,
                    data: pd.DataFrame) -> List[Population]:
        """
        Fit the gate using a given dataframe and then associate predicted Population objects to
        existing children. If no children exist, an AssertionError will be raised prompting the
        user to call 'fit' method.

        Parameters
        ----------
        data: Pandas.DataFrame
            Population data to fit gate to

        Returns
        -------
        List
            List of predicted Population objects, labelled according to the gates child objects
        """
        assert len(self.children) > 0, "No children defined for gate, call 'fit' before calling 'fit_predict'"
        data = self._transform(data=data)
        data = self._dim_reduction(data=data)
        return self._match_to_children(self._generate_populations(data=data.copy(),
                                                                  polygons=self._fit(data=data)))

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
        List
            List of Population objects
        """
        data = self._transform(data=data)
        data = self._dim_reduction(data=data)
        polygons = [create_polygon(c.geom.x_values, c.geom.y_values) for c in self.children]
        populations = self._generate_populations(data=data, polygons=polygons)
        for p, name in zip(populations, [c.name for c in self.children]):
            p.population_name = name
        return populations


class EllipseGate(PolygonGate):
    """
    EllipseGate inherits from PolygonGate. A Gate attempts to separate single cell data in one or
    two-dimensional space using unsupervised learning algorithms. The algorithm is fitted
    to example data to generate "children"; the populations of cells a user expects to
    identify. These children are stored and then when the gate is 'fitted' to new data,
    the resulting populations are matched to the expected children.

    The EllipseGate uses probabilistic mixture models to subset data into "populations". For
    each component of the mixture model the covariance matrix is used to generate a confidence
    ellipse, surrounding data and emulating a gate. EllipseGate can use any of the methods
    from the Scikit-Learn mixture module. Keyword arguments for the initiation of a class
    from this module can be given in "method_kwargs".

    Attributes
    -----------
    gate_name: str (required)
        Name of the gate
    parent: str (required)
        Parent population that this gate is applied to
    x: str (required)
        Name of the x-axis variable forming the one/two dimensional space this gate
        is applied to
    y: str (required)
        Name of the y-axis variable forming the two dimensional space this gate
        is applied to
    transformations: dict (optional)
        Transform method to be applied to each dimension, should be a dictionary with
        keys corresponding to each variable (e.g. "x" and/or "y") and values the
        transform to apply (e.g. {"x": "logicle"} for logicle transform of x-axis)
    sampling: dict (optional)
         Options for downsampling data prior to application of gate. Should contain a
         key/value pair for desired method e.g ({"method": "uniform"). Available methods
         are: 'uniform', 'density' or 'faithful'. See CytoPy.flow.sampling for details. Additional
         keyword arguments should be provided in the sampling dictionary.
    dim_reduction: dict (optional)
        Experimental feature. Allows for dimension reduction to be performed prior to
        applying gate. Gate will be applied to the resulting embeddings. Provide a dictionary
        with a key "method" and the value as any supported method in CytoPy.flow.dim_reduction.
        Additional keyword arguments should be provided in this dictionary.
    method: str (required)
        Name of the underlying algorithm to use. Should have a value of: "manual", or correspond
        to the name of an existing class in Scikit-Learn mixture module..
        If you have a method that follows the Scikit-Learn template but isn't currently present
        in CytoPy and you would like it to be, please contribute to the repository on GitHub
        or contact burtonrj@cardiff.ac.uk
    method_kwargs: dict
        Keyword arguments for initiation of the above method.
    """
    children = mongoengine.EmbeddedDocumentListField(ChildPolygon)

    def __init__(self, *args, **values):
        method = values.get("method", None)
        self.conf = values.get("method_kwargs", {}).pop("conf", 0.95)
        valid = ["manual", "GaussianMixture", "BayesianGaussianMixture"]
        assert method in valid, f"Elliptical gating method should be one of {valid}"
        super().__init__(*args, **values)

    def _manual(self) -> ShapelyPoly:
        """
        Wrapper for manual elliptical gating. Searches method kwargs for centroid, width, height, and angle,
        and returns polygon.

        Returns
        -------
        Shapely.geometry.Polygon
        """
        centroid = self.method_kwargs.get("centroid", None)
        width = self.method_kwargs.get("width", None)
        height = self.method_kwargs.get("height", None)
        angle = self.method_kwargs.get("angle", None)
        assert all([x is not None for x in [centroid, width, height, angle]]), \
            "Manual elliptical gate requires the following keyword arguments; width, height, angle and centroid"
        assert len(centroid) == 2 and all(isinstance(x, float) for x in centroid), \
            "Centroid should be a list of two float values"
        assert all(isinstance(x, float) for x in [width, height, angle]), \
            "Width, height, and angle should be of type float"
        return ellipse_to_polygon(centroid=centroid,
                                  width=width,
                                  height=height,
                                  angle=angle)

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
        if not self.method_kwargs.get("probabilistic_ellipse", True):
            return super()._fit(data=data)
        self._xy_in_dataframe(data=data)
        if self.sampling.get("method", None) is not None:
            data = self._downsample(data=data)
        self.model.fit_predict(data[[self.x, self.y]])
        ellipses = [probablistic_ellipse(covar, conf=self.conf)
                    for covar in self.model.covariances_]
        polygons = [ellipse_to_polygon(centroid, *ellipse)
                    for centroid, ellipse in zip(self.model.means_, ellipses)]
        return polygons


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
        merged_poly = cascaded_union([c.geom.shape for c in children])
        new_signature = pd.DataFrame([c.signature for c in children]).mean().to_dict()
        x, y = merged_poly.exterior.xy[0], merged_poly.exterior.xy[1]
        return ChildPolygon(name=children[0].name,
                            signature=new_signature,
                            geom=PolygonGeom(x=children[0].geom.x,
                                             y=children[0].geom.y,
                                             transform_x=children[0].geom.transform_x,
                                             transform_y=children[0].geom.transform_y,
                                             x_values=x,
                                             y_values=y))
    return children[0]


def _child_similarity_score(child: ChildPolygon,
                            population: Population) -> dict:
    """
    Compare a population to a ChildPolygon contained and return the similarity score.
    The similarity score is given by: fraction area overlap * (1 - absolute euclidean distance between signatures).
    This generates a score between 0 and 1, where 1 is identical populations and 0 is absolutely
    different populations.

    Parameters
    ----------
    child: ChildPolygon
    population: Population

    Returns
    -------
    str
    """
    assert isinstance(population.geom, PolygonGeom), "Similarity score is only for comparison of Polygon type gates"
    area = polygon_overlap(child.geom.shape, population.geom.shape)
    common_features = set(population.signature.keys()).intersection(child.signature.keys())
    vector_signatures = np.array([[population.signature.get(i), child.signature.get(i)]
                                  for i in common_features]).T
    dist_score = 1. - abs(euclidean(vector_signatures[:, 0], vector_signatures[:, 1]))
    return dist_score * area


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


def _reset_population(population: Population):
    """
    When a population geometry is updated, this function removes associated
    clusters and control indexes.

    Parameters
    ----------
    population: Population

    Returns
    -------
    None
    """
    if len(population.list_clusters()) > 0:
        warn(f"{population.population_name} has associated clusters which will be removed")
        population.delete_all_clusters()
    for ctrl_id in population.ctrl_index.keys():
        warn(f"{ctrl_id} index for {population.population_name} will be reset")
        population.ctrl_index.pop(ctrl_id)


def update_threshold(population: Population,
                     parent_data: pd.DataFrame,
                     x_threshold: float,
                     y_threshold: float or None = None):
    """
    Given an existing population and some new threshold(s) (different to what is already
    associated to the Population), update the Population index and geom accordingly.
    Any associated clusters will be removed and any controls will have to be estimated
    again.

    Parameters
    ----------
    population: Population
    parent_data: Pandas.DataFrame
    x_threshold: float
    y_threshold: float (optional)
        Required if 2D threshold geometry

    Returns
    -------
    None
    """
    if population.geom.y_threshold is None:
        new_data = threshold_1d(data=parent_data,
                                x=population.geom.x,
                                x_threshold=x_threshold).get(population.definition)
        population.index = new_data.index.values
        population.geom.x_threshold = x_threshold
    else:
        assert y_threshold is not None, "2D threshold requires y_threshold"
        new_data = threshold_2d(data=parent_data,
                                x=population.geom.x,
                                x_threshold=x_threshold,
                                y=population.geom.y,
                                y_threshold=y_threshold)
        definitions = population.definition.split(",")
        new_data = pd.concat([new_data.get(d) for d in definitions])
        population.index = new_data.index.values
        population.geom.x_threshold = x_threshold
        population.geom.y_threshold = y_threshold
        _reset_population(population)


def update_polygon(population: Population,
                   parent_data: pd.DataFrame,
                   x_values: list,
                   y_values: list):
    """
    Given an existing population and some new definition for it's polygon gate
    (different to what is already associated to the Population), update the Population
    index and geom accordingly. Any associated clusters will be removed and any controls
    will have to be estimated
    again.
    Parameters
    ----------
    population
    parent_data
    x_values
    y_values

    Returns
    -------

    """
    poly = create_polygon(x=x_values, y=y_values)
    new_data = inside_polygon(df=parent_data,
                              x=population.geom.x,
                              y=population.geom.y,
                              poly=poly)
    population.index = new_data.index.values
    _reset_population(population)
