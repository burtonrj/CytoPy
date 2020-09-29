from ..flow.dim_reduction import dimensionality_reduction
from ..flow.transforms import apply_transform, scaler
from ..flow.sampling import density_dependent_downsampling, faithful_downsampling, upsample_knn, upsample_svm
from ..flow.gating_analyst import ManualGate, DensityGate, Analyst
from ..utilities import inside_polygon
from ..feedback import vprint
from .populations import PopulationGeometry, Population, merge_populations
from scipy.spatial.distance import euclidean
from functools import reduce
from typing import List
from warnings import warn
import pandas as pd
import numpy as np
import mongoengine


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
    idx = idx or data.index.values
    # ToDo this should be more robust
    for x in ["Time", "time"]:
        if x in data.columns:
            data.drop(x, 1, inplace=True)
    summary_method = summary_method or np.median
    signature = data.loc[idx].apply(summary_method)
    return {x[0]: x[1] for x in zip(signature.index, signature.values)}


def _merge(new_children: List[Population],
           assignments: list):
    # Group the children by their assigned population
    groups = {name: [] for name in assignments}
    merged_children = list()
    for assignment, child in zip(assignments, new_children):
        groups[assignment].append(child)
    for assignment, children in groups.items():
        # If only one child, ignore
        if len(children) == 1:
            children[0].population_name = assignment
            merged_children.append(children[0])
            continue
        # Merge the populations
        new_child = reduce(lambda p1, p2: merge_populations(p1, p2), children)
        new_child.population_name = assignment
        merged_children.append(new_child)
    return merged_children


class ChildDefinition(mongoengine.EmbeddedDocument):
    """
    Each child describes a single population that a Gate generates. If the Gate property 'binary' is True,
    then definition will be one of "+" or "-"; population exists either within the gate, or outside the gate.

    If the Gate property geom_type is 'threshold' then definition will be "+" or "-", in the case that
    'binary' is also true, describing that the population is either to the right or to the left of the
    threshold, respectively. If 'binary' is not True, the definition will be "++", "--", "-+" or "+-",
    under the assumption that the threshold is 2 dimensional, generating a 4 dimensional grid. In this case
    a Child can have more than one definition, which is stored as a comma separated string.

    For non-threshold gates, a template_geometry attribute will be expected. These attributes
    correspond to the shape of the gate when first defined. Although when applied in the future an algorithm
    will be used to mold the gate to the new data the gate is exposed too, this template geometry will be
    used to assign child population labels (see Gate.label_populations)

    Parameters
    -----------
    population_name: str
        Name of the child population
    definition: str
        Either  "+" or "-" for a binary Gate, or one or more of the following: "++", "--", "-+" or "+-" for a
        non-binary threshold gate. Not required for other non-binary Gates.
    x_values: list
        Required for non-binary, non-threshold Gates. Corresponds to the X-axis coordinates of the geometric
        object defining the population.
    y_values: list
        Required for non-binary, non-threshold Gates. Corresponds to the Y-axis coordinates of the geometric
        object defining the population.
    """
    population_name = mongoengine.StringField()
    definition = mongoengine.StringField(required=False)
    geom = mongoengine.EmbeddedDocumentField(PopulationGeometry)
    signature = mongoengine.DictField()

    def match_definition(self, query: str):
        return query in self.definition.split(",")


def population_likeness(new_population: Population,
                        template_population: ChildDefinition):
    """
    Given two Populations, generate a score of their likeness, where a smaller value indicates greater
    likeness. This is a composite score of the euclidean distance between their signatures (their vector means),
    the euclidean distance between their centroids, and the overlap of their areas.
    between them
    Parameters
    ----------
    new_population: list
    template_population: list

    Returns
    -------
    float
        composite score
    """
    # Signature score
    vector_avgs = np.array([[new_population.signature[i], template_population.signature[i]]
                            for i in set(new_population.signature.keys()).intersection(template_population.signature.keys())]).T
    sig_score = abs(euclidean(vector_avgs[0, :], vector_avgs[1, :]))
    if hasattr(new_population.geom, "shape") and hasattr(template_population.geom, "shape"):
        # Centroid score
        new_pop_poly = new_population.geom.shape
        template_poly = template_population.geom.shape
        x = [new_pop_poly.centroid.coords[0][0], template_poly.centroid.coords[0][0]]
        y = [new_pop_poly.centroid.coords[0][1], template_poly.centroid.coords[0][1]]
        norm = scaler(data=np.array([x, y]),
                      scale_method="norm",
                      return_scaler=False)
        centroid_score = abs(euclidean(norm[0, :], norm[1, :]))
        # Area score
        area_score = 1. - template_population.geom.overlap(comparison_poly=new_population.geom.shape)
        return sum([sig_score, centroid_score, area_score])
    return sig_score


class PreProcess(mongoengine.EmbeddedDocument):
    downsample_method = mongoengine.StringField(required=False, choices=["uniform",
                                                                         "faithful",
                                                                         "density"])
    downsample_kwargs = mongoengine.DictField()
    transform_x = mongoengine.StringField(required=False,
                                          choices=["logicle",
                                                   "log_transform",
                                                   "hyperlog",
                                                   "asinh",
                                                   "percentile_rank",
                                                   "Yeo-Johnson",
                                                   "RobustScale"])
    transform_y = mongoengine.StringField(required=False,
                                          choices=["logicle",
                                                   "log_transform",
                                                   "hyperlog",
                                                   "asinh",
                                                   "percentile_rank",
                                                   "Yeo-Johnson",
                                                   "RobustScale"])
    scale = mongoengine.StringField(required=False,
                                    choices=["standard",
                                             "norm",
                                             "power",
                                             "robust"])
    scale_kwargs = mongoengine.DictField()
    dim_reduction = mongoengine.StringField(required=False,
                                            choices=["UMAP",
                                                     "tSNE",
                                                     "PCA",
                                                     "KernelPCA",
                                                     "PHATE"])
    dim_reduction_kwargs = mongoengine.DictField()


class PostProcess(mongoengine.EmbeddedDocument):
    upsample_method = mongoengine.StringField(required=False,
                                              choices=["knn", "svm"])
    upsample_kwargs = mongoengine.DictField()
    signature_transform = mongoengine.StringField(default="logicle")
    signature_method = mongoengine.StringField(default="median", choices=["median", "mean"])


class Gate(mongoengine.Document):
    """
    Document representation of a Gate
    """
    gate_name = mongoengine.StringField(required=True)
    parent = mongoengine.StringField(required=True)
    children = mongoengine.EmbeddedDocumentListField(ChildDefinition)
    binary = mongoengine.BooleanField(default=True)
    shape = mongoengine.StringField(required=True, choices=["threshold", "polygon", "ellipse"])
    x = mongoengine.StringField(required=True)
    y = mongoengine.StringField(required=False)
    ctrl_id = mongoengine.StringField(required=False)
    preprocessing = mongoengine.EmbeddedDocumentField(PreProcess)
    method = mongoengine.StringField()
    method_kwargs = mongoengine.DictField()
    postprocessing = mongoengine.EmbeddedDocumentField(PostProcess)

    meta = {
        'db_alias': 'core',
        'collection': 'gates'
    }

    def __init__(self,
                 *args,
                 **values):
        super().__init__(*args, **values)
        self.defined = True
        self.labelled = True
        if not self.children:
            self.defined = False
            self.labelled = False

    def clear_children(self):
        self.defined = False
        self.labelled = False
        self.children = []

    def label_children(self,
                       labels: dict):
        assert not self.labelled, "Children already labelled. To clear children and relabel call 'clear_children'"
        drop = [c.population_name for c in self.children if c.population_name not in labels.keys()]
        assert len(drop) != len(self.children), "No keys in label match existing child populations"
        if drop:
            warn(f"The following populations are not in labels and will be dropped: {drop}")
        self.children = [c for c in self.children if c.population_name not in drop]

        if self.binary and self.shape != "threshold":
            assert len(labels) == 1, "Non-threshold binary gate's should only have a single population"
        elif self.binary and self.shape == "threshold":
            assert set(labels.keys()) == {'+', '-'}, "For a binary threshold gate, labels should be provided " \
                                                     "with the keys: '+' and '-'"
        elif self.shape == "threshold":
            assert set(labels.keys()) == {'++', '--',
                                          '-+', '+-'}, "For a non-binary threshold gate, labels should be " \
                                                       "provided with the keys: '++', '-+', '+-' and '--'"
        for child in self.children:
            child.population_name = labels.get(child.population_name)
        self.labelled = True

    def _scale(self,
               data: pd.DataFrame):
        return pd.DataFrame(scaler(data.values, self.preprocessing.scale, **self.preprocessing.scale_kwargs),
                            columns=data.columns)

    def _dim_reduction(self,
                       data: pd.DataFrame):
        err = "Invalid Gate: when performing dimensionality reduction, transform_x and transform_y must be equal"
        assert self.preprocessing.transform_x == self.preprocessing.transform_y, err
        data = apply_transform(data=data,
                               transform_method=self.preprocessing.transform_x,
                               features_to_transform=data.columns)
        if self.preprocessing.scale:
            data = self._scale(data)
        data = pd.DataFrame(dimensionality_reduction(data=data,
                                                     features=data.columns,
                                                     method=self.preprocessing.dim_reduction,
                                                     n_components=2,
                                                     return_reducer=False,
                                                     return_embeddings_only=True,
                                                     **self.preprocessing.dim_reduction_kwargs),
                            columns=["embedding1", "embedding2"])
        return data

    def _init_method(self):
        kwargs = {x[0]: x[1] for x in self.method_kwargs}
        if self.method == "ManualGate":
            return ManualGate(x=self.x,
                              y=self.y,
                              shape=self.shape,
                              parent=self.parent,
                              **kwargs)
        if self.method == "DensityGate":
            return DensityGate(x=self.x,
                               y=self.y,
                               shape=self.shape,
                               parent=self.parent,
                               binary=self.binary,
                               **kwargs)
        return Analyst(x=self.x,
                       y=self.y,
                       shape=self.shape,
                       parent=self.parent,
                       binary=self.binary,
                       model=self.method,
                       **kwargs)

    def _transform(self,
                   data: pd.DataFrame):
        if self.preprocessing.transform_x and self.x is not None:
            data = apply_transform(data=data,
                                   transform_method=self.preprocessing.transform_x,
                                   features_to_transform=[self.x])
        if self.preprocessing.transform_y and self.y is not None:
            data = apply_transform(data=data,
                                   transform_method=self.preprocessing.transform_y,
                                   features_to_transform=[self.y])
        return data

    def _downsample(self,
                    data: pd.DataFrame):
        if self.method == "DensityGate":
            warn("DensityGate handles downsampling internally. Downsampling params ignored. To control "
                 "downsampling methodology alter the method_kwargs as per documentation")
            return data, None
        if self.preprocessing.downsample_method == "uniform":
            n = self.preprocessing.downsample_kwargs.get("sample_n", None)
            assert n, "Invalid Gate: for uniform downsampling expect 'sample_n' in downsample_kwargs"
            if type(n) == int:
                return data, data.sample(n=n)
            return data, data.sample(frac=float(n))
        if self.preprocessing.downsample_method == "density":
            return data, density_dependent_downsampling(data=data,
                                                        features=data.columns,
                                                        **self.preprocessing.downsample_kwargs)
        if self.preprocessing.downsample_method == "faithful":
            h = self.preprocessing.downsample_kwargs.get("h", None)
            assert "h", "Invalid Gate: for faithful downsampling, 'h' expected in downsampling_kwargs"
            return data, pd.DataFrame(faithful_downsampling(data=data, h=h), columns=data.columns)
        raise ValueError("Invalid Gate: downsampling_method must be one of: uniform, density, or faithful")

    def _apply_preprocessing(self,
                             data: pd.DataFrame):
        data = data.copy()
        if self.preprocessing.dim_reduction:
            data = self._dim_reduction(data)
        else:
            # Transform the x and y dimensions
            data = self._transform(data=data)
            features = [x for x in [self.x, self.y] if x is not None]
            data = data[features]
        # Perform additional scaling if requested
        if self.preprocessing.scale:
            data = self._scale(data)
        # Perform downsampling if requested
        if self.preprocessing.downsample_method:
            return self._downsample(data)
        return data, None

    def apply(self,
              data: pd.DataFrame,
              ctrl: pd.DataFrame or None = None,
              verbose: bool = True):
        original_data = data.copy()
        feedback = vprint(verbose)
        feedback("---- Applying gate ----")
        method = self._init_method()
        if ctrl is not None:
            assert "DensityGate" in str(self.method.__class__), \
                "Control driven gates are currently only supported for DensityGate method"
            data, _ = self._apply_preprocessing(data=data)
            ctrl, _ = self._apply_preprocessing(data=ctrl)
            populations = self._apply_postprocessing(method.ctrl_gate(data=data, ctrl=ctrl),
                                                     original_data=data, data=data)
        else:
            data, sample = self._apply_preprocessing(data=data)
            if sample is not None:
                feedback("Downsampling applied prior to fit...")
                populations = self._apply_postprocessing(method.fit_predict(sample),
                                                         original_data=original_data,
                                                         sample=sample,
                                                         verbose=verbose,
                                                         data=data)
            else:
                populations = self._apply_postprocessing(method.fit_predict(data),
                                                         original_data=original_data,
                                                         verbose=verbose,
                                                         data=data)
        if not self.defined:
            feedback("This gate has not been previously defined. Gate will be applied to example data "
                     "and child population definitions populated. Labels will need to be provided for "
                     "resulting child populations")
            feedback("Adding definitions of child populations to gate...")
            self.children = []
            for pop in populations:
                self._add_child(pop)
            self.defined = True
            return populations
        else:
            assert self.labelled, "Gate children are unlabelled, call `label_children prior to calling `apply``"
            feedback("Matching detected populations to expected children...")
            return self._match_to_children(populations)

    def _upsample(self,
                  new_populations: List[Population],
                  data: pd.DataFrame,
                  sample: pd.DataFrame,
                  verbose: bool = True):
        if not self.postprocessing.upsample_method:
            warn("Downsampling was performed yet not upsampling method has been defined, "
                 "defaulting to infer_from_gate")
            self.postprocessing.upsample_method = 'infer_from_gate'
        if self.postprocessing.upsample_method == "infer_from_gate":
            for p in new_populations:
                p.index = inside_polygon(data,
                                         p.geom.x,
                                         p.geom.y,
                                         p.geom.shape).index
                p.n = len(p.index)
            return new_populations
        if self.postprocessing.upsample_method == "knn":
            return upsample_knn(populations=new_populations,
                                features=[self.x, self.y],
                                original=data,
                                sample=sample,
                                verbose=verbose)
        if self.postprocessing.upsample_method == "svm":
            return upsample_svm(populations=new_populations,
                                features=[self.x, self.y],
                                original=data,
                                sample=sample,
                                verbose=verbose)
        raise ValueError("Upsample method should be one of: 'infer_from_gate', 'knn' or 'svm'")

    def _apply_postprocessing(self,
                              new_populations: List[Population],
                              data: pd.DataFrame,
                              original_data: pd.DataFrame,
                              sample: pd.DataFrame or None = None,
                              verbose: bool = True):
        # Upsample if necessary
        if sample is not None:
            new_populations = self._upsample(new_populations=new_populations, data=data, sample=sample, verbose=verbose)
        # Add transformation information to Populations
        if self.preprocessing.transform_x:
            for p in new_populations:
                p.geom.transform_x = self.preprocessing.transform_x
        if self.preprocessing.transform_y:
            for p in new_populations:
                p.geom.transform_y = self.preprocessing.transform_y
        # Add population signatures
        sig_data = apply_transform(data=original_data,
                                   features_to_transform="all",
                                   transform_method=self.postprocessing.signature_transform)
        summary_method = np.median
        if self.postprocessing.signature_method == "mean":
            summary_method = np.mean
        for p in new_populations:
            p.signature = create_signature(data=sig_data, idx=p.index, summary_method=summary_method)
        return new_populations

    def _add_child(self,
                   population: Population):
        name = population.population_name
        if self.shape == "threshold":
            name = population.definition
        new_child = ChildDefinition(population_name=name,
                                    definition=population.definition,
                                    template_geometry=population.geom,
                                    signature=population.signature)
        self.children.append(new_child)

    def _label_binary_threshold(self,
                                new_children: List[Population]):
        """
        Given a list of Population's newly generated by applying this Gate, label these Population's according
        to a binary match of their associated definitions: either + or -

        Parameters
        ----------
        new_children: list
            A list of newly generated Population objects

        Returns
        -------
        list
        """
        assert len(new_children) == 2, f"{self.gate_name} is binary and expects exactly two child populations"
        err = f"{self.gate_name} is binary and as such, two new child populations should exist, " \
              f"one of definition '+', the other '-' "
        assert set([c.definition for c in new_children]) == {"+", "-"}, err
        pos = [child for child in self.children if child.definition == "+"][0]
        neg = [child for child in self.children if child.definition == "-"][0]
        for new_child in new_children:
            if new_child.definition == "+":
                new_child.population_name = pos.population_name
            else:
                new_child.population_name = neg.population_name
        return new_children

    def _label_threshold(self,
                         new_children: list):
        for i, new_child in enumerate(new_children):
            match = [c for c in self.children if c.match_definition(new_child.definition)]
            if len(match) == 0:
                err = f"New child population at index {i} matches no definition in given gate; " \
                      f"new definition = {new_child.definition}. Existing definitions: " \
                      f"{[c.definition for c in self.children]}"
                raise ValueError(err)
            elif len(match) > 1:
                err = f"New child population at index {i} matches multiple expected child populations"
                raise ValueError(err)
            else:
                new_child.population_name = match[0].population_name
        if len(set([c.population_name for c in new_children])) < len(new_children):
            return _merge(new_children, assignments=[c.population_name for c in new_children])
        return new_children

    def _label_binary_other(self,
                            new_children: List[Population]):
        # Binary gates that are not thresholds only have one child
        pos = self.children[0]
        ranking = [population_likeness(c, pos) for c in new_children]
        new_children[int(np.argmin(ranking))].population_name = pos.population_name
        return [new_children[int(np.argmin(ranking))]]

    def _match_to_children(self,
                           new_children: List[Population],
                           errors: str = "warn",
                           overlaps: str = "merge"):
        # Binary threshold gate?
        if self.binary and self.shape == "threshold":
            return self._label_binary_threshold(new_children=new_children)
        # Non-binary threshold gate?
        if not self.binary and self.shape == "threshold":
            return self._label_threshold(new_children=new_children)
        # Is the gate binary and non-threshold? More than one child population? Then match by overlap
        if self.binary and self.shape != "threshold":
            return self._label_binary_other(new_children)
        # All other gate types (multiple populations generated by an ellipse or polygon gate)
        if len(new_children) < len(self.children):
            err = "Number of new child populations does not match expected number of child populations; " \
                  f"{len(new_children)} != {len(self.children)}"
            if errors == "warn":
                warn(err)
            else:
                raise ValueError(err)
        # Assign new populations to children based on overlapping geometries
        assignments = self._compare_populations(new_children=new_children)
        # Are any of the expected populations missing from assignments?
        for expected_population in [c.population_name for c in self.children]:
            if expected_population not in assignments:
                err = f"Expected population {expected_population} could not be located"
                if errors == "warn":
                    warn(err)
                else:
                    raise ValueError(err)
            # Are there duplicate assignments? If so, merge populations, if allowed
        if len(set(assignments)) < len(assignments):
            if overlaps == "merge":
                new_children = _merge(new_children, assignments)
            else:
                err = f"Some template populations assigned to multiple populations in new data: {assignments}"
                raise ValueError(err)
        else:
            for i, child in enumerate(new_children):
                child.population_name = assignments[i]
        return new_children

    def _compare_populations(self,
                             new_children: List[Population]):
        # Compare the signatures of each of the new children to the template signatures
        assignments = list()
        for child in new_children:
            ranking = [population_likeness(new_population=child,
                                           template_population=template) for template in self.children]
            assignments.append(self.children[int(np.argmin(ranking))].population_name)
        return assignments

    def save(self, *args, **kwargs):
        assert self.defined, f"Gate {self.gate_name} is newly created and has not been defined. " \
                             f"Call 'label_children' to complete gating definition"
        super().save(*args, **kwargs)
