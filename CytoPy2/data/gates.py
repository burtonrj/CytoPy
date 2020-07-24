from ..flow.dim_reduction import dimensionality_reduction
from ..flow.transforms import apply_transform, scaler
from ..flow.sampling import density_dependent_downsampling, faithful_downsampling, upsample_knn, upsample_svm
from .fcs import Population, PopulationGeometry, merge_populations
from functools import reduce
from typing import List
from warnings import warn
import pandas as pd
import numpy as np
import mongoengine


def _merge(new_children: List[Population],
           assignments: list):
    # Group the children by their assigned population
    groups = {name: [] for name in assignments}
    merged_children = list()
    for assignment, child in zip(assignments, new_children):
        groups[assignments].append(child)
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
    template_geometry = mongoengine.EmbeddedDocument(PopulationGeometry)

    def match_definition(self, query: str):
        return query in self.definition.split("_")


class PreProcess(mongoengine.EmbeddedDocument):
    downsample_method = mongoengine.StringField(required=False, choices=["uniform",
                                                                         "faithful",
                                                                         "density"])
    downsample_kwargs = mongoengine.ListField()
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
    scale_kwargs = mongoengine.ListField()
    dim_reduction = mongoengine.StringField(required=False,
                                            choices=["UMAP",
                                                     "tSNE",
                                                     "PCA",
                                                     "KernelPCA",
                                                     "PHATE"])
    dim_reduction_kwargs = mongoengine.ListField()


class PostProcess(mongoengine.EmbeddedDocument):
    upsample_method = mongoengine.StringField(required=False,
                                              choices=["knn", "svm"])
    upsample_kwargs = mongoengine.ListField()


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
    y = mongoengine.StringField(required=True)
    preprocessing = mongoengine.EmbeddedDocument(PreProcess)
    method = mongoengine.StringField()
    method_kwargs = mongoengine.ListField()
    postprocessing = mongoengine.EmbeddedDocument(PostProcess)

    meta = {
        'db_alias': 'core',
        'collection': 'gates'
    }

    def __init__(self,
                 *args,
                 **values):
        super().__init__(*args, **values)
        self.model = values.get("model", None)
        self._defined = False

    def initialise_model(self):
        assert self.method, "No method set"
        if not self.method_kwargs:
            method_kwargs = {}
        else:
            method_kwargs = {k: v for k, v in self.method_kwargs}
        if self.method == "ManualGate":
            assert not self.binary, "ManualGate is for use with binary gates only"
            self.model = ManualGate(x=self.x,
                                    y=self.y,
                                    shape=self.shape,
                                    parent=self.parent,
                                    **method_kwargs)
        elif self.method == "DensityGate":
            self.model = DensityGate(x=self.x,
                                     y=self.y,
                                     shape=self.shape,
                                     parent=self.parent,
                                     **method_kwargs)
        else:
            if self.method in ["DBSCAN", "HDBSCAN"] and not self.preprocessing.downsample_method:
                warn("DBSCAN and HDBSCAN do not scale well and it is recommended that downsampling "
                     "is performed")
            self.model = Analyst(x=self.x,
                                 y=self.y,
                                 shape=self.shape,
                                 parent=self.parent,
                                 model=self.method,
                                 **method_kwargs)

    def clear_children(self):
        self._defined = False
        self.children = []

    def label_children(self,
                       labels: dict):
        assert not self._defined, "Children already defined. To clear children and relabel call 'clear_children'"
        if self.binary and self.shape != "threshold":
            assert len(labels) == 1, "Non-threshold binary gate's should only have a single population"
        elif self.binary and self.shape == "threshold":
            assert {"+", "-"} == set(labels.keys()), "Binary threshold gate has the following populations: '+' or '-'"
        elif self.shape == "threshold":
            assert {"++", "--", "-+", "+-"} == set(labels.keys()), \
                "Binary threshold gate has the following populations: '++', '--', '+-', or '-+'"
        drop = [c.population_name for c in self.children if c.population_name not in labels.keys()]
        assert len(drop) != len(self.children), "No keys in label match existing child populations"
        if drop:
            warn(f"The following populations are not in labels and will be dropped: {drop}")
        self.children = [c for c in self.children if c.population_name not in drop]
        for child in self.children:
            child.population_name = labels.get(child)
        self._defined = True

    def _scale(self,
               data: pd.DataFrame):
        kwargs = {k: v for k, v in self.preprocessing.scale_kwargs}
        return pd.DataFrame(scaler(data.values, self.preprocessing.scale, **kwargs), columns=data.columns)

    def _dim_reduction(self,
                       data: pd.DataFrame):
        err = "Invalid Gate: when performing dimensionality reduction, transform_x and transform_y must be equal"
        assert self.preprocessing.transform_x == self.preprocessing.transform_y, err
        data = apply_transform(data=data,
                               transform_method=self.preprocessing.transform_x,
                               features_to_transform=data.columns)
        if self.preprocessing.scale:
            data = self._scale(data)
        kwargs = {k: v for k, v in self.preprocessing.dim_reduction_kwargs}
        data = pd.DataFrame(dimensionality_reduction(data=data,
                                                     method=self.preprocessing.dim_reduction,
                                                     n_components=2,
                                                     return_reducer=False,
                                                     return_embeddings_only=True,
                                                     **kwargs),
                            columns=["embedding1", "embedding2"])
        return data

    def _apply_preprocessing(self,
                             data: pd.DataFrame):
        data = data.copy()
        if self.preprocessing.dim_reduction:
            data = self._dim_reduction(data)
        else:
            # Transform the x and y dimensions
            if self.preprocessing.transform_x:
                data = apply_transform(data=data,
                                       transform_method=self.preprocessing.transform_x,
                                       features_to_transform=[self.x])
            if self.preprocessing.transform_y:
                data = apply_transform(data=data,
                                       transform_method=self.preprocessing.transform_y,
                                       features_to_transform=[self.y])
            data = data[[self.x, self.y]]
        # Perform additional scaling if requested
        if self.preprocessing.scale:
            data = self._scale(data)
        # Perform downsampling if requested
        if self.preprocessing.downsample_method:
            kwargs = {k: v for k, v in self.preprocessing.downsample_kwargs}
            if self.preprocessing.downsample_method == "uniform":
                assert "sample_n" in kwargs.keys(), "Invalid Gate: for uniform downsampling expect 'sample_n' in " \
                                                    "downsample_kwargs"
                n = kwargs.get("sample_n")
                if type(n) == int:
                    return data, data.sample(n=n)
                return data, data.sample(frac=float(n))
            elif self.preprocessing.downsample_method == "density":
                return data, density_dependent_downsampling(data=data,
                                                            features=data.columns,
                                                            **kwargs)
            elif self.preprocessing.downsample_method == "faithful":
                assert "h" in kwargs.keys(), "Invalid Gate: for faithful downsampling, 'h' expected in " \
                                             "downsampling_kwargs"
                return data, pd.DataFrame(faithful_downsampling(data=data, h=kwargs.get("h")),
                                          columns=data.columns)
            raise ValueError("Invalid Gate: downsampling_method must be one of: uniform, density, or faithful")
        return data, None

    def apply(self,
              data: pd.DataFrame):
        if not self._defined:
            # Applying for the first time, resulting populations should populate the child definitions
            data, sample = self._apply_preprocessing(data=data)
            if sample is not None:
                populations = self._apply_postprocessing(self.model.fit_predict(sample), original_data=data)
            else:
                populations = self._apply_postprocessing(self.model.fit_predict(data))
            for pop in populations:
                self._add_child(pop)
        else:
            # Pre-defined gate, resulting populations should be matched to the child definitions
            data, sample = self._apply_preprocessing(data=data)
            if sample is not None:
                populations = self._apply_postprocessing(self.model.fit_predict(sample), original_data=data)
            else:
                populations = self._apply_postprocessing(self.model.fit_predict(data))
            return self._match_to_children(populations)

    def _apply_postprocessing(self,
                              new_populations: List[Population],
                              original_data: pd.DataFrame or None = None):
        if self.preprocessing.downsample_method:
            upsample_method = self.preprocessing.upsample_method
            if not upsample_method:
                warn("Downsampling was performed yet not upsampling method has been defined, defaulting to KNN")
                upsample_method = 'knn'
            if upsample_method == "knn":
                new_populations = upsample_knn(new_populations, original_data)
            else:
                new_populations = upsample_svm(new_populations, original_data)
        return new_populations

    def _add_child(self,
                   population: Population):
        name = population.population_name
        if self.shape == "threshold":
            name = population.definition
        new_child = ChildDefinition(population_name=name,
                                    definition=population.definition,
                                    template_geom=population.geom)
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
                new_child.definition = pos.population_name
            else:
                new_child.definition = neg.population_name
        return new_children

    def _label_threshold(self,
                         new_children: List[children]):
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
        pos = [c for c in self.children if c.definition == "+"][0]
        neg = [c for c in self.children if c.definition == "-"][0]
        ranking = [pos.geom.overlap(comparison_poly=c.geom.shape) for c in new_children]
        new_children[int(np.argmax(ranking))].population_name = self.children[0]
        negative_new = [c for i, c in enumerate(new_children) if i != int(np.argmax(ranking))]
        for child in negative_new:
            child.population_name = neg.population_name
        return new_children

    def _match_to_children(self,
                           new_children: List[Population],
                           errors: str = "warn",
                           overlaps: str = "merge",
                           overlap_threshold: float = .1):
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
        assignments = self._compare_geometries(new_children=new_children,
                                               overlap_threshold=overlap_threshold)
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

    def _compare_geometries(self,
                            new_children: List[Population],
                            overlap_threshold: float):
        assignments = list()
        for child in new_children:
            ranking = [child.geom.overlap(comparison_poly=template.template_geometry.shape,
                                          threshold=overlap_threshold) for template in self.children]
            if all(x == 0. for x in ranking):
                ranking = [child.geom.shape.centroid.distance(template.template_geometry.shape.centroid)
                           for template in self.children]
                assignments.append(self.children[int(np.argmin(ranking))].population_name)
            else:
                assignments.append(self.children[int(np.argmax(ranking))].population_name)
        return assignments

    def save(self, *args, **kwargs):
        assert self._defined, "Gate is newly created and has not been defined. Call 'label_children' to complete " \
                              "gating definition"
        super().save(*args, **kwargs)
