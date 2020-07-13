from .fcs import PopulationGeometry, Population, merge_populations
from ..flow.gates import Base, ManualGate, DensityGate
from functools import reduce
from typing import List
from warnings import warn
import mongoengine
import numpy as np
import importlib
import datetime


class ControlGate(mongoengine.Document):
    gate_name = mongoengine.StringField()
    population = mongoengine.StringField()
    method = mongoengine.StringField()
    kwargs = mongoengine.ListField()


class PreProcess(mongoengine.EmbeddedDocument):
    downsample_method = mongoengine.StringField()
    downsample_kwargs = mongoengine.ListField()
    transform_x = mongoengine.StringField()
    transform_y = mongoengine.StringField()
    scale = mongoengine.StringField()
    scale_kwargs = mongoengine.ListField()
    dim_reduction = mongoengine.StringField()
    dim_reduction_kwargs = mongoengine.ListField()


class PostProcess(mongoengine.EmbeddedDocument):
    upsample_method = mongoengine.StringField()
    upsample_kwargs = mongoengine.ListField()


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
    definition: str or list
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
    _definition = mongoengine.StringField(db_field="definition")
    _template_geometry = mongoengine.EmbeddedDocument(PopulationGeometry,
                                                      db_field="template_geometry")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "definition" in kwargs:
            self.definition = kwargs.get("definition")

    @property
    def definition(self):
        return self._definition

    @property
    def template_geometry(self):
        return self._template_geometry

    @definition.setter
    def definition(self,
                   value: list or str):
        if self._instance.binary:
            assert value in ["+", "-"], "Binary gate assumes Child definition is either '+' or '-'"
        elif self._instance.geom == "threshold":
            acceptable = ["++", "--", "+-", "-+"]
            err = f"Non-binary gate of type threshold assumes Child " \
                  f"definition is one or more of {acceptable}"
            if type(value) is list:
                assert all([x in acceptable for x in value]), err
                self._definition = ",".join(value)
            else:
                assert value in acceptable, err
                self._definition = value
        else:
            warn("Definition given for a non-binary Gate that does not generate a threshold. Definition will be"
                 "ignored")

    @template_geometry.setter
    def template_geometry(self,
                          properties: dict):
        new_template = PopulationGeometry()
        if self._instance.geom == "threshold":
            warn("Threshold gate does not require template geometry. Input will be ignored.")
        elif self._instance.geom == "polygon":
            for required in ["x_values", "y_values"]:
                assert required in properties.keys(), f"{required} required for polygon gate"
                new_template[required] = properties.get(required)
            self._template_geometry = new_template
        elif self._instance.geom == "ellipse":
            for required in ["width", "height", "center", "angle"]:
                assert required in properties.keys(), f"{required} required for ellipse gate"
                new_template[required] = properties.get(required)
            self._template_geometry = new_template


class Gate(mongoengine.Document):
    """
    Document representation of a Gate
    """
    gate_name = mongoengine.StringField(required=True)
    parent = mongoengine.StringField(required=True)
    children = mongoengine.EmbeddedDocumentListField(ChildDefinition)
    binary = mongoengine.BooleanField(default=True)
    shape = mongoengine.StringField(required=True, choices=["threshold",
                                                            "polygon",
                                                            "ellipse"])
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

    def initialise_model(self):
        assert self.method, "No method set"
        if not self.method_kwargs:
            method_kwargs = {}
        else:
            method_kwargs = {k: v for k, v in self.method_kwargs}
        if self.method == "ManualGate":
            self.model = ManualGate(x=self.x,
                                    y=self.y,
                                    shape=self.shape,
                                    parent=self.parent,
                                    binary=self.binary,
                                    **method_kwargs)
        elif self.method == "DensityGate":
            self.model = DensityGate(x=self.x,
                                     y=self.y,
                                     shape=self.shape,
                                     parent=self.parent,
                                     binary=self.binary,
                                     **method_kwargs)
        else:
            self.model = Base(x=self.x,
                              y=self.y,
                              shape=self.shape,
                              parent=self.parent,
                              binary=self.binary,
                              **method_kwargs)

    def get_child_by_definition(self,
                                definition: str) -> ChildDefinition or None:
        """
        Given a definition (assume a binary gate or threshold gate) return the corresponding ChildDefinition i.e.
        the child population this gate associates to this definition. Returns None if no match found.

        Parameters
        ----------
        definition: str
        Returns
        -------
        ChildDefinition or None
        """
        if self.binary:
            assert definition in ["+", "-"], "Binary gate: definition should be a string of value '+' or '-'"
        for child in self.children:
            if child.definition == definition:
                return child
        return None

    def _label_binary(self,
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
        for new_child in new_children:
            match = self.get_child_by_definition(new_child.definition)
            new_child.population_name = match.population_name
        return new_children

    def _label_threshold(self,
                         new_children: List[Population],
                         errors: str):
        """
        Given a list of newly generated Population's, label the Population's according to their given
        definitions under the assumption that this Gate is a 2D threshold gate

        Parameters
        ----------
        new_children: list
        errors: str
            How to handle new child populations that do not match any expected populations. If "warn", a warning
            is returned and the population ignored. Otherwise, a ValueError will be raised.

        Returns
        -------
        list
        """
        for i, new_child in enumerate(new_children):
            err = f"New child population at index {i} matches no definition in given gate; " \
                  f"new definition = {new_child.definition}. Existing definitions: " \
                  f"{[c.definition for c in self.children]}"
            match = self.get_child_by_definition(new_child.definition)
            if match is not None:
                new_child.population_name = match.population_name
            elif errors == "warn":
                warn(err)
            else:
                raise ValueError(err)
        # Merge where a population has multiple definitions
        if len(set([c.population_name for c in new_children])) < len(new_children):
            return self._merge(new_children, assignments=[c.population_name for c in new_children])
        return new_children

    def _assign_new_populations(self,
                                new_children: List[Population],
                                errors: str = "warn",
                                overlaps: str = "merge",
                                overlap_threshold: float = .1):
        """
        Given a list of new Population's generated by the application of this Gate, label the Population's according
        to the expected population's (as defined in the ChildDefinition's embedded in this document) and their
        associated PopulationGeometry's.

        Population's are assigned child population names by the % of overlap between their own geometry and the
        geometry of the child populations generated when this Gate was first defined. If no overlap exists, then
        association is by nearest centroid.

        Parameters
        ----------
        new_children: list
            List of newly generated Population objects
        errors: str
            How to handle errors. If "warn", then a warning will be given and errors ignored. If "raise", then
            a ValueError exception will be raised.
        overlaps: str
            How to handle overlapping populations, given that a gate is non-binary and non-threshold. If "merge", then
            populations will be merged, otherwise a ValueError is risen.
        overlap_threshold: float (default = 0.1)
            Minimum overlap acknowledged when comparing geometries (default is 10%)

        Returns
        -------
        list
            List of Population's with population_name assigned from comparison to expected child populations
        """
        if self.binary:
            new_children = self._label_binary(new_children=new_children)
        # Non-binary gate. First check length of new populations matches what is expected
        if len(new_children) < len(self.children):
            err = "Number of new child populations does not match expected number of child populations; " \
                  f"{len(new_children)} != {len(self.children)}"
            if errors == "warn":
                warn(err)
            else:
                raise ValueError(err)
        elif self.shape == "threshold":
            new_children = self._label_threshold(new_children=new_children,
                                                 errors=errors)
        else:
            # Non-binary polygon gate. Assign populations based on overlapping geometries or nearest centroid
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
                    new_children = self._merge(new_children,
                                               assignments)
                else:
                    raise ValueError("Some template populations assigned to multiple populations in new "
                                     f"data: {assignments}")
            else:
                for i, child in enumerate(new_children):
                    child.population_name = assignments[i]
        return new_children

    def _compare_geometries(self,
                            new_children: List[Population],
                            overlap_threshold: float):
        # Build geometries
        assignments = list()
        for child in new_children:
            ranking = [child.geom.overlap(comparison_poly=template.template_geometry.geom,
                                          threshold=overlap_threshold) for template in self.children]
            if all(x == 0. for x in ranking):
                ranking = [child.geom.centroid.distance(template.template_geometry.geom.centroid)
                           for template in self.children]
                assignments.append(self.children[int(np.argmin(ranking))].population_name)
            else:
                assignments.append(self.children[int(np.argmax(ranking))].population_name)
        return assignments

    @staticmethod
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


class GatingStrategy(mongoengine.Document):
    """
    Document representation of a gating template; a gating template is a collection of gating objects
    that can be applied to multiple fcs files or an entire experiment in bulk

    Parameters
    -----------
    template_name: str, required
        unique identifier for template
    gates: EmbeddedDocumentList
        list of Gate documents; see Gate
    creation_date: DateTime
        date of creation
    last_edit: DateTime
        date of last edit
    flags: str, optional
        warnings associated to this gating template
    notes: str, optional
        free text comments
    """
    template_name = mongoengine.StringField(required=True)
    gates = mongoengine.EmbeddedDocumentListField(Gate)
    control_gates = mongoengine.EmbeddedDocumentListField(ControlGate)
    creation_date = mongoengine.DateTimeField(default=datetime.datetime.now)
    last_edit = mongoengine.DateTimeField(default=datetime.datetime.now)
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)
    meta = {
        'db_alias': 'core',
        'collection': 'gating_strategy'
    }
