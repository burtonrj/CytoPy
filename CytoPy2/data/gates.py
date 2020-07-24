from .fcs import Population, PopulationGeometry
from warnings import warn
import pandas as pd
import mongoengine


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
        if "definition" in kwargs.keys():
            self._definition = kwargs.pop("definition")
        if "template_geometry" in kwargs.keys():
            self._template_geometry = kwargs.pop("template_geometry")

    @property
    def definition(self):
        return self._definition

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

    @property
    def template_geometry(self):
        return self._template_geometry

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


class PreProcess(mongoengine.EmbeddedDocument):
    downsample_method = mongoengine.StringField(required=False, choices=["uniform",
                                                                         "faithful",
                                                                         "density",
                                                                         "probabilities",
                                                                         "custom"])
    downsample_kwargs = mongoengine.ListField()
    transform_x = mongoengine.StringField(default="linear",
                                          choices=["linear",
                                                   "logicle",
                                                   "log_transform",
                                                   "hyperlog",
                                                   "asinh",
                                                   "percentile_rank",
                                                   "Yeo-Johnson",
                                                   "RobustScale"])
    transform_y = mongoengine.StringField(default="linear",
                                          choices=["linear",
                                                   "logicle",
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

    def _add_child(self,
                   population: Population):
        name = population.population_name
        if self.shape == "threshold":
            name = population.definition
        new_child = ChildDefinition(population_name=name,
                                    )

    def _apply_preprocessing(self,
                             data: pd.DataFrame):
        sample = None
        return data, sample

    def apply(self,
              data: pd.DataFrame):
        if not self._defined:
            # Applying for the first time, resulting populations should populate the child definitions
            data, sample = self._apply_preprocessing(data=data)
            if sample is not None:
                populations = self.model.fit_predict(sample)
            else:
                populations = self.model.fit_predict(data)
            for pop in populations:
                self._add_child(pop)
        else:
            # Pre-defined gate, resulting populations should be matched to the child definitions
            data, sample = self._apply_preprocessing(data=data)
            if sample is not None:
                populations = self.model.fit_predict(sample)
            else:
                populations = self.model.fit_predict(data)
            return self._match_to_children(populations)

    def _apply_postprocessing(self):
        pass

    def save(self, *args, **kwargs):
        assert self._defined, "Gate is newly created and has not been defined. Call 'label_children' to complete " \
                              "gating definition"
        super().save(*args, **kwargs)

