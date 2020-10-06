from .geometry import ThresholdGeom, PolygonGeom
from ..flow.transforms import scaler
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
    if idx is None:
        idx = data.index.values
    # ToDo this should be more robust
    for x in ["Time", "time"]:
        if x in data.columns:
            data.drop(x, 1, inplace=True)
    summary_method = summary_method or np.median
    signature = data.loc[idx].apply(summary_method)
    return {x[0]: x[1] for x in zip(signature.index, signature.values)}


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
    preprocessing = mongoengine.DictField()
    postprocessing = mongoengine.DictField()
    method = mongoengine.StringField()
    method_kwargs = mongoengine.DictField()

    meta = {
        'db_alias': 'core',
        'collection': 'gates',
        'allow_inheritance': True
    }

    def _preprocessing(self,
                       data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing procedures to the given dataframe, as per the preprocessing
        kwargs stored in self.preprocessing. This can include:
            * Transformations
            * Dimensionality reduction
            * Downsampling

        Parameters
        ===========
        data: Pandas.DataFrame

        Returns
        -------
        Pandas.DataFrame
        """
        pass

    def _postprocessing(self,
                        data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply post-processing to the given data dataframe following the keyword arguments
        in self.postprocessing. Currently only supporting upsampling.

        Parameters
        ----------
        data: Pandas.DataFrame

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

    def _add_child(self,
                   child: ChildThreshold):
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

    def label_children(self):
        pass

    def _match_to_children(self):
        pass

class PolygonGate(Gate):
    """
    Polygon gates generate polygon shapes that capture populations of varying shapes. These can
    be generated by any number of clustering algorithms.
    """
    children = mongoengine.EmbeddedDocumentListField(ChildPolygon)


class EllipseGate(Gate):
    """
    Ellipse gates generate circular or elliptical gates and can be generated from algorithms that are
    centroid based (like K-means) or probabilistic methods that estimate the covariance matrix of one
    or more gaussian components such as mixture models.
    """
    children = mongoengine.EmbeddedDocumentListField(ChildPolygon)
