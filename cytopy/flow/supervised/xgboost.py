from .cell_classifier import CellClassifier
from xgboost import XGBClassifier
import numpy as np


class XGBoostClassifier(CellClassifier):
    """
    Implement an XGBoost classifier for the purpose of predicting cell classification in cytometry data
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefix = 'XGBoost'
        self.objective = 'binary:logistic'
        if self.multi_label:
            self.objective = 'multi:softprob'

    def build_model(self, **kwargs):
        """
        Build an XGBoost model. Implements XGBoost classifier using the Sklearn wrapper, see
        https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

        Parameters
        -----------
        kwargs:
            additional keyword arguments for XGBClassifier
            (see https://xgboost.readthedocs.io/en/latest/parameter.html)

        Returns
        --------
        None
        """
        if self.multi_label:
            self.classifier = XGBClassifier(objective=self.objective,
                                            num_class=len(self.population_labels),
                                            **kwargs)
        else:
            self.classifier = XGBClassifier(objective=self.objective, **kwargs)

    def _fit(self,
             x: np.array,
             y: np.array,
             **kwargs):
        """
        Overwrites fit from base class for XGBoost specific functionality

        Parameters
        ----------
        x: Numpy.array
            Feature space
        y: Numpy.array
            Labels
        kwargs:
            Additional keyword arguments to pass to call to MODEL.fit()
            (see see https://xgboost.readthedocs.io)

        Returns
        -------
        None
        """
        assert self.classifier is not None, 'Must construct classifier prior to calling `fit` using the `build` method'
        if self.class_weights is not None:
            self.classifier.fit(x, y, sample_weight=self.class_weights, **kwargs)
        else:
            self.classifier.fit(x, y, **kwargs)
