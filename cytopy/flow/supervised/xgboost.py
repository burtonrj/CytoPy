from cytopy.flow.supervised.cell_classifier import CellClassifier
from xgboost import XGBClassifier


class XGBoostClassifier(CellClassifier):
    """
    Implement an XGBoost classifier for the purpose of predicting cell classification in flow cytometry data
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
        :param kwargs: additional keyword arguments for XGBClassifier
        (see https://xgboost.readthedocs.io/en/latest/parameter.html)
        :return: None
        """
        if self.multi_label:
            self.classifier = XGBClassifier(objective=self.objective,
                                            num_class=len(self.population_labels),
                                            **kwargs)
        else:
            self.classifier = XGBClassifier(objective=self.objective, **kwargs)

    def _fit(self, x, y, **kwargs):
        assert self.classifier is not None, 'Must construct classifier prior to calling `fit` using the `build` method'
        if self.class_weights is not None:
            self.classifier.fit(x, y, sample_weight=self.class_weights, **kwargs)
        else:
            self.classifier.fit(x, y, **kwargs)
