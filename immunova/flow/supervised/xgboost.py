from immunova.flow.supervised.cell_classifier import CellClassifier, CellClassifierError
from xgboost import XGBClassifier, DMatrix


class XGBoostClassifier(CellClassifier):
    """
    Implement an XGBoost classifier for the purpose of predicting cell classification in flow cytometry data
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.objective = 'binary:logistic'
        if self.multi_label:
            self.objective = 'multi:softmax'

    def build_model(self, **kwargs):
        """
        Build an XGBoost model. Implements XGBoost classifier using the Sklearn wrapper, see
        https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
        :param kwargs: additional keyword arguments for XGBClassifier
        (see https://xgboost.readthedocs.io/en/latest/parameter.html)
        :return: None
        """
        if self.class_weights is not None:
            self._build_weighted_model(**kwargs)
        if self.multi_label:
            self.classifier = XGBClassifier(objective=self.objective,
                                            num_class=len(self.population_labels),
                                            **kwargs)
        else:
            self.classifier = XGBClassifier(objective=self.objective, **kwargs)

    def _build_weighted_model(self, **kwargs):
        """
        Internal use only. Converts train_X parameter to type DMatrix and associates weights based on given
        class weights
        :param kwargs: additional keyword arguments for XGBClassifier
        (see https://xgboost.readthedocs.io/en/latest/parameter.html)
        :return: None
        """
        weights = list(map(lambda x: self.class_weights[x], self.train_y))
        self.train_X = DMatrix(self.train_X, weight=weights)
        if self.multi_label:
            self.classifier = XGBClassifier(objective=self.objective,
                                            num_class=len(self.population_labels),
                                            **kwargs)
        else:
            self.classifier = XGBClassifier(objective=self.objective, **kwargs)
