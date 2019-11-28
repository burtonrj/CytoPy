from immunova.flow.supervised_algo.cell_classifier import CellClassifier, CellClassifierError
from sklearn.svm import LinearSVC, SVC


class SupportVectorMachine(CellClassifier):
    """
    Implement a Support Vector Machine for the purpose of predicting cell classification in flow cytometry data
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_linear(self, **kwargs):
        if self.class_weights is None:
            self.classifier = LinearSVC(**kwargs)
        else:
            self.classifier = LinearSVC(class_weight=self.class_weights, **kwargs)

    def build_nonlinear(self, kernel: str = 'poly', cache_size: float = 4000, **kwargs):
        if kernel not in ['poly', 'rbf', 'sigmoid']:
            raise CellClassifierError("Error: unsupported kernel type, must be one of 'poly', 'rbf', or 'sigmoid'")
        if self.class_weights is None:
            self.classifier = SVC(kernel=kernel, cache_size=cache_size, **kwargs)
        else:
            self.classifier = SVC(kernel=kernel, cache_size=cache_size,
                                  class_weight=self.class_weights, **kwargs)


