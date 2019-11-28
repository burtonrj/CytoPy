from immunova.flow.supervised_algo.cell_classifier import CellClassifier, CellClassifierError
from sklearn.svm import LinearSVC, SVC


class SupportVectorMachine(CellClassifier):
    """
    Implement a Support Vector Machine for the purpose of predicting cell classification in flow cytometry data
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.multi_label_method == 'one hot encode':
            raise CellClassifierError('Error: Support Vector Machine does not support multi-label classification '
                                      'set method to `convert`')

    def build_linear(self, **kwargs) -> None:
        """
        Build linear SVM. Implements Scikit-Learn's Linear Support Vector Classification, see
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
        :param kwargs: additional keyword arguments for LinearSVC (see sklearn documentation)
        :return: None
        """
        if self.class_weights is None:
            self.classifier = LinearSVC(**kwargs)
        else:
            self.classifier = LinearSVC(class_weight=self.class_weights, **kwargs)

    def build_nonlinear(self, kernel: str or callable = 'poly', cache_size: float = 4000, **kwargs) -> None:
        """
        Build non-linear SVM. Implements Scikit-Learn's C-Support Vector Classification, see
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        :param kernel: kernel type to be used by the algorithm, must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
        or a function (custom kernel).
        :param cache_size: Size of kernel cache (default = 4000 MB)
        :param kwargs: additional keyword arguments for SVC (see sklearn documentation)
        :return: None
        """
        if type(kernel) != callable:
            if kernel not in ['poly', 'rbf', 'sigmoid']:
                raise CellClassifierError("Error: unsupported kernel type, must be one of 'poly', 'rbf', or 'sigmoid'")
        if self.class_weights is None:
            self.classifier = SVC(kernel=kernel, cache_size=cache_size, **kwargs)
        else:
            self.classifier = SVC(kernel=kernel, cache_size=cache_size,
                                  class_weight=self.class_weights, **kwargs)


