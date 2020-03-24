from .cell_classifier import CellClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier


class SupportVectorMachine(CellClassifier):
    """
    Implement a Support Vector Machine for the purpose of predicting cell classification in flow cytometry data
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefix = 'SVM'

    def build_linear(self, **kwargs) -> None:
        """
        Build linear SVM. Implements Scikit-Learn's Linear Support Vector Classification, see
        https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
        Implements a one-verus-rest strategy for handling multi-class classificaiton.
        :param kwargs: additional keyword arguments for LinearSVC (see sklearn documentation)
        :return: None
        """
        if self.class_weights is None:
            self.classifier = LinearSVC(multi_class='ovr', **kwargs)
        else:
            self.classifier = LinearSVC(multi_class='ovr', class_weight=self.class_weights, **kwargs)

    def build_nonlinear(self, kernel: str or callable = 'poly', cache_size: float = 4000, **kwargs) -> None:
        """
        Build non-linear SVM. Implements Scikit-Learn's C-Support Vector Classification, see
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

        Implements one-versus-rest strategy for handling multi-class classification.
        :param kernel: kernel type to be used by the algorithm, must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
        or a function (custom kernel).
        :param cache_size: Size of kernel cache (default = 4000 MB)
        :param kwargs: additional keyword arguments for SVC (see sklearn documentation)
        :return: None
        """
        if type(kernel) != callable:
            assert kernel in ['poly', 'rbf', 'sigmoid'], "unsupported kernel type, must be one of " \
                                                         "'poly', 'rbf', or 'sigmoid'"
        if self.class_weights is None:
            self.classifier = OneVsRestClassifier(SVC(kernel=kernel, cache_size=cache_size,
                                                      probability=True, **kwargs), n_jobs=-1)
        else:
            self.classifier = OneVsRestClassifier(SVC(kernel=kernel, cache_size=cache_size, probability=True,
                                                      class_weight=self.class_weights, **kwargs), n_jobs=-1)


