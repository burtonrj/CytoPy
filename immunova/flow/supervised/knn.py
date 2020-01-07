from immunova.flow.supervised.cell_classifier import CellClassifier
from sklearn.neighbors import KNeighborsClassifier


class KNN(CellClassifier):
    """
    Implement a K Nearest Neighbours classifier for the purpose of predicting cell classification in flow cytometry data
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prefix = 'KNN'
        if self.class_weights is not None:
            print('Warning: KNN classifier does not support class weights and so they will be ignored. If you are '
                  'handling an imbalanced dataset, we suggest you perform resampling.')

    def build_model(self, **kwargs) -> None:
        """
        Build KNN model. Implements Scikit-Learn's  k-nearest neighbors classifier, see
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        :param kwargs: additional keyword arguments for KNeighborsClassifier (see sklearn documentation)
        :return: None
        """
        self.classifier = KNeighborsClassifier(**kwargs)
