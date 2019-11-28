from immunova.flow.supervised_algo.cell_classifier import CellClassifier, CellClassifierError
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


class DiscriminantAnalysis(CellClassifier):
    """
    Implement a discriminant analysis classifier for the purpose of predicting cell classification in
    flow cytometry data
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.class_weights is not None:
            print('Warning: discriminant analysis does not support class weights and so they will be ignored. '
                  'If you are handling an imbalanced dataset, we suggest you perform resampling.')
        if self.multi_label_method == 'one hot encode':
            raise CellClassifierError('Error: discriminant analysis does not support multi-label classification '
                                      'set multi_label_method to `convert`')

    def build_linear(self, **kwargs):
        """
        Build LDA model. Implements Scikit-Learn's  Linear Discriminant Analysis classifier, see
        https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
        :param kwargs: additional keyword arguments for LinearDiscriminantAnalysis (see sklearn documentation)
        :return: None
        """
        self.classifier = LinearDiscriminantAnalysis(**kwargs)

    def build_quadratic(self, **kwargs):
        """
        Build QDA model. Implements Scikit-Learn's  Quadratic Discriminant Analysis classifier, see
        https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
        :param kwargs: additional keyword arguments for QuadraticDiscriminantAnalysis (see sklearn documentation)
        :return: None
        """
        self.classifier = QuadraticDiscriminantAnalysis(**kwargs)
