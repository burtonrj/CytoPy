from .cell_classifier import CellClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


class DiscriminantAnalysis(CellClassifier):
    """
    Implement a discriminant analysis classifier for the purpose of predicting cell classification in
    cytometry data
    """
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.prefix = 'DiscriminantAnalysis'
        if self.class_weights is not None:
            print('Warning: discriminant analysis does not support class weights and so they will be ignored. '
                  'If you are handling an imbalanced dataset, we suggest you perform resampling.')

    def build_linear(self, **kwargs):
        """
        Build LDA model. Implements Scikit-Learn's  Linear Discriminant Analysis classifier, see
        https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html

        Parameters
        -----------
        kwargs:
            additional keyword arguments for LinearDiscriminantAnalysis (see sklearn documentation)

        Returns
        --------
        None
        """
        self.classifier = LinearDiscriminantAnalysis(**kwargs)

    def build_quadratic(self, **kwargs):
        """
        Build QDA model. Implements Scikit-Learn's  Quadratic Discriminant Analysis classifier, see
        https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html

        Parameters
        ----------
        kwargs:
            additional keyword arguments for QuadraticDiscriminantAnalysis (see sklearn documentation)

        Returns
        --------
        None
        """
        self.classifier = QuadraticDiscriminantAnalysis(**kwargs)
