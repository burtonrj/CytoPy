from ...data.supervised_classifier import Classifier


class CellClassifier:
    def __init__(self,
                 classifier: Classifier,
                 verbose: bool = True):
        assert classifier.klass in globals().keys(), f"Module {classifier.klass} not found, have you imported it into " \
                                                     f"the working environment?"
        self.classifier = classifier
        kwargs = {k: v for k, v in classifier.params}
        self.model = globals()[classifier.klass](**kwargs)

    def fit(self):
        pass

    def predict(self):
        pass