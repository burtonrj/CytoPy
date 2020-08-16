from ...data.supervised_classifier import Classifier
from ...data.experiments import Experiment
from ...feedback import vprint, progress_bar
from ..sampling import density_dependent_downsampling, faithful_downsampling
from ..gating_tools import Gating, check_population_tree
from ..transforms import scaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from warnings import warn
import pandas as pd
import numpy as np


# class Classifier(mongoengine.document):
#    klass = mongoengine.StringField(required=True)
#    params = mongoengine.ListField()
#    features = mongoengine.ListField()
#    multi_label = mongoengine.BooleanField(default=True)
#    test_frac
#    transform = mongoengine.StringField(default="logicle")
#    threshold = mongoengine.FloatField(default=0.5)
#    scale = mongoengine.StringField()
#    scale_kwargs = mongoengine.ListField()
#    balance = mongoengine.StringField()
#    balance_dict
#    downsample = mongoengine.StringField()
#    downsample_kwargs = mongoengine.ListField()


class CellClassifier:
    def __init__(self,
                 classifier: Classifier,
                 experiment: Experiment,
                 ref_sample: str,
                 population_labels: list or None = None,
                 verbose: bool = True):
        assert classifier.klass in globals().keys(), f"Module {classifier.klass} not found, have you imported it into " \
                                                     f"the working environment?"
        self.classifier = classifier
        self.experiment = experiment
        kwargs = {k: v for k, v in classifier.params}
        self.model = globals()[classifier.klass](**kwargs)
        self.verbose = verbose
        self.print = vprint(verbose)
        self.class_weights = None
        self.threshold = classifier.threshold or 0.5
        self.print("----- Building CellClassifier -----")
        assert ref_sample in experiment.list_samples(), "Invalid reference sample, could not be found in given " \
                                                        "experiment"
        self.print("Loading reference sample...")
        ref = Gating(experiment=experiment,
                     sample_id=ref_sample,
                     include_controls=False)
        self.population_labels = ref.valid_populations(populations=population_labels)
        assert len(self.population_labels) > 2, "Reference sample does not contain any gated populations"
        if len(self.population_labels) != len(population_labels):
            warn("One or more given population labels does not tie up with the populations in the reference "
                 f"sample, defaulting to the following valid labels: {self.population_labels}")
        else:
            self.print(f"CellClassifier will attempt to predict the following populations: {self.population_labels}")
        check_population_tree(gating=ref, populations=self.population_labels)
        features = [x for x in ref.data.get("primary").columns if x in classifier.features]
        if len(features) != len(classifier.features):
            warn(f"One or more features missing from reference sample, "
                 f"proceeding with the following features: {features}")
        self.classifier.features = features
        downstream = ref.list_downstream_populations(self.population_labels[0])
        assert all([x in downstream for x in self.population_labels[1:]]), \
            "The first population in population_labels should be the 'root' population, with all further populations " \
            "being downstream from this 'root'. The given population_labels has one or more populations that is not " \
            "downstream from the given root."
        self.print("Preparing training and testing data...")
        if classifier.multi_label:
            self.threshold = None
            (self.train_X, self.train_y,
             self.test_X, self.test_y,
             self.mappings) = self._multiclass_labels(ref=ref)
        else:
            (self.train_X, self.train_y,
             self.test_X, self.test_y,
             self.mappings) = self._singleclass_labels(ref=ref)
        if classifier.scale:
            self.print("Scaling data...")
            if classifier.scale not in ["standard", "norm", "robust", "power"]:
                warn('Scale method must be one of: "standard", "norm", "robust", "power", defaulting to standard '
                     'scaling (unit variance)')
                classifier.scale = "standard"
            kwargs = classifier.scale_kwargs or {}
            self.train_X = scaler(self.train_X, scale_method=classifier.scale, **kwargs)
            self.test_X = scaler(self.test_X, scale_method=classifier.scale, **kwargs)
        else:
            warn("For the majority of classifiers it is recommended to scale the data (exception being tree-based "
                 "algorithms)")
        if classifier.balance:
            self._balance()
        if classifier.downsample:
            self._downsample()
        self.print('Ready for training!')

    def _binarize_labels(self,
                         ref: Gating):
        root = ref.get_population_df(population_name=self.population_labels[0],
                                     transform=self.classifier.transform)
        for pop in self.population_labels[1:]:
            root[pop] = 0
            root.loc[ref.populations.get(pop).index, pop] = 1
        return root[self.classifier.features].values, root[self.population_labels[1:]].values

    def _multiclass_labels(self,
                           ref: Gating):
        X, y = self._binarize_labels(ref=ref)
        train_X = X.sample(frac=1 - self.classifier.test_frac)[self.classifier.features]
        test_X = X[~X.index.isin(train_X.index.values)][self.classifier.features]
        labels = y.reset_index().melt(id_vars="index", var_name="label", value_name="binary")
        labels = labels[labels.binary == 1].groupby("index").apply(lambda x: ",".join(x.label.values))
        y["label"] = "None"
        y.loc[labels.index, "label"] = labels
        mappings = {x: i for x, i in enumerate(y.label.unique())}
        y = y.label.apply(lambda x: mappings.get(x))
        mappings = {i: x for x, i in mappings.items()}
        train_y = y.loc[train_X.index.values]
        test_y = y.loc[test_X.index.values]
        return train_X, train_y, test_X, test_y, mappings

    def _singleclass_labels(self,
                            ref: Gating):
        root = ref.get_population_df(population_name=self.population_labels[0],
                                     transform=self.classifier.transform)
        y = np.zeros(root.shape[0])
        mappings = dict()
        mappings[0] = "None"
        for i, pop in enumerate(self.population_labels[1:]):
            pop_idx = ref.populations[pop].index
            np.put(y, pop_idx, i + 1)
            mappings[i + 1] = pop
        train_X = root.sample(frac=1 - self.classifier.test_frac)[self.classifier.features]
        test_X = root[~root.index.isin(train_X.index.values)][self.classifier.features]
        train_y = y[train_X.index.values]
        test_y = y[test_X.index.values]
        return train_X, train_y, test_X, test_y, mappings

    def _balance(self):
        if self.classifier.balance == "oversample":
            ros = RandomOverSampler(random_state=42)
            self.train_X, self.train_y = ros.fit_resample(self.train_X, self.train_y)
        elif self.classifier.balance == "auto-weights":
            weights = compute_class_weight('balanced',
                                           classes=np.array(list(self.mappings.keys())),
                                           y=self.train_y)
            class_weights = {k: w for k, w in zip(self.mappings.keys(), weights)}
            self.class_weights = list(map(lambda x: class_weights[x], self.train_y))
        elif self.classifier.balance_dict:
            class_weights = {k: w for k, w in self.classifier.balance_dict}
            self.class_weights = list(map(lambda x: class_weights[x], self.train_y))
        else:
            raise ValueError("Balance should have a value 'oversample' or 'auto-weights', alternatively, "
                             "populate balance_dict with (label, weight) pairs")

    def _downsample(self):
        kwargs = {k: v for k, v in self.classifier.downsample_kwargs}
        if self.classifier.downsample == "uniform":
            frac = kwargs.get("frac", 0.5)
            self.train_X = self.train_X.sample(frac=frac)
        elif self.classifier.downsample == "density":
            self.train_X = density_dependent_downsampling(data=self.train_X,
                                                          features=self.classifier.features,
                                                          **kwargs)
        elif self.classifier.downsample == "faithful":
            self.train_X = faithful_downsampling(data=self.train_X, **kwargs)
        raise ValueError("Downsample should have a value of: 'uniform', 'density', or 'faithful'")

    def fit(self,
            test: bool = True):
        pass

    def predict(self,
                sample_id: str):
        g = Gating(experiment=self.experiment,
                   sample_id=sample_id,
                   include_controls=False)
        assert self.population_labels[0] in g.populations.keys(), f"Root population {self.population_labels[0]} " \
                                                                  f"missing from {sample_id}"
        pass

    def fit_cv(self,
               test: bool = True):
        pass

    def gridsearchcv(self,
                     test: bool = True):
        pass

    def randomsearchcv(self,
                       test: bool = True):
        pass

    def predict_and_validate(self,
                             validation_id: str,
                             print_report_card: bool = True):
        pass

    def save_classifier(self):
        pass

    def _generate_gating_obj(self):
        pass

    def _build_tree(self,
                    tree: dict,
                    root_pop: str):
        for pop in self.population_labels[1:]:
            if not self.classifier.multi_label:
                parent = self.population_labels[0]
            else:
                parent =
            tree[pop] = Node(pop, parent)
                tree[f'{self.prefix}_{pop}'] = Node(f'{self.prefix}_{pop}',
                                                    parent=tree[f'{self.prefix}_{label[i - 1]}'],
                                                    collection=ChildPopulationCollection(gate_type='sml'))
        return tree

    def _create_population(self):
        pass
