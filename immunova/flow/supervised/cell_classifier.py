from immunova.data.fcs_experiments import FCSExperiment
from immunova.data.fcs import FileGroup, File, ChannelMap, Population
from immunova.data.panel import Panel
from immunova.flow.gating.actions import Gating
from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.supervised.utilities import standard_scale, norm_scale, find_common_features, \
    predict_class, random_oversampling
from immunova.flow.gating.utilities import density_dependent_downsample, check_downstream_overlaps
from immunova.flow.supervised.evaluate import evaluate_model, report_card
from immunova.flow.utilities import progress_bar
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer
from umap import UMAP
from phate import PHATE
from multiprocessing import Pool, cpu_count
from seaborn import heatmap
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _check_columns(data: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Given a dataframe and a list of expected features, print missing columns and return new dataframe
    with only valid features
    :param data: DataFrame for checking
    :param features: list of features (column names)
    :return: new 'valid' DataFrame
    """
    valid_features = [f for f in features if f in data.columns]
    if len(valid_features) < len(features):
        print(f'The following features are missing from the training data and will be excluded from the '
              f'model {list(set(features) - set(valid_features))}')
    return data[valid_features]


def multi_process_ordered(func: callable, x: iter, chunksize: int = 100) -> list:
    """
    Map a function (func) to some iterable (x) using multiprocessing. Iterable will be divided into chunks
    of size `chunksize`. The chunks will be given as a list of tuples, where the first value is the index of
    the chunk and the second value the chunk itself; this allows for ordered reassembly once processing has
    completed. For this reason, the function `func` MUST handle and return the index of the chunk upon which
    it acts.
    :param func: callable function to applied in parallel
    :param x: some iterable to apply the function to
    :param chunksize: size of chunks for multiprocessing
    :return: ordered list of funciton output
    """
    def chunks(l: iter):
        for i in range(0, len(l), chunksize):
            yield l[i:i + chunksize]
    indexed_chunks = [(i, xc) for i, xc in enumerate(chunks(x))]
    pool = Pool(cpu_count())
    results = pool.map(func, indexed_chunks)
    results = [x[1] for x in sorted(results, key=lambda t: t[0])]
    results = [x for l in results for x in l]
    pool.close()
    pool.join()
    return results


def _assign_labels(x: tuple, labels: dict) -> tuple:
    """
    Internal function. Used for assigning a unique 'fake' label to each multi-label sequence in a set in
    a parallel process
    :param x: tuple; (index of chunk, list of multi-label sequence's)
    :param labels: dicitonary; {multi-label sequence: unique 'fake' label}
    :return: tuple; (index of chunk, list of 'fake' labels)
    """
    return x[0], [labels[np.array_str(s)] for s in x[1]]


def __channel_mappings(features: list, panel: Panel) -> list:
    """
    Internal function. Given a list of features and a Panel object, return a list of ChannelMapping objects
    that correspond with the given Panel.
    :param features: list of features to compare to Panel object
    :param panel: Panel object of channel mappings
    :return: list of ChannelMappings
    """
    mappings = list()
    panel_mappings = panel.mappings
    for f in features:
        channel = list(filter(lambda x: x.channel == f, panel_mappings))
        marker = list(filter(lambda x: x.marker == f, panel_mappings))
        if (len(channel) > 1 or len(marker) > 1) or (len(channel) == 1 and len(marker) == 1):
            raise ValueError(f'Feature {f} found in multiple channel_marker mappings in panel {panel.panel_name}: '
                             f'{channel}; {marker}.')
        if len(channel) == 0:
            if len(marker) == 0:
                raise ValueError(f'Feature {f} not found in associated panel')
            mappings.append(ChannelMap(channel=marker[0].channel,
                                       marker=marker[0].marker))
            continue
        mappings.append(ChannelMap(channel=channel[0].channel,
                                   marker=channel[0].marker))
    return mappings


def create_reference_sample(experiment: FCSExperiment,
                            root_population='root',
                            exclude: list or None = None,
                            new_file_name: str or None = None,
                            sampling_method: str = 'uniform',
                            sample_n: int = 1000,
                            sample_frac: float or None = None) -> None:
    """
    Given some experiment and a root population that is common to all fcs file groups within this experiment, take
    a sample from each and create a new file group from the concatenation of these data. New file group will be created
    and associated to the given FileExperiment object.
    If no file name is given it will default to '{Experiment Name}_sampled_data'
    :param experiment: FCSExperiment object for corresponding experiment to sample
    :param root_population: if the files in this experiment have already been gated, you can specify to sample
    from a particular population e.g. Live CD3+ cells or Live CD45- cells
    :param exclude: list of sample IDs for samples to be excluded from sampling process
    :param new_file_name: name of file group generated
    :param sampling_method: method to use for sampling files (currently only supports 'uniform')
    :param sample_n: number of events to sample from each file
    :param sample_frac: fraction of events to sample from each file (default = None, if not None then sample_n is
    ignored)
    :return: None
    """
    def sample(d):
        if sampling_method == 'uniform':
            if sample_frac is None:
                if d.shape[0] > sample_n:
                    return d.sample(sample_n)
                return d
            return d.sample(frac=sample_frac)
        raise ValueError('Error: currently only uniform sampling is implemented in this version of immunova')

    print('-------------------- Generating Reference Sample --------------------')
    if exclude is None:
        exclude = []
    if new_file_name is None:
        new_file_name = f'{experiment.experiment_id}_sampled_data'
    print('Finding features common to all fcs files...')
    features = find_common_features(experiment=experiment, exclude=exclude)
    channel_mappings = __channel_mappings(features,
                                          experiment.panel)
    files = [f for f in experiment.list_samples() if f not in exclude]
    data = pd.DataFrame()
    for f in files:
        print(f'Sampling {f}...')
        g = Gating(experiment, f, include_controls=False)
        if root_population not in g.populations.keys():
            print(f'Skipping {f} as {root_population} is absent from gated populations')
            continue
        df = g.get_population_df(root_population)[features]
        data = pd.concat([data, sample(df)])
    data = data.reset_index(drop=True)
    print('Sampling complete!')
    new_filegroup = FileGroup(primary_id=new_file_name)
    new_filegroup.flags = 'sampled data'
    new_file = File(file_id=new_file_name,
                    compensated=True,
                    channel_mappings=channel_mappings)
    print('Inserting sampled data to database...')
    new_file.put(data.values)
    new_filegroup.files.append(new_file)
    root_p = Population(population_name='root',
                        prop_of_parent=1.0, prop_of_total=1.0,
                        warnings=[], geom=[['shape', None], ['x', 'FSC-A'], ['y', 'SSC-A']])
    root_p.save_index(data.index.values)
    new_filegroup.populations.append(root_p)
    print('Saving changes...')
    mid = new_filegroup.save()
    experiment.fcs_files.append(new_filegroup)
    experiment.save()
    print(f'Complete! New file saved to database: {new_file_name}, {mid}')
    print('-----------------------------------------------------------------')


class BoxCox:
    """
    Wrapper to bypass transform call when BoxCox is chosen as scaling method
    """
    def __init__(self):
        pass

    @staticmethod
    def transform(x):
        x, _ = norm_scale(x, range=(0.1, 1.1))
        pp = PowerTransformer(method='box-cox', standardize=False)
        return pp.fit_transform(x)


class CellClassifier:
    """
    Class for performing classification of cells by supervised machine learning.
    """
    def __init__(self, experiment: FCSExperiment, reference_sample: str, population_labels: list, features: list,
                 multi_label: bool = True, transform: str = 'logicle', root_population: str = 'root',
                 threshold: float = 0.5, scale: str or None = 'Standardise',
                 balance_method: None or str or dict = None, frac: float or None = None,
                 **downsampling_kwargs):
        """
        Constructor for CellClassifier
        :param experiment: FCSExperiment for classification
        :param reference_sample: sample ID for training sample (see 'create_reference_sample')
        :param population_labels: list of populations for prediction (populations must be valid gated populations
        that exist in the reference sample)
        :param features: list of features (channel/marker column names) to include
        :param multi_label: If True, cells can belong to multiple classes and the problem is treated as a
        'multi-label' classification task. Labels will be binarised and a new 'fake' label generated for each
        unique instance of a cell label. This is important to account for correlations between individual labels.
        :param transform: name of transform method to use (see flow.gating.transforms for info)
        :param root_population: name of root population i.e. the population to derive training and test data from
        :param threshold: minimum probability threshold to class as positive (default = 0.5)
        :param scale: how to scale the data prior to learning; either 'Standardise', 'Normalise' or None.
        Standardise scales using the standard score, removing the mean and scaling to unit variance
        (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        Normalise scales data between 0 and 1 using the MinMaxScaler
        (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
        """
        print('Constructing cell classifier object...')
        self.experiment = experiment
        self.transform = transform
        self.multi_label = multi_label
        self.classifier = None
        self.preprocessor = None
        self.features = features
        self.root_population = root_population
        self.threshold = threshold
        self.mappings = None
        self.class_weights = None
        self.prefix = 'sml'
        print('Loading information on reference sample...')
        ref = Gating(self.experiment, reference_sample, include_controls=False)

        self.population_labels = ref.valid_populations(population_labels)
        assert len(self.population_labels) > 2, f'Error: reference sample {reference_sample} does not contain any '\
                                                f'gated populations, please ensure that the reference sample has ' \
                                                f'been gated prior to training.'
        print('Preparing training data and labels...')
        if multi_label:
            self.threshold = None
            self.train_X, self.train_y, self.mappings = self.multiclass_labels(ref, features, root_population)
        else:
            self.train_X, self.train_y, self.mappings = self.singleclass_labels(ref, features, root_population)
        print('Scaling data...')
        if scale == 'Standardise':
            self.train_X, self.preprocessor = standard_scale(self.train_X)
        elif scale == 'Normalise':
            self.train_X, self.preprocessor = norm_scale(self.train_X)
        elif scale == 'Box-Cox':
            self.train_X, _ = norm_scale(self.train_X, range=(0.1, 1.1))
            pp = PowerTransformer(method='box-cox', standardize=False)
            self.train_X = pp.fit_transform(self.train_X)
            self.preprocessor = BoxCox()
        elif scale is None:
            print('Warning: it is recommended that data is scaled prior to training. Unscaled data can result '
                  'in some weights updating faster than others, having a negative effect on classifier performance')
        else:
            raise ValueError('Error: scale method not recognised, must be either `Standardise` or `Normalise`')

        if type(balance_method) == str:
            assert balance_method in ['density', 'oversample', 'auto_weights'], \
                'Balance method must be one of ["density", "oversample", "auto_weights"] or custom weights given as ' \
                'a python dictionary object'
            if balance_method in ['density', 'oversample']:
                print('Balancing dataset by sampling...')
                if frac:
                    self.balance_dataset(method=balance_method, frac=frac, **downsampling_kwargs)
                else:
                    self.balance_dataset(method=balance_method, **downsampling_kwargs)
            else:
                weights = compute_class_weight('balanced',
                                               classes=np.array(list(self.mappings.keys())),
                                               y=self.train_y)
                class_weights = {k: w for k, w in zip(self.mappings.keys(), weights)}
                self.class_weights = list(map(lambda x: class_weights[x], self.train_y))
        elif type(balance_method) == dict:
            self.class_weights = list(map(lambda x: balance_method[x], self.train_y))
        print('Ready for training!')

    def _binarize_labels(self, ref: Gating, features: list, root_pop: str) -> (pd.DataFrame, np.array):
        """
        Generate feature space and labels when a cell can belong to multiple populations. Labels are returned as a
        one-hot encoded sequence for each cell that represents the populations that cell belongs to (e.g for the
        population labels ['CD3+', 'CD4+', 'CD8+'] an encoding of [1,0,1] would be a CD3+CD8+ cell type.
        (multi-label learning)
        :param ref: Gating object to retrieve data from
        :param features: list of features for training
        :return: DataFrame of feature space, array of target labels
        """
        root = _check_columns(ref.get_population_df(root_pop,
                                                    transform=True,
                                                    transform_method=self.transform),
                              features)
        self.features = list(root.columns)
        for pop in self.population_labels:
            root[pop] = 0
            root.loc[ref.populations[pop].index, pop] = 1
        return root[self.features].values, root[self.population_labels].values

    def multiclass_labels(self, ref: Gating, features: list, root_pop: str) -> (pd.DataFrame, np.array, dict):
        """
        Generate feature space and labels for a multi-label multi-class classification problem and handle the
        multi-label aspect of this problem by converting multi-label signatures to single labels. This can be important
        if there are correlations between labels e.g. the classification CD3+/- and CD4+/- are not independent because
        a cell that is CD4+, CD3- will have a very different meaning to a cell that is CD4+ CD3+. The problem is
        converted to a multi class, single label classification by converting each unique one-hot encoded representation
        to a 'fake label'.
        :param ref: Gating object to retrieve data from
        :param features: list of features for training
        :return: DataFrame of feature space, array of target labels, and a dictionary of cell population mappings
        """
        train_X, train_y = self._binarize_labels(ref, features, root_pop)
        labels = {np.array2string(x): i for i, x in enumerate(np.unique(train_y, axis=0))}
        label_f = partial(_assign_labels, labels=labels)
        train_y = np.array(multi_process_ordered(label_f, train_y))
        labels = {i: list(map(lambda x: int(x), a.replace('[', '').replace(']', '').split(' ')))
                  for a, i in labels.items()}
        pops = np.array(self.population_labels)
        mappings = {i: pops[np.where(np.array(x) == 1)] for i, x in labels.items()}
        mappings = {k: v if len(v) > 0 else np.array(['None']) for k, v in mappings.items()}
        return train_X, train_y, mappings

    def singleclass_labels(self, ref: Gating, features: list, root_pop: str) -> (pd.DataFrame, np.array, dict):
        """
        Generate feature space and labels where a cell can belong to only one population
        :param ref: Gating object to retrieve data from
        :param features: list of features for training
        :return: DataFrame of feature space, array of target labels, mappings of cell population labels
        """
        if check_downstream_overlaps(ref, root_pop, self.population_labels):
            raise ValueError('Error: one or more population dependency errors')
        root = ref.get_population_df(root_pop, transform=True, transform_method=self.transform)[features]
        y = np.zeros((root.shape[0], len(self.population_labels)))
        mappings = dict()
        for i, pop in enumerate(self.population_labels):
            pop_idx = ref.populations[pop].index
            np.put(y, pop_idx, i + 1)
            mappings[i+1] = np.array([pop])
        return root, y, mappings

    def train_test_split(self, test_size=0.3) -> list:
        """
        Create train/test split of data using Sklearn's train_test_split function
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
        :param test_size: size of test population as a proportion of the the dataset (default = 0.3)
        :return: List containing train-test split of inputs.
        """
        return train_test_split(self.train_X, self.train_y, test_size=test_size, random_state=42)

    def __save_gating(self, target: Gating, y_hat: np.array, root_pop: str) -> Gating:
        """
        Internal method. Given some Gating object of the target file for prediction and the predicted labels,
        generate new population objects and insert them into the Gating object
        :param target: Gating object of the target file
        :param y_hat: array of predicted population labels
        :return: None
        """
        parent = target.get_population_df(population_name=root_pop)
        new_populations = ChildPopulationCollection(gate_type='sml')
        for i, pops in self.mappings.items():
            y_ = np.where(y_hat == i)
            idx = target.populations[root_pop].index[y_]
            for p in pops:
                l = f'{self.prefix}_{p}'
                if l not in new_populations.populations.keys():
                    new_populations.add_population(name=l)
                    new_populations.populations[l].update_geom(shape='sml', x=None, y=None)
                new_populations.populations[l].update_index(idx)
        target.update_populations(new_populations, parent, parent_name=root_pop, warnings=[])
        return target

    def predict(self, target_sample: str, return_gating: bool = True,
                root_population: str or None = None) -> Gating or (np.array, np.array):
        """
        Given a sample ID, predict cell populations. Model must already be trained. Results are saved as new
        populations in a Gating object returned to the user.
        :param return_gating:
        :param root_population:
        :param target_sample: Name of file for prediction. Must belong to experiment associated to CellClassifier obj.
        :return: Gating object containing new predicted populations.
        """
        assert self.classifier is not None, 'Model must be trained prior to prediction'
        root_pop = root_population
        if root_pop is None:
            root_pop = self.root_population
        target = Gating(self.experiment, target_sample, include_controls=False)
        x = target.get_population_df(root_pop, transform=True, transform_method=self.transform)
        x = _check_columns(x, self.features).values
        # Standardise/normalise if necessary
        if self.preprocessor is not None:
            x = self.preprocessor.transform(x)
        # Make prediction
        y_probs = self.classifier.predict_proba(x)
        y_hat = np.array(predict_class(y_probs, self.threshold))
        if return_gating:
            return self.__save_gating(target, y_hat, root_pop)
        return y_probs, y_hat

    @staticmethod
    def _exclude_class(x, y, mappings, exclude_class):
        exclude_class = [np.array(x) for x in exclude_class]
        mappings = {k: v for k, v in mappings.items() if not any([np.array2string(v) == np.array2string(x) for x in exclude_class])}
        mask = [c in mappings.keys() for c in y]
        y = y[mask]
        x = x[mask]
        return x, y, mappings

    def _fit(self, x, y, **kwargs):
        """
        Fit the model. Should be called internally.
        :param x: feature space
        :param y: target labels
        :param kwargs: keyword arguments t
        :return:
        """
        assert self.classifier is not None, 'Must construct classifier prior to calling `fit` using the `build` method'
        self.classifier._fit(x, y, **kwargs)

    def train_cv(self, k: int = 5, **kwargs) -> pd.DataFrame:
        """
        Fit classifier to training data using cross-validation
        :param k: Number of folds for cross-validation (default = 5)
        :param kwargs: kwargs: Optional additional kwargs for model fit.
        :return: Pandas DataFrame detailing performance
        """
        kf = KFold(n_splits=k)
        train_performance = list()
        test_performance = list()
        print(f'----------- Cross Validation: {k} folds -----------')
        idx = kf.split(self.train_X)
        for i in progress_bar(range(k)):
            train_index, test_index = next(idx)
            train_x, test_x = self.train_X[train_index], self.train_X[test_index]
            train_y, test_y = self.train_y[train_index], self.train_y[test_index]
            self._fit(train_x, train_y, **kwargs)
            p = evaluate_model(self.classifier, train_x, train_y, self.threshold)
            p['k'] = i
            train_performance.append(p)
            p = evaluate_model(self.classifier, test_x, test_y, self.threshold)
            p['k'] = i
            test_performance.append(p)

        train_performance = pd.concat(train_performance)
        train_performance['test_train'] = 'train'
        test_performance = pd.concat(test_performance)
        test_performance['test_train'] = 'test'
        return pd.concat([train_performance, test_performance])

    def train(self, **kwargs):
        self._fit(self.train_X, self.train_y, **kwargs)

    def train_holdout(self, holdout_frac: float = 0.3, print_report_card: bool = False, **kwargs) -> pd.DataFrame:
        """
        Fit classifier to training data and evaluate on holdout data.
        :param holdout_frac: Proportion of data to keep as holdout
        :param kwargs: kwargs: Optional additional kwargs for model fit.
        :return: Pandas DataFrame detailing performance
        """
        train_x, test_x, train_y, test_y = self.train_test_split(test_size=holdout_frac)
        self._fit(train_x, train_y, **kwargs)
        if not print_report_card:
            train_performance = evaluate_model(self.classifier, train_x, train_y, self.threshold)
            train_performance['test_train'] = 'train'
            test_performance = evaluate_model(self.classifier, test_x, test_y, self.threshold)
            test_performance['test_train'] = 'test'
            return pd.concat([test_performance, train_performance])
        print('TRAINING PERFORMANCE')
        report_card(self.classifier, train_x, train_y, mappings=self.mappings, threshold=self.threshold)
        print('HOLDOUT PERFORMANCE')
        report_card(self.classifier, test_x, test_y, mappings=self.mappings, threshold=self.threshold)

    def manual_validation(self, sample_id: str, print_report_card: bool = False,
                          root_population: str or None = None) -> pd.DataFrame or None:
        """
        Perform manual validation of the classifier using a sample associated to the same experiment as the
        training data. Important: the sample given MUST be pre-gated with the same populations as the training dataset
        :param root_population:
        :param sample_id: sample ID for file group to classify
        :param print_report_card:
        :return: Pandas DataFrame of classification performance
        """
        assert self.classifier is not None, 'Model must be trained prior to prediction'
        root_pop = root_population
        if root_pop is None:
            root_pop = self.root_population
        # Load the sample and prepare training data and labels
        # Note: data is transformed when labels are extracted; transform method == self.transform
        val_sample = Gating(self.experiment, sample_id, include_controls=False)
        if self.multi_label:
            x, y, mappings = self.multiclass_labels(val_sample, self.features, root_pop)
        else:
            x, y, mappings = self.singleclass_labels(val_sample, self.features, root_pop)

        # Check mappings match that expected from training data
        assert len(mappings.keys()) == len(self.mappings.keys()), \
            f'Class mappings do not match training data: {len(mappings.keys())} classes in ' \
            f'validation data, but {len(self.mappings.keys())} classes in training data'
        valm = set([np.array2string(x) for x in mappings.values()])
        trainm = set([np.array2string(x) for x in self.mappings.values()])
        assert valm == trainm,  f'Class mappings do not match training data: {valm} != {trainm}'

        # Standardise/normalise if necessary
        if self.preprocessor is not None:
            x = self.preprocessor.transform(x)
        if not print_report_card:
            performance = evaluate_model(self.classifier, x, y, threshold=self.threshold)
            return performance
        report_card(self.classifier, x, y, threshold=self.threshold, mappings=mappings)

    def balance_dataset(self, method: str = 'oversample', frac: float = 0.5,
                        **kwargs) -> (pd.DataFrame or np.array, np.array):
        """
        Given an imbalanced dataset, generate a new dataset with class balance attenuated. Method can either be
        'oversample' whereby the RandomOverSampler class of Imbalance Learn is implemented to sample with replacement
        in such a way that classes become balanced, or 'density' where density dependent downsampling is performed;
        see immunova.flow.gating.utilities.density_dependent_downsampling.
        :param x: Feature space
        :param y: Target labels
        :param method: Either 'oversample' or 'density' (default = 'oversample'
        :param frac: Ignored if method = 'oversample'. Density dependent downsampling is an absolute sampling technique
        that reduces the size of the dataset, this parameter indicates how large the resulting feature space (as a
        percentage of the original) should be
        :param kwargs: Keyword arguments to pass to density_dependent_downsample
        :return: Balanced feature space and labels
        """
        if method == 'oversample':
            return random_oversampling(self.train_X, self.train_y)
        elif method == 'density':
            if type(self.train_X) == np.array:
                x = pd.DataFrame(self.train_X, columns=self.features)
            self.train_X['labels'] = self.train_y
            x = density_dependent_downsample(data=self.train_X, features=self.features, frac=frac, **kwargs)
            return x[self.features], x['labels'].values

    def compare_to_training_set(self, sample_id: str, plot_umap: bool = False,
                                plot_phate: bool = False, root_population: str or None = None,
                                sample_n: int = 50000):

        def scatter_plot(te, ve, title):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(x=te[:, 0], y=te[:, 1], s=2, alpha=0.25, c='blue',
                       label='Training Data')
            ax.scatter(x=ve[:, 0], y=ve[:, 1], s=2, alpha=0.25, c='red',
                       label='Validation Data')
            ax.legend()
            ax.set_title(title)
            fig.show()

        root_pop = root_population
        if root_pop is None:
            root_pop = self.root_population
        # Fetch and sample data
        val_data = Gating(self.experiment, sample_id, include_controls=False)
        if self.multi_label:
            val_x, y, mappings = self.multiclass_labels(val_data, self.features, root_pop=root_pop)
        else:
            val_x, y, mappings = self.singleclass_labels(val_data, self.features, root_pop=root_pop)
        if val_x.shape[0] > sample_n:
            idx = np.random.randint(val_x.shape[0], size=sample_n)
            val_x = val_x[idx, :]
        train_x = np.copy(self.train_X)
        if train_x.shape[0] > sample_n:
            idx = np.random.randint(train_x.shape[0], size=sample_n)
            train_x = train_x[idx, :]
        assert val_x.shape[0] == train_x.shape[0], f'Row length for val data != length of train data; ' \
                                                   f'{val_x.shape[0]} != {train_x.shape[0]}. Try altering sample size.'
        print('Performing PCA...')
        pca = PCA(random_state=42, n_components=len(self.mappings.keys()))
        val_embeddings = pca.fit_transform(val_x)
        train_embeddings = pca.fit_transform(train_x)
        scatter_plot(train_embeddings, val_embeddings, title='Comparison of PCA embeddings; Training Vs PCA')
        pca_matrix = np.zeros((train_embeddings.shape[1], val_embeddings.shape[1]))
        n = train_embeddings.shape[1]
        for ti in range(n):
            train_c = train_embeddings[:, ti].reshape(-1, 1)
            for vi in range(n):
                val_c = val_embeddings[:, vi].reshape(-1, 1)
                lr = LinearRegression(n_jobs=-1)
                pca_matrix[ti, vi] = lr.fit(train_c, val_c).score(train_c, val_c)
        hfig, hax = plt.subplots(figsize=(5, 5))
        hax = heatmap(pca_matrix, annot=True, fmt='.1f', linewidths=.5, cmap="YlGnBu", ax=hax)
        hax.set_title('R-squared from regression of PCA components Train vs Validation')
        hfig.show()
        if plot_umap:
            print('Performing UMAP...')
            umap = UMAP(n_components=2, random_state=42)
            val_embeddings = umap.fit_transform(val_x)
            train_embeddings = umap.fit_transform(train_x)
            scatter_plot(train_embeddings, val_embeddings, title='Comparison of UMAP embeddings; Training Vs PCA')
        if plot_phate:
            print('Performing PHATE...')
            ph = PHATE(n_jobs=-2, verbose=0)
            val_embeddings = ph.fit_transform(val_x)
            train_embeddings = ph.fit_transform(train_x)
            scatter_plot(train_embeddings, val_embeddings, title='Comparison of PHATE embeddings; Training Vs PCA')







