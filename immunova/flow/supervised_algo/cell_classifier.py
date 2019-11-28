from immunova.data.fcs_experiments import FCSExperiment
from immunova.data.fcs import FileGroup, File, ChannelMap, Population
from immunova.data.panel import Panel
from immunova.flow.gating.actions import Gating
from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.supervised_algo.utilities import standard_scale, norm_scale, find_common_features, \
    predict_class, random_oversampling
from immunova.flow.gating.utilities import density_dependent_downsample
from immunova.flow.supervised_algo.evaluate import evaluate_model
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np


class CellClassifierError(Exception):
    pass


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
        raise CellClassifierError('Error: currently only uniform sampling is implemented in this version of immunova')

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
    print('Sampling complete!')
    new_filegroup = FileGroup(primary_id=new_file_name)
    new_filegroup.flags = 'sampled data'
    new_file = File(file_id=new_file_name,
                    compensated=True,
                    channel_mappings=channel_mappings)
    print('Inserting sampled data to database...')
    new_file.put(data.values)
    new_filegroup.files.append(new_file)
    root_p = Population(population_name=root_population,
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


class CellClassifier:
    """
    Class for performing classification of cells by supervised machine learning.
    """
    def __init__(self, experiment: FCSExperiment, reference_sample: str, population_labels: list, features: list,
                 multi_label: bool = True, multi_label_method: str = 'convert',
                 transform: str = 'log_transform', root_population: str = 'root',
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
        'multi-label' classification task
        :param multi_label_method: method by which to handle the multi-label classification problem. Can be either
        'convert' (default) or 'one hot encode'. The former will convert the problem into a multi-class classification
        problem  by performing one-hot encoding and creating a new label for each unique class. 'one hot encode' will
        simply assign a one-hot encoded label to each cells that represents the multiple populations it belongs to.
        :param transform: name of transform method to use (see flow.gating.transforms for info)
        :param root_population: name of root population i.e. the population to derive training and test data from
        :param threshold: minimum probability threshold to class as positive (default = 0.5)
        :param scale: how to scale the data prior to learning; either 'Standardise', 'Normalise' or None.
        Standardise scales using the standard score, removing the mean and scaling to unit variance
        (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        Normalise scales data between 0 and 1 using the MinMaxScaler
        (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
        """

        self.experiment = experiment
        self.transform = transform
        self.multi_label = multi_label
        self.multi_label_method = multi_label_method
        self.classifier = None
        self.preprocessor = None
        self.features = features
        self.root_population = root_population
        self.threshold = threshold
        self.mappings = None
        self.class_weights = None

        ref = Gating(self.experiment, reference_sample)

        self.population_labels = ref.valid_populations(population_labels)
        if len(self.population_labels) < 2:
            raise CellClassifierError(f'Error: reference sample {reference_sample} does not contain any '
                                      f'gated populations, please ensure that the reference sample has '
                                      f'been gated prior to training.')

        if multi_label:
            if multi_label_method == 'one hot encode':
                self.train_X, self.train_y = self.one_hot_labels(ref, features)
            else:
                self.train_X, self.train_y = self.multiclass_to_singleclass_labels(ref, features)
        else:
            self.train_X, self.train_y = self.singleclass_labels(ref, features)

        if scale == 'Standardise':
            self.train_X, self.preprocessor = standard_scale(self.train_X)
        elif scale == 'Normalise':
            self.train_X, self.preprocessor = norm_scale(self.train_X)
        elif scale is None:
            print('Warning: it is recommended that data is scaled prior to training. Unscaled data can result '
                  'in some weights updating faster than others, having a negative effect on classifier performance')
        else:
            raise CellClassifierError('Error: scale method not recognised, must be either `Standardise` or `Normalise`')

        if type(balance_method) == str:
            if frac:
                self.balance_dataset(method=balance_method, frac=frac, **downsampling_kwargs)
            else:
                self.balance_dataset(method=balance_method, **downsampling_kwargs)
        elif type(balance_method) == dict:
            self.class_weights = balance_method

    def one_hot_labels(self, ref: Gating, features: list) -> (pd.DataFrame, np.array):
        """
        Generate feature space and labels when a cell can belong to multiple populations. Labels are returned as a
        one-hot encoded sequence for each cell that represents the populations that cell belongs to (e.g for the
        population labels ['CD3+', 'CD4+', 'CD8+'] an encoding of [1,0,1] would be a CD3+CD8+ cell type.
        (multi-label learning)
        :param ref: Gating object to retrieve data from
        :param features: list of features for training
        :return: DataFrame of feature space, array of target labels
        """
        root = ref.get_population_df(self.root_population, transform=True, transform_method=self.transform)[features]
        y = np.zeros((root.shape[0], len(self.population_labels)))
        for pi, pop in enumerate(self.population_labels):
            pop_idx = ref.populations[pop].index
            for ci in pop_idx:
                y[ci, pi] = 1
        return root, y

    def multiclass_to_singleclass_labels(self, ref: Gating, features: list) -> (pd.DataFrame, np.array):
        """
        Generate feature space and labels for a multi-label multi-class classification problem and handle the
        multi-label aspect of this problem by converting multi-label signatures to single labels. This can be important
        if there are correlations between labels e.g. the classification CD3+/- and CD4+/- are not independent because
        a cell that is CD4+, CD3- will have a very different meaning to a cell that is CD4+ CD3+. The problem is
        converted to a multi class, single label classification by converting each unique one-hot encoded representation
        to a 'fake label'.
        :param ref: Gating object to retrieve data from
        :param features: list of features for training
        :return: DataFrame of feature space, array of target labels
        """
        train_X, train_y = self.one_hot_labels(ref, features)
        labels = {x: i for i, x in enumerate(np.unique(train_y, axis=0))}
        train_y = np.array(map(lambda x: labels[x], labels))
        pops = np.array(self.population_labels)
        self.mappings = {i: pops[np.where(x == 1)] for x, i in labels.items()}
        return train_X, train_y

    def singleclass_labels(self, ref: Gating, features: list) -> (pd.DataFrame, np.array):
        """
        Generate feature space and labels where a cell can belong to only one population
        :param ref: Gating object to retrieve data from
        :param features: list of features for training
        :return: DataFrame of feature space, array of target labels
        """
        if self.__check_downstream_overlaps(ref):
            raise CellClassifierError('Error: one or more population dependency errors')
        root = ref.get_population_df(self.root_population, transform=True, transform_method=self.transform)[features]
        y = np.zeros(root.shape[0])
        for i, pop in enumerate(self.population_labels):
            pop_idx = ref.populations[pop].index
            np.put(y, pop_idx, i + 1)
        return root, y

    def __check_downstream_overlaps(self, ref: Gating) -> bool:
        """
        Internal method. Check if a chosen root population is downstream of target populations for classification.
        This is a problem because if the root population is downstream then the model won't have access to the events
        it needs to classify.
        :param ref: Gating object whom's populations you wish to check
        :return: True if overlaps exist, otherwise False
        """
        downstream_overlaps = False
        for pop_i in self.population_labels:
            dependencies = ref.find_dependencies(pop_i)
            if self.root_population in dependencies:
                print(f'Error: population {pop_i} is upstream from the chosen root population {self.root_population}')
                downstream_overlaps = True
            for pop_j in self.population_labels:
                if pop_j == pop_i:
                    continue
                if pop_j in dependencies:
                    print(f'Error: population {pop_j} is a dependency of population {pop_i} (i.e. it is downstream '
                          f'from this population). This will result in invalid labelling. If you wish to continue '
                          f'with these population targets, please set multi_label parameter to True')
                    downstream_overlaps = True
        return downstream_overlaps

    def train_test_split(self, test_size=0.3) -> list:
        """
        Create train/test split of data using Sklearn's train_test_split function
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
        :param test_size: size of test population as a proportion of the the dataset (default = 0.3)
        :return: List containing train-test split of inputs.
        """
        return train_test_split(self.train_X, self.train_y, test_size=test_size, random_state=42)

    def __save_gating(self, target: Gating, y_hat: np.array) -> Gating:
        """
        Internal method. Given some Gating object of the target file for prediction and the predicted labels,
        generate new population objects and insert them into the Gating object
        :param target: Gating object of the target file
        :param y_hat: array of predicted population labels
        :return: None
        """
        parent = target.get_population_df(population_name=self.root_population)
        new_populations = ChildPopulationCollection(gate_type='sml')
        if self.multi_label:
            y_hat = pd.DataFrame(y_hat, columns=self.population_labels)
            for label in y_hat.columns:
                x = y_hat[label].values
                new_populations.add_population(name=label)
                new_populations.populations[label].update_index(x.nonzero())
                new_populations.populations[label].update_geom(shape='sml', x=None, y=None)
        else:
            for i, label in enumerate(self.population_labels):
                y_ = np.where(y_hat == i+1)[0]
                new_populations.add_population(name=label)
                new_populations.populations[label].update_index(y_)
                new_populations.populations[label].update_geom(shape='sml', x=None, y=None)
        target.update_populations(new_populations, parent, parent_name=self.root_population, warnings=[])
        return target

    def __preprocess_target(self, sample_gates: Gating) -> pd.DataFrame:
        """
        Internal method. Transform and scale data from the given Gating object.
        :param sample_gates: Gating object
        :return: Transformed and scaled DataFrame.
        """
        if self.preprocessor is not None:
            return self.preprocessor.fit_transform(sample_gates.get_population_df(self.root_population,
                                                                                  transform=True,
                                                                                  transform_method=self.transform)[self.features])
        return sample_gates.get_population_df(self.root_population,
                                              transform=True,
                                              transform_method=self.transform)[self.features]

    def predict(self, target_sample: str) -> Gating:
        """
        Given a sample ID, predict cell populations. Model must already be trained. Results are saved as new
        populations in a Gating object returned to the user.
        :param target_sample: Name of file for prediction. Must belong to experiment associated to CellClassifier obj.
        :return: Gating object containing new predicted populations.
        """
        sample_gates = Gating(self.experiment, target_sample)
        target_sample = self.__preprocess_target(sample_gates)
        if self.classifier is None:
            raise CellClassifierError('Error: cell classifier has not been trained, either '
                                      'load an existing model using the `load_model` method or train '
                                      'the classifier using the `train_classifier` method')

        y_probs = self.classifier.predict(target_sample)
        y_hat = predict_class(y_probs, self.threshold)
        return self.__save_gating(sample_gates, y_hat)

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

        for i, (train_index, test_index) in enumerate(kf.split(self.train_X)):
            train_x, test_x = self.train_X[train_index], self.train_X[test_index]
            train_y, test_y = self.train_y[train_index], self.train_y[test_index]
            self.classifier.fit(train_x, train_y)
            p = evaluate_model(self.classifier, train_x, train_y, self.multi_label_method, self.threshold)
            p['k'] = i
            train_performance.append(p)
            p = evaluate_model(self.classifier, test_x, test_y, self.multi_label_method, self.threshold)
            p['k'] = i
            test_performance.append(p)

        train_performance = pd.concat(train_performance)
        train_performance['average_performance'] = train_performance.mean(axis=1)
        train_performance['test_train'] = 'train'
        test_performance = pd.concat(test_performance)
        test_performance['average_performance'] = test_performance.mean(axis=1)
        test_performance['test_train'] = 'test'
        return pd.concat([train_performance, test_performance])

    def train_holdout(self, holdout_frac: float = 0.3, **kwargs) -> pd.DataFrame:
        """
        Fit classifier to training data and evaluate on holdout data.
        :param holdout_frac: Proportion of data to keep as holdout
        :param kwargs: kwargs: Optional additional kwargs for model fit.
        :return: Pandas DataFrame detailing performance
        """
        train_x, test_x, train_y, test_y = self.train_test_split(test_size=holdout_frac)
        self.classifier.fit(train_x, train_y)
        train_performance = evaluate_model(self.classifier, train_x, train_y, self.multi_label_method, self.threshold)
        train_performance['test_train'] = 'train'
        test_performance = evaluate_model(self.classifier, test_x, test_y, self.multi_label_method, self.threshold)
        test_performance['test_train'] = 'test'
        return pd.concat([test_performance, test_performance])

    def manual_validation(self, sample_id: str) -> pd.DataFrame:
        """
        Perform manual validation of the classifier using a sample associated to the same experiment as the
        training data. Important: the sample given MUST be pre-gated with the same populations as the training dataset
        :param sample_id: sample ID for file group to classify
        :return: Pandas DataFrame of classification performance
        """
        if self.classifier is None:
            raise CellClassifierError('Error: model must be trained prior to validation')
        ref = Gating(self.experiment, sample_id)
        if self.multi_label:
            if self.multi_label_method == 'one hot encode':
                x, y = self.one_hot_labels(ref, self.features)
            else:
                x, y = self.multiclass_to_singleclass_labels(ref, self.features)
        else:
            x, y = self.singleclass_labels(ref, self.features)

        if self.preprocessor is not None:
            x = self.preprocessor.transform(x)

        return evaluate_model(self.classifier, x, y, multi_label=self.multi_label_method, threshold=self.threshold)

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
        if self.multi_label and self.multi_label_method == 'one hot encode':
            raise CellClassifierError('Error: density dependent downsampling is not supported for multi-label '
                                      'classification. If you wish to still use density dependent downsampling '
                                      'set `multi_label_method` to `convert`')
        if method == 'oversample':
            return random_oversampling(self.train_X, self.train_y)
        elif method == 'density':
            if type(self.train_X) == np.array:
                x = pd.DataFrame(self.train_X, columns=self.features)
            self.train_X['labels'] = self.train_y
            x = density_dependent_downsample(data=self.train_X, features=self.features, frac=frac, **kwargs)
            return x[self.features], x['labels'].values
