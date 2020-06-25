from ...data.fcs_experiments import FCSExperiment
from ...data.fcs import FileGroup, File, ChannelMap, Population
from ...data.panel import Panel
from ...flow.gating.actions import Gating
from ...flow.gating.defaults import ChildPopulationCollection
from ...flow.supervised.utilities import find_common_features, predict_class, random_oversampling
from ..transforms import scaler
from ...flow.gating.utilities import density_dependent_downsample
from ...flow.supervised.evaluate import evaluate_model, report_card
from ...flow.feedback import progress_bar
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LinearRegression
from umap import UMAP
from phate import PHATE
from multiprocessing import Pool, cpu_count
from seaborn import heatmap
from functools import partial
from anytree import Node
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _check_columns(data: pd.DataFrame, 
                   features: list) -> pd.DataFrame:
    """
    Given a dataframe and a list of expected features, print missing columns and return new dataframe
    with only valid features

    Parameters
    -----------
    data: Pandas.DataFrame
        DataFrame for checking
    features: list
        list of features (column names)

    Returns
    ---------
    Pandas.DataFrame
        new 'valid' DataFrame
    """
    valid_features = [f for f in features if f in data.columns]
    if len(valid_features) < len(features):
        print(f'The following features are missing from the training data and will be excluded from the '
              f'model {list(set(features) - set(valid_features))}')
    return data[valid_features]


def multi_process_ordered(func: callable,
                          x: iter,
                          chunksize: int = 100) -> list:
    """
    Map a function (func) to some iterable (x) using multiprocessing. Iterable will be divided into chunks
    of size `chunksize`. The chunks will be given as a list of tuples, where the first value is the index of
    the chunk and the second value the chunk itself; this allows for ordered reassembly once processing has
    completed. For this reason, the function `func` MUST handle and return the index of the chunk upon which
    it acts.

    Parameters
    -----------
    func: callable
        callable function to applied in parallel
    x: iterable
        some iterable to apply the function to
    chunksize: int, (default=100)
        size of chunks for multiprocessing

    Returns
    --------
    list
        ordered list of funciton output
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


def _assign_labels(x: tuple,
                   labels: dict) -> tuple:
    """
    Internal function. Used for assigning a unique 'fake' label to each multi-label sequence in a set in
    a parallel process

    Parameters
    -----------
    x: tuple
        (index of chunk, list of multi-label sequence's)
    labels: dict
        {multi-label sequence: unique 'fake' label}

    Returns
    ---------
    tuple
        (index of chunk, list of 'fake' labels)
    """
    return x[0], [labels[np.array_str(s)] for s in x[1]]


def _channel_mappings(features: list,
                      panel: Panel) -> list:
    """
    Internal function. Given a list of features and a Panel object, return a list of ChannelMapping objects
    that correspond with the given Panel.

    Parameters
    -----------
    features: list
        list of features to compare to Panel object
    panel: Panel
        Panel object of channel mappings

    Returns
    --------
    list
        list of ChannelMappings
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
                            samples: list or None = None,
                            new_file_name: str or None = None,
                            sampling_method: str = 'uniform',
                            sample_n: int or float = 1000,
                            include_population_labels: bool = False,
                            verbose: bool = True) -> None:
    """
    Given some experiment and a root population that is common to all fcs file groups within this experiment, take
    a sample from each and create a new file group from the concatenation of these data. New file group will be created
    and associated to the given FileExperiment object.
    If no file name is given it will default to '{Experiment Name}_sampled_data'

    Parameters
    -----------
    experiment: FCSExperiment
        FCSExperiment object for corresponding experiment to sample
    root_population: str
        if the files in this experiment have already been gated, you can specify to sample
        from a particular population e.g. Live CD3+ cells or Live CD45- cells
    samples: list, optional
        list of sample IDs for samples to be included (default = all samples in experiment)
    new_file_name: str
        name of file group generated
    sampling_method: str, (default='uniform')
        method to use for sampling files (currently only supports 'uniform')
    sample_n: int, (default=1000)
        number or fraction of events to sample from each file
    include_population_labels: bool, (default=False)
        If True, populations in the new file generated are inferred from the existing samples
    verbose: bool, (default=True)
        Whether to provide feedback

    Returns
    --------
    None
    """
    def sample(d):
        if sampling_method == 'uniform':
            if type(sample_n) == int:
                if d.shape[0] > sample_n:
                    return d.sample(sample_n)
                return d
            return d.sample(frac=sample_n)
        raise ValueError('Error: currently only uniform sampling is implemented in this version of cytopy')

    vprint = print if verbose else lambda *a, **k: None
    assert all([s in experiment.list_samples() for s in samples]), 'One or more samples specified do not belong to experiment'
    vprint('-------------------- Generating Reference Sample --------------------')
    if new_file_name is None:
        new_file_name = f'{experiment.experiment_id}_sampled_data'
    vprint('Finding features common to all fcs files...')
    features = find_common_features(experiment=experiment, samples=samples)
    channel_mappings = _channel_mappings(features,
                                         experiment.panel)
    data = pd.DataFrame()
    if include_population_labels:
        features.append('label')
    for s in samples:
        vprint(f'Sampling {s}...')
        g = Gating(experiment, s, include_controls=False)
        if root_population not in g.populations.keys():
            vprint(f'Skipping {s} as {root_population} is absent from gated populations')
            continue
        df = g.get_population_df(root_population, label=include_population_labels)[features]
        data = pd.concat([data, sample(df)])
    data = data.reset_index(drop=True)
    vprint('Sampling complete!')
    new_filegroup = FileGroup(primary_id=new_file_name)
    new_filegroup.flags = 'sampled data'
    new_file = File(file_id=new_file_name,
                    compensated=True,
                    channel_mappings=channel_mappings)
    vprint('Inserting sampled data to database...')
    new_file.put(data[[f for f in features if f != 'label']].values)
    new_filegroup.files.append(new_file)
    root_p = Population(population_name='root',
                        prop_of_parent=1.0, prop_of_total=1.0,
                        warnings=[], geom=[['shape', None], ['x', 'FSC-A'], ['y', 'SSC-A']])
    root_p.save_index(data.index.values)
    new_filegroup.populations.append(root_p)
    if include_population_labels:
        vprint('Warning: new concatenated sample will inherit population labels but NOT gates or '
               'population hierarchy')
        for pop in data.label.unique():
            idx = data[data.label == pop].index.values
            n = len(idx)
            p = Population(population_name=pop, prop_of_parent=n/data.shape[0],
                           prop_of_total=n/data.shape[0], warnings=[], parent='root',
                           geom=[['shape', None], ['x', 'FSC-A'], ['y', 'SSC-A']])
            p.save_index(idx)
            new_filegroup.populations.append(p)
    vprint('Saving changes...')
    mid = new_filegroup.save()
    experiment.fcs_files.append(new_filegroup)
    experiment.save()
    vprint(f'Complete! New file saved to database: {new_file_name}, {mid}')
    vprint('-----------------------------------------------------------------')


class CellClassifier:
    """
    Base class for performing classification of cells by supervised machine learning.

    Parameters
    -----------
    experiment: FCSExperiment
        FCSExperiment for classification
    reference_sample: str
        sample ID for training sample (see 'create_reference_sample')
    population_labels: list
        list of populations for prediction (populations must be valid gated populations
        that exist in the reference sample)
    features: list
        list of features (channel/marker column names) to include
    multi_label: bool, (default=True)
        If True, cells can belong to multiple classes and the problem is treated as a
        'multi-label' classification task. Labels will be binarised and a new 'fake' label generated for each
        unique instance of a cell label. This is important to account for correlations between individual labels.
    transform: str, (default='logicle')
        name of transform method to use (see flow.gating.transforms for info)
    root_population: str, (default='root')
        name of root population i.e. the population to derive training and test data from
    threshold: float, (default=0.5)
        minimum probability threshold to class as positive
    scale: str, (default='standard')
        how to scale the data prior to learning; either 'Standardise', 'Normalise' or None.
        Standardise scales using the standard score, removing the mean and scaling to unit variance
        (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
        Normalise scales data between 0 and 1 using the MinMaxScaler
        (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
    balance_method: str or dict, optional
        Methodology to use to combat class imbalance; either a dictionary of class weights or one of the following:
        - 'density': density dependent downsampling is performed using keyword arguments provided in 'downsample_kwargs'
        - 'oversample': classes balanced by oversampling underrepresented classes
        'auto_weights': weights are calculated automatically using scikit-learn compute_class_weight function
    scale_kwargs: dict, optional
        keyword arguments to pass to scaling function
    frac: float, optional
        If provided, data will be downsampled to given fraction prior to classification
    downsampling_kwargs: dict, optional
        keyword arguments to be passed to density dependent downsampling
    """
    def __init__(self,
                 experiment: FCSExperiment,
                 reference_sample: str,
                 population_labels: list,
                 features: list,
                 multi_label: bool = True,
                 transform: str = 'logicle',
                 root_population: str = 'root',
                 threshold: float = 0.5,
                 scale: str or None = 'standard',
                 balance_method: None or str or dict = None,
                 frac: float or None = None,
                 downsampling_kwargs: dict or None = None,
                 scale_kwargs: dict or None = None,
                 verbose=True):
        self.vprint = print if verbose else lambda *a, **k: None
        self.vprint('Constructing cell classifier object...')
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
        self.vprint('Loading information on reference sample...')
        ref = Gating(self.experiment, reference_sample, include_controls=False)

        self.population_labels = ref.valid_populations(population_labels)
        assert len(self.population_labels) > 2, f'Error: reference sample {reference_sample} does not contain any '\
                                                f'gated populations, please ensure that the reference sample has ' \
                                                f'been gated prior to training.'
        self.vprint('Preparing training data and labels...')
        if multi_label:
            self.threshold = None
            self.train_X, self.train_y, self.mappings = self.multiclass_labels(ref, features, root_population)
        else:
            self.train_X, self.train_y, self.mappings = self.singleclass_labels(ref, features, root_population)
        self.vprint('Scaling data...')
        if scale is not None:
            if scale_kwargs is not None:
                self.train_X, self.preprocessor = scaler(self.train_X, scale_method=scale, **scale_kwargs)
            else:
                self.train_X, self.preprocessor = scaler(self.train_X, scale_method=scale)
        else:
            self.vprint('Warning: it is recommended that data is scaled prior to training. Unscaled data can result '
                  'in some weights updating faster than others, having a negative effect on classifier performance')

        if type(balance_method) == str:
            assert balance_method in ['density', 'oversample', 'auto_weights'], \
                'Balance method must be one of ["density", "oversample", "auto_weights"] or custom weights given as ' \
                'a python dictionary object'
            if balance_method in ['density', 'oversample']:
                self.vprint('Balancing dataset by sampling...')
                if frac:
                    self.balance_dataset(method=balance_method, frac=frac, downsampling_kwargs=downsampling_kwargs)
                else:
                    self.balance_dataset(method=balance_method, downsampling_kwargs=downsampling_kwargs)
            else:
                weights = compute_class_weight('balanced',
                                               classes=np.array(list(self.mappings.keys())),
                                               y=self.train_y)
                class_weights = {k: w for k, w in zip(self.mappings.keys(), weights)}
                self.class_weights = list(map(lambda x: class_weights[x], self.train_y))
        elif type(balance_method) == dict:
            self.class_weights = list(map(lambda x: balance_method[x], self.train_y))
        self.vprint('Ready for training!')

    def _binarize_labels(self,
                         ref: Gating,
                         features: list,
                         root_pop: str) -> (pd.DataFrame, np.array):
        """
        Generate feature space and labels when a cell can belong to multiple populations. Labels are returned as a
        one-hot encoded sequence for each cell that represents the populations that cell belongs to (e.g for the
        population labels ['CD3+', 'CD4+', 'CD8+'] an encoding of [1,0,1] would be a CD3+CD8+ cell type.
        (multi-label learning)

        Parameters
        -----------
        ref: Gating
            Gating object to retrieve data from
        features: list
            list of features for training
        root_pop: str
            name of the root population

        Returns
        --------
        Pandas.DataFrame
            DataFrame of feature space, array of target labels
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

    def multiclass_labels(self,
                          ref: Gating,
                          features: list,
                          root_pop: str) -> (pd.DataFrame, np.array, dict):
        """
        Generate feature space and labels for a multi-label multi-class classification problem and handle the
        multi-label aspect of this problem by converting multi-label signatures to single labels. This can be important
        if there are correlations between labels e.g. the classification CD3+/- and CD4+/- are not independent because
        a cell that is CD4+, CD3- will have a very different meaning to a cell that is CD4+ CD3+. The problem is
        converted to a multi class, single label classification by converting each unique one-hot encoded representation
        to a 'fake label'.

        Parameters
        -----------
        ref: Gating
            Gating object to retrieve data from
        features: list
            list of features for training
        root_pop: str
            name of the root population

        Returns
        --------
        Pandas.DataFrame
            DataFrame of feature space, array of target labels, and a dictionary of cell population mappings
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

    def singleclass_labels(self,
                           ref: Gating,
                           features: list,
                           root_pop: str) -> (pd.DataFrame, np.array, dict):
        """
        Generate feature space and labels where a cell can belong to only one population

        ref: Gating
            Gating object to retrieve data from
        features: list
            list of features for training
        root_pop: str
            name of the root population

        Returns
        --------
        (Pandas.DataFrame, Numpy.array, dict)
            DataFrame of feature space, array of target labels, mappings of cell population labels
        """
        if ref.check_downstream_overlaps(root_pop, self.population_labels):
            raise ValueError('Error: one or more population dependency errors')
        root = ref.get_population_df(root_pop, transform=True, transform_method=self.transform)[features]
        y = np.zeros((root.shape[0], len(self.population_labels)))
        mappings = dict()
        for i, pop in enumerate(self.population_labels):
            pop_idx = ref.populations[pop].index
            np.put(y, pop_idx, i + 1)
            mappings[i+1] = np.array([pop])
        return root, y, mappings

    def train_test_split(self,
                         test_size: float = 0.3) -> list:
        """
        Create train/test split of data using Sklearn's train_test_split function
        (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

        Parameters
        -----------
        test_size: float
            size of test population as a proportion of the the dataset (default = 0.3)

        Returns
        --------
        list
            List containing train-test split of inputs.
        """
        return train_test_split(self.train_X, self.train_y, test_size=test_size, random_state=42)

    def _build_tree(self,
                    label: str,
                    tree: dict,
                    root_pop: str):
        """
        Internal function. Build population tree as a valid dictionary of anytree objects

        Parameters
        ----------
        label: str
            Name of population
        tree: dict
            Classification tree
        root_pop: str
            Name of the root population

        Returns
        -------
        dict
        """
        if not tree.get(f'{self.prefix}_{label[0]}'):
            tree[f'{self.prefix}_{label[0]}'] = Node(f'{self.prefix}_{label[0]}',
                                                     parent=tree[root_pop],
                                                     collection=ChildPopulationCollection(gate_type='sml'))
        for i, pop in enumerate(label[1:]):
            i += 1
            if not tree.get(f'{self.prefix}_{pop}'):
                tree[f'{self.prefix}_{pop}'] = Node(f'{self.prefix}_{pop}',
                                                    parent=tree[f'{self.prefix}_{label[i - 1]}'],
                                                    collection=ChildPopulationCollection(gate_type='sml'))
        return tree

    def _create_populations(self,
                            tree: dict,
                            branch: Node,
                            y_hat: np.array,
                            target: Gating,
                            root_pop: str,
                            mappings: dict):
        """
        Internal function. Recursively transverse population tree and update index

        Parameters
        ----------
        tree: dict
        branch: Node
        y_hat: Numpy.array
        target: Gating
        root_pop: str
        mappings: dict

        Returns
        -------
        dict
            Completed tree
        """
        tree[branch.name].collection.add_population(name=branch.name)
        tree[branch.name].collection.populations[branch.name].update_geom(shape='sml', x=None, y=None)
        labels = [x[0] for x in mappings.items() if any([branch.name == l for l in x[1]])]
        assert labels, f'Population {branch.name} does not appear in mappings'
        idx = np.array([])
        for i in labels:
            idx_ = target.populations[root_pop].index[np.where(y_hat == i)]
            idx = np.unique(np.concatenate((idx_, idx)))
        tree[branch.name].collection.populations[branch.name].update_index(idx)
        for child in tree[branch.name].children:
            tree = self._create_populations(tree, child, y_hat, target, root_pop, mappings)
        return tree

    def _save_gating(self,
                     target: Gating,
                     y_hat: np.array,
                     root_pop: str) -> Gating:
        """
        Internal method. Given some Gating object of the target file for prediction and the predicted labels,
        generate new population objects and insert them into the Gating object

        Parameters
        -----------

        target: Gating
            Gating object of the target file
        y_hat: Numpy.array
            array of predicted population labels
        root_pop: str
            Name of the root population

        Returns
        --------
        Gating
        """
        tree = {root_pop: Node(root_pop)}
        for pops in self.mappings.values():
            tree = self._build_tree(pops, tree, root_pop=root_pop)
        prefix_mappings = {k: [f'{self.prefix}_{x}' for x in v] for k, v in self.mappings.items()}
        trunk = tree[root_pop].children
        for branch in trunk:
            tree = self._create_populations(tree, branch, y_hat, target, root_pop, prefix_mappings)
        for node in tree.keys():
            if node == root_pop:
                continue
            parent_name = tree[node].parent.name
            parent = target.get_population_df(parent_name)
            target.update_populations(tree[node].collection, parent, parent_name=parent_name, warnings=[])
        return target

    def predict(self,
                target_sample: str,
                return_gating: bool = True,
                root_population: str or None = None) -> Gating or (np.array, np.array):
        """
        Given a sample ID, predict cell populations. Model must already be trained. Results are saved as new
        populations in a Gating object returned to the user.

        Parameters
        ------------
        target_sample: str
            Name of file for prediction. Must belong to experiment associated to CellClassifier obj.
        return_gating: bool, (default=True)
            If True, returns the modified Gating object with predicted populations contained within
        root_population: str
            Name of root population

        Returns
        --------
        Gating or (Numpy.array, Numpy.array)
            Gating object containing new predicted populations or tuple of (predicted probabilities, predicted labels)
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
            return self._save_gating(target, y_hat, root_pop)
        return y_probs, y_hat

    def _fit(self,
             x: np.array,
             y: np.array,
             **kwargs):
        """
        Fit the model. Should be called internally.

        Parameters
        -----------
        x: Numpy.array
            feature space
        y: Numpy.array
            target labels
        kwargs:
            keyword arguments to pass to call to MODEL.fit()

        Returns
        --------
        None
        """
        assert self.classifier is not None, 'Must construct classifier prior to calling `fit` using the `build` method'
        self.classifier.fit(x, y, **kwargs)

    def train_cv(self,
                 k: int = 5,
                 **kwargs) -> pd.DataFrame:
        """
        Fit classifier to training data using cross-validation

        Parameters
        -----------
        k: int, (default=5)
            Number of folds for cross-validation (default = 5)
        kwargs:
            Optional additional kwargs for model fit.

        Returns
        --------
        Pandas.DataFrame
            Pandas DataFrame detailing performance
        """
        kf = KFold(n_splits=k)
        train_performance = list()
        test_performance = list()
        self.vprint(f'----------- Cross Validation: {k} folds -----------')
        idx = kf.split(self.train_X)
        for i in progress_bar(range(k)):
            train_index, test_index = next(idx)
            train_x, test_x = self.train_X[train_index], self.train_X[test_index]
            train_y, test_y = self.train_y[train_index], self.train_y[test_index]
            self._fit(train_x, train_y, **kwargs)
            test_y, train_y = self._flatten_one_hot(test_y, train_y)
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

    def gridsearch(self,
                   param_grid: dict or list,
                   scoring: str = 'f1_weighted',
                   n_jobs: int = -1,
                   cv: int = 5,
                   **kwargs):
        """
        Perform Grid-Search for hyper-parameter optimisation using Scikit-Learn GridSearchCV class.

        Parameters
        ----------
        param_grid: dict or list
            (see https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
        scoring: str, (default='f1_weighted')
            Scaring function to use
        n_jobs: int, (default=-1)
            Number of jobs to run in parallel (-1 will use all available cores)
        cv: int, (default=5)
            Number of folds for cross-validation
        kwargs:
            Keyword arguments to be passed to GridSearchCV

        Returns
        -------
        GridSearchCV
        """
        assert self.classifier, 'Call to build_model must be made prior to grid search'
        grid = GridSearchCV(estimator=self.classifier, param_grid=param_grid,
                            scoring=scoring, n_jobs=n_jobs, cv=cv, **kwargs)
        grid.fit(self.train_X, self.train_y)
        return grid

    def train(self, **kwargs):
        """
        Train classifier

        Parameters
        ----------
        kwargs:
            Additional keyword arguments to be passed to call to MODEL.train()

        Returns
        -------
        None
        """
        self._fit(self.train_X, self.train_y, **kwargs)

    @staticmethod
    def _flatten_one_hot(test: np.array,
                         train: np.array):
        """
        Internal function. Flatten one-hot-encoded array

        Parameters
        ----------
        test: Numpy.array
        train: Numpy.array

        Returns
        -------
        Numpy.array
            Flattened train and test array
        """
        if train.ndim != 1:
            train = np.argmax(train, axis=1)
        if test.ndim != 1:
            test = np.argmax(test, axis=1)
        return test, train

    def train_holdout(self, holdout_frac: float = 0.3,
                      print_report_card: bool = False,
                      **kwargs) -> pd.DataFrame:
        """
        Fit classifier to training data and evaluate on holdout data.

        Parameters
        -----------
        holdout_frac: float, (default=0.3)
            Proportion of data to keep as holdout
        print_report_card: bool, (default=False)
            If True, detailed classification report printed to stdout
        kwargs:
            Optional additional kwargs for model fit.

        Returns
        --------
        Pandas.DataFrame
            Pandas DataFrame detailing performance
        """
        train_x, test_x, train_y, test_y = self.train_test_split(test_size=holdout_frac)
        self._fit(train_x, train_y, **kwargs)
        test_y, train_y = self._flatten_one_hot(test_y, train_y)
        if not print_report_card:
            train_performance = evaluate_model(self.classifier, train_x, train_y, self.threshold)
            train_performance['test_train'] = 'train'
            test_performance = evaluate_model(self.classifier, test_x, test_y, self.threshold)
            test_performance['test_train'] = 'test'
            return pd.concat([test_performance, train_performance])
        self.vprint('TRAINING PERFORMANCE')
        report_card(self.classifier, train_x, train_y, mappings=self.mappings, threshold=self.threshold)
        self.vprint('HOLDOUT PERFORMANCE')
        report_card(self.classifier, test_x, test_y, mappings=self.mappings, threshold=self.threshold)

    def manual_validation(self,
                          sample_id: str,
                          print_report_card: bool = False,
                          root_population: str or None = None) -> pd.DataFrame or None:
        """
        Perform manual validation of the classifier using a sample associated to the same experiment as the
        training data. Important: the sample given MUST be pre-gated with the same populations as the training dataset

        Parameters
        -----------
        root_population: str, (optional)
            Name of root population. If none given, defaults to root population used in training.
        sample_id: str
            sample ID for file group to classify
        print_report_card: bool
            If True, detailed classification report printed to stdout

        Returns
        --------
        Pandas.DataFrame
            Pandas DataFrame of classification performance
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
        for m in mappings.values():
            assert np.array2string(m) in set([np.array2string(i) for i in self.mappings.values()]), \
            f'Invalid mappings; label {np.array2string(m)} present in validation data but not in training data'

        # Standardise/normalise if necessary
        if self.preprocessor is not None:
            x = self.preprocessor.transform(x)
        if not print_report_card:
            performance = evaluate_model(self.classifier, x, y, threshold=self.threshold)
            return performance
        report_card(self.classifier, x, y, threshold=self.threshold, mappings=mappings)

    def balance_dataset(self,
                        method: str = 'oversample',
                        frac: float = 0.5,
                        downsampling_kwargs: dict or None = None) -> (pd.DataFrame or np.array, np.array):
        """
        Given an imbalanced dataset, generate a new dataset with class balance attenuated. Method can either be
        'oversample' whereby the RandomOverSampler class of Imbalance Learn is implemented to sample with replacement
        in such a way that classes become balanced, or 'density' where density dependent downsampling is performed;
        see cytopy.flow.gating.utilities.density_dependent_downsampling.

        Parameters
        -----------
        method: str
            Either 'oversample' or 'density' (default = 'oversample'
            frac: Ignored if method = 'oversample'. Density dependent downsampling is an absolute sampling technique
            that reduces the size of the dataset, this parameter indicates how large the resulting feature space (as a
            percentage of the original) should be
        frac: float, (default=0.5)
            If 'density' given for method, this is passed as the fraction of data to be sampled
        downsampling_kwargs: dict
            Additional keyword arguments passed to call to density_dependent_downsample

        Returns
        --------
        Pandas.DataFrame or (Numpy.array, Numpy.array)
            Balanced feature space and labels
        """
        if method == 'oversample':
            return random_oversampling(self.train_X, self.train_y)
        elif method == 'density':
            if type(self.train_X) == np.array:
                x = pd.DataFrame(self.train_X, columns=self.features)
            self.train_X['labels'] = self.train_y
            if downsampling_kwargs is None:
                x = density_dependent_downsample(data=self.train_X, features=self.features, frac=frac)
            else:
                x = density_dependent_downsample(data=self.train_X, features=self.features, frac=frac,
                                                 **downsampling_kwargs)
            return x[self.features], x['labels'].values

    def compare_to_training_set(self,
                                sample_id: str,
                                plot_umap: bool = False,
                                plot_phate: bool = False,
                                root_population: str or None = None,
                                sample_n: int = 50000):
        """
        Utility function. Given some sample ID for a sample associated to experiment, compare the outcome to
        the training data; outcome given as a labelled scatter plot, with classified single cell data shown as
        components following some dimensionality reduction procedure (default = PCA).

        Parameters
        ----------
        sample_id: str
            Name of sample to classify and compare to training data
        plot_umap: bool, (default=False)
            If True, plot UMAP instead of PCA
        plot_phate: bool, (default=False)
            If True, plot PHATE instead of PCA
        root_population: str, optional
            Name of root population. If none given, use same population as training data
        sample_n: int, (default=50000)
            Number of cells to sample prior to dimensionality reduction procedure

        Returns
        -------
        None
        """

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
        self.vprint('Performing PCA...')
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
            self.vprint('Performing UMAP...')
            umap = UMAP(n_components=2, random_state=42)
            val_embeddings = umap.fit_transform(val_x)
            train_embeddings = umap.fit_transform(train_x)
            scatter_plot(train_embeddings, val_embeddings, title='Comparison of UMAP embeddings; Training Vs PCA')
        if plot_phate:
            self.vprint('Performing PHATE...')
            ph = PHATE(n_jobs=-2, verbose=0)
            val_embeddings = ph.fit_transform(val_x)
            train_embeddings = ph.fit_transform(train_x)
            scatter_plot(train_embeddings, val_embeddings, title='Comparison of PHATE embeddings; Training Vs PCA')







