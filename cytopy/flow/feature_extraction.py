from ..data.fcs_experiments import FCSExperiment
from ..data.fcs import FileGroup
from ..data.subject import Subject
from ..data.fcs import ClusteringDefinition
from ..flow.gating.actions import Gating
from ..flow.dim_reduction import dimensionality_reduction
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from .feedback import progress_bar
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class ControlComparisons:
    """
    This class is for making experiment wide comparisons of controls to the primary data collected
    on each sample. As an example, people commonly use Full-Minus-One controls (FMO) in cytometry experiments.
    Where a common FMO is present across all samples of an experiment, a ControlComparison object will allow
    the extraction of a population from both primary samples and control samples and make comparisons.

    Parameters
    ----------
    experiment: FCSExperiment
        Instance of FCSExperiment to investigate
    samples: list, optional
        List of samples in experiment to investigate, if left as None all samples will be included
    gating_model: str, (default='knn')
        Type of model to use to estimate the populations in control samples using primary data as training data, if
        control samples have not been gated previously and saved to the database.
        By default it uses a KNearestNeighbours model, but user has the option to use a Support Vector Machine instead (svm).
    tree_map: dict, optional
        Only required if one or more samples in experiment is lacking populations for control data and populations in sample
        were derived using a supervised machine learning model. See cytopy.gating.actions.Gating.control_gating for more information.
    model_kwargs:
        Additional keyword arguments will be passed to instance scikit-learn classifier
    """
    def __init__(self,
                 experiment: FCSExperiment,
                 samples: list or None = None,
                 gating_model: str = 'knn',
                 tree_map: dict or None = None,
                 **model_kwargs):
        if samples is None:
            samples = experiment.list_samples()
        self.experiment = experiment
        print('---- Preparing for control comparisons ----')
        print('If control files have not been gated prior to initiation, control gating will '
              'be performed automatically (gating methodology can be specified in ControlComparison arguments '
              'see documentation for details). Note: control gating can take some time so please be patient. '
              'Results will be saved to the database to reduce future computation time.')
        print('\n')
        self.samples = self._gate_samples(self._check_samples(samples), tree_map, gating_model, **model_kwargs)

    def _gate_samples(self,
                      samples: dict,
                      tree_map,
                      gating_model,
                      **model_kwargs) -> dict:
        """
        Internal function. Pass samples in experiment. If they have not been previously gated, perform control gating.

        Parameters
        ----------
        samples: dict
            Dictionary output of self._check_samples
        tree_map: dict
            See cytopy.gating.actions.Gating.control_gating for more information.
        gating_model: str
            Type of gating model to use for estimating populations in controls
        model_kwargs:
            Additional keyword arguments will be passed to instance scikit-learn classifier

        Returns
        -------
        dict
            Returns modified sample dictionary where each key is a sample name and the value a list of control file IDs
            present for this sample
        """
        not_gated = [s for s, ctrls in samples.items() if len(ctrls['gated']) < len(ctrls['all'])]
        if len(not_gated) == 0:
            return {s: ctrls['all'] for s, ctrls in samples.items()}
        print(f'The following samples have not been previously gated for one or more controls: {not_gated}')
        print('\n')

        if tree_map is None:
            tree_map = dict()
            print('No tree_map provided. If one or more of the gates to be calculated for control files '
                  'derives from a supervised learning method, this will raise an Assertion error; tree_map '
                  'must provide details of the "x" and "y" dimensions for model fitting for each gate')
            print('\n')

        for s in progress_bar(not_gated):
            for ctrl in samples[s]['all']:
                if ctrl in samples[s]['gated']:
                    continue
                g = Gating(self.experiment, s, include_controls=True)
                g.control_gating(ctrl, tree_map=tree_map, model=gating_model, verbose=False, **model_kwargs)
                g.save(feedback=False)
                samples[s]['gated'].append(ctrl)
        return {s: ctrls['gated'] for s, ctrls in samples.items()}

    def _check_samples(self,
                       samples: list):
        """
        Internal function. Parses samples and checks that they are valid, that they have associated
        control data, and then returns a dictionary where the key is the sample name and the value a list
        of control IDs associated to this sample, e.g. {'sample_1': ['CXCR3', 'CD27']} would indicate
        one file with controls for CXCR3 and CD27.

        Parameters
        ----------
        samples: list
            List of samples to pass from experiment

        Returns
        -------
        dict
            Returns modified sample dictionary where each key is a sample name and the value a list of control file IDs
            present for this sample
        """
        if not samples:
            samples = self.experiment.list_samples()
        else:
            assert all([s in self.experiment.list_samples() for s in samples]), 'One or more given sample IDs is ' \
                                                                                'invalid or not associated to the ' \
                                                                                'given experiment'

        filtered_samples = list(filter(lambda s: any([f.file_type == 'control' for
                                                      f in self.experiment.pull_sample(s).files]), samples))
        assert filtered_samples, 'No files in the provided experiment have associated control data'
        no_ctrls = [s for s in samples if s not in filtered_samples]
        if no_ctrls:
            print(f'The following samples are missing control data and therefore will be omitted from '
                  f'analysis: {no_ctrls}')
            print('\n')
        return {s: {'all': self.experiment.pull_sample(s).list_controls(),
                    'gated': self.experiment.pull_sample(s).list_gated_controls()} for s in filtered_samples}

    def _has_control_data(self,
                          sample_id: str,
                          marker: str,
                          verbose: bool = False) -> bool:
        """
        Check if a sample has control data

        Parameters
        ----------
        sample_id: str
            Sample ID of sample to check
        marker: str
            Name of control to check
        verbose: bool, (default=False)
            Feedback

        Returns
        -------
        bool
            True if control data present, else False
        """
        assert sample_id in self.samples.keys(), 'Invalid sample_id'
        present_ctrls = self.samples.get(sample_id)
        if marker in present_ctrls:
            return True
        if verbose:
            print(f'Warning: {sample_id} missing control data for {marker}')
        return False

    def _fetch_data(self,
                    sample_id: str,
                    markers: list,
                    population: str,
                    transform: bool = True,
                    transform_method: str = 'logicle') -> dict:
        """
        Internal function. Parse all samples and fetch the given population, collecting data from both
        the control(s) and the primary data. Return a dictionary of Pandas DataFrames, where the key is the data type
        (either primary or control) and the value the DataFrame of events data.

        Parameters
        ----------
        sample_id: str
        markers: list
            List of one or many controls to fetch for each sample
        population: str
            Name of the population that data should be derived from
        transform: bool, (default=True)
            If True, transformation will be applied
        transform_method: str, (default='logicle')
            Name of the transformation method to be applied (see cytopy.flow.transforms)

        Returns
        -------
        dict
            Pandas DataFrame, where samples are identifiable by
            the 'sample_id' column, controls from the 'data_source' column, and the events data itself transformed
            as specified.
        """
        assert all(list(filter(lambda x: self._has_control_data(sample_id=sample_id, marker=x), self.samples.keys()))), \
            f'One or more controls {markers} missing from {sample_id}'
        data = dict()
        g = Gating(self.experiment, sample_id, include_controls=True)
        data['primary'] = g.get_population_df(population,
                                              transform=transform,
                                              transform_method=transform_method,
                                              transform_features=markers)[markers].melt(value_name='primary',
                                                                                        var_name='marker')
        for m in markers:
            data[m] = g.get_population_df(population,
                                          transform=transform,
                                          transform_method=transform_method,
                                          transform_features=markers,
                                          ctrl_id=m)[markers].melt(value_name=m,
                                                                   var_name='marker')
        return data

    @staticmethod
    def _summarise_fi(data: dict, method: callable):
        summary = list()
        for k in list(data.keys()):
            summary.append(data[k].groupby('marker').apply(method).reset_index())
        return reduce(lambda left, right: pd.merge(left, right, on=['marker'],
                                                   how='outer'), summary)

    def _fold_change(self,
                     data: dict,
                     center_func: callable or None = None,
                     sample_id: str or None = None) -> pd.DataFrame:
        """
        Internal function. Calculates the relative fold change in MFI between a control and
        the primary data. NOTE: this function expects only one control present in data.

        Parameters
        ----------
        data: Pandas.DataFrame
            As generated from self._get_data
        center_func: callable, (default=Numpy.mean)
            Function for calculating center of data
        sample_id: str
        Returns
        -------
        Pandas.DataFrame
            Pandas DataFrame where each sample is identifiable from the 'sample_id' column, the control
            identifiable from the 'data_source' column, and the relative fold change given in a column
            aptly named 'relative_fold_change'
        """
        def check_fold_change(x):
            if x.loc[0, f'relative_fold_change_{x.loc[0, "marker"]}'] < 1:
                print(f'WARNING {sample_id}: primary MFI < control {x.loc[0, "marker"]}, interpret results with caution')
        if center_func is None:
            center_func = np.mean
        mfi = self._summarise_fi(data, method=center_func)
        for m in mfi.marker:
            mfi[f'relative_fold_change_{m}'] = mfi['primary'] / mfi[m]
        mfi.groupby('marker').apply(check_fold_change)
        return mfi

    def statistic_1d(self,
                     markers: list,
                     population: str,
                     transform: bool = True,
                     transform_method: str = 'logicle',
                     stat: str = 'fold_change',
                     center_function: callable or None = None) -> pd.DataFrame:
        """
        Calculate a desired statistic for each marker currently acquired

        Parameters
        ----------
        markers: list
            List of markers to calculate statistic for
        population: str
            Population to calculate statistic for
        transform: bool, (default=True)
            If True, apply transfomration to data
        transform_method: str, (default='logicle')
            Transformation method to apply
        stat: str, (default = 'fold_change')
            Name of the statistic to calculate. Currently CytoPy version 0.0.1 only supports fold_change, which is
            the relative fold change in MFI between control and primary data. Future releases hope to include more.
        center_function: callable, (default=Numpy.mean)
            Function to use for calculating center of data

        Returns
        -------
        Pandas.DataFrame
        """
        assert stat in ['fold_change'], f'Invalid statistic, CytoPy currently supports: ["fold_change"]'
        results = pd.DataFrame()
        print('Collecting data...')
        for s in progress_bar(self.samples.keys()):
            valid_markers = list(filter(lambda x: self._has_control_data(sample_id=s, marker=x, verbose=True), markers))
            data = self._fetch_data(sample_id=s,
                                    population=population,
                                    markers=valid_markers,
                                    transform=transform,
                                    transform_method=transform_method)
            if stat == 'fold_change':
                fold_change = self._fold_change(data, center_func=center_function, sample_id=s)
                fold_change['sample_id'] = s
                results = pd.concat([results, fold_change])
        return results


def meta_labelling(experiment: FCSExperiment,
                   dataframe: pd.DataFrame,
                   meta_label: str):
    """
    Given a Pandas DataFrame and an FCSExperiment object, on the assumption that the DataFrame contains
    a column named 'sample_id' with the ID of samples associated to the given FCSExperiment, parse the sample
    ID's and retrieve patient meta-data. Returns modified DataFrame with a new column corresponding to the requested
    meta-data.

    Parameters
    ----------
    experiment: FCSExperiment
        Instance of FCSExperiment to search
    dataframe: Pandas.DataFrame
        DataFrame with a minimum of one column named 'sample_id' that will be passed to fetch
        associated patient meta-data using the given FCSExperiment object
    meta_label: str
        Name of a meta-data variable, associated to the Subject document of each patient, that should be
        retrieved and inserted into the DataFrame

    Returns
    -------
    Pandas.DataFrame
        Modified DataFrame with a new column corresponding to the requested meta-data

    """
    def fetch_meta(sample_id):
        subject = Subject.objects(files=experiment.pull_sample(sample_id))
        assert subject, f'Sample {sample_id} is not associated to any subject!'
        try:
            return subject[0][meta_label]
        except KeyError:
            return None
    assert 'sample_id' in dataframe.columns, 'DataFrame must contain a column of valid sample IDs with ' \
                                             'column name "sample_id"'
    assert all([s in experiment.list_samples() for s in dataframe['sample_id']]), 'One or more sample IDs ' \
                                                                                  'are not associated to the ' \
                                                                                  'given experiment'
    df = dataframe.copy()
    df[meta_label] = df['sample_id'].apply(fetch_meta)
    return df


class ExperimentProportions:
    """
    Calculate proportion of populations/clusters for an experiment

    Parameters
    -----------
    experiment: FCSExperiment
        Experiment to calculate proportions for
    samples: list, optional
        List of samples to include, if not provided, will analyse all samples in experiment
    """
    def __init__(self,
                 experiment: FCSExperiment,
                 samples: list or None = None):
        self.experiment = experiment
        if not samples:
            self.samples = self.experiment.list_samples()
        else:
            assert all([s in experiment.list_samples() for s in samples]), 'One or more given sample IDs is ' \
                                                                           'invalid or not associated to the ' \
                                                                           'given experiment'
            self.samples = samples

    def raw_counts(self,
                   populations: list):
        """
        Collect raw counts of events in populations. Returns a DataFrame with columns 'sample_id',
        '{population_name}'. Where the population column contains the raw counts.

        Parameters
        ----------
        populations: list
            Population names to collect raw counts of

        Returns
        -------
        Pandas.DataFrame
        """
        populations = list(set(populations))
        results = {p: list() for p in populations}
        for s in self.samples:
            results['sample_id'].append(s)
            try:
                prop = Proportions(experiment=self.experiment, sample_id=s)
                for p in populations:
                    results[p].append(prop.get_population_n(p))
            except AssertionError as e:
                print(f'Failed to retrieve data for {s}: {e}')
                continue
        return pd.DataFrame(results)

    def population_proportions(self,
                               parent: str,
                               populations_of_interest: list):
        """
        Retrieve the proportion of populations relative to some parent population

        Parameters
        ----------
        parent: str
            Name of parent population; proportion = population/parent
        populations_of_interest: list
            List of populations to retrieve proportional data for

        Returns
        -------
        Pandas.DataFrame
        """
        populations_of_interest = list(set(populations_of_interest))
        results = pd.DataFrame()
        for s in self.samples:
            try:
                prop = Proportions(experiment=self.experiment, sample_id=s)
                results = pd.concat([results, prop.get_pop_proportions(parent, populations_of_interest)])
            except AssertionError as e:
                print(f'Failed to retrieve data for {s}: {e}')
        return results

    def cluster_proportions(self,
                            comparison_population: str,
                            clusters_of_interest: list,
                            clustering_definition: ClusteringDefinition,
                            merge_associated_clusters: bool = False):
        """
        Retrieve proportion of clusters relative to some comparison population

        Parameters
        ----------
        comparison_population: str
            Population to compare clusters to: proportion = cluster/comparison_population
        clusters_of_interest: list
            Clusters to collect proportions for
        clustering_definition: ClusteringDefinition
            ClusteringDefinition for clusters of interest
        merge_associated_clusters: bool, (default=False)
            If True, values in clusters_of_interest treated as like terms and clusters are merged on like terms
            e.g. a value of 'CD4' would result in all clusters containing the term 'CD4' into one cluster

        Returns
        -------
        Pandas.DataFrame
        """
        clusters_of_interest = list(set(clusters_of_interest))
        results = pd.DataFrame()
        for s in self.samples:
            prop = Proportions(experiment=self.experiment, sample_id=s)
            try:
                results = pd.concat([results, prop.get_cluster_proportions(comparison_population,
                                                                           clusters_of_interest,
                                                                           clustering_definition,
                                                                           merge_associated_clusters)])
            except AssertionError as e:
                print(f'Failed to retrieve data for {s}: {e}')
        return results


class Proportions:
    """
    Calculate proportion of populations/clusters for a single sample

    Parameters
    -----------
    experiment: FCSExperiment
        Experiment sample associate to
    sample_id: str
    """
    def __init__(self,
                 experiment: FCSExperiment,
                 sample_id: str):
        assert sample_id in experiment.list_samples(), f'{sample_id} not found for {experiment.experiment_id}, ' \
                                                       f'are you sure this is a valid sample?'
        self.experiment = experiment
        self.sample_id = sample_id
        self.file_group: FileGroup = self.experiment.pull_sample(self.sample_id)

    def get_population_n(self,
                         population: str):
        """
        Get raw counts of a single population

        Parameters
        ----------
        population: str
            Name of population

        Returns
        -------
        Int
            N events
        """
        return self.file_group.get_population(population).n

    def get_cluster_n(self,
                      clustering_definition: ClusteringDefinition,
                      cluster_id: str,
                      merge_on_like_term: bool = False):
        """
        Get number of events in a single cluster

        Parameters
        ----------
        clustering_definition: ClusteringDefinition
            ClusteringDefinition for cluster of interest
        cluster_id: str
            ID for cluster of interest
        merge_on_like_term: bool, (default=False)
            If True, cluster_id treated as like terms and clusters are merged on like terms
            e.g. a value of 'CD4' would result in all clusters containing the term 'CD4' into one cluster

        Returns
        -------
        Int
            N events
        """
        root = self.file_group.get_population(clustering_definition.root_population)
        if merge_on_like_term:
            # Get a list of meta cluster IDs that contain the term of interest
            meta_clusters = [c.meta_cluster_id for c in root.clustering]
            filtered_clusters = set(filter(lambda c: cluster_id in c, meta_clusters))
            # Use the filtered list to generate a list of relevant cluster objects
            clusters = [c for c in root.clustering if c.meta_cluster_id in filtered_clusters]
            # If empty return 0
            if not clusters:
                return 0
            # Sum the resulting cluster n's
            return sum([c.n_events for c in clusters])
        try:
            return len(root.get_cluster(cluster_id=cluster_id, meta=clustering_definition.meta_method)[1])
        except AssertionError:
            return 0

    def _as_dataframe(self,
                      results: dict):
        """
        Convert results to DataFrame

        Parameters
        ----------
        results: dict
            Dictionary of results

        Returns
        -------
        Pandas.DataFrame
        """
        results['sample_id'] = [self.sample_id]
        try:
            return pd.DataFrame(results)
        except ValueError as e:
            print(f'Unable to convert dictionary to DataFrame: {results}')
            raise ValueError(e)

    def get_pop_proportions(self,
                            parent: str,
                            populations_of_interest: list):
        """
        Get the proportion of events for populations

        Parameters
        ----------
        parent: str
            Parent population for comparison; proportion = population/parent
        populations_of_interest: list
            List of populations to calculate proportions for

        Returns
        -------
        Pandas.DataFrame
        """
        results = {i: list() for i in populations_of_interest}
        population_n = {p: self.get_population_n(p) for p in populations_of_interest}
        parent_n = self.get_population_n(parent)
        for p in populations_of_interest:
            assert population_n.get(p) < parent_n, f'Population {p} is larger than the given parent {parent}, ' \
                                                   f'are you sure {parent} is upstream of {p}'
            results[p].append(population_n.get(p)/parent_n)
        return self._as_dataframe(results)

    def get_cluster_proportions(self,
                                comparison_population: str,
                                clusters_of_interest: list,
                                clustering_definition: ClusteringDefinition,
                                merge_associated_clusters: bool = False):
        """
        Retrieve proportion of clusters relative to some comparison population

        Parameters
        ----------
        comparison_population: str
            Population to compare clusters to: proportion = cluster/comparison_population
        clusters_of_interest: list
            Clusters to collect proportions for
        clustering_definition: ClusteringDefinition
            ClusteringDefinition for clusters of interest
        merge_associated_clusters: bool, (default=False)
            If True, values in clusters_of_interest treated as like terms and clusters are merged on like terms
            e.g. a value of 'CD4' would result in all clusters containing the term 'CD4' into one cluster

        Returns
        -------
        Pandas.DataFrame
        """
        results = {i: list() for i in clusters_of_interest}
        cluster_n = {c: self.get_cluster_n(clustering_definition,
                                           cluster_id=c,
                                           merge_on_like_term=merge_associated_clusters) for c in clusters_of_interest}
        denominator = self.file_group.get_population(comparison_population).n
        for c in clusters_of_interest:
            results[c].append(cluster_n[c]/denominator)
        return self._as_dataframe(results)


def sort_variance(summary: pd.DataFrame):
    """
    Given the output of either population_proportions or cluster_proportions of the class ExperimentProportions,
    rank the clusters or propulations by variance (highest to lowest)

    Parameters
    ----------
    summary: Pandas.DataFrame

    Returns
    -------
    Pandas.DataFrame
        Column 'feature' ranked by column 'variance' (highest to lowest)

    """

    x = summary.melt(id_vars='sample_id',
                     value_name='prop',
                     var_name='feature')
    return x.groupby('feature').var().reset_index().sort_values('prop',
                                                                ascending=False).rename(columns={'prop': 'variance'})


def multicolinearity_matrix(summary: pd.DataFrame,
                            features: list,
                            **kwargs):
    """
    Given the output of either population_proportions or cluster_proportions of the class ExperimentProportions,
    or a concatenation of results (where the features are each contained within their own column) plot a
    multicolinearity matrix

    Parameters
    ----------
    summary: Pandas.DataFrame
        Columns correspond to features
    features: List
        List of features to include
    kwargs:
        Additional keyword arguments to pass to call to Seaborn.clustermap

    Returns
    -------
    Seaborn.ClusterGrid
    """
    corr = summary[features].corr()
    return sns.clustermap(corr, **kwargs)


def dim_reduction(summary: pd.DataFrame,
                  label: str,
                  features: list or None = None,
                  scale: bool = True,
                  method: str = 'PCA',
                  **plotting_kwargs):
    """
    Given a dataframe of features where each column is a feature, each row a subject, and a column whom's name is equal
    to the value of the label argument is the target label that colours our data points in our plot, perform
    dimensionality reduction and plot outcome.

    Parameters
    ----------
    summary: Pandas.DataFrame
        Dataframe of features index by column 'sample_id' and containing a column whom's name is equal to the argument
        label
    features: list, optional
        If given, only these columns will be used as features
    label: str
        Column to colour data points by
    scale: bool, (default=True)
        If True, values of features are scaled (standard scale) prior to dimensionality reduction
    method: str, (default='PCA')
        Method used for dimensionality reduction (see cytopy.flow.dim_reduction)
    plotting_kwargs:
        Additional keyword arguments to pass to call to Seaborn.scatterplot

    Returns
    -------
    Matplotlib.axes

    """
    if features is None:
        features = [f for f in summary.columns if f not in ['sample_id', label]]
    feature_space = summary[features].drop_na()
    if scale:
        scaler = StandardScaler()
        feature_space = pd.DataFrame(scaler.fit_transform(feature_space.values),
                                     columns=feature_space.columns,
                                     index=feature_space.index)
    feature_space = dimensionality_reduction(feature_space,
                                             features=features,
                                             method=method,
                                             n_components=2)
    fig, ax = plt.subplots(figsize=(8, 8))
    feature_space['label'] = summary[label]
    ax = sns.scatterplot(data=feature_space,
                         x='PCA_0',
                         y='PCA_1',
                         hue='label',
                         ax=ax,
                         **plotting_kwargs)
    return ax


def radar_plot(summary: pd.DataFrame,
               features: list,
               figsize: tuple = (10, 10)):
    """
    Given a Pandas DataFrame where columns are features and each row is a different subject
    (indexed by a column named 'subject_id'), generate a radar plot of all the features
    Parameters
    ----------
    summary: Pandas.DataFrame
    features: List
        Features to be included in the plot
    figsize: tuple, (default=(10,10))

    Returns
    -------
    Matplotlib.axes

    """
    summary = summary.melt(id_vars='subject_id',
                           value_name='stat',
                           var_name='feature')
    summary = summary[summary.feature.isin(features)]
    labels = summary.feature.values
    stats = summary.stat.values

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))  # Closed
    angles = np.concatenate((angles, [angles[0]]))  # Closed

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)  # Set polar axis
    ax.plot(angles, stats, 'o-', linewidth=2, c='blue')
    ax.set_thetagrids(angles * 180 / np.pi, labels)  # Set the label for each axis
    ax.tick_params(pad=30)
    ax.set_rlabel_position(145)
    ax.grid(True)
    return ax


def l1_feature_selection(feature_space: pd.DataFrame,
                         features: list,
                         label: str,
                         scale: bool = True,
                         search_space: tuple = (-2, 0, 50),
                         model: callable or None = None,
                         figsize: tuple = (10, 5)):
    """
    Perform L1 regularised classification over a defined search space for the L1 parameter and plot the
    coefficient of each feature in respect to the change in L1 parameter.

    Parameters
    ----------
    feature_space: Pandas.DataFrame
        A dataframe of features where each column is a feature, each row a subject, and a column whom's name is equal
        to the value of the label argument is the target label for prediction
    features: List
        List of features to include in model
    label: str
        The target label to predict
    scale: bool, (default=True)
        if True, features are scaled (standard scale) prior to analysis
    search_space: tuple, (default=(-2, 0, 50))
        Search range for L1 parameter
    model: callable, optional
        Must be a Scikit-Learn classifier that accepts an L1 regularisation parameter named 'C'. If left as None,
        a linear SVM is used
    figsize: tuple, (default=(10,5))

    Returns
    -------
    Matplotlib.axes

    """

    y = feature_space[label].values
    feature_space = feature_space[features].drop_na()
    if scale:
        scaler = StandardScaler()
        feature_space = pd.DataFrame(scaler.fit_transform(feature_space.values),
                                     columns=feature_space.columns,
                                     index=feature_space.index)
    x = feature_space.values
    cs = np.logspace(*search_space)
    coefs = []
    if model is None:
        model = LinearSVC(penalty='l1',
                          loss='squared_hinge',
                          dual=False,
                          tol=1e-3)
    for c in cs:
        model.set_params(C=c)
        model.fit(x, y)
        coefs.append(list(model.coef_[0]))
    coefs = np.array(coefs)

    # Plot result
    fig, ax = plt.subplots(figsize=figsize)
    for i, col in enumerate(range(len(features))):
        ax.plot(cs, coefs[:, col], label=features[i])
    ax.xscale('log')
    ax.title('L1 penalty')
    ax.xlabel('C')
    ax.ylabel('Coefficient value')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    return ax
