from ..data.fcs_experiments import FCSExperiment
from ..data.fcs import FileGroup
from ..data.subject import Subject
from ..data.fcs import ClusteringDefinition
from ..flow.gating.actions import Gating
from .feedback import progress_bar
from functools import reduce
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
    def __init__(self, experiment: FCSExperiment,
                 samples: list or None = None,
                 gating_model: str = 'knn',
                 tree_map: dict or None = None,
                 **model_kwargs):
        self.experiment = experiment
        print('---- Preparing for control comparisons ----')
        print('If control files have not been gated prior to initiation, control gating will '
              'be performed automatically (gating methodology can be specified in ControlComparison arguments '
              'see documentation for details). Note: control gating can take some time so please be patient. '
              'Results will be saved to the database to reduce future computation time.')
        print('\n')
        self.samples = self._gate_samples(self._check_samples(samples), tree_map, gating_model, **model_kwargs)

    def _gate_samples(self, samples: dict, tree_map, gating_model, **model_kwargs) -> dict:
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

    def _check_samples(self, samples: list):
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

    def _has_control_data(self, sample_id: str, marker: str, verbose: bool = False) -> bool:
        assert sample_id in self.samples.keys(), 'Invalid sample_id'
        present_ctrls = self.samples.get(sample_id)
        if marker in present_ctrls:
            return True
        if verbose:
            print(f'Warning: {sample_id} missing control data for {marker}')
        return False

    def _fetch_data(self, sample_id: str, markers: list, population: str,
                    transform: bool = True, transform_method: str = 'logicle') -> pd.DataFrame:
        """
        Internal function. Parse all samples and fetch the given population, collecting data from both
        the control(s) and the primary data. Return a Pandas DataFrame, where samples are identifiable by
        the 'sample_id' column, controls from the 'data_source' column, and the events data itself transformed
        as specified.

        Parameters
        ----------
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
        Pandas.DataFrame
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

    def _fold_change(self, data: dict, center_func: callable or None = None, sample_id: str or None = None) -> pd.DataFrame:
        """
        Internal function. Calculates the relative fold change in MFI between a control and
        the primary data. NOTE: this function expects only one control present in data.

        Parameters
        ----------
        data: Pandas.DataFrame
            As generated from self._get_data

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
                     center_function: callable or None =  None) -> pd.DataFrame:
        """
        Calculate a desired statistic for each marker currently acquired

        Parameters
        ----------
        stat: str, (default = 'fold_change')
            Name of the statistic to calculate. Currently CytoPy version 0.0.1 only supports fold_change, which is
            the relative fold change in MFI between control and primary data. Future releases hope to include more.

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


def meta_labelling(experiment: FCSExperiment, dataframe: pd.DataFrame, meta_label: str):
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
    def __init__(self, experiment: FCSExperiment, samples: list or None = None):
        self.experiment = experiment
        if not samples:
            self.samples = self.experiment.list_samples()
        else:
            assert all([s in experiment.list_samples() for s in samples]), 'One or more given sample IDs is ' \
                                                                           'invalid or not associated to the ' \
                                                                           'given experiment'
            self.samples = samples

    def raw_counts(self, populations: list):
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

    def population_proportions(self, parent: str, populations_of_interest: list):
        populations_of_interest = list(set(populations_of_interest))
        results = pd.DataFrame()
        for s in self.samples:
            try:
                prop = Proportions(experiment=self.experiment, sample_id=s)
                results = pd.concat([results, prop.get_pop_proportions(parent, populations_of_interest)])
            except AssertionError as e:
                print(f'Failed to retrieve data for {s}: {e}')
        return results

    def cluster_proportions(self, comparison_population: str, clusters_of_interest: list,
                            clustering_definition: ClusteringDefinition,
                            merge_associated_clusters: bool = False):
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
    def __init__(self, experiment: FCSExperiment, sample_id: str):
        assert sample_id in experiment.list_samples(), f'{sample_id} not found for {experiment.experiment_id}, ' \
                                                       f'are you sure this is a valid sample?'
        self.experiment = experiment
        self.sample_id = sample_id
        self.file_group: FileGroup = self.experiment.pull_sample(self.sample_id)

    def get_population_n(self, population: str):
        return self.file_group.get_population(population).n

    def get_cluster_n(self, clustering_definition: ClusteringDefinition, cluster_id: str,
                      merge_on_like_term: bool = False):
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

    def _as_dataframe(self, results: dict):
        results['sample_id'] = [self.sample_id]
        try:
            return pd.DataFrame(results)
        except ValueError as e:
            print(f'Unable to convert dictionary to DataFrame: {results}')
            raise ValueError(e)

    def get_pop_proportions(self, parent: str, populations_of_interest: list):
        results = {i: list() for i in populations_of_interest}
        population_n = {p: self.get_population_n(p) for p in populations_of_interest}
        parent_n = self.get_population_n(parent)
        for p in populations_of_interest:
            assert population_n.get(p) < parent_n, f'Population {p} is larger than the given parent {parent}, ' \
                                                   f'are you sure {parent} is upstream of {p}'
            results[p].append(population_n.get(p)/parent_n)
        return self._as_dataframe(results)

    def get_cluster_proportions(self, comparison_population: str, clusters_of_interest: list,
                                clustering_definition: ClusteringDefinition, merge_associated_clusters: bool = False):
        results = {i: list() for i in clusters_of_interest}
        cluster_n = {c: self.get_cluster_n(clustering_definition,
                                           cluster_id=c,
                                           merge_on_like_term=merge_associated_clusters) for c in clusters_of_interest}
        denominator = self.file_group.get_population(comparison_population).n
        for c in clusters_of_interest:
            results[c].append(cluster_n[c]/denominator)
        return self._as_dataframe(results)


