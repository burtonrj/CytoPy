from mongoengine.base.datastructures import EmbeddedDocumentList
from immunova.data.fcs_experiments import FCSExperiment
from immunova.data.patient import Patient, Bug
from immunova.flow.gating.actions import Gating
from immunova.flow.utilities import progress_bar
from anytree.node import Node
import matplotlib.pyplot as plt
import phenograph
import pandas as pd
import numpy as np
import umap
import phate
import scprep


class Explorer:
    """
    Using a dimensionality reduction technique, explore high dimensional flow cytometry data.
    """
    def __init__(self, root_population: str = 'root', transform: str or None = 'logicle'):
        """
        :param root_population: data included in the indicated population will be pulled from each file
        :param transform: transform to be applied to each sample, provide a value of None to use raw data
        (default = 'logicle')
        population of each file group (optional)
        """
        self.data = pd.DataFrame()
        self.transform = transform
        self.root_population = root_population

    def clear_data(self):
        """
        Clear all existing data, gate labels, and cluster labels.
        """
        self.data = None

    def load_data(self, experiment: FCSExperiment, samples: list, sample_n: None or int = None):
        """
        Load fcs file data, including any associated gates or clusters
        :param experiment: FCSExperiment to load samples from
        :param samples: list of sample IDs
        :param sample_n: if an integer value is provided, each file will be downsampled to the indicated
        amount (optional)
        """
        print(f'------------ Loading flow data: {experiment.experiment_id} ------------')
        for sid in progress_bar(samples):
            g = Gating(experiment, sid, include_controls=False)
            if self.transform is not None:
                fdata = g.get_population_df(population_name=self.root_population,
                                            transform=True,
                                            transform_method=self.transform)
            else:
                fdata = g.get_population_df(population_name=self.root_population)
            if fdata is None:
                raise ValueError(f'Population {self.root_population} does not exist for {sid}')
            if sample_n is not None:
                if sample_n < fdata.shape[0]:
                    fdata = fdata.sample(n=sample_n)
            fdata = self.__population_labels(fdata, g.populations[self.root_population])
            fdata = fdata.reset_index()
            fdata = fdata.rename({'index': 'original_index'}, axis=1)
            pt = Patient.objects(files__contains=g.mongo_id)
            if pt:
                fdata['pt_id'] = pt[0].patient_id
            else:
                print(f'File group {g.id} in experiment {experiment.experiment_id} is not associated to any patient')
                fdata['pt_id'] = 'NONE'
            self.data = pd.concat([self.data, fdata])
        print('------------ Completed! ------------')

    def __population_labels(self, data: pd.DataFrame, root_node: Node) -> pd.DataFrame:
        """
        Internal function. Called when loading data. Populates DataFrame column named 'population_label' with the
        name of the node associated with each event most downstream of the root population.
        :param data: Pandas DataFrame of events corresponding to root population from single patient
        :param root_node: anytree Node object of root population
        :return: Pandas DataFrame with 'population_label' column
        """
        def recursive_label(d, n):
            mask = d.index.isin(n.index)
            d.loc[mask, 'population_label'] = n.name
            if len(n.children) == 0:
                return d
            for c in n.children:
                recursive_label(d, c)
            return d
        data = data.copy()
        data['population_label'] = self.root_population
        data = recursive_label(data, root_node)
        return data

    def load_meta(self, variable: str):
        """
        Load meta data for each patient. Must be provided with a variable that is a field with a single value
        NOT an embedded document. A column will be generated in the Pandas DataFrame stored in the attribute 'data'
        that pertains to the variable given and the value will correspond to that of the patients.
        :param variable: field name to populate data with
        """
        self.data[variable] = 'NONE'
        for pt_id in progress_bar(self.data.pt_id.unique()):
            if pt_id == 'NONE':
                continue
            p = Patient.objects(patient_id=pt_id).get()
            if type(p[variable]) == EmbeddedDocumentList:
                raise TypeError('Chosen variable is an embedded document.')
            self.data.loc[self.data.pt_id == pt_id, variable] = p[variable]

    def load_infectious_data(self, multi_org: str = 'list'):
        """
        Load the bug data from each patient and populate 'data' accordingly. As default variables will be created as
        follows:
        * organism_name = If 'multi_org' equals 'list' then multiple organisms will be stored as a comma separated list
        without duplicates, whereas if the value is 'mixed' then multiple organisms will result in a value of 'mixed'.
        * organism_type = value of either 'gram positive', 'gram negative', 'virus', 'mixed' or 'fungal'
        * hmbpp = True or False based on HMBPP status (Note: it only takes one positive organism for this value to be
        True)
        * ribo = True or False based on Ribo status (Note: it only takes one positive organism for this value to be
        True)
        """
        self.data['organism_name'] = 'Unknown'
        self.data['organism_type'] = 'Unknown'
        self.data['hmbpp'] = 'Unknown'
        self.data['ribo'] = 'Unknown'

        for pt_id in progress_bar(self.data.pt_id.unique()):
            if pt_id == 'NONE':
                continue
            p = Patient.objects(patient_id=pt_id).get()
            self.data.loc[self.data.pt_id == pt_id, 'organism_name'] = self.__bugs(patient=p, multi_org=multi_org)
            self.data.loc[self.data.pt_id == pt_id, 'organism_type'] = self.__org_type(patient=p)
            self.data.loc[self.data.pt_id == pt_id, 'hmbpp'] = self.__hmbpp_ribo(patient=p, field='hmbpp_status')
            self.data.loc[self.data.pt_id == pt_id, 'ribo'] = self.__hmbpp_ribo(patient=p, field='ribo_status')

    @staticmethod
    def __bugs(patient: Patient, multi_org: str) -> str:
        """
        Internal function. Fetch the name of isolated organisms for each patient.
        :param patient: Patient model object
        :param multi_org: If 'multi_org' equals 'list' then multiple organisms will be stored as a comma separated list
        without duplicates, whereas if the value is 'mixed' then multiple organisms will result in a value of 'mixed'.
        :return: string of isolated organisms comma seperated, or 'mixed' if multi_org == 'mixed' and multiple organisms
        listed for patient
        """
        if not patient.infection_data:
            return 'Unknown'
        orgs = [b.org_name for b in patient.infection_data if b.org_name]
        if not orgs:
            return 'Unknown'
        if len(orgs) == 1:
            return orgs[0]
        if multi_org == 'list':
            return ','.join(orgs)
        return 'mixed'

    @staticmethod
    def __org_type(patient: Patient) -> str:
        """
        Parse all infectious isolates for each patient and return the organism type isolated, one of either:
        'gram positive', 'gram negative', 'virus', 'mixed' or 'fungal'
        :param patient: Patient model object
        :return: common organism type isolated for patient
        """
        def bug_type(b: Bug):
            if not b.organism_type:
                return 'Unknown'
            if b.organism_type == 'bacteria':
                return b.gram_status
            return b.organism_type
        bugs = list(set(map(bug_type, patient.infection_data)))
        if len(bugs) == 0:
            return 'Unknown'
        if len(bugs) == 1:
            return bugs[0]
        return 'mixed'

    @staticmethod
    def __hmbpp_ribo(patient: Patient, field: str) -> str:
        """
        Given a value of either 'hmbpp' or 'ribo' for 'field' argument, return True if any Bug has a positive status
        for the given patient ID.
        :param patient: Patient model object
        :param field: field name to search for; expecting either 'hmbpp_status' or 'ribo_status'
        :return: common value of hmbpp_status/ribo_status
        """
        if all([b[field] is None for b in patient.infection_data]):
            return 'Unknown'
        if all([b[field] == 'P+ve' for b in patient.infection_data]):
            return 'P+ve'
        if all([b[field] == 'N-ve' for b in patient.infection_data]):
            return 'N-ve'
        return 'mixed'

    def load_biology_data(self, test_name: str, summary_method: str = 'average'):
        """
        Load the pathology results of a given test from each patient and populate 'data' accordingly. As multiple
        results may exist for one particular test, a summary method should be provided, this should have a value as
        follows:
        * average - the average test result is generated and stored
        * max - the maximum value is stored
        * min - the minimum value is stored
        * median - the median test result is generated and stored
        """
        self.data[test_name] = self.data['pt_id'].apply(lambda x: self.__biology(x, test_name, summary_method))

    @staticmethod
    def __biology(pt_id: str, test_name: str, method: str) -> np.float or None:
        """
        Given some test name, return a summary statistic of all results for a given patient ID
        :param pt_id: patient identifier
        :param test_name: name of test to search for
        :param method: summary statistic to use
        """
        if pt_id == 'NONE':
            return None
        tests = Patient.objects(patient_id=pt_id).get().patient_biology
        tests = [t.result for t in tests if t.test == test_name]
        if not tests:
            return None
        if method == 'max':
            return np.max(tests)
        if method == 'min':
            return np.min(tests)
        if method == 'median':
            return np.median(tests)
        return np.average(tests)

    def __plotting_labels(self, label: str, populations: list or None):
        """
        Internal function called by plotting functions to generate array that will be used for the colour property
        of each data point.
        :param label: string value of the label option (see scatter_plot method)
        :param populations: list of populations to include if label = 'gated populations' (optional)
        """
        if label in self.data.columns:
            return self.data[label].values
        elif label == 'global clusters':
            if 'PhenoGraph labels' not in self.data.columns:
                raise ValueError('Must call phenograph_clustering method prior to plotting PhenoGraph clusters')
            return self.data['PhenoGraph labels'].values
        elif label == 'gated populations':
            if populations:
                return np.array(list(map(lambda x: x if x in populations else 'None',
                                         self.data['population_label'].values)))
            return self.data['population_label'].values
        raise ValueError(f'Label {label} is invalid; must be either a column name in the existing dataframe '
                         f'({self.data.columns.tolist()}), "global clusters" which labels events according to '
                         f'clustering on the concatenated dataset, or "gated populations" which are common populations '
                         f'gated on a per sample basis.')

    def scatter_plot(self, primary_label: str, features: list, secondary_label: str or None = None,
                     populations: list or None = None, n_components: int = 2,
                     dim_reduction_method: str = 'UMAP', **kwargs) -> plt.Axes:
        """
        Generate a 2D/3D scatter plot (dimensions depends on the number of components chosen for dimensionality
        reduction. Each data point is labelled according to the option provided to the label arguments. If a value
        is given to both primary and secondary label, the secondary label colours the background and the primary label
        colours the foreground of each datapoint.
        :param primary_label: option for the primary label, must be one of the following:
        * A valid column name in Explorer attribute 'data' (check valid column names using Explorer.data.columns)
        * 'global clusters' - requires that phenograph_clustering method has been called prior to plotting. Each data
        point will be coloured according to cluster association.
        * 'gated populations' - each data point is coloured according to population identified by prior gating
        :param features: list of column names used as feature space for dimensionality reduction
        :param secondary_label: option for the secondary label, options same as primary_label (optional)
        :param populations: if primary/secondary label has value of 'gated populations', only populations in this
        list will be included (events with no population associated will be labelled 'None')
        :param n_components: number of components to produce from dimensionality reduction, valid values are 2 or 3
        (default = 2)
        :param dim_reduction_method: method to use for dimensionality reduction, valid values are 'UMAP' or 'PHATE'
        (default = 'UMAP')
        :param kwargs: additional keyword arguments to pass to dimensionality reduction algorithm
        :return: matplotlib subplot axes object
        """
        #ToDo Add caching
        fig, ax = plt.subplots(figsize=(12, 8))
        if n_components not in [2, 3]:
            raise ValueError('n_components must have a value of 2 or 3')
        if dim_reduction_method == 'UMAP':
            embeddings = umap.UMAP(n_components=n_components, **kwargs).fit_transform(self.data[features])
        else:
            phate_operator = phate.PHATE(n_jobs=-2, **kwargs)
            embeddings = phate_operator.fit_transform(self.data[features])
        plabel = self.__plotting_labels(primary_label, populations)
        if secondary_label is not None:
            slabel = self.__plotting_labels(secondary_label, populations)
            if n_components == 2:
                ax = scprep.plot.scatter2d(embeddings, c=slabel, ticks=False,
                                           label_prefix=dim_reduction_method, ax=ax, s=50)
            else:
                ax = scprep.plot.scatter3d(embeddings, c=slabel, ticks=False,
                                           label_prefix=dim_reduction_method, ax=ax, s=50)
        if n_components == 2:
            ax = scprep.plot.scatter2d(embeddings, c=plabel, ticks=False,
                                       label_prefix=dim_reduction_method, ax=ax, s=50)
        else:
            ax = scprep.plot.scatter3d(embeddings, c=plabel, ticks=False,
                                       label_prefix=dim_reduction_method, ax=ax, s=50)
        return ax

    def launch_bokeh_dashboard(self):
        pass

    def phenograph_clustering(self, features: list, **kwargs):
        """
        Using the PhenoGraph clustering algorithm, cluster all events in concatenated dataset.
        :param features: list of features to perform clustering on
        :param kwargs: keyword arguments to pass to PhenoGraph clustering object
        """
        communities, graph, q = phenograph.cluster(self.data[features], **kwargs)
        self.data['PhenoGraph labels'] = communities



