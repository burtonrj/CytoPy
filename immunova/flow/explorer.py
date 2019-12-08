from mongoengine.base.datastructures import EmbeddedDocumentList
from immunova.data.fcs_experiments import FCSExperiment
from immunova.data.patient import Patient, Bug
from immunova.flow.gating.actions import Gating
import pandas as pd


class Explorer:
    """
    Using a dimensionality reduction technique, explore high dimensional flow cytometry data.
    """
    def __init__(self, root_population: str = 'root', transform: str or None = 'logicle',
                 clustering_uid: str or None = None):
        """
        :param root_population: data included in the indicated population will be pulled from each file
        :param transform: transform to be applied to each sample, provide a value of None to use raw data
        (default = 'logicle')
        :param clustering_uid: if a clustering_uid is provided, then will pull clustering experiment from the root
        population of each file group (optional)
        """
        self.data = pd.DataFrame()
        self.gate_labels = dict()
        self.cluster_labels = dict()
        self.transform = transform
        self.root_population = root_population
        self.clustering_uid = clustering_uid

    def clear_data(self):
        """
        Clear all existing data, gate labels, and cluster labels.
        """
        self.data = None
        self.cluster_labels = dict()
        self.gate_labels = dict()

    def load_data(self, experiment: FCSExperiment, samples: list, sample_n: None or int = None):
        """
        Load fcs file data, including any associated gates or clusters
        :param experiment: FCSExperiment to load samples from
        :param samples: list of sample IDs
        :param sample_n: if an integer value is provided, each file will be downsampled to the indicated
        amount (optional)
        """
        print(f'------------ Loading flow data: {experiment.experiment_id} ------------')
        for sid in samples:
            print(f'Loading {sid}...')
            g = Gating(experiment, sid, include_controls=False)
            if self.transform is not None:
                print(f'...applying {self.transform} transformation...')
                fdata = g.get_population_df(population_name=self.root_population,
                                            transform=True,
                                            transform_method=self.transform)
            else:
                fdata = g.get_population_df(population_name=self.root_population)
            if fdata is None:
                raise ValueError(f'Population {self.root_population} does not exist for {sid}')
            if sample_n is not None:
                print('...sampling...')
                fdata = fdata.sample(n=sample_n)
            if self.clustering_uid is not None:
                print('...loading clusters...')
                self.__load_clusters(experiment, sid)
            print('...loading gated populations...')
            self.__load_gates(g)
            fdata = fdata.reset_index()
            print('...associating to patient...')
            pt = Patient.objects(files__contains=g.mongo_id)
            if pt:
                fdata['pt_id'] = pt[0].patient_id
                print(f'...patient ID = {pt[0].patient_id}...')
            else:
                print(f'File group {g.id} in experiment {experiment.experiment_id} is not associated to any patient')
                fdata['pt_id'] = 'NONE'
            self.data = pd.concat([self.data, fdata])
    print('------------ Completed! ------------')

    def __load_clusters(self, exp: FCSExperiment, sid: str):
        """
        Internal method. Load root population from given sample and update cluster_labels.
        :param exp: experiment object sample is associated to
        :param sid: sample ID
        """
        fg = exp.pull_sample(sid)
        root_p = [p for p in fg.populations if p.population_name == self.root_population][0]
        clustering = root_p.pull_clustering_experiment(self.clustering_uid)
        self.cluster_labels[sid] = dict()
        for c in clustering.clusters:
            self.cluster_labels[sid][c.cluster_id] = c.load_index()

    def __load_gates(self, gating: Gating):
        """
        Given some gating object for a sample, update the gating labels with all populations downstream of the root
        population.
        :param gating: gating object of the sample.
        """
        populations = gating.find_dependencies(self.root_population)
        self.gate_labels[gating.id] = {p: gating.populations[p].index for p in populations}

    def load_meta(self, variable: str):
        """
        Load meta data for each patient. Must be provided with a variable that is a field with a single value
        NOT an embedded document. A column will be generated in the Pandas DataFrame stored in the attribute 'data'
        that pertains to the variable given and the value will correspond to that of the patients.
        :param variable: field name to populate data with
        """
        self.data[variable] = self.data['pt_id'].apply(lambda x: self.__meta(pt_id=x, variable=variable), axis=1)

    def __meta(self, pt_id: str, variable: str) -> str:
        """
        Internal method. Applied to 'pt_id' column of 'data' to create new variable that is patient specific.
        """
        if pt_id == 'NONE':
            return 'NONE'
        p = Patient.objects(patient_id=pt_id).get()
        if type(p[variable]) == EmbeddedDocumentList:
            raise TypeError('Chosen variable is an embedded document.')
        return p[variable]

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
        self.data['organism_name'] = self.data['pt_id'].apply(lambda x: self.__bugs(x, multi_org=multi_org))
        self.data['organism_type'] = self.data['pt_id'].apply(self.__org_type)
        self.data['hmbpp'] = self.data['pt_id'].apply(lambda x: self.__hmbpp_ribo(x, field='hmbpp_status'))
        self.data['ribo'] = self.data['pt_id'].apply(lambda x: self.__hmbpp_ribo(x, field='ribo_status'))

    @staticmethod
    def __bugs(pt_id: str, multi_org: str) -> str:
        """
        Internal function. Fetch the name of isolated organisms for each patient.
        :param pt_id: patient identifier
        :param multi_org: If 'multi_org' equals 'list' then multiple organisms will be stored as a comma separated list
        without duplicates, whereas if the value is 'mixed' then multiple organisms will result in a value of 'mixed'.
        """
        if pt_id == 'NONE':
            return 'NONE'
        p = Patient.objects(patient_id=pt_id).get()
        if not p.infection_data:
            return 'NONE'
        orgs = [b.org_name for b in p.infection_data]
        if len(orgs) == 1:
            return orgs[0]
        if multi_org == 'list':
            return ','.join(orgs)
        return 'mixed'

    @staticmethod
    def __org_type(pt_id: str) -> str:
        """
        Parse all infectious isolates for each patient and return the organism type isolated, one of either:
        'gram positive', 'gram negative', 'virus', 'mixed' or 'fungal'
        :param pt_id: patient identifier
        """
        def bug_type(b: Bug):
            if b.organism_type == 'bacteria':
                return b.gram_status
            return b.organism_type
        if pt_id == 'NONE':
            return 'NONE'
        p = Patient.objects(patient_id=pt_id).get()
        bugs = set(map(bug_type, p.infection_data))
        if len(bugs) == 1:
            return bugs[0]
        return 'mixed'

    @staticmethod
    def __hmbpp_ribo(pt_id: str, field: str) -> bool or None:
        """
        Given a value of either 'hmbpp' or 'ribo' for 'field' argument, return True if any Bug has a positive status
        for the given patient ID.
        """
        if pt_id == 'NONE':
            return None
        p = Patient.objects(patient_id=pt_id).get()
        return any([b[field] for b in p.infection_data])

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
        pass

    def static_umap(self):
        pass

    def static_phate(self):
        pass

    def launch_bokeh_dashboard(self):
        pass



