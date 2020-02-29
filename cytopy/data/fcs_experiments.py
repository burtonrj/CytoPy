from .fcs import FileGroup, File
from .subject import Subject
from .gating import GatingStrategy
from .utilities import data_from_file
from .panel import Panel, ChannelMap
from cytopy.flow.read_write import FCSFile
from multiprocessing import Pool, cpu_count
from functools import partial
import mongoengine


class FCSExperiment(mongoengine.Document):
    """
    Document representation of Flow Cytometry experiment

    Attributes:
        experiment_id - unique identifier for experiment
        panel - Panel object describing associated channel/marker pairs
        fcs_files - reference field for associated files
        flags - warnings associated to experiment
        notes - additional free text comments
        gating_templates - reference to gating templates associated to this experiment
        meta_cluster_ids - list of IDs for meta clusters belonging to this experiment
    Methods:
        pull_sample_data - Given a sample ID, associated to this experiment, fetch the fcs data
        pull_sample - Given a sample ID, returns the FileGroup object
        list_samples - Generate a list IDs of file groups associated to experiment
        list_invalid - Lists all samples that have an 'invalid' flag
        remove_sample - Remove sample (FileGroup) from experiment.
        add_new_sample - Add a new sample (FileGroup) to this experiment
        fetch_sample_mid - Given a sample_id, return it's corresponding mongo ObjectID
        pull_sample_mappings - Given a sample ID, return a dictionary of channel/marker mappings for
        all associated fcs files
        delete_all_populations - deletes all population data associated to a given sample; value of 'all' will delete population data for every sample
        delete_gating_templates - deletes a gating template associated to experiment; value of 'all' will delete all gating templates
        sample_exists - checks if sample is associated to experiment

    """
    experiment_id = mongoengine.StringField(required=True, unique=True)
    panel = mongoengine.ReferenceField(Panel, reverse_delete_rule=4)
    fcs_files = mongoengine.ListField(mongoengine.ReferenceField(FileGroup, reverse_delete_rule=4))
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)
    gating_templates = mongoengine.ListField(mongoengine.ReferenceField(GatingStrategy, reverse_delete_rule=4))
    meta_cluster_ids = mongoengine.ListField()

    meta = {
        'db_alias': 'core',
        'collection': 'fcs_experiments'
    }

    def delete_all_populations(self, sample_id: str, remove_gates: bool = False) -> None:
        """
        Delete population data associated to experiment. Give a value of 'all' for sample_id to remove all population data for every sample.
        :param sample_id: name of sample to remove populations from'; give a value of 'all' for sample_id to remove all population data for every sample.
        :param remove_gates: If True, all stored gating information will also be removed
        :return: None
        """
        for f in self.fcs_files:
            if sample_id == 'all' or f.primary_id == sample_id:
                f.populations = []
                if remove_gates:
                    f.gates = []
                f.save()

    def delete_gating_templates(self, template_name: str) -> None:
        """
        Remove association and delete gating template. If template_name is 'all', then all associated gating templates will be deleted and removed
        :param template_name: name of template to remove; if 'all', then all associated gating templates will be deleted and removed
        :return: None
        """
        for g in self.gating_templates:
            if template_name == 'all' or g.template_name == template_name:
                g.delete()
        if template_name == 'all':
            self.gating_templates = []
        else:
            self.gating_templates = [g for g in self.gating_templates if g.template_name != template_name]
        self.save()

    def sample_exists(self, sample_id: str) -> bool:
        """
        Returns True if the given sample_id exists in FCSExperiment
        :param sample_id: name of sample to search for
        :return: True if exists, else False
        """
        if sample_id not in self.list_samples():
            print(f'Error: invalid sample_id, {sample_id} not associated to this experiment')
            return False
        return True
    
    def pull_sample(self, sample_id: str) -> FileGroup or None:
        """
        Given a sample ID, return the corresponding FileGroup object
        :param sample_id: sample ID for search
        :return: FileGroup object; if sample does not belong to experiment, returns Null
        """
        if not self.sample_exists(sample_id):
            return None
        file_grp = [f for f in self.fcs_files if f.primary_id == sample_id][0]
        return FileGroup.objects(id=file_grp.id).get()
        
    def list_samples(self, valid_only=True) -> list:
        """
        Generate a list IDs of file groups associated to experiment
        :return: List of IDs of file groups associated to experiment
        """
        if valid_only:
            return [f.primary_id for f in self.fcs_files if f.validity()]
        return [f.primary_id for f in self.fcs_files]

    def list_invalid(self) -> list:
        """
        Generate list of sample IDs for samples that have the 'invalid' flag in their flag attribute
        :return: List of sample IDs for invalid samples
        """
        return [f.primary_id for f in self.fcs_files if not f.validity()]

    def fetch_sample_mid(self, sample_id: str) -> str or None:
        """
        Given a sample ID (for a sample belonging to this experiment) return it's mongo ObjectID as a string
        :param sample_id: sample ID for sample of interest
        :return: string value for ObjectID
        """
        if not self.sample_exists(sample_id):
            return None
        file_grp = [f for f in self.fcs_files if f.primary_id == sample_id][0]
        return file_grp.id.__str__()
    
    def pull_sample_mappings(self, sample_id):
        """
        Given a sample ID, return a dictionary of channel/marker mappings for all associated fcs files
        :param sample_id: sample ID for search 
        :return: dictionary of channel/marker mappings for each associated fcs file
        """
        file_grp = self.pull_sample(sample_id)
        if not file_grp:
            return None
        mappings = dict()
        for f in file_grp.files:
            mappings[f.file_id] = [m.to_python() for m in f.channel_mappings]
        return mappings

    def pull_sample_data(self, sample_id: str, sample_size: int or None = None, include_controls: bool = True,
                         output_format: str = 'dataframe', columns_default: str = 'marker') -> None or list:
        """
        Given a sample ID, associated to this experiment, fetch the fcs data
        :param sample_id: ID of sample to fetch data for
        :param sample_size: if provided with an integer value, a sample of data of given size will be returned
        (sample drawn from a uniform distribution)
        :param include_controls: if True (default) then control files associated to sample are included in the result
        :param output_format: preferred format of output; can either be 'dataframe' for a pandas dataframe, or 'matrix'
        for a numpy array
        :param columns_default: naming convention for returned dataframes; must be either 'marker' or 'channel'
        (default = marker)
        :return: list of dictionaries {id: file id, typ: data type, either raw or normalised,
        data: dataframe/matrix}
        """
        file_grp = self.pull_sample(sample_id)
        if not file_grp:
            return None
        files = file_grp.files
        # Fetch data
        if not include_controls:  # Fetch data for primary file only
            f = [f for f in files if f.file_type == 'complete'][0]
            complete = data_from_file(file=f, sample_size=sample_size, output_format=output_format,
                                      columns_default=columns_default)
            return [complete]
        # Fetch data for primary file & controls
        pool = Pool(cpu_count())
        f = partial(data_from_file, sample_size=sample_size, output_format=output_format,
                    columns_default=columns_default)
        data = pool.map(f, files)
        pool.close()
        pool.join()
        return data

    def remove_sample(self, sample_id: str) -> bool:
        """
        Remove sample (FileGroup) from experiment.
        :param sample_id: ID of sample to remove
        :return: True if successful
        """
        assert sample_id in self.list_samples(), f'{sample_id} not associated to this experiment'
        fg = self.pull_sample(sample_id)
        self.fcs_files = [f for f in self.fcs_files if f.primary_id != sample_id]
        fg.delete()
        self.save()
        return True

    def _create_file_entry(self, path: str, file_id: str, comp_matrix: str or None,
                           compensate: bool, catch_standardisation_errors: bool,
                           control: bool = False) -> File or None:
        """
        Internal method. Create a new File object.
        :param path: file path of the primary fcs file (e.g. the fcs file that is of primary interest such as the
        file with complete staining)
        :param file_id: identifier for file
        :param comp_matrix: compensation matrix (if Null, linked matrix expected)
        :param compensate: if True, compensation will be applied else False
        :param catch_standardisation_errors: If True, standardisation errors will result in no File generation
        and function will return Null
        :param control: if True, File will be created as file type 'control'
        :return: File Object
        """
        try:
            fcs = FCSFile(path, comp_matrix=comp_matrix)
        except ValueError as e:
            print(f'Unable to load data from {path}; encountered the following exception: {e}')
            return None
        new_file = File()
        new_file.file_id = file_id
        if compensate:
            fcs.compensate()
            new_file.compensated = True
        if control:
            new_file.file_type = 'control'
        data = fcs.dataframe
        column_mappings = self.panel.standardise(data, catch_standardisation_errors)
        if column_mappings is None:
            print(f'Error: invalid channel/marker mappings for {file_id}, at path {path}, aborting.')
            return None
        new_file.put(data.values)
        new_file.channel_mappings = [ChannelMap(channel=c, marker=m) for c, m in column_mappings]
        return new_file

    def add_new_sample(self, sample_id: str, file_path: str, controls: list, subject_id: str or None = None,
                       comp_matrix: str or None = None, compensate: bool = True,
                       feedback: bool = True, catch_standardisation_errors: bool = False,
                       processing_datetime: str or None = None,
                       collection_datetime: str or None = None) -> None or str:
        """
        Add a new sample (FileGroup) to this experiment
        :param sample_id: primary ID for identification of sample (FileGroup.primary_id)
        :param subject_id: ID for patient to associate sample too
        :param file_path: file path of the primary fcs file (e.g. the fcs file that is of primary interest such as the
        file with complete staining)
        :param controls: list of file paths for control files e.g. a list of file paths for associated FMO controls
        :param comp_matrix: (optional) numpy array for spillover matrix for compensation calculation; if not supplied
        the matrix linked within the fcs file will be used, if not present will present an error
        :param compensate: boolean value as to whether compensation should be applied before data entry (default=True)
        :param feedback: boolean value, if True function will provide feedback in the form of print statements
        (default=True)
        :param catch_standardisation_errors: if True, standardisation errors will cause process to abort
        :param processing_datetime:
        :param collection_datetime:
        :return: MongoDB ObjectID string for new FileGroup entry
        """
        if self.panel is None:
            print('Error: no panel design assigned to this experiment')
            return None
        if sample_id in self.list_samples():
            print(f'Error: a file group with id {sample_id} already exists')
            return None
        if feedback:
            print('Generating main file entry...')
        file_collection = FileGroup()
        if processing_datetime is not None:
            file_collection.processing_datetime = processing_datetime
        if collection_datetime is not None:
            file_collection.collection_datetime = collection_datetime
        file_collection.primary_id = sample_id
        primary_file = self._create_file_entry(file_path, sample_id, comp_matrix=comp_matrix, compensate=compensate,
                                               catch_standardisation_errors=catch_standardisation_errors)
        if not primary_file:
            return None
        file_collection.files.append(primary_file)
        if feedback:
            print('Generating file entries for controls...')
        for c in controls:
            control = self._create_file_entry(c['path'], f"{sample_id}_{c['control_id']}",
                                              comp_matrix=comp_matrix, compensate=compensate,
                                              catch_standardisation_errors=catch_standardisation_errors,
                                              control=True)
            if not control:
                return None
            file_collection.files.append(control)
        file_collection.save()
        self.fcs_files.append(file_collection)
        if subject_id is not None:
            p = Subject.objects(subject_id=subject_id)
            if len(p) == 0:
                print(f'Error: no such patient {subject_id}, continuing without association.')
            else:
                p = p[0]
                p.files.append(file_collection)
                p.save()
        if feedback:
            print(f'Successfully created {sample_id} and associated to {self.experiment_id}')
        self.save()
        return file_collection.id.__str__()
