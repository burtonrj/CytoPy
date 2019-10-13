from immunova.data.fcs import FileGroup, File
from immunova.data.gating import GatingStrategy
from immunova.data.utilities import data_from_file
from immunova.data.panel import Panel, ChannelMap
from immunova.flow.readwrite.read_fcs import FCSFile
from multiprocessing import Pool, cpu_count
from functools import partial
import mongoengine
import numpy as np


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
    Methods:
        pull_sample_data - Given a sample ID, associated to this experiment, fetch the fcs data
        list_samples - Generate a list IDs of file groups associated to experiment
        remove_sample - Remove sample (FileGroup) from experiment.
        add_new_sample - Add a new sample (FileGroup) to this experiment

    """
    experiment_id = mongoengine.StringField(required=True, unique=True)
    panel = mongoengine.ReferenceField(Panel, reverse_delete_rule=4)
    fcs_files = mongoengine.ListField(mongoengine.ReferenceField(FileGroup, reverse_delete_rule=4))
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)
    gating_templates = mongoengine.ListField(mongoengine.ReferenceField(GatingStrategy, reverse_delete_rule=4))

    meta = {
        'db_alias': 'core',
        'collection': 'fcs_experiments'
    }
    
    def pull_sample(self, sample_id: str) -> FileGroup or None:
        """
        Given a sample ID, return the corresponding FileGroup object
        :param sample_id: sample ID for search
        :return: FileGroup object; if sample does not belong to experiment, returns Null
        """
        if sample_id not in self.list_samples():
            print(f'Error: invalid sample_id, {sample_id} not associated to this experiment')
            return None
        file_grp = [f for f in self.fcs_files if f.primary_id == sample_id][0]
        return file_grp
        
    def list_samples(self) -> list:
        """
        Generate a list IDs of file groups associated to experiment
        :return: List of IDs of file groups associated to experiment
        """
        return [f.primary_id for f in self.fcs_files]
    
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

    def pull_sample_data(self, sample_id: str, sample_size: int or None = None,
                         data_type: str = 'raw', include_controls: bool = True,
                         output_format: str = 'dataframe', columns_default: str = 'marker') -> None or list:
        """
        Given a sample ID, associated to this experiment, fetch the fcs data
        :param sample_id: ID of sample to fetch data for
        :param sample_size: if provided with an integer value, a sample of data of given size will be returned
        (sample drawn from a uniform distribution)
        :param data_type: type of data to retrieve; either 'raw' or 'norm' normalised
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
            complete = f.data_from_file(data_type=data_type, sample_size=sample_size, output_format=output_format,
                                        columns_default=columns_default)
            return [complete]
        # Fetch data for primary file & controls
        pool = Pool(cpu_count())
        f = partial(data_from_file, data_type=data_type, sample_size=sample_size, output_format=output_format,
                    columns_default=columns_default)
        data = pool.map(f, files)
        pool.close()
        pool.join()
        return data

    def remove_sample(self, sample_id: str, delete=False) -> bool:
        """
        Remove sample (FileGroup) from experiment.
        :param sample_id: ID of sample to remove
        :param delete: if True, the FileGroup entry will be deleted from the database
        :return: True if successful
        """
        fg = FileGroup.objects(primary_id=sample_id)
        if not fg:
            print(f'Error: {sample_id} does not exist')
            return False
        self.fcs_files = [f for f in self.fcs_files if f.primary_id != sample_id]
        if delete:
            fg[0].delete()
        return True

    def __create_file_entry(self, path: str, file_id: str, comp_matrix: np.array,
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
        fcs = FCSFile(path, comp_matrix=comp_matrix)
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

    def add_new_sample(self, sample_id: str, file_path: str, controls: list,
                       comp_matrix: np.array or None = None, compensate: bool = True,
                       feedback: bool = True, catch_standardisation_errors: bool = False) -> None or str:
        """
        Add a new sample (FileGroup) to this experiment
        :param sample_id: primary ID for identification of sample (FileGroup.primary_id)
        :param file_path: file path of the primary fcs file (e.g. the fcs file that is of primary interest such as the
        file with complete staining)
        :param controls: list of file paths for control files e.g. a list of file paths for associated FMO controls
        :param comp_matrix: (optional) numpy array for spillover matrix for compensation calculation; if not supplied
        the matrix linked within the fcs file will be used, if not present will present an error
        :param compensate: boolean value as to whether compensation should be applied before data entry (default=True)
        :param feedback: boolean value, if True function will provide feedback in the form of print statements
        (default=True)
        :param catch_standardisation_errors: if True, standardisation errors will cause process to abort
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
        file_collection.primary_id = sample_id
        primary_file = self.__create_file_entry(file_path, sample_id, comp_matrix=comp_matrix, compensate=compensate,
                                                catch_standardisation_errors=catch_standardisation_errors)
        if not primary_file:
            return None
        file_collection.files.append(primary_file)
        if feedback:
            print('Generating file entries for controls...')
        for c in controls:
            control = self.__create_file_entry(c['path'], f"{sample_id}_{c['control_id']}",
                                               comp_matrix=comp_matrix, compensate=compensate,
                                               catch_standardisation_errors=catch_standardisation_errors,
                                               control=True)
            if not control:
                return None
            file_collection.files.append(control)
        file_collection.save()
        self.fcs_files.append(file_collection)
        if feedback:
            print(f'Successfully created {sample_id} and associated to {self.experiment_id}')
        self.save()
        return file_collection.id.__str__()
