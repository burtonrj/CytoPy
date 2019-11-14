from immunova.data.fcs_experiments import FCSExperiment
from immunova.flow.gating.transforms import apply_transform
from immunova.flow.normalisation.MMDResNet import MMDNet
from immunova.flow.deepcytof.deep_gating import calculate_reference_sample
import pandas as pd


class CalibrationError(Exception):
    pass


class Normalise:
    """
    Class for normalising a flow cytometry file using a reference target file
    """
    def __init__(self, experiment: FCSExperiment, source_id: str,
                 features: list, reference_sample: str or None = None, transform: str = 'logicle',
                 **mmdresnet_kwargs):
        """
        Constructor for Normalise object
        :param experiment: FCSExperiment object
        :param source_id: sample ID for the file to normalise
        :param reference_sample: sample ID to use as target distribution (leave as 'None' if unknown and use the
        `calculate_reference_sample` method to find an optimal reference sample)
        :param transform: transformation to apply to raw FCS data (default = 'logicle')
        :param mmdresnet_kwargs: keyword arguments for MMD-ResNet
        """
        self.experiment = experiment
        self.source_fg = self.experiment.pull_sample(source_id)
        self.transform = transform
        self.model = MMDNet(data_dim=len(features), **mmdresnet_kwargs)
        self.reference_sample = reference_sample or None

        if source_id not in self.experiment.list_samples():
            raise CalibrationError(f'Error: invalid target sample {source_id}; '
                                   f'must be one of {self.experiment.list_samples()}')
        else:
            self.source = self.__load_and_transform(source_id)

    def load_model(self, model_path:str) -> None:
        """
        Load an existing MMD-ResNet model from .h5 file
        :param model_path: path to model .h5 file
        :return: None
        """
        self.model.load_model(path=model_path)

    def build_model(self) -> None:
        """
        Construct the MMD-ResNet model infrastructure.
        :return:
        """
        self.model.build_model()

    def calculate_reference_sample(self):
        """
        Calculate the optimal reference sample. This is performed as described in Li et al paper
        (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860171/) on DeepCyTOF: for every 2 samples i, j compute
        the Frobenius norm of the difference between their covariance matrics and then select the sample
         with the smallest average distance to all other samples. Optimal sample assigned to self.reference_sample.
        :return: None
        """
        self.reference_sample = calculate_reference_sample(self.experiment)
        print(f'{self.reference_sample} chosen as optimal reference sample.')

    def __load_and_transform(self, sample_id: str) -> list:
        """
        Given a sample ID, retrieve the sample data and apply transformation
        :param sample_id: ID corresponding to sample for retrieval
        :return: transformed data as a list of dictionary objects:
        {id: file id, typ: type of file (either 'complete' or 'control'), data: Pandas DataFrame}
        """
        data = self.experiment.pull_sample_data(sample_id=sample_id, data_type='raw')
        if data is None:
            raise CalibrationError(f'Error: unable to load data for {sample_id}')
        features = [c for c in data[0]['data'] if c.lower() != 'time']
        transformed = [dict(id=x['id'],
                            typ=x['typ'],
                            data=apply_transform(x['data'],
                                                 features_to_transform=features,
                                                 transform_method=self.transform)) for x in data]
        return transformed

    def __put_norm_data(self, file_id: str, data: pd.DataFrame):
        """
        Given a file ID and a Pandas DataFrame, fetch the corresponding File document and insert the normalised data.
        :param file_id: ID for file for insert
        :param data: Pandas DataFrame of normalised and transformed data
        :return:
        """
        file = [f for f in self.source_fg.files if f.file_id == file_id][0]
        file.put(data=data.values, typ='norm')

    def normalise_and_save(self) -> None:
        """
        Apply normalisation to source sample and save result to the database.
        :return:
        """
        if self.model.net is None:
            print('Error: normalisation model has not yet been calibrated')
            return None
        features = [x for x in self.source[0]['data'].columns if x.lower() != 'time']
        for x in self.source:
            print(f'Saving normalised data for {x["file_id"]}')
            data = self.model.net.predict(x['data'])
            data = pd.DataFrame(data, columns=features)
            if 'Time' in data.columns:
                data['Time'] = x['data']['Time']
            self.__put_norm_data(x['file_id'], data)
        print('Save complete!')

    def calibrate(self, initial_lr=1e-3, lr_decay=0.97, evaluate=False, save=False) -> None:
        """
        Train the MMD-ResNet to minimise the Maximum Mean Discrepancy between our target and source sample.
        :param initial_lr: initial learning rate (default = 1e-3)
        :param lr_decay: decay rate for learning rate (default = 0.97)
        :param evaluate: If True, the performance of the training is evaluated and a PCA plot of aligned distributions
        is generated (default = False).
        :param save: If True, normalisation is applied to source sample and saved to database.
        :return: None
        """
        if self.reference_sample is None:
            print('Error: must provide a reference sample for training. This can be provided during initialisation, '
                  'by assigning a valid value to self.reference_sample, or by calling `calculate_reference_sample`.')
            return
        if self.reference_sample not in self.experiment.list_samples():
            print(f'Error: invalid reference sample {self.reference_sample}; must be one of '
                  f'{self.experiment.list_samples()}')
            return
        # Load and transform data
        target = self.__load_and_transform(self.reference_sample)
        target = [x for x in target if x['typ'] == 'complete'][0]['data']
        features = [x for x in target.columns if x.lower() != 'time']
        print('Warning: calibration can take some time and is dependent on the sample size')
        s = [x for x in self.source if x['typ'] == 'complete'][0]['data']
        self.model.fit(s[features], target[features], initial_lr, lr_decay)
        print('Calibration complete!')
        if evaluate:
            print('Evaluating calibration...')
            self.model.evaluate(s[features], target[features])
        if save:
            self.normalise_and_save()


