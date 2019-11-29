from immunova.data.fcs_experiments import FCSExperiment
from immunova.data.fcs import Normalisation
from immunova.flow.gating.actions import Gating
from immunova.flow.normalisation.MMDResNet import MMDNet
from immunova.flow.supervised.utilities import calculate_reference_sample
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
np.random.seed(42)


class CalibrationError(Exception):
    pass


class Normalise:
    """
    Class for normalising a flow cytometry file using a reference target file
    """
    def __init__(self, experiment: FCSExperiment, source_id: str, root_population: str,
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
        self.source_id = source_id
        self.root_population = root_population
        self.transform = transform
        self.features = [c for c in features if c.lower() != 'time']
        self.calibrator = MMDNet(data_dim=len(self.features), **mmdresnet_kwargs)
        self.reference_sample = reference_sample or None

        if source_id not in self.experiment.list_samples():
            raise CalibrationError(f'Error: invalid target sample {source_id}; '
                                   f'must be one of {self.experiment.list_samples()}')
        else:
            self.source = self.__load_and_transform(sample_id=source_id)

    def load_model(self, model_path: str) -> None:
        """
        Load an existing MMD-ResNet model from .h5 file
        :param model_path: path to model .h5 file
        :return: None
        """
        self.calibrator.load_model(path=model_path)

    def calculate_reference_sample(self) -> None:
        """
        Calculate the optimal reference sample. This is performed as described in Li et al paper
        (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5860171/) on DeepCyTOF: for every 2 samples i, j compute
        the Frobenius norm of the difference between their covariance matrics and then select the sample
         with the smallest average distance to all other samples. Optimal sample assigned to self.reference_sample.
        :return: None
        """
        self.reference_sample = calculate_reference_sample(self.experiment)
        print(f'{self.reference_sample} chosen as optimal reference sample.')

    def __load_and_transform(self, sample_id) -> pd.DataFrame:
        """
        Given a sample ID, retrieve the sample data and apply transformation
        :param sample_id: ID corresponding to sample for retrieval
        :return: transformed data as a list of dictionary objects:
        {id: file id, typ: type of file (either 'complete' or 'control'), data: Pandas DataFrame}
        """
        gating = Gating(experiment=self.experiment, sample_id=sample_id)
        data = gating.get_population_df(self.root_population,
                                        transform=True,
                                        transform_method=self.transform,
                                        transform_features=self.features)
        if data is None:
            raise CalibrationError(f'Error: unable to load data for population {self.root_population}')
        return data[self.features]

    def __put_norm_data(self, file_id: str, data: pd.DataFrame):
        """
        Given a file ID and a Pandas DataFrame, fetch the corresponding File document and insert the normalised data.
        :param file_id: ID for file for insert
        :param data: Pandas DataFrame of normalised and transformed data
        :return:
        """
        source_fg = self.experiment.pull_sample(self.source_id)
        file = [f for f in source_fg.files if f.file_id == file_id][0]
        norm = Normalisation()
        norm.put(data.values, root_population=self.root_population, method='MMD-ResNet')
        file.norm = norm
        source_fg.save()

    def normalise_and_save(self) -> None:
        """
        Apply normalisation to source sample and save result to the database.
        :return:
        """
        if self.calibrator.model is None:
            print('Error: normalisation model has not yet been calibrated')
            return None
        print(f'Saving normalised data for {self.source_id} population {self.root_population}')
        data = self.calibrator.model.predict(self.source)
        data = pd.DataFrame(data, columns=self.source.columns)
        self.__put_norm_data(self.source_id, data)
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
        print('Warning: calibration can take some time and is dependent on the sample size')
        self.calibrator.fit(self.source, target, initial_lr, lr_decay, evaluate=evaluate)
        print('Calibration complete!')
        if save:
            self.normalise_and_save()


class EvaluateBatchEffects:
    def __init__(self, experiment: FCSExperiment, transform='logicle'):
        self.experiment = experiment
        self.transform = transform

    def __load_and_transform(self, sample_id, root_population) -> pd.DataFrame:
        """
        Given a sample ID, retrieve the sample data and apply transformation
        :param sample_id: ID corresponding to sample for retrieval
        :return: transformed data as a list of dictionary objects:
        {id: file id, typ: type of file (either 'complete' or 'control'), data: Pandas DataFrame}
        """
        gating = Gating(experiment=self.experiment, sample_id=sample_id, include_controls=False)
        data = gating.get_population_df(root_population,
                                        transform=True,
                                        transform_method=self.transform,
                                        transform_features='all')
        if data is None:
            raise CalibrationError(f'Error: unable to load data for population {root_population}')
        return data

    def marker_variance(self, reference_id, root_population, marker):
        reference = self.__load_and_transform(reference_id, root_population).sample(500)[marker]
        for sample in self.experiment.list_samples():
            try:
                d = self.__load_and_transform(sample, root_population).sample(500)[marker]
                ax = sns.kdeplot(d, color='b', shade=False, alpha=0.5)
            except ValueError:
                print(f'{sample} has less than 500 events, this sample will not be included in the comparison')
            except CalibrationError:
                print(f'{sample} does not contain the population {root_population}, skipping')
        ax = sns.kdeplot(reference, shade=True, color="r")
        ax.set_title(f'Total variance in {marker}')
        ax.get_legend().remove()
        return ax

    def pca(self, sample_id, reference_id, root_population, sample_n=1000):
        reference = self.__load_and_transform(reference_id, root_population).sample(sample_n)
        sample = self.__load_and_transform(sample_id, root_population).sample(sample_n)
        ref_features = set(reference.columns)
        features = ref_features.intersection(list(sample.columns))
        # project data onto PCs
        reducer = PCA()
        reducer.fit(reference[features])

        ref_pca = pd.DataFrame(reducer.transform(reference[features])[:, 0:2], columns=['PCA1', 'PCA2'])
        ref_pca['Label'] = 'Target'
        sample_pca = pd.DataFrame(reducer.transform(sample[features])[:, 0:2], columns=['PCA1', 'PCA2'])
        sample_pca['Label'] = 'Source before calibration'
        pca_df = pd.concat([ref_pca, sample_pca])

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(ref_pca['PCA1'], ref_pca['PCA2'], c='blue', s=4, alpha=0.4, label=f'Target: {reference_id}')
        ax.scatter(sample_pca['PCA1'], sample_pca['PCA2'], c='red', s=4, alpha=0.4, label=f'Sample: {sample_id}')
        ax.set_xlabel('PCA1')
        ax.set_ylabel('PCA2')
        ax.legend()
        plt.show()
