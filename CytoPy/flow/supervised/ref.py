from ...data.experiment import Experiment
from ...data.fcs import FileGroup
from ...feedback import vprint
from ..variance import load_and_sample, _common_features
import pandas as pd


def create_ref_sample(experiment: Experiment,
                      sample_size: int or float = 2500,
                      sampling_method: str = "uniform",
                      sampling_kwargs: dict or None = None,
                      root_population='root',
                      sample_ids: list or None = None,
                      new_file_name: str or None = None,
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
    sample_ids: list, optional
        list of sample IDs for samples to be included (default = all samples in experiment)
    new_file_name: str
        name of file group generated
    sampling_method: str, (default='uniform')
        method to use for sampling files (currently only supports 'uniform')
    sample_size: int or float, (default=1000)
        number or fraction of events to sample from each file
    sampling_kwargs: dict
        Additional keyword arguments passed to sampling method
    verbose: bool, (default=True)
        Whether to provide feedback
    Returns
    --------
    None
    """
    vprint_ = vprint(verbose)
    new_file_name = new_file_name or f'{experiment.experiment_id}_sampled_data'
    assert all([s in experiment.list_samples() for s in sample_ids]), \
        'One or more samples specified do not belong to experiment'

    vprint_('-------------------- Generating Reference Sample --------------------')
    vprint_('Sampling experiment data...')
    data = load_and_sample(experiment=experiment,
                           population=root_population,
                           sample_size=sample_size,
                           sample_ids=sample_ids,
                           sampling_method=sampling_method,
                           transform=None,
                           **sampling_kwargs)
    features = _common_features(data)
    data = pd.concat([x[features] for x in data.values()])
    data = data.reset_index(drop=True)
    vprint_('Creating new file entry...')
    new_filegroup = FileGroup(primary_id=new_file_name,
                              data_directory=experiment.data_directory,
                              data=data,
                              channels=features,
                              markers=features)
    new_filegroup.notes = 'sampled data'
    vprint_('Inserting sampled data to database...')
    new_filegroup.save()
    experiment.fcs_files.append(new_filegroup)
    experiment.save()
    vprint_(f'Complete! New file saved to database: {new_file_name}, {new_filegroup.id}')
    vprint_('-----------------------------------------------------------------')