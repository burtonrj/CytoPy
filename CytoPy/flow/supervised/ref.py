from ...data.experiments import Experiment
from ...data.fcs import FileGroup
from ...feedback import progress_bar, vprint
from ..gating_tools import Gating
from warnings import warn
import pandas as pd


def _sample(df: pd.DataFrame,
            n: int):
    if df.shape[0] <= n:
        return df
    return df.sample(n=n)


def create_reference_sample(experiment: Experiment,
                            sample_n: int = 2500,
                            root_population='root',
                            samples: list or None = None,
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
    vprint_ = vprint(verbose)
    samples = samples or list(experiment.list_samples())
    assert all([s in experiment.list_samples() for s in samples]), \
        'One or more samples specified do not belong to experiment'

    vprint_('-------------------- Generating Reference Sample --------------------')
    vprint_('Sampling experiment data...')
    new_file_name = new_file_name or f'{experiment.experiment_id}_sampled_data'
    data = list()
    for _id in progress_bar(samples, verbose=verbose):
        g = Gating(experiment, sample_id=_id, include_controls=False)
        if root_population not in g.populations.keys():
            warn(f'Skipping {_id} as {root_population} is absent from gated populations')
            continue
        df = _sample(g.get_population_df(population_name=root_population, transform=None),
                     n=sample_n)
        data.append(df)
    all_columns = list(df.columns.tolist() for df in data)
    features = list()
    for c in set([x for sl in all_columns for x in sl]):
        if all([c in x for x in all_columns]):
            features.append(c)
    data = pd.concat([df[features] for df in data])
    data = data.reset_index(drop=True)
    vprint_('Sampling complete!')

    vprint_('Creating new file entry...')
    mappings = [dict(channel=f, marker=f) for f in features]
    new_filegroup = FileGroup(primary_id=new_file_name,
                              data_directory=experiment.data_directory)
    new_filegroup.notes = 'sampled data'
    new_filegroup.add_file(data=data[features],
                           channel_mappings=mappings)
    vprint_('Inserting sampled data to database...')
    new_filegroup.save()
    experiment.fcs_files.append(new_filegroup)
    experiment.save()
    vprint_(f'Complete! New file saved to database: {new_file_name}, {new_filegroup.id}')
    vprint_('-----------------------------------------------------------------')