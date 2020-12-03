#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
When creating training data for supervised classification it can be useful
to generate a new example FileGroup by sampling many or all the FileGroups
present in an Experiment. This can also be useful if we have suitable data
to be modelled as after concatenation of all available events (say the data
was all measured within the same batch). This module contains the
create_ref_sample function for merging multiple FileGroups to form
a new FileGroup saved to the experiment.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from CytoPy.data.experiment import Experiment
from CytoPy.data.fcs import FileGroup
from CytoPy.feedback import vprint
from CytoPy.flow.variance import load_and_sample, _common_features
import pandas as pd
import numpy as np
np.random.seed(42)

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def create_ref_sample(experiment: Experiment,
                      sample_size: int or float = 2500,
                      sampling_method: str = "uniform",
                      sampling_kwargs: dict or None = None,
                      root_population='root',
                      sample_ids: list or None = None,
                      new_file_name: str or None = None,
                      verbose: bool = True,
                      save_sample_id: bool = True,
                      include_ctrls: bool = True) -> None:
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
    save_sample_id: bool (default=True)
        If True, the sample ID that each cell originates from is saved to the
        FileGroup cell_meta_labels attribute
    include_ctrls: bool (default=True)
        If True, the control files are amalgamated and stored within the new file
    Returns
    --------
    None
    """
    vprint_ = vprint(verbose)
    sampling_kwargs = sampling_kwargs or {}
    sample_ids = sample_ids or experiment.list_samples()
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
    sample_id_idx = []
    if save_sample_id:
        for _id, x in data.items():
            sample_id_idx = sample_id_idx + [_id for _ in range(x.shape[0])]
    data = pd.concat([x[features] for x in data.values()]).reset_index(drop=True)
    vprint_('Creating new file entry...')
    new_filegroup = FileGroup(primary_id=new_file_name,
                              data_directory=experiment.data_directory,
                              data=data,
                              channels=features,
                              markers=features)
    new_filegroup.notes = 'sampled data'
    if save_sample_id:
        new_filegroup.cell_meta_labels["original_filegroup"] = np.array(sample_id_idx, dtype="U")
    if include_ctrls:
        ctrls = [experiment.get_sample(x).controls for x in sample_ids]
        ctrls = list(set([c for sl in ctrls for c in sl]))
        new_filegroup = add_ctrl_data(experiment=experiment,
                                      sample_ids=list(sample_ids),
                                      sampling_method=sampling_method,
                                      sample_size=sample_size,
                                      sampling_kwargs=sampling_kwargs,
                                      ctrls=ctrls,
                                      verbose=verbose,
                                      population=root_population,
                                      new_file=new_filegroup)
    vprint_('Inserting sampled data to database...')
    new_filegroup.save()
    experiment.fcs_files.append(new_filegroup)
    experiment.save()
    vprint_(f'Complete! New file saved to database: {new_file_name}, {new_filegroup.id}')
    vprint_('-----------------------------------------------------------------')


def add_ctrl_data(experiment: Experiment,
                  sample_ids: list,
                  new_file: FileGroup,
                  ctrls: list,
                  verbose: bool = True,
                  population: str = "root",
                  sample_size: float or int = 2500,
                  sampling_method: str = "uniform",
                  sampling_kwargs: dict or None = None):
    """
    Add the amalgamation of control data for the given file IDs and add to the new file

    Parameters
    ----------
    experiment: Experiment
    sample_ids: list
    new_file: FileGroup
    ctrls: list
    verbose: bool (default=True)
    population: str (default="root")
    sample_size: float or int (default=2500)
    sampling_method: str (default="uniform")
    sampling_kwargs: dict (optional)

    Returns
    -------
    FileGroup
    """
    vprint_ = vprint(verbose)
    vprint_("Adding control data...")
    sampling_kwargs = sampling_kwargs or {}
    for c in ctrls:
        vprint_(f"...{c}...")
        ctrl_data = load_and_sample(experiment=experiment,
                                    population=population,
                                    sample_ids=sample_ids,
                                    ctrl=c,
                                    sampling_method=sampling_method,
                                    sample_size=sample_size,
                                    transform=None,
                                    **sampling_kwargs)
        features = _common_features(ctrl_data)
        ctrl_data = pd.concat([x[features] for x in ctrl_data.values()]).reset_index(drop=True)
        new_file.add_ctrl_file(ctrl_id=c,
                               data=ctrl_data.values,
                               channels=features,
                               markers=features)
    return new_file
