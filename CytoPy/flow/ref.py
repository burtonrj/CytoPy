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
from CytoPy.flow.variance import load_and_sample
import numpy as np
np.random.seed(42)

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def create_ref_sample(experiment: Experiment,
                      new_file_name: str,
                      sample_size: int or float = 2500,
                      sampling_method: str = "uniform",
                      sampling_kwargs: dict or None = None,
                      root_population='root',
                      sample_ids: list or None = None) -> None:
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
    Returns
    --------
    None

    Raises
    ------
    AssertionError
        One or more samples specified do not belong to experiment
    """
    sampling_kwargs = sampling_kwargs or {}
    sample_ids = sample_ids or experiment.list_samples()
    new_file_name = new_file_name or f'{experiment.experiment_id}_sampled_data'
    assert all([s in experiment.list_samples() for s in sample_ids]), \
        'One or more samples specified do not belong to experiment'
    data = load_and_sample(experiment=experiment,
                           population=root_population,
                           sample_size=sample_size,
                           sample_ids=sample_ids,
                           sampling_method=sampling_method,
                           transform=None,
                           **sampling_kwargs)[0]
    features = [x for x in data.columns if x != "sample_id"]
    new_filegroup = FileGroup(primary_id=new_file_name)
    new_filegroup.data_directory = experiment.get_data_directory()
    new_filegroup.init_new_file(data=data[features].values,
                                channels=features,
                                markers=features)
    new_filegroup.notes = 'sampled data'
    new_filegroup.save()
    experiment.fcs_files.append(new_filegroup)
    experiment.save()

