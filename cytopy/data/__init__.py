"""
CytoPy is built around a data model housed in a MongoDB database. See 'data model' in documentation for details.
In this module you will find the constituents of this model, each modelled as a MongoDB document or embedded
document:
    - Project: the main access point for analysis that gives reference to Experiments and Subjects
    - Experiment: defines a single staining panel and houses FileGroups
    - FileGroup: defines one or more related cytometry files (primary staining and controls)
    - Subject: houses subject meta-data and references one or more FileGroup's

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
from .experiment import Experiment
from .experiment import single_cell_anndata
from .experiment import single_cell_dataframe
from .fcs import FileGroup
from .panel import Panel
from .population import Population
from .project import Project
from .setup import global_init
from .subject import Subject

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "3.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"
