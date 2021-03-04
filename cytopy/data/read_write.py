#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
The read_write module contains tools for accessing *.fcs files and
relies on the Python library FlowIO by Scott White. This is used by
Experiment to population FileGroups.

Projects also house the subjects (represented by the Subject class;
see cytopy.data.subject) of an analysis which can contain multiple
meta-data.

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

from multiprocessing import Pool, cpu_count
import flowio
import dateutil.parser as date_parser
import numpy as np
import pandas as pd
import json
import os

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Scott White", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def filter_fcs_files(fcs_dir: str,
                     exclude_comps: bool = True,
                     exclude_dir: str = 'DUPLICATES') -> list:
    """
    Given a directory, return file paths for all fcs files in directory and subdirectories contained within

    Parameters
    ----------
    fcs_dir: str
        path to directory for search
    exclude_comps: bool
        if True, compensation files will be ignored (note: function searches for 'comp' in file name
        for exclusion)
    exclude_dir: str (default = 'DUPLICATES')
        Will ignore any directories with this name
    Returns
    --------
    List
        list of fcs file paths
    """
    fcs_files = []
    for root, dirs, files in os.walk(fcs_dir):
        if os.path.basename(root) == exclude_dir:
            continue
        if exclude_comps:
            fcs = [f for f in files if f.lower().endswith('.fcs') and f.lower().find('comp') == -1]
        else:
            fcs = [f for f in files if f.lower().endswith('.fcs')]
        fcs = [os.path.join(root, f) for f in fcs]
        fcs_files = fcs_files + fcs
    return fcs_files


def get_fcs_file_paths(fcs_dir: str,
                       control_names: list,
                       ctrl_id: str,
                       ignore_comp: bool = True,
                       exclude_dir: str = "DUPLICATE") -> dict:
    """
    Generate a standard dictionary object of fcs files in given directory
    Parameters
    -----------
    fcs_dir: str
        target directory for search
    control_names: list
        names of expected control files (names must appear in filenames)
    ctrl_id: str
        global identifier for control file e.g. 'FMO' (must appear in filenames)
    ignore_comp: bool, (default=True)
        If True, files with 'compensation' in their name will be ignored (default = True)
    exclude_dir: str (default = 'DUPLICATES')
        Will ignore any directories with this name
    Returns
    --------
    dict
        standard dictionary of fcs files contained in target directory
    """
    file_tree = dict(primary=[], controls={})
    fcs_files = filter_fcs_files(fcs_dir, exclude_comps=ignore_comp, exclude_dir=exclude_dir)
    ctrl_files = [f for f in fcs_files if f.find(ctrl_id) != -1]
    primary = [f for f in fcs_files if f.find(ctrl_id) == -1]
    for c_name in control_names:
        matched_controls = list(filter(lambda x: x.find(c_name) != -1, ctrl_files))
        if not matched_controls:
            print(f'Warning: no file found for {c_name} control')
            continue
        if len(matched_controls) > 1:
            print(f'Warning: multiple files found for {c_name} control')
        file_tree['controls'][c_name] = matched_controls

    if len(primary) > 1:
        print('Warning! Multiple non-control (primary) files found in directory. Check before proceeding.')
    file_tree['primary'] = primary
    return file_tree


def chunks(df_list: list,
           n: int) -> pd.DataFrame:
    """
    Yield successive n-sized chunks from l.
    ref: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

    Parameters
    -----------
    df_list: list
        list of DataFrames to generated 'chunks' from
    n: int
        number of chunks to generate
    Returns
    --------
    generator
        Yields successive n-sized DataFrames
    """
    for i in range(0, len(df_list), n):
        yield df_list[i:i + n]


def fcs_mappings(path: str) -> list or None:
    """
    Fetch channel mappings from fcs file.

    Parameters
    ------------
    path: str
        path to fcs file

    Returns
    --------
    List or None
        List of channel mappings. Will return None if file fails to load.
    """
    try:
        fo = FCSFile(path)
    except ValueError as e:
        print(f'Failed to load file {path}; {e}')
        return None
    return fo.channel_mappings


def explore_channel_mappings(fcs_dir: str,
                             exclude_comps: bool = True) -> list:
    """
    Given a directory, explore all fcs files and find all permutations of channel/marker mappings

    Parameters
    ----------
    fcs_dir: str
        root directory to search
    exclude_comps: bool, (default=True)
        exclude compentation files (must have 'comp' in filename)

    Returns
    --------
    List
        list of all unique channel/marker mappings
    """
    fcs_files = filter_fcs_files(fcs_dir, exclude_comps)
    with Pool(cpu_count()) as pool:
        mappings = list(pool.map(fcs_mappings, fcs_files))
        mappings = list(pool.map(json.dumps, mappings))
    return [json.loads(x) for x in mappings]


def _get_spill_matrix(matrix_string: str) -> pd.DataFrame:
    """
    Generate pandas dataframe for the fluorochrome spillover matrix used for compensation calc

    Code is modified from: https://github.com/whitews/FlowUtils
    Pedersen NW, Chandran PA, Qian Y, et al. Automated Analysis of Flow Cytometry
    Data to Reduce Inter-Lab Variation in the Detection of Major Histocompatibility
    Complex Multimer-Binding T Cells. Front Immunol. 2017;8:858.
    Published 2017 Jul 26. doi:10.3389/fimmu.2017.00858

    Parameters
    -----------
    matrix_string: str
        string value extracted from the 'spill' parameter of the FCS file

    Returns
    --------
    Pandas.DataFrame
    """
    matrix_list = matrix_string.split(',')
    n = int(matrix_list[0])
    header = matrix_list[1:(n+1)]
    header = [i.strip().replace('\n', '') for i in header]
    values = [i.strip().replace('\n', '') for i in matrix_list[n+1:]]
    matrix = np.reshape(list(map(float, values)), (n, n))
    matrix_df = pd.DataFrame(matrix)
    matrix_df = matrix_df.rename(index={k: v for k, v in zip(matrix_df.columns.to_list(), header)},
                                 columns={k: v for k, v in zip(matrix_df.columns.to_list(), header)})
    return matrix_df


def _get_channel_mappings(fluoro_dict: dict) -> list:
    """
    Generates a list of dictionary objects that describe the fluorochrome mappings in this FCS file

    Parameters
    -----------
    fluoro_dict: dict
        dictionary object from the channels param of the fcs file

    Returns
    --------
    List
        List of dict obj with keys 'channel' and 'marker'. Use to map fluorochrome channels to
    corresponding marker
    """
    fm = [(int(k), x) for k, x in fluoro_dict.items()]
    fm = [x[1] for x in sorted(fm, key=lambda x: x[0])]
    mappings = []
    for fm_ in fm:
        channel = fm_['PnN'].replace('_', '-')
        if 'PnS' in fm_.keys():
            marker = fm_['PnS'].replace('_', '-')
        else:
            marker = ''
        mappings.append({'channel': channel, 'marker': marker})
    return mappings


class FCSFile:
    """
    Utilising FlowIO to generate an object for representing an FCS file

    Attributes
    ----------
    filepath: str
        location of fcs file to parse
    comp_matrix: str
        csv file containing compensation matrix (optional, not required if a
        spillover matrix is already linked to the file)
    """
    def __init__(self, filepath, comp_matrix=None):
        fcs = flowio.FlowData(filepath)
        self.filename = fcs.text.get('fil', 'Unknown_filename')
        self.sys = fcs.text.get('sys', 'Unknown_system')
        self.total_events = int(fcs.text.get('tot', 0))
        self.tube_name = fcs.text.get('tube name', 'Unknown')
        self.exp_name = fcs.text.get('experiment name', 'Unknown')
        self.cytometer = fcs.text.get('cyt', 'Unknown')
        self.creator = fcs.text.get('creator', 'Unknown')
        self.operator = fcs.text.get('export user name', 'Unknown')
        self.channel_mappings = _get_channel_mappings(fcs.channels)
        self.cst_pass = False
        self.data = fcs.events
        self.event_data = np.reshape(np.array(fcs.events, dtype=np.float32), (-1, fcs.channel_count))
        if 'threshold' in fcs.text.keys():
            self.threshold = [{'channel': c, 'threshold': v} for c, v in chunks(fcs.text["threshold"].split(','), 2)]
        else:
            self.threshold = 'Unknown'
        try:
            self.processing_date = date_parser.parse(fcs.text['date'] +
                                                     ' ' + fcs.text['etim']).isoformat()
        except KeyError:
            self.processing_date = 'Unknown'
        if comp_matrix is not None:
            self.spill = pd.read_csv(comp_matrix)
            self.spill_txt = None
        else:
            if 'spill' in fcs.text.keys():
                self.spill_txt = fcs.text['spill']

            elif 'spillover' in fcs.text.keys():
                self.spill_txt = fcs.text['spillover']
            else:
                self.spill_txt = None
            if self.spill_txt is not None:
                if(len(self.spill_txt)) < 1:
                    print("""Warning: no spillover matrix found, please provide
                    path to relevant csv file with 'comp_matrix' argument if compensation is necessary""")
                    self.spill = None
                else:
                    self.spill = _get_spill_matrix(self.spill_txt)
            else:
                self.spill = None
        if 'cst_setup_status' in fcs.text:
            if fcs.text['cst setup status'] == 'SUCCESS':
                self.cst_pass = True

    def compensate(self):
        """
        Apply compensation to event data

        Returns
        -------
        None
        """
        assert self.spill is not None, f'Unable to locate spillover matrix, please provide a compensation matrix'
        channel_idx = [i for i, x in enumerate(self.channel_mappings) if x['marker'] != '']
        if len(channel_idx) == 0:
            # No markers defined in file
            channel_idx = [i for i, x in enumerate(self.channel_mappings) if all([z not in x['channel'].lower()
                                                                                  for z in ['fsc', 'ssc', 'time']])]
        comp_data = self.event_data[:, channel_idx]
        comp_data = np.linalg.solve(self.spill.values.T, comp_data.T).T
        self.event_data[:, channel_idx] = comp_data
