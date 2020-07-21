from dask import bag as db
import flowio
import dateutil.parser as date_parser
import numpy as np
import pandas as pd
import json
import os


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
    list
        list of fcs file paths
    """
    fcs_files = []
    for root, dirs, files in os.walk(fcs_dir):
        if os.path.basename(root) == exclude_dir:
            continue
        if exclude_comps:
            fcs = [f for f in files if f.endswith('.fcs') and f.lower().find('comp') == -1]
        else:
            fcs = [f for f in files if f.endswith('.fcs')]
        fcs = [os.path.join(root, f) for f in fcs]
        fcs_files = fcs_files + fcs
    return fcs_files


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
    return fo.fluoro_mappings


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
    fcs_files = db.from_sequence(filter_fcs_files(fcs_dir, exclude_comps))
    mappings = set(fcs_files.map(fcs_mappings).map(json.dumps).compute())
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
    fm = [x for k, x in fluoro_dict.items()]
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

    Parameters
    -----------
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
        comp_data = self.event_data[:, channel_idx]
        comp_data = np.linalg.solve(self.spill.values.T, comp_data.T).T
        self.event_data[:, channel_idx] = comp_data

