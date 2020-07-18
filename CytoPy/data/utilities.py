from mongoengine import connection
from .fcs import File, FileGroup
import pandas as pd
import numpy as np
import os


def get_fcs_file_paths(fcs_dir: str, control_names: list, ctrl_id: str, ignore_comp: bool = True) -> dict:
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

    Returns
    --------
    dict
        standard dictionary of fcs files contained in target directory
    """
    file_tree = dict(primary=[], controls=[])
    fcs_files = filter_fcs_files(fcs_dir, exclude_comps=ignore_comp)
    ctrl_files = [f for f in fcs_files if f.find(ctrl_id) != -1]
    primary = [f for f in fcs_files if f.find(ctrl_id) == -1]
    for c_name in control_names:
        matched_controls = list(filter(lambda x: x.find(c_name) != -1, ctrl_files))
        if not matched_controls:
            print(f'Warning: no file found for {c_name} control')
            continue
        if len(matched_controls) > 1:
            print(f'Warning: multiple files found for {c_name} control')
            file_tree['controls'].append(dict(control_id=c_name, path=matched_controls))
            continue
        file_tree['controls'].append(dict(control_id=c_name, path=matched_controls[0]))
    if len(primary) > 1:
        print('Warning! Multiple non-control (primary) files found in directory. Check before proceeding.')
    file_tree['primary'] = primary
    return file_tree


def data_from_file(file_id: str,
                   filegrp_id: str,
                   db_name: str,
                   sample_size: int or None,
                   output_format: str = 'dataframe',
                   columns_default: str = 'marker') -> None or dict:
    """
    Pull data from a given file document (Used for multi-process pull)

    Parameters
    -----------
    file_id: str
        ID for file of interest
    filegrp_id: str
        MongoDB unique identifier for fcs file
    db_name: str
        Name of database
    sample_size: int, optional
        return a sample of given integer size
    output_format: str, (default='dataframe')
        preferred format of output; can either be 'dataframe' for a pandas dataframe, or 'matrix' for a numpy array
    columns_default: str, (default='marker')
        how to name columns if output_format='dataframe'; either 'marker' or 'channel' (default = 'marker')

    Returns
    --------
    dict
        Dictionary output {id: file_id, typ: file_type, data: dataframe/matrix}
    """
    db = connection.connect(db_name, alias='core')

    fg = FileGroup.objects(id=filegrp_id).get()
    file = [f for f in fg.files if f.file_id == file_id]
    assert file, f'Invalid file ID {file_id} for FileGroup {fg.primary_id}'
    assert len(file) == 1, f'Multiple files of ID {file_id} found in FileGroup {fg.primary_id}'
    data = file[0].pull(sample=sample_size)
    if output_format == 'dataframe':
        data = as_dataframe(data, column_mappings=file[0].channel_mappings, columns_default=columns_default)
    data = dict(id=file[0].file_id, typ=file[0].file_type, data=data)
    db.close()
    connection._connections = {}
    connection._connection_settings = {}
    connection._dbs = {}
    FileGroup._collection = None

    return data


def as_dataframe(matrix: np.array, column_mappings: list, columns_default: str = 'marker'):
    """
    Generate a pandas dataframe using a given numpy multi-dim array with specified column defaults
    (Used for multi-process pull)

    Parameters
    -----------
    matrix: Numpy.array
        numpy array to convert to dataframe
    column_mappings: list
        Channel/marker mappings for each columns in matrix
    columns_default: str
        how to name columns; either 'marker' or 'channel' (default = 'marker')

    Returns
    --------
    Pandas.DataFrame
    """
    columns = []
    if columns_default == 'channel':
        for i, m in enumerate(column_mappings):
            if m.channel:
                columns.append(m.channel)
            else:
                columns.append(f'Unnamed: {i}')
    else:
        for i, m in enumerate(column_mappings):
            if m.marker:
                columns.append(m.marker)
            elif m.channel:
                columns.append(m.channel)
            else:
                columns.append(f'Unnamed: {i}')
    return pd.DataFrame(matrix, columns=columns, dtype='float32')
