from immunova.data.fcs import File
import pandas as pd
import numpy as np
import os


def filter_fcs_files(fcs_dir: str, exclude_comps: bool = True) -> list:
    """
    Given a directory, return file paths for all fcs files in directory and subdirectories contained within
    :param fcs_dir:
    :param exclude_comps:
    :return: list of fcs file paths
    """
    fcs_files = []
    for root, dirs, files in os.walk(fcs_dir):
        if os.path.basename(root) == 'DUPLICATES':
            continue
        if exclude_comps:
            fcs = [f for f in files if f.endswith('.fcs') and f.lower().find('comp') == -1]
        else:
            fcs = [f for f in files if f.endswith('.fcs')]
        fcs = [f'{root}/{f}' for f in fcs]
        fcs_files = fcs_files + fcs
    return fcs_files


def get_fcs_file_paths(fcs_dir: str, control_names: list, ctrl_id: str, ignore_comp=True) -> dict:
    """
    Generate a standard dictionary object of fcs files in given directory
    :param fcs_dir: target directory for search
    :param control_names: names of expected control files (names must appear in filenames)
    :param ctrl_id: global identifier for control file e.g. 'FMO' (must appear in filenames)
    :param ignore_comp: If True, files with 'compensation' in their name will be ignored (default = True)
    :return: standard dictionary of fcs files contained in target directory
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


def data_from_file(file: File, sample_size: int, output_format: str = 'dataframe',
                   columns_default: str = 'marker') -> None or dict:
    """
    Pull data from a given file document
    :param file: File object
    :param data_type: data type to retrieve; either 'raw' or 'norm' (normalised)
    :param sample_size: return a sample of given integer size
    :param output_format: preferred format of output; can either be 'dataframe' for a pandas dataframe, or 'matrix'
    for a numpy array
    :param columns_default: how to name columns if output_format='dataframe';
    either 'marker' or 'channel' (default = 'marker')
    :return: Dictionary output {id: file_id, typ: file_type, data: dataframe/matrix}
    """
    data = file.pull(sample=sample_size)
    if output_format == 'dataframe':
        data = as_dataframe(data, column_mappings=file.channel_mappings, columns_default=columns_default)
    return dict(id=file.file_id, typ=file.file_type, data=data)


def as_dataframe(matrix: np.array, column_mappings, columns_default: str = 'marker'):
    """
    Generate a pandas dataframe using a given numpy multi-dim array with specified column defaults
    :param matrix: numpy matrix to convert to dataframe
    :param column_mappings: Channel/marker mappings for each columns in matrix
    :param columns_default: how to name columns; either 'marker' or 'channel' (default = 'marker')
    :return: Pandas dataframe
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
