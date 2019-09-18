import os


def get_fcs_file_paths(fcs_dir, control_names, ctrl_id, ignore_comp=True):
    """
    Generate a standard dictionary object of fcs files in given directory
    :param fcs_dir: target directory for search
    :param control_names: names of expected control files (names must appear in filenames)
    :param ctrl_id: global identifier for control file e.g. 'FMO' (must appear in filenames)
    :param ignore_comp:
    :return: standard dictionary of fcs files contained in target directory
    """
    file_tree = dict(primary=[], controls=[])
    files = os.listdir(fcs_dir)
    files = [f for f in files if f.endswith('.fcs')]
    if ignore_comp:
        files = [f for f in files if f.find('Comp') == -1]
    primary = [f for f in files if f.find(ctrl_id) == -1]
    ctrl_files = [f for f in files if f.find(ctrl_id) != -1]
    for file in ctrl_files:
        for c in control_names:
            file_tree['controls'].append(dict(control_id=c, path=f'{fcs_dir}/{file}'))
    if len(primary) > 1:
        print('Warning! Multiple non-control (primary) files found in directory. Check before proceeding.')
    file_tree['primary'] = [f'{fcs_dir}/{p}' for p in primary]
    return file_tree
