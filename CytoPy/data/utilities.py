import os


def filter_fcs_files(fcs_dir: str, exclude_comps: bool = True) -> list:
    """
    Given a directory, return file paths for all fcs files in directory and subdirectories contained within
    Parameters
    ----------
    fcs_dir: str
        path to directory for search
    exclude_comps: bool
        if True, compensation files will be ignored (note: function searches for 'comp' in file name
        for exclusion)
    Returns
    --------
    list
        list of fcs file paths
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
