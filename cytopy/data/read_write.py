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
import json
import logging
import os
import re
from collections import defaultdict
from multiprocessing import cpu_count
from multiprocessing import Pool
from typing import List
from typing import Optional

import flowio
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import s3fs

logger = logging.getLogger(__name__)


def filter_fcs_files(fcs_dir: str, exclude_files: Optional[str] = None, exclude_dir: Optional[str] = None) -> list:
    """
    Given a directory, return file paths for all fcs files in directory and subdirectories contained within

    Parameters
    ----------
    fcs_dir: str
        path to directory for search
    exclude_files: str, optional
        Regex pattern - matching files will be ignored
    exclude_dir: str, optional
        Regex pattern - will ignore any matching subdirectories
    Returns
    --------
    List
        list of fcs file paths
    """
    fcs_files = []
    for root, dirs, files in os.walk(fcs_dir):
        if exclude_dir:
            if re.match(exclude_dir, os.path.basename(root)):
                continue
        if exclude_files:
            fcs = [f for f in files if f.lower().endswith(".fcs") and not re.match(exclude_files, f)]
        else:
            fcs = [f for f in files if f.lower().endswith(".fcs")]
        fcs = [os.path.join(root, f) for f in fcs]
        fcs_files = fcs_files + fcs
    return fcs_files


def parse_directory_for_fcs_files(
    fcs_dir: str,
    control_names: list,
    ctrl_id: str,
    ignore_comp: bool = True,
    exclude_dir: str = "DUPLICATE",
) -> dict:
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
    filetree = defaultdict(defaultdict(list))

    file_tree = dict(primary=[], controls={})
    fcs_files = filter_fcs_files(fcs_dir, exclude_comps=ignore_comp, exclude_dir=exclude_dir)
    ctrl_files = [f for f in fcs_files if f.find(ctrl_id) != -1]
    primary = [f for f in fcs_files if f.find(ctrl_id) == -1]
    for c_name in control_names:
        matched_controls = list(filter(lambda x: x.find(c_name) != -1, ctrl_files))
        if not matched_controls:
            print(f"Warning: no file found for {c_name} control")
            continue
        if len(matched_controls) > 1:
            print(f"Warning: multiple files found for {c_name} control")
        file_tree["controls"][c_name] = matched_controls

    if len(primary) > 1:
        print("Warning! Multiple non-control (primary) files found in directory. Check before proceeding.")
    file_tree["primary"] = primary
    return file_tree


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
        fo = flowio.FlowData(filename_or_handle=path)
    except ValueError as e:
        logger.error(f"Failed to load file {path}; {e}")
        return None
    return fo.channels


def explore_channel_mappings(fcs_dir: str, exclude_comps: bool = True) -> list:
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
        channel = fm_["PnN"].replace("_", "-")
        if "PnS" in fm_.keys():
            marker = fm_["PnS"].replace("_", "-")
        else:
            marker = ""
        mappings.append({"channel": channel, "marker": marker})
    return mappings


def match_file_ext(path: str, ext: str):
    return os.path.splitext(path)[1].lower() == ext


def load_compensation_matrix(fcs: flowio.FlowData) -> pl.DataFrame:
    """
    Extract a compensation matrix from an FCS file using FlowIO.

    Parameters
    ----------
    fcs: flowio.FlowData

    Returns
    -------
    polars.DataFrame or None
        Returns None if no compensation matrix is found; will log warning.
    """
    spill_txt = None
    if "spill" in fcs.text.keys():
        spill_txt = fcs.text["spill"]
    elif "spillover" in fcs.text.keys():
        spill_txt = fcs.text["spillover"]
    if spill_txt is None or len(spill_txt) < 1:
        logger.warning("No compensation matrix found")
        return None
    matrix_list = spill_txt.split(",")
    n = int(matrix_list[0])
    header = matrix_list[1 : (n + 1)]
    header = [i.strip().replace("\n", "") for i in header]
    values = [i.strip().replace("\n", "") for i in matrix_list[n + 1 :]]
    matrix = np.reshape(list(map(float, values)), (n, n))
    matrix_df = pl.DataFrame(matrix, columns=header)
    return matrix_df


def fcs_to_polars(fcs: flowio.FlowData) -> pl.DataFrame:
    """
    Return the events of a FlowData objects as a polars.DataFrame

    Parameters
    ----------
    fcs: flowio.FlowData

    Returns
    -------
    polars.DataFrame

    Raises
    ------
    ValueError
        Incorrect number of columns provided
    """
    channels = {int(k): v["PnN"] for k, v in fcs.channels.items()}
    columns = [x[1] for x in sorted(channels.items())]
    data = pl.DataFrame(np.reshape(np.array(fcs.events, dtype=np.float32), (-1, fcs.channel_count)), columns=columns)
    data = data[pl.col("*").cast(pl.Float64)]
    data["Index"] = np.arange(0, data.shape[0], dtype=np.int32)
    return data


def read_headers(path: str, s3_bucket: Optional[str] = None) -> List[str]:
    if s3_bucket is not None:
        if match_file_ext(path, ".csv"):
            data = read_from_remote(s3_bucket=s3_bucket, path=path, stop_after_n_rows=3)
        else:
            data = read_from_remote(s3_bucket=s3_bucket, path=path)
    else:
        if match_file_ext(path, ".csv"):
            data = read_from_disk(path=path, stop_after_n_rows=3)
        elif match_file_ext(path, ".fcs"):
            fcs = flowio.FlowData(filename_or_handle=path)
            return [x["PnN"] for _, x in fcs.channels.items()] + ["Index"]
        else:
            data = read_from_disk(path=path)
    return data.columns


def read_from_disk(path: str, **kwargs) -> pl.DataFrame:
    """
    Read cytometry data from disk. Must be either fcs, csv, or parquet file

    Parameters
    ----------
    path: str

    Returns
    -------
    polars.DataFrame

    Raises
    ------
    ValueError
        Invalid file extension
    """
    if match_file_ext(path=path, ext=".fcs"):
        return fcs_to_polars(flowio.FlowData(filename_or_handle=path))
    elif match_file_ext(path, ext=".csv"):
        data = pl.read_csv(path, **kwargs)[pl.col("*").cast(pl.Float64)]
    elif match_file_ext(path, ext=".parquet"):
        data = pl.read_parquet(source=path, **kwargs)[pl.col("*").cast(pl.Float64)]
    else:
        raise ValueError("Currently only support fcs, csv, or parquet file extensions")
    data["Index"] = np.arange(0, data.shape[0], dtype=np.int32)
    return data


def read_from_remote(s3_bucket: str, path: str, **kwargs) -> pl.DataFrame:
    """
    Read cytometry data from S3. Target file must be csv or parquet file type.

    Parameters
    ----------
    s3_bucket: str
    path: str

    Returns
    -------
    polars.DataFrame

    Raises
    ------
    ValueError
        Invalid file extension
    """
    fs = s3fs.S3FileSystem()
    if match_file_ext(path=path, ext=".csv"):
        with fs.open(f"s3://{s3_bucket}/{path}") as f:
            data = pl.read_csv(file=f, **kwargs)[pl.col("*").cast(pl.Float64)]
    elif match_file_ext(path=path, ext=".parquet"):
        data = pq.ParquetDataset(f"s3://{s3_bucket}/{path}", filesystem=fs)
        data = pl.from_arrow(data.read(**kwargs))[pl.col("*").cast(pl.Float64)]
    else:
        raise ValueError("Currently only support csv or parquet file extensions")
    data["Index"] = np.arange(0, data.shape[0], dtype=np.int32)
    return data


def pandas_to_polars(data: pd.DataFrame) -> pl.DataFrame:
    data = data.reset_index().rename({"index": "Index"}, axis=1)
    return pl.DataFrame(data)


def polars_to_pandas(data: pl.DataFrame) -> pd.DataFrame:
    assert "Index" in data.columns, "Missing 'Index' column"
    return data.to_pandas().set_index("Index")
