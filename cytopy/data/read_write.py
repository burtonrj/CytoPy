#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
The read_write module contains tools for accessing *.fcs files and
relies on the Python library FlowIO by Scott White. This is used by
Experiment to populate FileGroups. Data can also be read from
parquet and CSV files with the Polars library is used for reading. This
also provides optimised data transformations handled in the utils.transform
module.

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
import pickle
import re
from multiprocessing import cpu_count
from multiprocessing import Pool
from typing import Union, Dict
from typing import List
from typing import Optional

import flowio
import mongoengine
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import s3fs
from bson import Binary

logger = logging.getLogger(__name__)


class BaseIndexDocument(mongoengine.EmbeddedDocument):
    """
    Base class for embedded documents that store an event index, such as Populations or
    Gate children. Indexes are stored in the MongoDB database in a FileField as a
    pickled array.
    """
    _index = mongoengine.FileField(db_alias="core", collection_name="event_index")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._index_cache = None

    def _load_index(self):
        try:
            idx = pickle.loads(self._index.read())
            self._index.seek(0)
            return idx
        except TypeError:
            logger.error(f"Index is empty for child/population!")
            return None

    def write_index(self):
        if self._index:
            self._index.replace(Binary(pickle.dumps(list(self.index), protocol=2)))
        else:
            self._index.new_file()
            self._index.write(Binary(pickle.dumps(list(self.index), protocol=2)))
            self._index.close()

    @property
    def index(self) -> List[int]:
        if self._index_cache is None:
            return self._load_index()
        return self._index_cache

    @index.setter
    def index(self, idx: Union[List[int], np.ndarray]):
        if isinstance(idx, np.ndarray):
            idx = idx.tolist()
        self._index_cache = idx

    meta = {"allow_inheritance": True}


def filter_fcs_files(fcs_dir: str, exclude_files: Optional[str] = None, exclude_dir: Optional[str] = None) -> List[str]:
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
    List[str]
        list of fcs file paths
    """
    fcs_files = []
    for root, dirs, files in os.walk(fcs_dir):
        if exclude_dir:
            if re.match(exclude_dir, os.path.basename(root)):
                continue
        if exclude_files:
            fcs = [f for f in files if f.lower().endswith(".fcs") and not re.search(exclude_files, f)]
        else:
            fcs = [f for f in files if f.lower().endswith(".fcs")]
        fcs = [os.path.join(root, f) for f in fcs]
        fcs_files = fcs_files + fcs
    return fcs_files


def parse_directory_for_cytometry_files(
    fcs_dir: str,
    control_id: Optional[str] = None,
    control_names: Optional[List[str]] = None,
    exclude_files: Optional[str] = None,
    exclude_dir: Optional[str] = None,
    compensation_file: Optional[str] = None,
) -> Dict[str, Union[str, List[str]]]:
    """
    Generate a standard dictionary object of fcs files in given directory. The resulting dictionary will contain the
    following:
        {
            "primary": The file path to the primary staining file,
            control_name: The file path to a control staining file, where control_name is a value in 'control_names',
            ...
            "compensation_file": Either None, if 'compensation_file' is None, otherwise the filepath to the compensation
            file
        }

    Parameters
    -----------
    fcs_dir: str
        Target directory for search
    control_names: list
        Names of expected control files (names must appear in filenames)
    control_id: str
        Global identifier for control file e.g. 'FMO' (must appear in filenames)
    exclude_files: str, optional
        Any files containing this string in their filename will be ignored
    compensation_file: str, optional
        If a file is encountered with this exact filename, its filepath will be stored under the key 'compensation_file'
    exclude_dir: str (default = 'DUPLICATES')
        Will ignore any directories with this name

    Returns
    --------
    Dict[str, Union[str, List[str]]]
        standard dictionary of fcs files contained in target directory

    Raises
    ------
    ValueError
        Multiple files identified for either the primary staining, a control file, or the compensation file
    """
    file_tree = {}
    fcs_files = filter_fcs_files(fcs_dir, exclude_files=exclude_files, exclude_dir=exclude_dir)
    ctrl_files = [f for f in fcs_files if f.find(control_id) != -1]
    primary = [f for f in fcs_files if f.find(control_id) == -1]
    for c_name in control_names:
        matched_controls = list(filter(lambda x: x.find(c_name) != -1, ctrl_files))
        if not matched_controls:
            logger.warning(f"No file found for {c_name} control in {fcs_dir}")
            continue
        if len(matched_controls) > 1:
            raise ValueError(f"Multiple files found for {c_name} control in {fcs_dir}: {matched_controls}")
        file_tree[c_name] = matched_controls[0]

    if len(primary) > 1:
        raise ValueError(f"Multiple non-control (primary) files found in directory {fcs_dir}: {primary}.")
    file_tree["primary"] = primary[0]

    compensation_file = [x for x in os.listdir(fcs_dir) if x == compensation_file]
    if len(compensation_file) > 1:
        raise ValueError(f"Multiple compensation files identified in {fcs_dir}: {compensation_file}")

    file_tree["compensation_file"] = None
    if compensation_file:
        file_tree["compensation_file"] = os.path.join(fcs_dir, compensation_file[0])

    return file_tree


def fcs_mappings(path: str) -> Union[List[str], None]:
    """
    Fetch channel mappings from fcs file.

    Parameters
    ------------
    path: str
        path to fcs file

    Returns
    --------
    Union[List[str], None]
        List of channel mappings. Will return None if file fails to load.
    """
    try:
        fo = flowio.FlowData(filename_or_handle=path)
    except ValueError as e:
        logger.error(f"Failed to load file {path}; {e}")
        return None
    return fo.channels


def explore_channel_mappings(fcs_dir: str, exclude_comps: bool = True) -> List[str]:
    """
    Given a directory, explore all fcs files and find all permutations of channel/marker mappings

    Parameters
    ----------
    fcs_dir: str
        Root directory to search
    exclude_comps: bool, (default=True)
        Exclude compentation files (must have 'comp' in filename)

    Returns
    --------
    List[str]
        list of all unique channel/marker mappings
    """
    if exclude_comps:
        fcs_files = filter_fcs_files(fcs_dir, exclude_files="comp")
    else:
        fcs_files = filter_fcs_files(fcs_dir)
    with Pool(cpu_count()) as pool:
        mappings = list(pool.map(fcs_mappings, fcs_files))
        mappings = list(pool.map(json.dumps, mappings))
    return [json.loads(x) for x in mappings]


def _get_channel_mappings(fluoro_dict: Dict) -> List[Dict[str, str]]:
    """
    Generates a list of dictionary objects that describe the fluorochrome mappings in this FCS file

    Parameters
    -----------
    fluoro_dict: Dict
        dictionary object from the channels param of the fcs file

    Returns
    --------
    List[Dict[str, str]]
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


def load_compensation_matrix(fcs: flowio.FlowData) -> Union[pl.DataFrame, None]:
    """
    Extract a compensation matrix from an FCS file using FlowIO and return as a polars DataFrame. If no spillover
    matrix found in file, will return None.

    Parameters
    ----------
    fcs: flowio.FlowData

    Returns
    -------
    Union[polars.DataFrame, None]
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
    """
    Read the headers from a csv or fcs file and return as a list

    Parameters
    ----------
    path: str
    s3_bucket: str

    Returns
    -------
    List[str]
    """
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
