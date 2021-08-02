from cytopy.tests import assets
from cytopy.data import read_write
from pathlib import Path
import pytest
import shutil
import os


@pytest.fixture()
def make_examples():
    os.mkdir(f"{os.getcwd()}/test_filter")
    os.mkdir(f"{os.getcwd()}/test_filter/ignore")
    Path(f"{os.getcwd()}/test_filter/primary.fcs").touch()
    Path(f"{os.getcwd()}/test_filter/FMO_CD40.fcs").touch()
    Path(f"{os.getcwd()}/test_filter/FMO_CD45.fcs").touch()
    Path(f"{os.getcwd()}/test_filter/FMO_CD80.fcs").touch()
    Path(f"{os.getcwd()}/test_filter/FMO_CD85.fcs").touch()
    for i in range(5):
        Path(f"{os.getcwd()}/test_filter/{i + 1}_Compensation.fcs").touch()
    for i in range(5):
        Path(f"{os.getcwd()}/test_filter/ignore/{i + 1}.fcs").touch()
    yield
    shutil.rmtree(f"{os.getcwd()}/test_filter")


def test_filter_fcs_files(make_examples):
    assert (
        len(
            read_write.filter_fcs_files(
                f"{os.getcwd()}/test_filter", exclude_comps=True, exclude_dir="ignore"
            )
        )
        == 5
    )
    assert (
        len(
            read_write.filter_fcs_files(
                f"{os.getcwd()}/test_filter", exclude_comps=False, exclude_dir="ignore"
            )
        )
        == 10
    )


def test_get_fcs_file_paths(make_examples):
    tree = read_write.get_fcs_file_paths(
        f"{os.getcwd()}/test_filter",
        control_names=["CD45", "CD80", "CD85", "CD40"],
        ctrl_id="FMO",
        ignore_comp=True,
        exclude_dir="ignore",
    )
    assert tree.get("primary")[0] == f"{os.getcwd()}/test_filter/primary.fcs"
    assert len(tree.get("controls")) == 4


def test_fcs_mappings():
    mappings = read_write.fcs_mappings(f"{assets.__path__._path[0]}/test.FCS")
    x = [i for i in mappings if i["channel"] == "FL2 Log"]
    assert x[0].get("marker") == "IgG1-PE"


def test_explore_channel_mappings():
    mappings = read_write.explore_channel_mappings(assets.__path__._path[0])[0]
    assert isinstance(mappings, list)
    x = [i for i in mappings if dict(i)["channel"] == "FL2 Log"]
    assert x[0].get("marker") == "IgG1-PE"
