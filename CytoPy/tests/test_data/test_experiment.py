from ...data.experiment import check_excel_template, NormalisedName, \
    _check_duplication, Panel, _data_dir_append_leading_char, Experiment, \
    _check_pairing
from ...tests import assets
import pandas as pd
import pytest
import os


def test_check_excel_template():
    mappings, nomenclature = check_excel_template(f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert isinstance(mappings, pd.DataFrame)
    assert isinstance(nomenclature, pd.DataFrame)


def test_check_duplicates():
    x = ["Marker1", "Marker1", "Marker2", "Marker3"]
    with pytest.warns(UserWarning) as warn:
        y = _check_duplication(x)
    assert str(warn.list[0].message) == "Duplicate channel/markers identified: ['Marker1']"
    assert y


def test_panel_create_from_excel():
    test = Panel(panel_name="Test Panel")
    test.create_from_excel(f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert all(x in [cm.marker for cm in test.mappings] for x in ["FS Lin", "SS Log", "IgG1-FITC", "IgG1-PE", "CD45-ECD", "IgG1-PC5", "IgG1-PC7"])
    assert all(x in [cm.channel for cm in test.mappings] for x in ["FS Lin", "SS Log", "FL1 Log", "FL2 Log", "FL3 Log", "FL4 Log", "FL5 Log"])
    assert all(x in [m.standard for m in test.markers] for x in ["FS Lin", "SS Log", "IgG1-FITC", "IgG1-PE", "CD45-ECD", "IgG1-PC5", "IgG1-PC7"])
    assert all(x in [c.standard for c in test.channels] for x in ["FS Lin", "SS Log", "FL1 Log", "FL2 Log", "FL3 Log", "FL4 Log", "FL5 Log"])


def test_data_dir_append_leading_char():
    assert _data_dir_append_leading_char("C:\\some\\path\\") == "C:\\some\path\\"
    assert _data_dir_append_leading_char("C:\\some\\path") == "C:\\some\path\\"
    assert _data_dir_append_leading_char("/some/path/") == "/some/path/"
    assert _data_dir_append_leading_char("/some/path") == "/some/path/"


def test_exp_init():
    exp = Experiment(experiment_id="test",
                     data_directory=f"{os.getcwd()}/test_data",
                     panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert exp.panel.panel_name  == "test_panel"


def test_exp_update_data_dir():
    exp = Experiment(experiment_id="test",
                     data_directory=f"{os.getcwd()}/test_data",
                     panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    with pytest.raises(AssertionError) as ex:
        exp.update_data_directory("not_a_path")
    assert str(ex.value) == "Invalid directory given for new_path"
    exp.update_data_directory(assets.__path__._path[0])
    assert exp.data_directory == assets.__path__._path[0]


def test_exp_add_new_sample():
    exp = Experiment(experiment_id="test",
                     data_directory=f"{os.getcwd()}/test_data",
                     panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    exp.add_new_sample(sample_id="test_sample",
                       primary_path=f"{assets.__path__._path[0]}/test.FCS",
                       compensate=False)
    assert "test_sample" in list(exp.list_samples())
    new_filegroup = exp.get_sample("test_sample")
    assert os.path.isfile(f"{os.getcwd()}/test_data/{new_filegroup.id}.hdf5")


