from ...data.experiments import check_excel_template, NormalisedName, _query, _check_duplication, Panel, _data_dir_append_leading_char, Experiment
from ...tests import assets
import pandas as pd
import pytest
import os


def test_check_excel_template():
    mappings, nomenclature = check_excel_template(f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert isinstance(mappings, pd.DataFrame)
    assert isinstance(nomenclature, pd.DataFrame)


def test_query():
    example_mappings = [NormalisedName(standard="IgG1-FITC", regex_str="\s*IgG[-\s.]*1[-\s.]+FITC\s*", case_sensitive=False),
                        NormalisedName(standard="CD45-ECD", regex_str="\s*CD[-\s.]*45[-\s.]+ECD\s*", case_sensitive=False)]
    with pytest.raises(AssertionError) as exp:
        _query("Invalid", example_mappings)
    assert str(exp.value) == f'Unable to normalise Invalid; no match in linked panel'
    assert _query("igg1-fitc", example_mappings) == "IgG1-FITC"
    assert _query("cd45 ECD", example_mappings) == "CD45-ECD"


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


def test_panel_check_pairing():
    test = Panel(panel_name="Test Panel")
    test.create_from_excel(f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert test._check_pairing(channel="FS Lin", marker="FS Lin")
    assert test._check_pairing(channel="FL2 Log", marker="IgG1-PE")
    assert not test._check_pairing(channel="FL2 Log", marker="CD45-ECD")


def test_panel_standardise_names():
    test = Panel(panel_name="Test Panel")
    test.create_from_excel(f"{assets.__path__._path[0]}/test_panel.xlsx")
    original_mappings = [("FS-Lin", ""), ("SS-Log", ""), ("FL1-Log", "IgG1 Fitc"),
                         ("FL2 Log", "IgG1 pe"), ("FL3 Log", "cd45 ecd"),
                         ("FL4 Log", "IgG1 pc5"), ("FL5 Log", "IgG1 pc7")]
    correct_mappings = [("FS Lin", "FS Lin"), ("SS Log", "SS Log"), ("FL1 Log", "IgG1-FITC"),
                        ("FL2 Log", "IgG1-PE"), ("FL3 Log", "CD45-ECD"), ("FL4 Log", "IgG1-PC5"),
                        ("FL5 Log", "IgG1-PC7")]
    corrected = test.standardise_names(original_mappings)
    assert all(x[0] == y[0] for x, y in zip(corrected, correct_mappings))
    assert all(x[1] == y[1] for x, y in zip(corrected, correct_mappings))


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


