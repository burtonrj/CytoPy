from ...data import experiment
from ...tests import assets
import pandas as pd
import pytest
import shutil
import os


@pytest.fixture()
def example_experiment():
    exp = experiment.Experiment(experiment_id="test",
                                data_directory=f"{os.getcwd()}/test_data",
                                panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    exp.save()
    yield exp
    exp.delete(delete_panel=True)


def test_check_excel_template():
    mappings, nomenclature = experiment.check_excel_template(f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert isinstance(mappings, pd.DataFrame)
    assert isinstance(nomenclature, pd.DataFrame)


def test_check_duplicates():
    x = ["Marker1", "Marker1", "Marker2", "Marker3"]
    with pytest.warns(UserWarning) as warn:
        y = experiment._check_duplication(x)
    assert str(warn.list[0].message) == "Duplicate channel/markers identified: ['Marker1']"
    assert y


@pytest.mark.parametrize("query,expected,regex,permutations,case",
                         [("CD-45 RA", "test", r"^CD[\-\s]*45[\-\s]*RA$", "CD--45RA", False),
                          ("CD--45RA", "test", r"^CD[\-\s]*45[\-\s]*RA$", "CD--45RA", False),
                          ("Cd-45 ra", None, r"^CD[\-\s]*45[\-\s]*RA$", "CD--45RA", True),
                          ("Cd-45 ra", "test", r"^CD[\-\s]*45[\-\s]*RA$", "CD--45RA", False),
                          ("45RA", "test", r"^CD[\-\s]*45[\-\s]*RA$", "45RA", False),
                          ("CD4", None, r"^CD[\-\s]*45[\-\s]*RA$", "CD--45RA", False)])
def test_query_normalised_name(query, expected, regex, permutations, case):
    test_standard = experiment.NormalisedName(standard="test",
                                              regex_str=regex,
                                              permutations=permutations,
                                              case_sensitive=case)
    result = test_standard.query(query)
    assert result == expected


def test_query_normalised_name_list():
    ref = [experiment.NormalisedName(standard="CD4",
                                     regex_str=r"^\s*CD[\-\s]*4\s*$"),
           experiment.NormalisedName(standard="HLA-DR",
                                     regex_str=r"^\s*HLA[\-\s]*DR\s*$"),
           experiment.NormalisedName(standard="CD45",
                                     regex_str=r"^\s*CD[\-\s]*45\s*$",
                                     permutations="FITC-CD45,FITC CD45")]
    with pytest.raises(AssertionError) as err:
        experiment._query_normalised_list("CD8", ref=ref)
    assert str(err.value) == f'Unable to normalise CD8; no match in linked panel'
    for x in ["CD4", "cd4", "CD-4"]:
        assert experiment._query_normalised_list(x, ref=ref) == "CD4"
    for x in ["CD45", "cd45", "FITC-CD45"]:
        assert experiment._query_normalised_list(x, ref=ref) == "CD45"
    for x in ["hla dr", "hla-dr", "HLA-dr"]:
        assert experiment._query_normalised_list(x, ref=ref) == "HLA-DR"


def test_panel_create_from_excel():
    test = experiment.Panel(panel_name="Test Panel")
    test.create_from_excel(f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert all(x in [cm.marker for cm in test.mappings] for x in
               ["FS Lin", "SS Log", "IgG1-FITC", "IgG1-PE", "CD45-ECD", "IgG1-PC5", "IgG1-PC7"])
    assert all(x in [cm.channel for cm in test.mappings] for x in
               ["FS Lin", "SS Log", "FL1 Log", "FL2 Log", "FL3 Log", "FL4 Log", "FL5 Log"])
    assert all(x in [m.standard for m in test.markers] for x in
               ["FS Lin", "SS Log", "IgG1-FITC", "IgG1-PE", "CD45-ECD", "IgG1-PC5", "IgG1-PC7"])
    assert all(x in [c.standard for c in test.channels] for x in
               ["FS Lin", "SS Log", "FL1 Log", "FL2 Log", "FL3 Log", "FL4 Log", "FL5 Log"])


def test_data_dir_append_leading_char():
    assert experiment._data_dir_append_leading_char("C:\\some\\path\\") == "C:\\some\path\\"
    assert experiment._data_dir_append_leading_char("C:\\some\\path") == "C:\\some\path\\"
    assert experiment._data_dir_append_leading_char("/some/path/") == "/some/path/"
    assert experiment._data_dir_append_leading_char("/some/path") == "/some/path/"


def test_exp_init(example_experiment):
    assert example_experiment.panel.panel_name == "test_panel"


def test_exp_add_new_sample(example_experiment):
    example_experiment.add_new_sample(sample_id="test_sample",
                                      primary_path=f"{assets.__path__._path[0]}/test.FCS",
                                      compensate=False)
    assert "test_sample" in list(example_experiment.list_samples())
    new_filegroup = example_experiment.get_sample("test_sample")
    assert os.path.isfile(f"{os.getcwd()}/test_data/{new_filegroup.id}.hdf5")


def test_delete(example_experiment):
    example_experiment.add_new_sample(sample_id="test_sample",
                                      primary_path=f"{assets.__path__._path[0]}/test.FCS",
                                      compensate=False)
    example_experiment.save()
    example_experiment.delete()
    assert len(experiment.Experiment.objects()) == 0
    assert len(experiment.FileGroup.objects()) == 0


def test_exp_update_data_dir(example_experiment):
    with pytest.raises(AssertionError) as ex:
        example_experiment.update_data_directory("not_a_path")
    assert str(ex.value) == "Invalid directory given for new_path"
    example_experiment.update_data_directory(assets.__path__._path[0])
    assert example_experiment.data_directory == assets.__path__._path[0]
    shutil.rmtree(f"{assets.__path__._path[0]}/test_data", ignore_errors=True)
