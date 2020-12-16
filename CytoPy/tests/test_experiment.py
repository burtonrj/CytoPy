from sklearn.datasets import make_blobs
from CytoPy.data.experiment import *
from CytoPy.tests.conftest import create_example_populations
from CytoPy.tests import assets
import pandas as pd
import pytest
import shutil
import h5py
import os


def test_check_excel_template():
    mappings, nomenclature = check_excel_template(f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert isinstance(mappings, pd.DataFrame)
    assert isinstance(nomenclature, pd.DataFrame)


def test_check_duplicates():
    x = ["Marker1", "Marker1", "Marker2", "Marker3"]
    with pytest.warns(UserWarning) as warn:
        y = check_duplication(x)
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
    test_standard = NormalisedName(standard="test",
                                   regex_str=regex,
                                   permutations=permutations,
                                   case_sensitive=case)
    result = test_standard.query(query)
    assert result == expected


def test_query_normalised_name_list():
    ref = [NormalisedName(standard="CD4",
                          regex_str=r"^\s*CD[\-\s]*4\s*$"),
           NormalisedName(standard="HLA-DR",
                          regex_str=r"^\s*HLA[\-\s]*DR\s*$"),
           NormalisedName(standard="CD45",
                          regex_str=r"^\s*CD[\-\s]*45\s*$",
                          permutations="FITC-CD45,FITC CD45")]
    with pytest.raises(AssertionError) as err:
        query_normalised_list("CD8", ref=ref)
    assert str(err.value) == f'Unable to normalise CD8; no match in linked panel'
    for x in ["CD4", "cd4", "CD-4"]:
        assert query_normalised_list(x, ref=ref) == "CD4"
    for x in ["CD45", "cd45", "FITC-CD45"]:
        assert query_normalised_list(x, ref=ref) == "CD45"
    for x in ["hla dr", "hla-dr", "HLA-dr"]:
        assert query_normalised_list(x, ref=ref) == "HLA-DR"


@pytest.mark.parametrize("example,expected", [({"channel": "  FITC  ", "marker": "CD4-FITC"},
                                               {"channel": "FITC", "marker": "CD4"}),
                                              ({"channel": "FSC-A", "marker": ""},
                                               {"channel": "FSC-A", "marker": None}),
                                              ({"channel": "APC/Fire-750", "marker": "cd3"},
                                               {"channel": "APC-Fire-750", "marker": "CD3"})])
def test_standardise_names(example, expected):
    ref_channels = [NormalisedName(standard="FITC",
                                   regex_str=r"\s*FITC\s*"),
                    NormalisedName(standard="FSC-A",
                                   regex_str=r"\s*FSC[\-\s]+A\s*"),
                    NormalisedName(standard="APC-Fire-750",
                                   regex_str=r"\s*APC[/\-\s]+Fire[\-\s]+750\s*")]
    ref_markers = [NormalisedName(standard="CD4",
                                  regex_str=r"\s*CD4\s*",
                                  permutations="CD4-FITC"),
                   NormalisedName(standard="CD3",
                                  regex_str=r"\s*CD3\s*")]
    ref_mappings = [ChannelMap(channel="FITC", marker="CD4"),
                    ChannelMap(channel="FSC-A"),
                    ChannelMap(channel="APC-Fire-750", marker="CD3")]
    standardised = standardise_names(example,
                                     ref_channels=ref_channels,
                                     ref_markers=ref_markers,
                                     ref_mappings=ref_mappings)
    assert standardised.get("channel") == expected.get("channel")
    assert standardised.get("marker") == expected.get("marker")


@pytest.mark.parametrize("example,expected", [({"channel": "FITC", "marker": "CD4"}, True),
                                              ({"channel": "FSC-A", "marker": "CD8"}, False),
                                              ({"channel": "APC-Fire-750", "marker": "CD3"}, True)])
def test_check_pairing(example, expected):
    ref_mappings = [ChannelMap(channel="FITC", marker="CD4"),
                    ChannelMap(channel="FSC-A"),
                    ChannelMap(channel="APC-Fire-750", marker="CD3")]
    assert check_pairing(example, ref_mappings) == expected


def test_duplicate_mappings():
    examples = [{"channel": "FSC-A", "marker": ""},
                {"channel": "SSC-A", "marker": ""},
                {"channel": "FITC", "marker": "CD3"},
                {"channel": "FITC", "marker": "CD8"}]
    with pytest.raises(AssertionError) as err:
        duplicate_mappings(examples)
    assert str(err.value) == "Duplicate channels provided"
    examples = [{"channel": "FSC-A", "marker": ""},
                {"channel": "SSC-A", "marker": ""},
                {"channel": "FITC", "marker": "CD3"},
                {"channel": "BV405", "marker": "CD3"}]
    with pytest.raises(AssertionError) as err:
        duplicate_mappings(examples)
    assert str(err.value) == "Duplicate markers provided"


def test_missing_channels():
    examples = [{"channel": "FSC-A", "marker": ""},
                {"channel": "SSC-A", "marker": ""},
                {"channel": "FITC", "marker": "CD3"},
                {"channel": "BV405", "marker": "CD4"}]
    ref_channels = [NormalisedName(standard="FITC"),
                    NormalisedName(standard="FSC-A"),
                    NormalisedName(standard="SSC-A"),
                    NormalisedName(standard="APC-Fire-750"),
                    NormalisedName(standard="BV405")]
    with pytest.raises(KeyError) as err:
        missing_channels(examples, ref_channels, errors="raise")
    assert str(err.value) == "'Missing channel APC-Fire-750'"
    with pytest.warns(UserWarning) as warning:
        missing_channels(examples, ref_channels, errors="warn")
    assert str(warning.list[0].message) == "Missing channel APC-Fire-750"


def test_panel_create_from_excel():
    test = Panel(panel_name="Test Panel")
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
    assert data_dir_append_leading_char("C:\\some\\path\\") == "C:\\some\path\\"
    assert data_dir_append_leading_char("C:\\some\\path") == "C:\\some\path\\"
    assert data_dir_append_leading_char("/some/path/") == "/some/path/"
    assert data_dir_append_leading_char("/some/path") == "/some/path/"


def test_exp_init(example_populated_experiment):
    print(os.getcwd())
    exp = example_populated_experiment
    assert exp.panel.panel_name == "test_panel"


def example_update_data_directory(exp: Experiment,
                                  path: str):
    exp.update_data_directory(new_path=path, move=False)


def test_exp_update_data_dir(example_populated_experiment):
    with pytest.raises(AssertionError) as ex:
        example_update_data_directory(example_populated_experiment, "not_a_path")
    assert str(ex.value) == "Invalid directory given for new_path"
    example_update_data_directory(example_populated_experiment, assets.__path__._path[0])
    assert example_populated_experiment.data_directory == assets.__path__._path[0]
    shutil.rmtree(f"{assets.__path__._path[0]}/test_data", ignore_errors=True)


def test_exp_delete_all_populations(example_populated_experiment):
    example_populated_experiment.delete_all_populations("test sample")
    fg = example_populated_experiment.get_sample("test sample")
    assert fg.list_populations() == ["root"]


def test_exp_sample_exists(example_populated_experiment):
    assert example_populated_experiment.sample_exists("test sample")
    assert not example_populated_experiment.sample_exists("doesn't exist")


def test_exp_get_sample(example_populated_experiment):
    fg = example_populated_experiment.get_sample("test sample")
    assert isinstance(fg, FileGroup)
    assert fg.primary_id == "test sample"


def test_exp_list_samples(example_populated_experiment):
    assert example_populated_experiment.list_samples() == ["test sample"]


def test_exp_remove_sample(example_populated_experiment):
    exp = example_populated_experiment
    filepath = exp.get_sample("test sample").h5path
    exp.remove_sample("test sample")
    assert "test sample" not in exp.list_samples()
    assert not os.path.isfile(filepath)
    x = FileGroup.objects(primary_id="test sample")
    assert len(x) == 0


def test_add_new_sample_exists_error(example_populated_experiment):
    with pytest.raises(AssertionError) as err:
        example_populated_experiment.add_dataframes(sample_id="test sample",
                                                    primary_path="dummy path",
                                                    mappings=[])
    assert str(err.value) == "A file group with id test sample already exists"
    with pytest.raises(AssertionError) as err:
        example_populated_experiment.add_fcs_files(sample_id="test sample",
                                                   primary="dummy path")
    assert str(err.value) == "A file group with id test sample already exists"


def assert_h5_setup_correctly(filepath: str,
                              exp: Experiment):
    assert os.path.isfile(filepath)
    with h5py.File(filepath, "r") as f:
        assert len(f["index/root/primary"][:]) == 30000
        assert len(f["index/root/ctrl1"][:]) == 30000
        assert set([x.decode("utf-8") for x in f["mappings/primary/channels"][:]]) == set(exp.panel.list_channels())
        assert set([x.decode("utf-8") for x in f["mappings/primary/markers"][:]]) == set(exp.panel.list_markers())


MAPPINGS = [{"channel": "FS-Lin", "marker": "FS-Lin"},
            {"channel": "SS-Log", "marker": "SS-Log"},
            {"channel": "fl1-log", "marker": "IgG1 FITC"},
            {"channel": "fl2-log", "marker": "IgG1 PE"},
            {"channel": "fl3-log", "marker": "CD45 ECD"},
            {"channel": "fl4-log", "marker": "IgG1 PC5"},
            {"channel": "fl5-log", "marker": "IgG1 PC7"}]


def test_exp_add_dataframes(example_populated_experiment):
    primary = pd.DataFrame(make_blobs(n_samples=30000, n_features=14, centers=8, random_state=42)[0])
    controls = {"ctrl1": pd.DataFrame(make_blobs(n_samples=30000, n_features=14, centers=8, random_state=42)[0])}
    exp = example_populated_experiment
    exp.add_dataframes(sample_id="test sample 2",
                       primary_data=primary,
                       controls=controls,
                       mappings=MAPPINGS)
    assert "test sample 2" in exp.list_samples()
    fg = exp.get_sample("test sample 2")
    path = f"{os.getcwd()}/test_data/{fg.id}.hdf5"
    assert_h5_setup_correctly(path, exp)


def test_exp_add_fcs_files(example_populated_experiment):
    exp = example_populated_experiment
    exp.add_fcs_files(sample_id="test_sample",
                      primary=f"{assets.__path__._path[0]}/test.FCS",
                      controls={"ctrl1": f"{assets.__path__._path[0]}/test.FCS"},
                      compensate=False)
    assert "test_sample" in list(exp.list_samples())
    new_filegroup = exp.get_sample("test_sample")
    path = f"{os.getcwd()}/test_data/{new_filegroup.id}.hdf5"
    assert_h5_setup_correctly(path, exp)


def test_exp_delete(example_populated_experiment):
    exp = example_populated_experiment
    exp.delete()
    assert len(Experiment.objects()) == 0
    assert len(FileGroup.objects()) == 0


def test_exp_standardise_mappings(example_populated_experiment):
    exp = example_populated_experiment
    standardised_mappings = exp._standardise_mappings(mappings=MAPPINGS,
                                                      missing_error="raise")
    assert set([x["channel"] for x in standardised_mappings]) == set(exp.panel.list_channels())
    assert set([x["marker"] for x in standardised_mappings]) == set(exp.panel.list_markers())


@pytest.mark.parametrize("pop_name", ["pop1", "pop2", "pop3"])
def test_load_clusters(example_populated_experiment, pop_name):
    fg = create_example_populations(example_populated_experiment.get_sample("test sample"))
    pop_df = fg.load_population_df(population=pop_name)
    clusters_df = load_clusters(pop_data=pop_df,
                                tag="testing",
                                filegroup=fg,
                                population=pop_name)
    assert clusters_df.dropna().shape[0] == (0.25 * clusters_df.shape[0])


def test_load_data(example_populated_experiment):
    exp = example_populated_experiment
    create_example_populations(exp.get_sample("test sample"))
    data = load_population_data_from_experiment(experiment=exp,
                                                population="pop1")


def test_fetch_subject_meta():
    pass


def test_fetch_subject():
    pass
