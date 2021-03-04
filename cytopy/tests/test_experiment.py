from sklearn.datasets import make_blobs
from cytopy.data.experiment import *
from cytopy.data.errors import *
from cytopy.tests.conftest import create_example_populations
from cytopy.tests import assets
import pandas as pd
import pytest
import h5py
import os


def test_check_excel_template():
    mappings, nomenclature = check_excel_template(f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert isinstance(mappings, pd.DataFrame)
    assert isinstance(nomenclature, pd.DataFrame)


def test_check_duplicates():
    x = ["Marker1", "Marker1", "Marker2", "Marker3"]
    with pytest.warns(UserWarning) as warn_:
        y = check_duplication(x)
    assert str(warn_.list[0].message) == "Duplicate channel/markers identified: ['Marker1']"
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
    test = Panel()
    test.create_from_excel(f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert all(x in [cm.marker for cm in test.mappings] for x in
               ["FS Lin", "SS Log", "IgG1-FITC", "IgG1-PE", "CD45-ECD", "IgG1-PC5", "IgG1-PC7"])
    assert all(x in [cm.channel for cm in test.mappings] for x in
               ["FS Lin", "SS Log", "FL1 Log", "FL2 Log", "FL3 Log", "FL4 Log", "FL5 Log"])
    assert all(x in [m.standard for m in test.markers] for x in
               ["FS Lin", "SS Log", "IgG1-FITC", "IgG1-PE", "CD45-ECD", "IgG1-PC5", "IgG1-PC7"])
    assert all(x in [c.standard for c in test.channels] for x in
               ["FS Lin", "SS Log", "FL1 Log", "FL2 Log", "FL3 Log", "FL4 Log", "FL5 Log"])


def test_exp_init(example_populated_experiment):
    print(os.getcwd())
    exp = example_populated_experiment
    exp.generate_panel(panel_definition=f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert exp.panel is not None
    assert exp.data_directory == f"{os.getcwd()}/test_data"


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
    with pytest.raises(DuplicateSampleError) as err:
        example_populated_experiment.add_dataframes(sample_id="test sample",
                                                    primary_data="dummy path",
                                                    mappings=[])
    assert str(err.value) == "A file group with id test sample already exists"
    with pytest.raises(DuplicateSampleError) as err:
        example_populated_experiment.add_fcs_files(sample_id="test sample",
                                                   primary="dummy path")
    assert str(err.value) == "A file group with id test sample already exists"


def assert_h5_setup_correctly(filepath: str,
                              exp: Experiment):
    assert os.path.isfile(filepath)
    with h5py.File(filepath, "r") as f:
        assert f["index/root/primary"][:].shape[0] == 30000
        assert f["test_ctrl"][:].shape[0] == 30000
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
    controls = {"test_ctrl": pd.DataFrame(make_blobs(n_samples=30000, n_features=14, centers=8, random_state=42)[0])}
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
    exp.add_fcs_files(sample_id="test sample 2",
                      primary=f"{assets.__path__._path[0]}/test.FCS",
                      controls={"test_ctrl": f"{assets.__path__._path[0]}/test.FCS"},
                      compensate=False)
    assert "test sample" in list(exp.list_samples())
    new_filegroup = exp.get_sample("test sample")
    path = f"{os.getcwd()}/test_data/{new_filegroup.id}.hdf5"
    assert_h5_setup_correctly(path, exp)


def test_exp_delete(example_populated_experiment):
    exp = example_populated_experiment
    exp.delete()
    assert len(FileGroup.objects()) == 0


def test_exp_standardise_mappings(example_populated_experiment):
    exp = example_populated_experiment
    standardised_mappings = exp._standardise_mappings(mappings=MAPPINGS,
                                                      missing_error="raise")
    assert set([x["channel"] for x in standardised_mappings]) == set(exp.panel.list_channels())
    assert set([x["marker"] for x in standardised_mappings]) == set(exp.panel.list_markers())


def test_load_data(example_populated_experiment):
    exp = example_populated_experiment
    exp.add_fcs_files(sample_id="test sample 2",
                      primary=f"{assets.__path__._path[0]}/test.FCS",
                      controls={"test_ctrl": f"{assets.__path__._path[0]}/test.FCS"},
                      compensate=False)
    create_example_populations(exp.get_sample("test sample"))
    create_example_populations(exp.get_sample("test sample 2"))
    data = load_population_data_from_experiment(experiment=exp,
                                                population="pop1")
    assert all([x in data.columns for x in ["sample_id", "subject_id", "original_index"]])
    assert data.shape[0] == 30084
    assert set(data["sample_id"].unique()) == {"test sample", "test sample 2"}
    test_sample_pop1 = exp.get_sample("test sample").get_population("pop1")
    test_sample2_pop1 = exp.get_sample("test sample 2").get_population("pop1")
    # Check original index
    assert np.array_equal(data[data.sample_id == "test sample"]["original_index"].values,
                          test_sample_pop1.index)
    assert np.array_equal(data[data.sample_id == "test sample 2"]["original_index"].values,
                          test_sample2_pop1.index)
    # Check that the population furthest from the root is populated correctly i.e. pop3
    test_sample_pop = exp.get_sample("test sample").get_population("pop3")
    test_sample2_pop = exp.get_sample("test sample 2").get_population("pop3")
    assert np.array_equal(data[(data.sample_id == "test sample") &
                               (data.population_label == "pop3")]["original_index"].sort_values().values,
                          np.sort(test_sample_pop.index))
    assert np.array_equal(data[(data.sample_id == "test sample 2") &
                               (data.population_label == "pop3")]["original_index"].sort_values().values,
                          np.sort(test_sample2_pop.index))
    # The next population up is pop2, cells belonging to pop2 but not pop3 will be labelled
    # in the population_label column
    test_sample_pop = exp.get_sample("test sample").get_population("pop2")
    idx = [x for x in test_sample_pop.index
           if x not in exp.get_sample("test sample").get_population("pop3").index]
    assert np.array_equal(data[(data.sample_id == "test sample") &
                               (data.population_label == "pop2")]["original_index"].sort_values().values,
                          np.sort(idx))
    test_sample2_pop = exp.get_sample("test sample 2").get_population("pop2")
    idx = [x for x in test_sample2_pop.index
           if x not in exp.get_sample("test sample 2").get_population("pop3").index]
    assert np.array_equal(data[(data.sample_id == "test sample 2") &
                               (data.population_label == "pop2")]["original_index"].sort_values().values,
                          np.sort(idx))

