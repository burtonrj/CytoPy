from CytoPy.tests import assets
from CytoPy.flow import ext_tools
import pandas as pd
import pytest


@pytest.fixture()
def load_example_data():
    return {"data": pd.read_csv(f"{assets.__path__._path[0]}/example_assay.csv"),
            "conc": pd.read_csv(f"{assets.__path__._path[0]}/example_assay_conc.csv")}


def test_subtract_background(load_example_data):
    corrected = ext_tools.subtract_background(load_example_data.get("data"),
                                              "Background0",
                                              ["beta-NGF", "CXCL10"])
    assert corrected.iloc[10]["beta-NGF"] == 19.5
    assert corrected.iloc[20]["beta-NGF"] == 1.5
    assert corrected.iloc[30]["beta-NGF"] == 1
    assert corrected.iloc[92]["beta-NGF"] == 0
    assert corrected.iloc[10]["CXCL10"] == (80.5 - 31)
    assert corrected.iloc[20]["CXCL10"] == (74 - 31)
    assert corrected.iloc[30]["CXCL10"] == (3908 - 31)
    assert corrected.iloc[40]["CXCL10"] == (1642 - 31)


def test_valid_assay_data(load_example_data):
    x = ext_tools.AssayTools(data=load_example_data.get("data"),
                             conc=load_example_data.get("conc"),
                             background_id="Background0",
                             standards=[f"Standard{i+1}" for i in range(6)])
    assert isinstance(x.raw, pd.DataFrame)


def test_invalid_assay_data(load_example_data):
    data = load_example_data.get("data")
    data_no_sample = data.drop("Sample", axis=1)
    with pytest.raises(AssertionError) as err:
        ext_tools.AssayTools(data=data_no_sample,
                             conc=load_example_data.get("conc"),
                             background_id="Backgroud0",
                             standards=[f"Standard{i+1}" for i in range(6)])
    assert str(err.value) == "Invalid DataFrame missing 'Sample' column"
    with pytest.raises(AssertionError) as err:
        ext_tools.AssayTools(data=load_example_data.get("data"),
                             conc=load_example_data.get("conc"),
                             background_id="Backgroud0",
                             standards=[f"Standard{i+1}" for i in range(9)])
    assert str(err.value) == "One or more listed standards missing from Sample column"


def test_invalid_conc(load_example_data):
    conc = load_example_data.get("conc")
    conc_no_analyte = conc.drop("analyte", axis=1)
    with pytest.raises(AssertionError) as err:
        ext_tools.AssayTools(data=load_example_data.get("data"),
                             conc=conc_no_analyte,
                             background_id="Backgroud0",
                             standards=[f"Standard{i+1}" for i in range(6)])
    assert str(err.value), "Analyte column missing from concentrations dataframe"
    conc_missing_analyte = conc[conc.analyte != "Ferritin"].copy()
    with pytest.raises(AssertionError) as err:
        ext_tools.AssayTools(data=load_example_data.get("data"),
                             conc=conc_missing_analyte,
                             background_id="Backgroud0",
                             standards=[f"Standard{i+1}" for i in range(6)])
    assert str(err.value), "One or more of the specified analytes missing from concentrations dataframe"
    conc_missing_standard = conc.drop("Standard6", axis=1)
    with pytest.raises(AssertionError) as err:
        ext_tools.AssayTools(data=load_example_data.get("data"),
                             conc=conc_missing_standard,
                             background_id="Backgroud0",
                             standards=[f"Standard{i+1}" for i in range(6)])
    assert str(err.value), "One or more of the specified standards missing from concentrations dataframe"
