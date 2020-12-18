from CytoPy.tests import assets
from CytoPy.flow import ext_tools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

FUNCTIONS = ["linear", "poly2", "poly3", "4pl", "5pl"]


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
    assert corrected.iloc[92]["beta-NGF"] == 0.00000001
    assert corrected.iloc[10]["CXCL10"] == (80.5 - 31)
    assert corrected.iloc[20]["CXCL10"] == (74 - 31)
    assert corrected.iloc[30]["CXCL10"] == (3908 - 31)
    assert corrected.iloc[40]["CXCL10"] == (1642 - 31)


def test_valid_assay_data(load_example_data):
    x = ext_tools.AssayTools(data=load_example_data.get("data"),
                             conc=load_example_data.get("conc"),
                             background_id="Background0",
                             standards=[f"Standard{i + 1}" for i in range(6)])
    assert isinstance(x.raw, pd.DataFrame)


def test_invalid_assay_data(load_example_data):
    data = load_example_data.get("data")
    data_no_sample = data.drop("Sample", axis=1)
    with pytest.raises(AssertionError) as err:
        ext_tools.AssayTools(data=data_no_sample,
                             conc=load_example_data.get("conc"),
                             background_id="Background0",
                             standards=[f"Standard{i + 1}" for i in range(6)])
    assert str(err.value) == "Invalid DataFrame missing 'Sample' column"
    with pytest.raises(AssertionError) as err:
        ext_tools.AssayTools(data=load_example_data.get("data"),
                             conc=load_example_data.get("conc"),
                             background_id="Background0",
                             standards=[f"Standard{i + 1}" for i in range(9)])
    assert str(err.value) == "One or more listed standards missing from Sample column"


def test_invalid_conc(load_example_data):
    conc = load_example_data.get("conc").copy()
    conc_no_analyte = conc.drop("analyte", axis=1)
    with pytest.raises(AssertionError) as err:
        ext_tools.AssayTools(data=load_example_data.get("data"),
                             conc=conc_no_analyte,
                             background_id="Background0",
                             standards=[f"Standard{i + 1}" for i in range(6)])
    assert str(err.value), "Analyte column missing from concentrations dataframe"
    conc_missing_analyte = conc[conc.analyte != "CXCL10"].copy()
    with pytest.raises(AssertionError) as err:
        ext_tools.AssayTools(data=load_example_data.get("data"),
                             conc=conc_missing_analyte,
                             background_id="Background0",
                             standards=[f"Standard{i + 1}" for i in range(6)])
    assert str(err.value), "One or more of the specified analytes missing from concentrations dataframe"
    conc = load_example_data.get("conc").copy()
    conc_missing_standard = conc.drop("Standard6", axis=1)
    with pytest.raises(AssertionError) as err:
        ext_tools.AssayTools(data=load_example_data.get("data"),
                             conc=conc_missing_standard,
                             background_id="Background0",
                             standards=[f"Standard{i + 1}" for i in range(6)])
    assert str(err.value), "One or more of the specified standards missing from concentrations dataframe"


@pytest.mark.parametrize("func,transform",
                         [(f, "log10") for f in FUNCTIONS] + [(f, None) for f in FUNCTIONS])
def test_fit_one_analyte(load_example_data, func, transform):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit(func=func,
                transform=transform,
                analyte="CXCL10")
    assert "CXCL10" in example.standard_curves.keys()
    assert isinstance(example.standard_curves.get("CXCL10"), dict)
    assert isinstance(example.standard_curves.get("CXCL10").get("params"), np.ndarray)
    assert example.standard_curves.get("CXCL10").get("transform") == transform
    assert hasattr(example.standard_curves.get("CXCL10").get("function"), '__call__')
    assert isinstance(example.standard_curves.get("CXCL10").get("residuals"), np.ndarray)
    assert isinstance(example.standard_curves.get("CXCL10").get("sigma"), np.ndarray)
    assert isinstance(example.standard_curves.get("CXCL10").get("r_squared"), float)


@pytest.mark.parametrize("func,transform",
                         [(f, "log10") for f in FUNCTIONS] + [(f, None) for f in FUNCTIONS])
def test_fit_all(load_example_data, func, transform):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit(func=func,
                transform=transform,
                analyte=None)
    assert all([x in example.standard_curves.keys() for x in example.analytes])


def test_assert_fitted(load_example_data):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    with pytest.raises(AssertionError) as err:
        example.predict(analyte="CXCL10")
    assert str(err.value) == "Standard curves have not been computed; call 'fit' prior to additional functions"
    example.fit(func="linear", analyte="CXCL10")
    with pytest.raises(AssertionError) as err:
        example.predict(analyte="beta-NGF")
    assert str(err.value) == "Standard curve not detected for beta-NGF; call 'fit' prior to additional functions"


@pytest.mark.parametrize("func,transform,analyte",
                         [(f, "log10", "CXCL10") for f in FUNCTIONS] + [(f, None, "CXCL10") for f in FUNCTIONS])
def test_fit_predict_one_analyte(load_example_data, func, transform, analyte):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit_predict(func=func, transform=transform, analyte=analyte)
    assert "CXCL10" in example.predictions.columns
    assert not example.predictions["CXCL10"].isna().any()


def test_plot_standard_curve(load_example_data):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit_predict(func="4pl", transform="log10", analyte="CXCL10")
    example.plot_standard_curve(analyte="CXCL10")
    plt.show()
