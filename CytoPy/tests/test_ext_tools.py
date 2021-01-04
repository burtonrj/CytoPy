from CytoPy.tests import assets
from CytoPy.flow import ext_tools
from lmfit.model import ModelResult
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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


@pytest.mark.parametrize("model,transform",
                         [("linear", "log10"),
                          ("poly", "log10"),
                          ("quad", "log10"),
                          ("logit", "log10"),
                          ("linear", None),
                          ("poly", None),
                          ("quad", None),
                          ("logit", None)])
def test_fit_one_analyte(load_example_data, model, transform):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit(model=model,
                transform=transform,
                analyte="CXCL10")
    assert "CXCL10" in example.standard_curves.keys()
    assert isinstance(example.standard_curves.get("CXCL10"), dict)
    assert example.standard_curves.get("CXCL10").get("transform") == transform
    assert isinstance(example.standard_curves.get("CXCL10").get("model_result"), ModelResult)


@pytest.mark.parametrize("model,transform",
                         [("linear", "log10"),
                          ("poly", "log10"),
                          ("quad", "log10"),
                          ("logit", "log10"),
                          ("linear", None),
                          ("poly", None),
                          ("quad", None),
                          ("logit", None)])
def test_fit_all(load_example_data, model, transform):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit(model=model,
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
    example.fit(model="linear", analyte="CXCL10")
    with pytest.raises(AssertionError) as err:
        example.predict(analyte="beta-NGF")
    assert str(err.value) == "Standard curve not detected for beta-NGF; call 'fit' prior to additional functions"


@pytest.mark.parametrize("model,transform",
                         [("linear", "log10"),
                          ("poly", "log10"),
                          ("quad", "log10"),
                          ("logit", "log10"),
                          ("linear", None),
                          ("poly", None),
                          ("quad", None),
                          ("logit", None)])
def test_fit_predict_one_analyte(load_example_data, model, transform):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit_predict(model=model, transform=transform, analyte="CXCL10")
    assert "CXCL10" in example.predictions.columns
    assert not example.predictions["CXCL10"].isna().any()


@pytest.mark.parametrize("overlay_predictions", [False, True])
def test_plot_standard_curve(load_example_data, overlay_predictions):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit_predict(model="logit", transform="log10", analyte="CXCL10")
    example.plot_standard_curve(analyte="CXCL10", overlay_predictions=overlay_predictions)
    plt.show()


def test_plot_repeat_measures(load_example_data):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit_predict(model="logit", transform="log10", analyte="CXCL10")
    example.plot_repeat_measures(analyte="CXCL10")
    plt.show()


@pytest.mark.parametrize("linear_scale", [True, False])
def test_cv(load_example_data, linear_scale):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit_predict(model="logit", transform="log10", analyte="CXCL10")
    x = example.coef_var(analyte="CXCL10", linear_scale=linear_scale)
    assert isinstance(x, pd.DataFrame)


def test_plot_shift(load_example_data):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit_predict(model="logit", transform="log10", analyte="CXCL10")
    example.raw["RandomMeta"] = np.random.randint(0, 2, size=example.raw.shape[0])
    example.plot_shift(analyte="CXCL10", factor="RandomMeta")
    plt.show()


def test_corr_matrix(load_example_data):
    example = ext_tools.AssayTools(data=load_example_data.get("data"),
                                   conc=load_example_data.get("conc"),
                                   background_id="Background0",
                                   standards=[f"Standard{i + 1}" for i in range(6)][::-1])
    example.fit_predict(model="logit", transform="log10", analyte="CXCL10")
    example.fit_predict(model="logit", transform="log10", analyte="beta-NGF")
    example.corr_matrix()
    plt.show()

