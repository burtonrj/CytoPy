from CytoPy.tests import assets
from CytoPy import assay_tools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest


@pytest.fixture()
def load_example_data():
    return {"data": pd.read_csv(f"{assets.__path__._path[0]}/example_assay.csv"),
            "conc": pd.read_csv(f"{assets.__path__._path[0]}/example_assay_conc.csv")}


def test_subtract_background(load_example_data):
    corrected = assay_tools.subtract_background(load_example_data.get("data"),
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


def test_wrangle_standard_data(load_example_data):
    data = assay_tools.wrangle_standard_data(analyte="CXCL10",
                                             response=load_example_data.get("data"),
                                             concentrations=load_example_data.get("conc"))
    assert isinstance(data, pd.DataFrame)
    assert all([x in data.columns for x in ["Sample", "Concentration", "Response"]])
    assert data["Response"].dtype == np.dtype("float")
    assert data["Concentration"].dtype == np.dtype("float")
    assert data["Sample"].dtype == np.dtype("O")


def test_generalised_hill_equation():
    x = np.array([4481., 4735., 2695.5,
                  1466.5, 521.5, 417.5,
                  133., 127.5, 43.,
                  49.5, 20., 25.])
    y = np.array([250., 250., 83.33333333,
                  83.33333333, 27.77777778, 27.77777778,
                  9.25925926, 9.25925926, 3.08641975,
                  3.08641975, 1.02880658, 1.02880658])
    xx = np.linspace(np.min(x) - (np.min(x) * 2),
                     np.max(x) + (np.max(x) * 2),
                     1000)
    y_hat = assay_tools.generalised_hill_equation(xx,
                                                  a=np.min(y),
                                                  d=np.max(y),
                                                  log_inflection_point=np.log10(2200),
                                                  slope=10,
                                                  symmetry=1.0)
    ax = plt.subplots()[1]
    ax.scatter(x, y)
    ax.plot(xx, y_hat)
    ax.set_xscale("log")
    plt.show()
    x_hat = assay_tools.inverse_generalised_hill_equation(np.array([50]),
                                                          a=np.min(y),
                                                          d=np.max(y),
                                                          log_inflection_point=np.log10(2200),
                                                          slope=10,
                                                          symmetry=1.0)
    assert 800 < x_hat[0] < 900


def test_estimate_inflection_point():
    x = np.array([4481., 4735., 2695.5,
                  1466.5, 521.5, 417.5,
                  133., 127.5, 43.,
                  49.5, 20., 25.])
    y = np.array([250., 250., 83.33333333,
                  83.33333333, 27.77777778, 27.77777778,
                  9.25925926, 9.25925926, 3.08641975,
                  3.08641975, 1.02880658, 1.02880658])
    assert assay_tools.estimate_inflection_point(x, y) == np.log10(2695.5)

