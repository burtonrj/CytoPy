import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from ..conftest import savefig
from cytopy.utils import batch_effects


def test_bw_optimisation():
    np.random.seed(42)
    data = pd.DataFrame({"x": np.random.normal(loc=1.0, scale=1.0, size=1000)})
    bandwidth = batch_effects.bw_optimisation(data=data, features=["x"], bandwidth=[0.1, 2.0])
    assert 0.0 < bandwidth < 0.5


def test_marker_variance():
    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "x": np.random.normal(loc=-1.0, scale=1.0, size=1000),
                    "y": np.random.normal(loc=-0.0, scale=1.0, size=1000),
                    "sample_id": ["A" for _ in range(1000)],
                }
            ),
            pd.DataFrame(
                {
                    "x": np.random.normal(loc=0, scale=2.0, size=1000),
                    "y": np.random.normal(loc=0, scale=1.0, size=1000),
                    "sample_id": ["B" for _ in range(1000)],
                }
            ),
            pd.DataFrame(
                {
                    "x": np.random.normal(loc=1.0, scale=1.0, size=1000),
                    "y": np.random.normal(loc=0.0, scale=1.0, size=1000),
                    "sample_id": ["C" for _ in range(1000)],
                }
            ),
        ]
    )
    fig = batch_effects.marker_variance(data=data, reference="B", markers=["x", "y"], figsize=(10, 5))
    savefig(fig, "test_marker_variance.png")
    assert isinstance(fig, plt.Figure)


def test_dim_reduction_grid_absent_ref_error(small_high_dim_dataframe):
    with pytest.raises(AssertionError):
        batch_effects.dim_reduction_grid(
            data=small_high_dim_dataframe,
            reference="sample_15",
            features=[x for x in small_high_dim_dataframe.columns if x != "sample_id"],
        )


def test_dim_reduction_grid(small_high_dim_dataframe):
    fig = batch_effects.dim_reduction_grid(
        data=small_high_dim_dataframe,
        reference="sample_0",
        features=[x for x in small_high_dim_dataframe.columns if x != "sample_id"],
    )
    savefig(fig, "dime_reduction_grid.png")
    assert isinstance(fig, plt.Figure)


def test_construct_harmony(small_high_dim_dataframe):
    harmony = batch_effects.Harmony(
        data=small_high_dim_dataframe,
        features=[x for x in small_high_dim_dataframe.columns if x not in ["sample_id", "original_index"]],
        transform=None,
        scale="standard",
    )
    assert isinstance(harmony, batch_effects.Harmony)
    assert isinstance(harmony.data, pd.DataFrame)
    assert isinstance(harmony.meta, pd.DataFrame)
    assert harmony.meta.columns == ["sample_id"]


def test_harmony_run_and_plot(small_high_dim_dataframe):
    harmony = batch_effects.Harmony(
        data=small_high_dim_dataframe,
        features=[x for x in small_high_dim_dataframe.columns if x not in ["sample_id", "original_index"]],
        transform=None,
        scale="standard",
    )
    x = harmony.run(var_use="sample_id")
    assert isinstance(x, batch_effects.Harmony)
    assert harmony.harmony is not None

    ax = harmony.batch_lisi_distribution(meta_var="sample_id", sample=0.25)
    assert isinstance(ax, plt.Axes)
    savefig(plt.gcf(), "batch_lisi_distribution.png")

    fig = harmony.plot_correction(n=1000, dim_reduction_method="UMAP", hue="sample_id")
    assert isinstance(fig, plt.Figure)
    savefig(fig, "plot_correction.png")

    corrected = harmony.batch_corrected()
    assert isinstance(corrected, pd.DataFrame)
