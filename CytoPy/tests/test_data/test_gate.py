from ...data.gate import Gate, ThresholdGate, PolygonGate, EllipseGate, ChildThreshold, ChildPolygon, \
    create_signature, Population
from ...data.geometry import ThresholdGeom, PolygonGeom
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import pytest


@pytest.mark.parametrize("klass,method", [(Gate, None),
                                          (ThresholdGate, "density"),
                                          (PolygonGate, "manual"),
                                          (EllipseGate, "GaussianMixture")])
def test_gate_init(klass, method):
    gate = klass(gate_name="test",
                 parent="test parent",
                 x="X",
                 y="Y",
                 method=method,
                 dim_reduction=dict(method="UMAP", kwargs={"n_neighbours": 100}))
    assert gate.gate_name == "test"
    assert gate.parent == "test parent"
    assert gate.x == "X"
    assert gate.y == "Y"
    assert gate.method == method
    assert gate.dim_reduction.get("method") == "UMAP"
    assert gate.dim_reduction.get("kwargs").get("n_neighbours") == 100


def test_transform_none():
    gate = Gate(gate_name="test",
                parent="test parent",
                x="X",
                y="Y",
                method="manual")
    data = pd.DataFrame({"X": np.random.normal(1, scale=0.5, size=1000),
                         "Y": np.random.normal(1, scale=0.5, size=1000)})
    transformed = gate._transform(data)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == 1000
    assert transformed.shape[1] == 2
    for i in ["X", "Y"]:
        assert transformed[i].mean() == pytest.approx(1., 0.1)
        assert transformed[i].std() == pytest.approx(0.5, 0.1)


def test_transform_x():
    gate = Gate(gate_name="test",
                parent="test parent",
                x="X",
                y="Y",
                method="manual",
                transformations={"x": "logicle"})
    data = pd.DataFrame({"X": np.random.normal(1, scale=0.5, size=1000),
                         "Y": np.random.normal(1, scale=0.5, size=1000)})
    transformed = gate._transform(data)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == 1000
    assert transformed.shape[1] == 2
    assert transformed["X"].mean() != pytest.approx(1., 0.1)
    assert transformed["X"].std() != pytest.approx(0.5, 0.1)
    assert transformed["Y"].mean() == pytest.approx(1., 0.1)
    assert transformed["Y"].std() == pytest.approx(0.5, 0.1)


def test_transform_xy():
    gate = Gate(gate_name="test",
                parent="test parent",
                x="X",
                y="Y",
                method="manual",
                transformations={"x": "logicle",
                                 "y": "logicle"})
    data = pd.DataFrame({"X": np.random.normal(1, scale=0.5, size=1000),
                         "Y": np.random.normal(1, scale=0.5, size=1000)})
    transformed = gate._transform(data)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == 1000
    assert transformed.shape[1] == 2
    assert transformed["X"].mean() != pytest.approx(1., 0.1)
    assert transformed["X"].std() != pytest.approx(0.5, 0.1)
    assert transformed["Y"].mean() != pytest.approx(1., 0.1)
    assert transformed["Y"].std() != pytest.approx(0.5, 0.1)


@pytest.mark.parametrize("kwargs", [{"method": "uniform",
                                     "n": 500},
                                    {"method": "faithful"},
                                    {"method": "density"},
                                    {"method": None}])
def test_downsample(kwargs):
    gate = Gate(gate_name="test",
                parent="test parent",
                x="X",
                y="Y",
                method="manual",
                sampling=kwargs)
    data = pd.DataFrame({"X": np.random.normal(1, scale=0.5, size=1000),
                         "Y": np.random.normal(1, scale=0.5, size=1000)})
    sample = gate._downsample(data=data)
    if kwargs.get("method") is None:
        assert sample is None
    else:
        assert sample.shape[0] < data.shape[0]


def test_upsample():
    data, labels = make_blobs(n_samples=3000,
                              n_features=2,
                              centers=3,
                              random_state=42)
    data = pd.DataFrame(data, columns=["X", "Y"])
    gate = Gate(gate_name="test",
                parent="test parent",
                x="X",
                y="Y",
                method="manual",
                sampling={"method": "uniform",
                          "frac": 0.5})
    sample = gate._downsample(data=data)
    sample_labels = labels[sample.index.values]
    pops = list()
    for x in np.unique(sample_labels):
        idx = sample.index.values[np.where(sample_labels == x)[0]]
        pops.append(Population(population_name=f"Pop_{x}",
                               parent="root",
                               index=idx[:498]))
    pops = gate._upsample(data=data, sample=sample, populations=pops)
    assert isinstance(pops, list)
    assert all([isinstance(p, Population) for p in pops])
    assert all([len(p.index) == 1000 for p in pops])
    for x in np.unique(labels):
        p = [i for i in pops if i.population_name == f"Pop_{x}"][0]
        assert np.array_equal(p.index, np.where(labels == x)[0])


def test_dim_reduction():
    gate = Gate(gate_name="test",
                parent="test parent",
                x="X",
                y="Y",
                method="manual",
                dim_reduction={"method": "UMAP",
                               "n_neighbors": 100})
    data = pd.DataFrame({"X": np.random.normal(1, 0.5, 1000),
                         "Y": np.random.normal(1, 0.5, 1000),
                         "Z": np.random.normal(1, 0.5, 1000),
                         "W": np.random.normal(1, 0.5, 1000)})
    data = gate._dim_reduction(data=data)
    assert gate.x == "UMAP1"
    assert gate.y == "UMAP2"
    assert data.shape == (1000, 6)
    assert all([f"UMAP{i+1}" in data.columns for i in range(2)])


@pytest.mark.parametrize("d", ["++", "--", "+-", "+++", "+ -"])
def test_threshold_add_child_invalid_1d(d):
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              x="X")
    child = ChildThreshold(name="test child",
                           definition=d,
                           geom=ThresholdGeom(x="X", x_threshold=0.56, y_threshold=0.75))
    with pytest.raises(AssertionError) as err:
        threshold.add_child(child)
    assert str(err.value) == "Invalid child definition, should be either '+' or '-'"



@pytest.mark.parametrize("d", ["+", "-", "+--", "+++", "+ -"])
def test_threshold_add_child_invalid_2d(d):
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              x="X",
                              y="Y")
    child = ChildThreshold(name="test child",
                           definition=d,
                           geom=ThresholdGeom(x_threshold=0.56, y_threshold=0.75))
    with pytest.raises(AssertionError) as err:
        threshold.add_child(child)
    assert str(err.value) == "Invalid child definition, should be one of: '++', '+-', '-+', or '--'"


def test_threshold_add_child():
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              x="X",
                              y="Y",
                              preprocessing=dict(transform_x="logicle"))
    child = ChildThreshold(name="test child",
                           definition="++",
                           geom=ThresholdGeom(x_threshold=0.56, y_threshold=0.75))
    threshold.add_child(child)
    assert len(threshold.children)
    assert threshold.children[0].geom.x == threshold.x
    assert threshold.children[0].geom.y == threshold.y
    assert threshold.children[0].geom.transform_x == "logicle"
    assert not threshold.children[0].geom.transform_y


def norm(x):
    return list(map(lambda i: (i - min(x)) / (max(x) - min(x)), x))


def test_create_signature():
    d = {"x": [15, 22, 80, 32],
         "y": [55, 32, 10, 11],
         "z": [42, 87, 91, 10]}
    d_norm = {k: norm(x) for k, x in d.items()}
    example = pd.DataFrame(d)
    x = create_signature(example)
    y = create_signature(example, summary_method=np.mean)
    z = create_signature(example, idx=[1, 2], summary_method=np.mean)
    assert isinstance(x, dict)
    assert isinstance(y, dict)
    assert isinstance(z, dict)
    for i in ["x", "y", "z"]:
        assert pytest.approx(x.get(i), 0.001) == np.median(d_norm.get(i))
        assert pytest.approx(y.get(i), 0.001) == np.mean(d_norm.get(i))
        assert pytest.approx(z.get(i), 0.001) == np.mean(np.array(d_norm.get(i))[[1, 2]])
