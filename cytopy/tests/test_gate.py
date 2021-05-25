from cytopy.data import gate
from cytopy.data.geometry import *
from scipy.spatial.distance import euclidean
from shapely.geometry import Polygon
from sklearn.datasets import make_blobs
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest

np.random.seed(42)


def test_child_init():
    test_child = gate.Child(name="test",
                            signature={"x": 2423, "y": 2232, "z": 4543})
    assert test_child.name == "test"
    assert test_child.signature.get("x") == 2423
    assert test_child.signature.get("y") == 2232
    assert test_child.signature.get("z") == 4543


def test_childthreshold_init():
    test_child = gate.ChildThreshold(name="test",
                                     signature={"x": 2423, "y": 2232, "z": 4543},
                                     definition="+",
                                     geom=ThresholdGeom(x="x",
                                                        y="y",
                                                        x_threshold=0.5,
                                                        y_threshold=0.5,
                                                        transform_x="logicle",
                                                        transform_y="logicle"))
    assert test_child.name == "test"
    assert test_child.signature.get("x") == 2423
    assert test_child.signature.get("y") == 2232
    assert test_child.signature.get("z") == 4543
    assert test_child.definition == "+"
    assert test_child.geom.x == "x"
    assert test_child.geom.y == "y"
    assert test_child.geom.x_threshold == 0.5
    assert test_child.geom.y_threshold == 0.5
    assert test_child.geom.transform_x == "logicle"
    assert test_child.geom.transform_x == "logicle"


@pytest.mark.parametrize("definition,expected", [("+", True),
                                                 ("-", False)])
def test_childthreshold_match_definition_1d(definition, expected):
    test_child = gate.ChildThreshold(name="test",
                                     signature={"x": 2423, "y": 2232, "z": 4543},
                                     definition=definition,
                                     geom=ThresholdGeom(x="x",
                                                        y="y",
                                                        x_threshold=0.5,
                                                        y_threshold=0.5,
                                                        transform_x="logicle",
                                                        transform_y="logicle"))
    assert test_child.match_definition("+") == expected


@pytest.mark.parametrize("definition,expected", [("++", True),
                                                 ("--", False),
                                                 ("++,+-", True),
                                                 ("--,-+", False),
                                                 ("+-,-+,++", True)])
def test_childthreshold_match_definition_2d(definition, expected):
    test_child = gate.ChildThreshold(name="test",
                                     signature={"x": 2423, "y": 2232, "z": 4543},
                                     definition=definition,
                                     geom=ThresholdGeom(x="x",
                                                        y="y",
                                                        x_threshold=0.5,
                                                        y_threshold=0.5,
                                                        transform_x="logicle",
                                                        transform_y="logicle"))
    assert test_child.match_definition("++") == expected


def test_childpolygon_init():
    test_child = gate.ChildPolygon(name="test",
                                   signature={"x": 2423, "y": 2232, "z": 4543},
                                   geom=PolygonGeom(x="x", y="y"))
    assert test_child.name == "test"
    assert test_child.signature.get("x") == 2423
    assert test_child.signature.get("y") == 2232
    assert test_child.signature.get("z") == 4543
    assert test_child.geom.x == "x"
    assert test_child.geom.y == "y"


@pytest.mark.parametrize("klass,method", [(gate.Gate, "manual"),
                                          (gate.ThresholdGate, "density"),
                                          (gate.PolygonGate, "manual"),
                                          (gate.EllipseGate, "GaussianMixture"),
                                          (gate.EllipseGate, "SMM")])
def test_gate_init(klass, method):
    g = klass(gate_name="test",
              parent="test parent",
              x="X",
              y="Y",
              method=method,
              dim_reduction=dict(method="UMAP", kwargs={"n_neighbours": 100}))
    assert g.gate_name == "test"
    assert g.parent == "test parent"
    assert g.x == "X"
    assert g.y == "Y"
    assert g.method == method
    assert g.dim_reduction.get("method") == "UMAP"
    assert g.dim_reduction.get("kwargs").get("n_neighbours") == 100


def test_transform_none():
    g = gate.Gate(gate_name="test",
                  parent="test parent",
                  x="X",
                  y="Y",
                  method="manual")
    data = pd.DataFrame({"X": np.random.normal(1, scale=0.5, size=1000),
                         "Y": np.random.normal(1, scale=0.5, size=1000)})
    transformed = g.transform(data)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == 1000
    assert transformed.shape[1] == 2
    for i in ["X", "Y"]:
        assert transformed[i].mean() == pytest.approx(1., 0.1)
        assert transformed[i].std() == pytest.approx(0.5, 0.1)


def test_transform_x():
    g = gate.Gate(gate_name="test",
                  parent="test parent",
                  x="X",
                  y="Y",
                  method="manual",
                  transform_x="logicle")
    data = pd.DataFrame({"X": np.random.normal(1, scale=0.5, size=1000),
                         "Y": np.random.normal(1, scale=0.5, size=1000)})
    transformed = g.transform(data)
    assert isinstance(transformed, pd.DataFrame)
    assert transformed.shape[0] == 1000
    assert transformed.shape[1] == 2
    assert transformed["X"].mean() != pytest.approx(1., 0.1)
    assert transformed["X"].std() != pytest.approx(0.5, 0.1)
    assert transformed["Y"].mean() == pytest.approx(1., 0.1)
    assert transformed["Y"].std() == pytest.approx(0.5, 0.1)


def test_transform_xy():
    g = gate.Gate(gate_name="test",
                  parent="test parent",
                  x="X",
                  y="Y",
                  method="manual",
                  transform_x="logicle",
                  transform_y="logicle")
    data = pd.DataFrame({"X": np.random.normal(1, scale=0.5, size=1000),
                         "Y": np.random.normal(1, scale=0.5, size=1000)})
    transformed = g.transform(data)
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
                                    {"method": "density"}])
def test_downsample(kwargs):
    g = gate.Gate(gate_name="test",
                  parent="test parent",
                  x="X",
                  y="Y",
                  method="manual",
                  sampling=kwargs)
    data = pd.DataFrame({"X": np.random.normal(1, scale=0.5, size=1000),
                         "Y": np.random.normal(1, scale=0.5, size=1000)})
    sample = g._downsample(data=data)
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
    g = gate.Gate(gate_name="test",
                  parent="test parent",
                  x="X",
                  y="Y",
                  method="manual",
                  sampling={"method": "uniform",
                            "frac": 0.5})
    sample = g._downsample(data=data)
    sample_labels = labels[sample.index.values]
    pops = list()
    for x in np.unique(sample_labels):
        idx = sample.index.values[np.where(sample_labels == x)[0]]
        pops.append(gate.Population(population_name=f"Pop_{x}",
                                    parent="root",
                                    index=idx[:498]))
    pops = g._upsample(data=data, sample=sample, populations=pops)
    assert isinstance(pops, list)
    assert all([isinstance(p, gate.Population) for p in pops])
    assert all([len(p.index) == 1000 for p in pops])
    for x in np.unique(labels):
        p = [i for i in pops if i.population_name == f"Pop_{x}"][0]
        assert np.array_equal(p.index, np.where(labels == x)[0])


def test_dim_reduction():
    g = gate.Gate(gate_name="test",
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
    data = g._dim_reduction(data=data)
    assert g.x == "UMAP1"
    assert g.y == "UMAP2"
    assert data.shape == (1000, 6)
    assert all([f"UMAP{i + 1}" in data.columns for i in range(2)])


@pytest.mark.parametrize("d", ["++", "--", "+-", "+++", "+ -"])
def test_threshold_add_child_invalid_1d(d):
    threshold = gate.ThresholdGate(gate_name="test",
                                   parent="test parent",
                                   method="manual",
                                   x="X")
    child = gate.ChildThreshold(name="test child",
                                definition=d,
                                geom=ThresholdGeom(x="X", x_threshold=0.56, y_threshold=0.75))
    with pytest.raises(AssertionError) as err:
        threshold.add_child(child)
    assert str(err.value) == "Invalid child definition, should be either '+' or '-'"


@pytest.mark.parametrize("d", ["+", "-", "+--", "+++", "+ -"])
def test_threshold_add_child_invalid_2d(d):
    threshold = gate.ThresholdGate(gate_name="test",
                                   parent="test parent",
                                   x="X",
                                   y="Y",
                                   method="manual")
    child = gate.ChildThreshold(name="test child",
                                definition=d,
                                geom=ThresholdGeom(x_threshold=0.56, y_threshold=0.75))
    with pytest.raises(AssertionError) as err:
        threshold.add_child(child)
    assert str(err.value) == "Invalid child definition, should be one of: '++', '+-', '-+', or '--'"


def test_threshold_add_child():
    threshold = gate.ThresholdGate(gate_name="test",
                                   parent="test parent",
                                   x="X",
                                   y="Y",
                                   method="manual",
                                   transform_x="logicle")
    child = gate.ChildThreshold(name="test child",
                                definition="++",
                                geom=ThresholdGeom(x_threshold=0.56, y_threshold=0.75))
    threshold.add_child(child)
    assert len(threshold.children)
    assert threshold.children[0].geom.x == threshold.x
    assert threshold.children[0].geom.y == threshold.y
    assert threshold.children[0].geom.transform_x == "logicle"
    assert not threshold.children[0].geom.transform_y


def test_threshold_match_children_1d():
    threshold = gate.ThresholdGate(gate_name="test",
                                   parent="test parent",
                                   x="X",
                                   method="density")
    data = np.random.normal(loc=1., scale=1.5, size=1000)
    threshold.add_child(gate.ChildThreshold(name="positive",
                                            definition="+",
                                            geom=ThresholdGeom(x_threshold=0.5)))
    threshold.add_child(gate.ChildThreshold(name="negative",
                                            definition="-",
                                            geom=ThresholdGeom(x_threshold=0.5)))
    pos = gate.Population(population_name="p1",
                          parent="root",
                          definition="+",
                          geom=ThresholdGeom(x_threshold=0.6),
                          index=data[np.where(data >= 0.6)])
    neg = gate.Population(population_name="p2",
                          parent="root",
                          definition="-",
                          geom=ThresholdGeom(x_threshold=0.6),
                          index=data[np.where(data >= 0.6)])
    pops = threshold._match_to_children([neg, pos])
    pos = [p for p in pops if p.definition == "+"][0]
    assert pos.population_name == "positive"
    neg = [p for p in pops if p.definition == "-"][0]
    assert neg.population_name == "negative"


def test_threshold_match_children_2d():
    threshold = gate.ThresholdGate(gate_name="test",
                                   parent="test parent",
                                   x="X",
                                   y="Y",
                                   method="density")
    x = np.random.normal(loc=1., scale=1.5, size=1000)
    y = np.random.normal(loc=1., scale=1.5, size=1000)
    data = pd.DataFrame({"X": x, "Y": y})
    threshold.add_child(gate.ChildThreshold(name="positive",
                                            definition="++,+-",
                                            geom=ThresholdGeom(x_threshold=0.5)))
    threshold.add_child(gate.ChildThreshold(name="negative",
                                            definition="--,-+",
                                            geom=ThresholdGeom(x_threshold=0.5)))
    pos = gate.Population(population_name="p1",
                          parent="root",
                          definition="++",
                          geom=ThresholdGeom(x_threshold=0.6),
                          index=data[data.X >= 0.6].index.values)
    neg = gate.Population(population_name="p2",
                          parent="root",
                          definition="--,-+",
                          geom=ThresholdGeom(x_threshold=0.6),
                          index=data[data.X < 0.6].index.values)
    pops = threshold._match_to_children([neg, pos])
    pos = [p for p in pops if p.definition == "++"][0]
    assert pos.population_name == "positive"
    neg = [p for p in pops if p.definition == "--,-+"][0]
    assert neg.population_name == "negative"


def test_threshold_1d():
    x = np.random.normal(loc=1., scale=1.5, size=1000)
    data = pd.DataFrame({"X": x})
    results = gate.threshold_1d(data=data, x="X", x_threshold=0.5)
    assert len(results.keys()) == 2
    assert all(isinstance(df, pd.DataFrame) for df in results.values())
    assert len(np.where(x >= 0.5)[0]) == results.get("+").shape[0]
    assert len(np.where(x < 0.5)[0]) == results.get("-").shape[0]


def test_threshold_2d():
    x = np.random.normal(loc=1., scale=1.5, size=1000)
    y = np.random.normal(loc=1., scale=1.5, size=1000)
    data = pd.DataFrame({"X": x,
                         "Y": y})
    results = gate.threshold_2d(data=data, x="X", y="Y", x_threshold=0.5, y_threshold=0.5)
    assert len(results.keys()) == 4
    assert all(isinstance(df, pd.DataFrame) for df in results.values())
    x_pos, y_pos = np.where(x >= 0.5)[0], np.where(y >= 0.5)[0]
    x_neg, y_neg = np.where(x < 0.5)[0], np.where(y < 0.5)[0]
    assert len(np.intersect1d(x_pos, y_pos)) == results.get("++").shape[0]
    assert len(np.intersect1d(x_pos, y_neg)) == results.get("+-").shape[0]
    assert len(np.intersect1d(x_neg, y_pos)) == results.get("-+").shape[0]
    assert len(np.intersect1d(x_neg, y_neg)) == results.get("--").shape[0]


def test_smoothed_peak_finding():
    n1 = np.random.normal(loc=0.2, scale=1, size=500)
    n2 = np.random.normal(loc=2.5, scale=0.2, size=250)
    n3 = np.random.normal(loc=6.5, scale=0.5, size=500)
    data = np.hstack([n1, n2, n3])
    smoothed, peaks = gate.smoothed_peak_finding(p=data)
    assert isinstance(smoothed, np.ndarray)
    assert isinstance(peaks, np.ndarray)
    assert len(peaks) == 2


def test_find_local_minima():
    n1 = np.random.normal(loc=2, scale=1, size=1000)
    n2 = np.random.normal(loc=10, scale=0.5, size=1000)
    data = np.hstack([n1, n2])
    x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(data).evaluate()
    peak1 = np.where(y == np.max(y[np.where(x < 6)]))[0][0]
    peak2 = np.where(y == np.max(y[np.where(x > 6)]))[0][0]
    minima = x[np.where(y == np.min(y[np.where((x > 4) & (x < 7))]))[0][0]]
    assert gate.find_local_minima(p=y, x=x, peaks=np.array([peak1, peak2])) == minima


def test_find_inflection_point():
    np.random.seed(42)
    n1 = np.random.normal(loc=2, scale=1, size=1000)
    x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(n1).evaluate()
    inflection_point = gate.find_inflection_point(x=x, p=y, peak_idx=int(np.argmax(y)),
                                                  incline=False)
    plt.plot(x, y)
    plt.axvline(inflection_point, c="r")
    plt.title("Test inflection point; incline=False")
    plt.show()
    assert 3 < inflection_point < 4
    inflection_point = gate.find_inflection_point(x=x, p=y, peak_idx=int(np.argmax(y)),
                                                  incline=True)
    plt.plot(x, y)
    plt.axvline(inflection_point, c="r")
    plt.title("Test inflection point; incline=True")
    plt.show()
    assert 0 < inflection_point < 1


@pytest.mark.parametrize("yeo_johnson", [True, False])
def test_threshold_fit_1d(yeo_johnson):
    np.random.seed(42)
    n1 = np.random.normal(loc=0.2, scale=1, size=500)
    n2 = np.random.normal(loc=2.5, scale=0.2, size=250)
    n3 = np.random.normal(loc=6.5, scale=0.5, size=500)
    data = pd.DataFrame({"X": np.hstack([n1, n2, n3])})
    method_kwargs = {"yeo_johnson": yeo_johnson}
    threshold = gate.ThresholdGate(gate_name="test",
                                   parent="test parent",
                                   x="X",
                                   method="density",
                                   method_kwargs=method_kwargs)
    threshold.fit(data=data)
    assert len(threshold.children) == 2
    assert threshold.children[0].geom.x_threshold == threshold.children[1].geom.x_threshold
    assert round(threshold.children[0].geom.x_threshold) == 4
    assert all([i in [c.definition for c in threshold.children] for i in ["+", "-"]])


@pytest.mark.parametrize("yeo_johnson", [True, False])
def test_threshold_fit_2d(yeo_johnson):
    data, labels = make_blobs(n_samples=3000,
                              n_features=2,
                              centers=[(1., 1.), (1., 5.), (5., 0.2)],
                              random_state=42)
    data = pd.DataFrame({"X": data[:, 0], "Y": data[:, 1]})
    method_kwargs = {"yeo_johnson": yeo_johnson}
    threshold = gate.ThresholdGate(gate_name="test",
                                   parent="test parent",
                                   x="X",
                                   y="Y",
                                   method="density",
                                   method_kwargs=method_kwargs)
    threshold.fit(data)
    assert len(threshold.children) == 4
    assert len(set([c.geom.x_threshold for c in threshold.children])) == 1
    assert len(set([c.geom.y_threshold for c in threshold.children])) == 1
    assert all([i in [c.definition for c in threshold.children] for i in ["++", "--",
                                                                          "+-", "-+"]])
    assert 2 < threshold.children[0].geom.x_threshold < 4
    assert 2 < threshold.children[0].geom.y_threshold < 4


@pytest.mark.parametrize("yeo_johnson", [True, False])
def test_threshold_predict_1d(yeo_johnson):
    n1 = np.random.normal(loc=0.2, scale=1, size=500)
    n2 = np.random.normal(loc=2.5, scale=0.2, size=250)
    n3 = np.random.normal(loc=6.5, scale=0.5, size=500)
    data = pd.DataFrame({"X": np.hstack([n1, n2, n3])})
    method_kwargs = {"yeo_johnson": yeo_johnson}
    threshold = gate.ThresholdGate(gate_name="test",
                                   parent="test parent",
                                   x="X",
                                   method="density",
                                   method_kwargs=method_kwargs)
    threshold.fit(data=data)
    new_data = pd.DataFrame({"X": np.hstack([np.random.normal(loc=0.2, scale=1, size=500),
                                             np.random.normal(loc=6.5, scale=0.5, size=500)])})
    pops = threshold.predict(new_data)
    assert len(pops) == 2
    assert all([isinstance(p, gate.Population) for p in pops])
    assert all([isinstance(p.geom, ThresholdGeom) for p in pops])
    assert all([p.geom.x == threshold.x for p in pops])
    assert all([p.geom.y == threshold.y for p in pops])
    assert all(p.geom.transform_x == threshold.transform_x for p in pops)
    assert all(p.geom.transform_y == threshold.transform_y for p in pops)
    assert all(i in [p.definition for p in pops] for i in ["+", "-"])
    neg_idx = new_data[new_data.X < threshold.children[0].geom.x_threshold].index.values
    pos_idx = new_data[new_data.X >= threshold.children[0].geom.x_threshold].index.values
    pos_pop = [p for p in pops if p.definition == "+"][0]
    neg_pop = [p for p in pops if p.definition == "-"][0]
    assert np.array_equal(neg_pop.index, neg_idx)
    assert np.array_equal(pos_pop.index, pos_idx)


@pytest.mark.parametrize("yeo_johnson", [True, False])
def test_threshold_predict_2d(yeo_johnson):
    data, _ = make_blobs(n_samples=3000,
                         n_features=2,
                         centers=[(1., 1.), (1., 5.), (5., 0.2)],
                         random_state=42)
    data = pd.DataFrame({"X": data[:, 0], "Y": data[:, 1]})
    method_kwargs = {"yeo_johnson": yeo_johnson}
    threshold = gate.ThresholdGate(gate_name="test",
                                   parent="test parent",
                                   x="X",
                                   y="Y",
                                   method="density",
                                   method_kwargs=method_kwargs)
    threshold.fit(data=data)
    new_data, _ = make_blobs(n_samples=3000,
                             n_features=2,
                             centers=[(1., 1.), (5., 0.2)],
                             random_state=42)
    new_data = pd.DataFrame({"X": new_data[:, 0], "Y": new_data[:, 1]})
    pops = threshold.predict(new_data)
    assert len(pops) == 4
    assert all([isinstance(p, gate.Population) for p in pops])
    assert all([isinstance(p.geom, ThresholdGeom) for p in pops])
    assert all([p.geom.x == threshold.x for p in pops])
    assert all([p.geom.y == threshold.y for p in pops])
    assert all(p.geom.transform_x == threshold.transform_x for p in pops)
    assert all(p.geom.transform_y == threshold.transform_y for p in pops)
    assert all(i in [p.definition for p in pops] for i in ["++", "--", "-+", "+-"])
    neg_idx = new_data[(new_data.X < threshold.children[0].geom.x_threshold) &
                       (new_data.Y < threshold.children[0].geom.y_threshold)].index.values
    pos_idx = new_data[(new_data.X >= threshold.children[0].geom.x_threshold) &
                       (new_data.Y >= threshold.children[0].geom.y_threshold)].index.values
    negpos_idx = new_data[(new_data.X < threshold.children[0].geom.x_threshold) &
                          (new_data.Y >= threshold.children[0].geom.y_threshold)].index.values
    posneg_idx = new_data[(new_data.X >= threshold.children[0].geom.x_threshold) &
                          (new_data.Y < threshold.children[0].geom.y_threshold)].index.values
    pos_pop = [p for p in pops if p.definition == "++"][0]
    neg_pop = [p for p in pops if p.definition == "--"][0]
    posneg_pop = [p for p in pops if p.definition == "+-"][0]
    negpos_pop = [p for p in pops if p.definition == "-+"][0]
    assert np.array_equal(neg_pop.index, neg_idx)
    assert np.array_equal(pos_pop.index, pos_idx)
    assert np.array_equal(negpos_pop.index, negpos_idx)
    assert np.array_equal(posneg_pop.index, posneg_idx)


@pytest.mark.parametrize("yeo_johnson", [True, False])
def test_threshold_fit_predict_1d(yeo_johnson):
    n1 = np.random.normal(loc=0.2, scale=1, size=500)
    n2 = np.random.normal(loc=2.5, scale=0.2, size=250)
    n3 = np.random.normal(loc=6.5, scale=0.5, size=500)
    data = pd.DataFrame({"X": np.hstack([n1, n2, n3])})
    method_kwargs = {"yeo_johnson": yeo_johnson}
    threshold = gate.ThresholdGate(gate_name="test",
                                   parent="test parent",
                                   x="X",
                                   method="density",
                                   method_kwargs=method_kwargs)
    threshold.fit(data=data)
    threshold.label_children({"+": "Positive",
                              "-": "Negative"})
    new_data = pd.DataFrame({"X": np.hstack([np.random.normal(loc=0.2, scale=1, size=200),
                                             np.random.normal(loc=6.5, scale=0.5, size=1000)])})
    pops = threshold.fit_predict(new_data)
    assert len(pops) == 2
    assert all([isinstance(p, gate.Population) for p in pops])
    assert all([isinstance(p.geom, ThresholdGeom) for p in pops])
    assert all([p.geom.x == threshold.x for p in pops])
    assert all([p.geom.y == threshold.y for p in pops])
    assert all(p.geom.transform_x == threshold.transform_x for p in pops)
    assert all(p.geom.transform_y == threshold.transform_y for p in pops)
    assert all(i in [p.definition for p in pops] for i in ["+", "-"])
    pos_pop = [p for p in pops if p.definition == "+"][0]
    assert pos_pop.population_name == "Positive"
    neg_pop = [p for p in pops if p.definition == "-"][0]
    assert neg_pop.population_name == "Negative"
    assert len(pos_pop.index) > len(neg_pop.index)
    assert len(pos_pop.index) > 800
    assert len(neg_pop.index) < 300


@pytest.mark.parametrize("yeo_johnson", [True, False])
def test_threshold_fit_predict_2d(yeo_johnson):
    data, _ = make_blobs(n_samples=4000,
                         n_features=2,
                         centers=[(1., 1.), (1., 7.), (7., 2.), (7., 6.2)],
                         random_state=42)
    data = pd.DataFrame({"X": data[:, 0], "Y": data[:, 1]})
    method_kwargs = {"yeo_johnson": yeo_johnson}
    threshold = gate.ThresholdGate(gate_name="test",
                                   parent="test parent",
                                   x="X",
                                   y="Y",
                                   method="density",
                                   method_kwargs=method_kwargs)
    threshold.fit(data)
    threshold.label_children({"++": "Top left",
                              "--": "Other",
                              "-+": "Other",
                              "+-": "Other"})
    data, _ = make_blobs(n_samples=3000,
                         n_features=2,
                         centers=[(1., 1.), (1., 7.), (7., 6.2)],
                         random_state=42)
    data = pd.DataFrame({"X": data[:, 0], "Y": data[:, 1]})
    pops = threshold.fit_predict(data=data)
    assert len(pops) == 2
    assert all([isinstance(p, gate.Population) for p in pops])
    assert all([isinstance(p.geom, ThresholdGeom) for p in pops])
    assert all([p.geom.x == threshold.x for p in pops])
    assert all([p.geom.y == threshold.y for p in pops])
    assert all(p.geom.transform_x == threshold.transform_x for p in pops)
    assert all(p.geom.transform_y == threshold.transform_y for p in pops)
    top_left = [p for p in pops if p.population_name == "Top left"][0]
    other = [p for p in pops if p.population_name == "Other"][0]
    assert top_left.definition == "++"
    assert {"+-", "-+", "--"} == set(other.definition.split(","))
    assert len(top_left.index) < len(other.index)
    assert len(top_left.index) > 900
    assert len(other.index) > 1900


def create_polygon_gate(klass,
                        method: str,
                        **kwargs):
    g = klass(gate_name="test",
              parent="test parent",
              x="X",
              y="Y",
              method=method,
              method_kwargs={k: v for k, v in kwargs.items()})
    return g


def test_polygon_add_child():
    g = create_polygon_gate(klass=gate.PolygonGate, method="MiniBatchKMeans")
    data, _ = make_blobs(n_samples=3000,
                         n_features=2,
                         centers=[(1., 1.), (1., 7.), (7., 6.2)],
                         random_state=42)
    g.add_child(gate.ChildPolygon(name="test",
                                  geom=PolygonGeom(x_values=np.linspace(0, 1000, 1).tolist(),
                                                   y_values=np.linspace(0, 1000, 1).tolist())))
    assert len(g.children) == 1
    assert g.children[0].name == "test"
    assert g.children[0].geom.x == g.x
    assert g.children[0].geom.y == g.y
    assert g.children[0].geom.transform_x == g.transform_x
    assert g.children[0].geom.transform_y == g.transform_y


def test_polygon_generate_populations():
    data, labels = make_blobs(n_samples=4000,
                              n_features=2,
                              cluster_std=0.5,
                              centers=[(1., 1.), (1., 7.), (7., 6.2), (6., 1.)],
                              random_state=42)
    data = pd.DataFrame(data, columns=["X", "Y"])
    g = create_polygon_gate(klass=gate.PolygonGate, method="MiniBatchKMeans")
    polys = [Polygon([(-1., -1), (-1, 10), (3, 10), (3, -1), (-1, -1)]),
             Polygon([(4, -1), (8, -1), (8, 3.8), (4, 3.8), (4, -1)]),
             Polygon([(4, 4), (4, 10), (10, 10), (10, 4), (4, 4)])]
    pops = g._generate_populations(data=data,
                                   polygons=polys)
    assert len(pops) == 3
    assert all([isinstance(p, gate.Population) for p in pops])
    assert all([isinstance(p.geom, PolygonGeom) for p in pops])
    for p in pops:
        assert p.geom.x == g.x
        assert p.geom.y == g.y
        assert p.geom.transform_x == g.transform_x
        assert p.geom.transform_y == g.transform_y
        assert p.parent == "test parent"
    for name, n in zip(["A", "B", "C"], [2000, 1000, 1000]):
        p = [p for p in pops if p.population_name == name][0]
        assert len(p.index) == n
        assert len(p.geom.x_values) == 5
        assert len(p.geom.y_values) == 5


def test_polygon_match_to_children():
    data, labels = make_blobs(n_samples=5000,
                              n_features=2,
                              cluster_std=1,
                              centers=[(1., 1.), (10., 6.2), (1.5, 2.), (11, 7.), (11.5, 7.5)],
                              random_state=42)
    data_dict = [{"data": data[np.where(labels == i)],
                  "signature": pd.DataFrame(data[np.where(labels == i)], columns=["X", "Y"]).mean().to_dict(),
                  "poly": create_envelope(data[np.where(labels == i)][:, 0], data[np.where(labels == i)][:, 1])}
                 for i in range(5)]
    g = create_polygon_gate(klass=gate.PolygonGate, method="MiniBatchKMeans")
    for i in [0, 1]:
        g.add_child(gate.ChildPolygon(name=f"Child{i + 1}",
                                      signature=data_dict[i].get("signature"),
                                      geom=PolygonGeom(x_values=data_dict[i].get("poly").exterior.xy[0],
                                                       y_values=data_dict[i].get("poly").exterior.xy[1])))
    pops = g._generate_populations(data=pd.DataFrame(data, columns=["X", "Y"]),
                                   polygons=[x.get("poly") for x in data_dict[2:]])
    pops = g._match_to_children(pops)
    assert len(pops) == 2
    assert {p.population_name for p in pops} == {"Child1", "Child2"}
    assert 1800 < len([p for p in pops if p.population_name == "Child1"][0].index) < 2200
    assert 2800 < len([p for p in pops if p.population_name == "Child2"][0].index) < 4200


@pytest.mark.parametrize("gate", [create_polygon_gate(klass=gate.PolygonGate,
                                                      method="MiniBatchKMeans",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="GaussianMixture",
                                                      n_components=2),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="SMM",
                                                      n_components=2),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="BayesianGaussianMixture",
                                                      n_components=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="SpectralClustering",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="AgglomerativeClustering",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="Birch",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="MiniBatchKMeans",
                                                      n_clusters=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="GaussianMixture",
                                                      n_components=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="SMM",
                                                      n_components=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="BayesianGaussianMixture",
                                                      n_components=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="SpectralClustering",
                                                      n_clusters=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="AgglomerativeClustering",
                                                      n_clusters=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="Birch",
                                                      n_clusters=2,
                                                      yeo_johnson=True)])
def test_polygon_fit(gate):
    data, labels = make_blobs(n_samples=5000,
                              n_features=2,
                              cluster_std=1,
                              centers=[(1., 1.), (10., 6.2)],
                              random_state=42)
    data = pd.DataFrame(data, columns=["X", "Y"])
    gate.fit(data=data)
    assert len(gate.children) == 2
    centroids = [create_envelope(data.loc[np.where(labels == i)]["X"].values,
                                 data.loc[np.where(labels == i)]["Y"].values).centroid.xy for i in
                 np.unique(labels)]
    child_centroids = [create_envelope(c.geom.x_values, c.geom.y_values).centroid.xy
                       for c in gate.children]
    for c in child_centroids:
        distances = [abs(euclidean(c, centroid)) for centroid in centroids]
        assert sum([x <= 1. for x in distances]) == 1


@pytest.mark.parametrize("gate", [create_polygon_gate(klass=gate.PolygonGate,
                                                      method="MiniBatchKMeans",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="GaussianMixture",
                                                      n_components=2),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="SMM",
                                                      n_components=2),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="BayesianGaussianMixture",
                                                      n_components=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="SpectralClustering",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="AgglomerativeClustering",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="Birch",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="MiniBatchKMeans",
                                                      n_clusters=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="GaussianMixture",
                                                      n_components=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="SMM",
                                                      n_components=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="BayesianGaussianMixture",
                                                      n_components=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="SpectralClustering",
                                                      n_clusters=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="AgglomerativeClustering",
                                                      n_clusters=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="Birch",
                                                      n_clusters=2,
                                                      yeo_johnson=True)])
def test_polygon_predict(gate):
    data, labels = make_blobs(n_samples=5000,
                              n_features=2,
                              cluster_std=1,
                              centers=[(1., 1.), (10., 6.2)],
                              random_state=42)
    data = pd.DataFrame(data, columns=["X", "Y"])
    gate.fit(data=data)
    gate.label_children(labels={"A": "Pop1", "B": "Pop2"})
    data, labels = make_blobs(n_samples=5000,
                              n_features=2,
                              cluster_std=1,
                              centers=[(1., 1.)],
                              random_state=42)
    data = pd.DataFrame(data, columns=["X", "Y"])
    pops = gate.predict(data=data)
    assert len(pops) == 2
    assert {"Pop1", "Pop2"} == {p.population_name for p in pops}
    p1 = [p for p in pops if p.population_name == "Pop1"][0]
    p2 = [p for p in pops if p.population_name == "Pop2"][0]
    assert sum([4700 < pl < 5000 for pl in [len(p1.index), len(p2.index)]]) == 1


@pytest.mark.parametrize("gate", [create_polygon_gate(klass=gate.PolygonGate,
                                                      method="MiniBatchKMeans",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="GaussianMixture",
                                                      n_components=2),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="SMM",
                                                      n_components=2),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="BayesianGaussianMixture",
                                                      n_components=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="SpectralClustering",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="AgglomerativeClustering",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="Birch",
                                                      n_clusters=2),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="MiniBatchKMeans",
                                                      n_clusters=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="GaussianMixture",
                                                      n_components=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="SMM",
                                                      n_components=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.EllipseGate,
                                                      method="BayesianGaussianMixture",
                                                      n_components=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="SpectralClustering",
                                                      n_clusters=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="AgglomerativeClustering",
                                                      n_clusters=2,
                                                      yeo_johnson=True),
                                  create_polygon_gate(klass=gate.PolygonGate,
                                                      method="Birch",
                                                      n_clusters=2,
                                                      yeo_johnson=True)])
def test_polygon_fit_predict(gate):
    data, labels = make_blobs(n_samples=5000,
                              n_features=2,
                              cluster_std=1,
                              centers=[(1., 1.), (10., 6.2), (1.5, 2.), (11, 7.), (11.5, 7.5)],
                              random_state=42)
    data = pd.DataFrame(data, columns=["X", "Y"])
    training = data.loc[np.where((labels == 0) | (labels == 1))]
    testing = data.loc[np.where((labels == 2) | (labels == 3) | (labels == 4))]
    gate.fit(data=training)
    gate.label_children(labels={"A": "Pop1", "B": "Pop2"})
    pops = gate.fit_predict(testing)
    assert len(pops) == 2
    assert {p.population_name for p in pops} == {"Pop1", "Pop2"}
    assert sum([1900 < len(p.index) < 2100 for p in pops]) == 1
    assert sum([900 < len(p.index) < 1100 for p in pops]) == 1
