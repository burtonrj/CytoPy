from ...data.gate import Gate, ThresholdGate, PolygonGate, EllipseGate, ChildThreshold, ChildPolygon, \
    create_signature, Population, threshold_1d, threshold_2d, smoothed_peak_finding, find_local_minima, \
    find_inflection_point
from ...data.geometry import ThresholdGeom, PolygonGeom
from sklearn.datasets import make_blobs
from KDEpy import FFTKDE
import pandas as pd
import numpy as np
import pytest

np.random.seed(42)


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
    assert all([f"UMAP{i + 1}" in data.columns for i in range(2)])


@pytest.mark.parametrize("d", ["++", "--", "+-", "+++", "+ -"])
def test_threshold_add_child_invalid_1d(d):
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              method="manual",
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
                              y="Y",
                              method="manual")
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
                              method="manual",
                              transformations={"x": "logicle"})
    child = ChildThreshold(name="test child",
                           definition="++",
                           geom=ThresholdGeom(x_threshold=0.56, y_threshold=0.75))
    threshold.add_child(child)
    assert len(threshold.children)
    assert threshold.children[0].geom.x == threshold.x
    assert threshold.children[0].geom.y == threshold.y
    assert threshold.children[0].geom.transform_x == "logicle"
    assert not threshold.children[0].geom.transform_y


def test_threshold_match_children_1d():
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              x="X",
                              method="density")
    data = np.random.normal(loc=1., scale=1.5, size=1000)
    threshold.add_child(ChildThreshold(name="positive",
                                       definition="+",
                                       geom=ThresholdGeom(x_threshold=0.5)))
    threshold.add_child(ChildThreshold(name="negative",
                                       definition="-",
                                       geom=ThresholdGeom(x_threshold=0.5)))
    pos = Population(population_name="p1",
                     parent="root",
                     definition="+",
                     geom=ThresholdGeom(x_threshold=0.6),
                     index=data[np.where(data >= 0.6)])
    neg = Population(population_name="p2",
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
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              x="X",
                              y="Y",
                              method="density")
    x = np.random.normal(loc=1., scale=1.5, size=1000)
    y = np.random.normal(loc=1., scale=1.5, size=1000)
    data = pd.DataFrame({"X": x, "Y": y})
    threshold.add_child(ChildThreshold(name="positive",
                                       definition="++,+-",
                                       geom=ThresholdGeom(x_threshold=0.5)))
    threshold.add_child(ChildThreshold(name="negative",
                                       definition="--,-+",
                                       geom=ThresholdGeom(x_threshold=0.5)))
    pos = Population(population_name="p1",
                     parent="root",
                     definition="++",
                     geom=ThresholdGeom(x_threshold=0.6),
                     index=data[data.X >= 0.6].index.values)
    neg = Population(population_name="p2",
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
    results = threshold_1d(data=data, x="X", x_threshold=0.5)
    assert len(results.keys()) == 2
    assert all(isinstance(df, pd.DataFrame) for df in results.values())
    assert len(np.where(x >= 0.5)[0]) == results.get("+").shape[0]
    assert len(np.where(x < 0.5)[0]) == results.get("-").shape[0]


def test_threshold_2d():
    x = np.random.normal(loc=1., scale=1.5, size=1000)
    y = np.random.normal(loc=1., scale=1.5, size=1000)
    data = pd.DataFrame({"X": x,
                         "Y": y})
    results = threshold_2d(data=data, x="X", y="Y", x_threshold=0.5, y_threshold=0.5)
    assert len(results.keys()) == 4
    assert all(isinstance(df, pd.DataFrame) for df in results.values())
    x_pos, y_pos = np.where(x >= 0.5)[0], np.where(y >= 0.5)[0]
    x_neg, y_neg = np.where(x < 0.5)[0], np.where(y < 0.5)[0]
    assert len(np.intersect1d(x_pos, y_pos)) == results.get("++").shape[0]
    assert len(np.intersect1d(x_pos, y_neg)) == results.get("+-").shape[0]
    assert len(np.intersect1d(x_neg, y_pos)) == results.get("-+").shape[0]
    assert len(np.intersect1d(x_neg, y_neg)) == results.get("--").shape[0]


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


def test_smoothed_peak_finding():
    n1 = np.random.normal(loc=0.2, scale=1, size=500)
    n2 = np.random.normal(loc=2.5, scale=0.2, size=250)
    n3 = np.random.normal(loc=6.5, scale=0.5, size=500)
    data = np.hstack([n1, n2, n3])
    smoothed, peaks = smoothed_peak_finding(p=data)
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
    assert find_local_minima(p=y, x=x, peaks=np.array([peak1, peak2])) == minima


def test_find_inflection_point():
    n1 = np.random.normal(loc=2, scale=1, size=1000)
    x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(n1).evaluate()
    inflection_point = find_inflection_point(x=x, p=y, peak_idx=np.argmax(y),
                                             incline=False)
    assert 3 < inflection_point < 4
    inflection_point = find_inflection_point(x=x, p=y, peak_idx=np.argmax(y),
                                             incline=True)
    assert 0 < inflection_point < 1


def test_threshold_fit_1d():
    n1 = np.random.normal(loc=0.2, scale=1, size=500)
    n2 = np.random.normal(loc=2.5, scale=0.2, size=250)
    n3 = np.random.normal(loc=6.5, scale=0.5, size=500)
    data = pd.DataFrame({"X": np.hstack([n1, n2, n3])})
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              x="X",
                              method="density")
    threshold.fit(data=data)
    assert len(threshold.children) == 2
    assert threshold.children[0].geom.x_threshold == threshold.children[1].geom.x_threshold
    assert round(threshold.children[0].geom.x_threshold) == 4
    assert all([i in [c.definition for c in threshold.children] for i in ["+", "-"]])


def test_threshold_fit_2d():
    data, labels = make_blobs(n_samples=3000,
                              n_features=2,
                              centers=[(1., 1.), (1., 5.), (5., 0.2)],
                              random_state=42)
    data = pd.DataFrame({"X": data[:, 0], "Y": data[:, 1]})
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              x="X",
                              y="Y",
                              method="density")
    threshold.fit(data)
    assert len(threshold.children) == 4
    assert len(set([c.geom.x_threshold for c in threshold.children])) == 1
    assert len(set([c.geom.y_threshold for c in threshold.children])) == 1
    assert all([i in [c.definition for c in threshold.children] for i in ["++", "--",
                                                                          "+-", "-+"]])
    assert 2 < threshold.children[0].geom.x_threshold < 4
    assert 2 < threshold.children[0].geom.y_threshold < 4


def test_threshold_predict_1d():
    n1 = np.random.normal(loc=0.2, scale=1, size=500)
    n2 = np.random.normal(loc=2.5, scale=0.2, size=250)
    n3 = np.random.normal(loc=6.5, scale=0.5, size=500)
    data = pd.DataFrame({"X": np.hstack([n1, n2, n3])})
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              x="X",
                              method="density")
    threshold.fit(data=data)
    new_data = pd.DataFrame({"X": np.hstack([np.random.normal(loc=0.2, scale=1, size=500),
                                             np.random.normal(loc=6.5, scale=0.5, size=500)])})
    pops = threshold.predict(new_data, parent="root")
    assert len(pops) == 2
    assert all([isinstance(p, Population) for p in pops])
    assert all([isinstance(p.geom, ThresholdGeom) for p in pops])
    assert all([p.geom.x == threshold.x for p in pops])
    assert all([p.geom.y == threshold.y for p in pops])
    assert all(p.geom.transform_x == threshold.transformations.get("x") for p in pops)
    assert all(p.geom.transform_y == threshold.transformations.get("y") for p in pops)
    assert all(i in [p.definition for p in pops] for i in ["+", "-"])
    neg_idx = new_data[new_data.X < threshold.children[0].geom.x_threshold].index.values
    pos_idx = new_data[new_data.X >= threshold.children[0].geom.x_threshold].index.values
    pos_pop = [p for p in pops if p.definition == "+"][0]
    neg_pop = [p for p in pops if p.definition == "-"][0]
    assert np.array_equal(neg_pop.index, neg_idx)
    assert np.array_equal(pos_pop.index, pos_idx)


def test_threshold_predict_2d():
    data, _ = make_blobs(n_samples=3000,
                         n_features=2,
                         centers=[(1., 1.), (1., 5.), (5., 0.2)],
                         random_state=42)
    data = pd.DataFrame({"X": data[:, 0], "Y": data[:, 1]})
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              x="X",
                              y="Y",
                              method="density")
    threshold.fit(data=data)
    new_data, _ = make_blobs(n_samples=3000,
                             n_features=2,
                             centers=[(1., 1.), (5., 0.2)],
                             random_state=42)
    new_data = pd.DataFrame({"X": new_data[:, 0], "Y": new_data[:, 1]})
    pops = threshold.predict(new_data, parent="root")
    assert len(pops) == 4
    assert all([isinstance(p, Population) for p in pops])
    assert all([isinstance(p.geom, ThresholdGeom) for p in pops])
    assert all([p.geom.x == threshold.x for p in pops])
    assert all([p.geom.y == threshold.y for p in pops])
    assert all(p.geom.transform_x == threshold.transformations.get("x") for p in pops)
    assert all(p.geom.transform_y == threshold.transformations.get("y") for p in pops)
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


def test_threshold_fit_predict_1d():
    n1 = np.random.normal(loc=0.2, scale=1, size=500)
    n2 = np.random.normal(loc=2.5, scale=0.2, size=250)
    n3 = np.random.normal(loc=6.5, scale=0.5, size=500)
    data = pd.DataFrame({"X": np.hstack([n1, n2, n3])})
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              x="X",
                              method="density")
    threshold.fit(data=data)
    threshold.label_children({"+": "Positive",
                              "-": "Negative"})
    new_data = pd.DataFrame({"X": np.hstack([np.random.normal(loc=0.2, scale=1, size=200),
                                             np.random.normal(loc=6.5, scale=0.5, size=1000)])})
    pops = threshold.fit_predict(new_data, parent="root")
    assert len(pops) == 2
    assert all([isinstance(p, Population) for p in pops])
    assert all([isinstance(p.geom, ThresholdGeom) for p in pops])
    assert all([p.geom.x == threshold.x for p in pops])
    assert all([p.geom.y == threshold.y for p in pops])
    assert all(p.geom.transform_x == threshold.transformations.get("x") for p in pops)
    assert all(p.geom.transform_y == threshold.transformations.get("y") for p in pops)
    assert all(i in [p.definition for p in pops] for i in ["+", "-"])
    pos_pop = [p for p in pops if p.definition == "+"][0]
    assert pos_pop.population_name == "Positive"
    neg_pop = [p for p in pops if p.definition == "-"][0]
    assert neg_pop.population_name == "Negative"
    assert len(pos_pop.index) > len(neg_pop.index)
    assert len(pos_pop.index) > 800
    assert len(neg_pop.index) < 300


def test_threshold_fit_predict_2d():
    data, _ = make_blobs(n_samples=4000,
                         n_features=2,
                         centers=[(1., 1.), (1., 7.), (7., 2.), (7., 6.2)],
                         random_state=42)
    data = pd.DataFrame({"X": data[:, 0], "Y": data[:, 1]})
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              x="X",
                              y="Y",
                              method="density")
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
    pops = threshold.fit_predict(data=data, parent="root")
    assert len(pops) == 2
    assert all([isinstance(p, Population) for p in pops])
    assert all([isinstance(p.geom, ThresholdGeom) for p in pops])
    assert all([p.geom.x == threshold.x for p in pops])
    assert all([p.geom.y == threshold.y for p in pops])
    assert all(p.geom.transform_x == threshold.transformations.get("x") for p in pops)
    assert all(p.geom.transform_y == threshold.transformations.get("y") for p in pops)
    top_left = [p for p in pops if p.population_name == "Top left"][0]
    other = [p for p in pops if p.population_name == "Other"][0]
    assert top_left.definition == "++"
    assert {"+-", "-+", "--"} == set(other.definition.split(","))
    assert len(top_left.index) < len(other.index)
    assert len(top_left.index) > 900
    assert len(other.index) > 1900


def create_polygon_gate():
    gate = PolygonGate(gate_name="test",
                       parent="test parent",
                       x="X",
                       y="Y",
                       method="MiniBatchKMeans")
    return gate


def test_polygon_add_child():
    gate = create_polygon_gate()
    gate.add_child(ChildPolygon(name="test",
                                geom=PolygonGeom(x_values=np.linspace(0, 1000, 1),
                                                 y_values=np.linspace(0, 1000, 1))))
    assert len(gate.children) == 1
    assert gate.children[0].name == "test"
    assert gate.children[0].geom.x == gate.x
    assert gate.children[0].geom.y == gate.y
    assert gate.children[0].geom.transform_x == gate.transformations.get("x", None)
    assert gate.children[0].geom.transform_y == gate.transformations.get("y", None)


def test_polygon_match_to_children():
    pass


def test_polygon_fit():
    pass


def test_polygon_predict():
    pass


def test_polygon_fit_predict():
    pass


def test_ellipse_add_child_invalid():
    pass


def test_ellipse_add_child():
    pass


def test_ellipse_match_to_children():
    pass


def test_ellipse_fit():
    pass


def test_ellipse_predict():
    pass


def test_ellipse_fit_predict():
    pass
