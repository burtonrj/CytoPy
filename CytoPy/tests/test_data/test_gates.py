from ...data.populations import Population
from ...data.geometry import ThresholdGeom, PolygonGeom
from ...data.gates import create_signature, ChildDefinition, population_likeness, Gate, PreProcess, PostProcess, DensityGate, \
    ManualGate, Analyst
from .test_population import generate_polygons
import pandas as pd
import numpy as np
import pytest


def create_population(name: str,
                      geom: ThresholdGeom or PolygonGeom,
                      idx: np.array or None = None,
                      signature: dict or None = None,
                      **kwargs):
    idx = idx or np.arange(0, 100, 1)
    signature = signature or {"x": 0.8, "y": 0.2, "z": 0.1}
    return Population(population_name=name,
                      geom=geom,
                      index=idx,
                      signature=signature,
                      **kwargs)


def create_child_definition(name: str,
                            geom: ThresholdGeom or PolygonGeom,
                            definition: str = "+",
                            signature: dict or None = None):
    signature = signature or {"x": 0.8, "y": 0.2, "z": 0.1}
    return ChildDefinition(population_name=name, geom=geom, definition=definition, signature=signature)


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


def test_pop_likeness():
    template = create_child_definition(name="template", geom=ThresholdGeom())
    pop = create_population(name="test", geom=ThresholdGeom())
    score = population_likeness(pop, template)
    assert score == 0.
    template = create_child_definition(name="template", geom=ThresholdGeom(), signature={"x": 0.15, "y": 0.5, "z": 0.2})
    score = population_likeness(pop, template)
    assert pytest.approx(0.7228, 0.001) == score
    poly1, poly2, poly3 = generate_polygons()
    template = create_child_definition(name="template", geom=poly1)
    p1 = create_population(name="p1", geom=poly2)
    p2 = create_population(name="p1", geom=poly3)
    p1_score = population_likeness(p1, template)
    p2_score = population_likeness(p2, template)
    assert p2_score > p1_score


def create_gate(**kwargs):
    shape = kwargs.pop("shape", "polygon")
    return Gate(gate_name="test",
                parent="root",
                shape=shape,
                x="x",
                y="y",
                **kwargs)


def test_gate_init():
    gate = create_gate()
    assert not gate.defined
    assert not gate.labelled


def test_clear_children():
    gate = create_gate()
    gate.children.append(create_child_definition(name="test", geom=ThresholdGeom()))
    assert len(gate.children) == 1
    gate.clear_children()
    assert len(gate.children) == 0
    assert not gate.defined
    assert not gate.labelled


def test_label_children_err():
    gate = create_gate()
    gate.labelled = True
    with pytest.raises(AssertionError) as exp:
        gate.label_children(labels={})
    assert str(exp.value) == "Children already labelled. To clear children and relabel call 'clear_children'"


def test_label_children_drop_msg():
    gate = create_gate()
    gate.children.append(create_child_definition(name="test1", geom=ThresholdGeom()))
    gate.children.append(create_child_definition(name="test2", geom=ThresholdGeom()))
    with pytest.warns(UserWarning) as warn:
        gate.label_children({"test1": None})
    assert str(warn.list[0].message) == "The following populations are not in labels and will be dropped: ['test2']"


def test_label_children_err_binary():
    gate = create_gate(binary=True, shape="polygon")
    gate.children.append(create_child_definition(name="test1", geom=PolygonGeom()))
    gate.children.append(create_child_definition(name="test2", geom=PolygonGeom()))
    with pytest.raises(AssertionError) as exp:
        gate.label_children(labels={"test1": None, "test2": None})
    assert str(exp.value) == "Non-threshold binary gate's should only have a single population"


def test_label_children_err_threhsold_binary():
    gate = create_gate(binary=True, shape="threshold")
    gate.children.append(create_child_definition(name="test1", geom=ThresholdGeom()))
    gate.children.append(create_child_definition(name="test2", geom=ThresholdGeom()))
    with pytest.raises(AssertionError) as exp:
        gate.label_children(labels={"test1": None, "test2": None})
    assert str(exp.value) == "For a binary threshold gate, labels should be provided with the keys: '+' and '-'"


def test_label_children_err_threhsold_nonbinary():
    gate = create_gate(binary=False, shape="threshold")
    gate.children.append(create_child_definition(name="test1", geom=ThresholdGeom()))
    gate.children.append(create_child_definition(name="test2", geom=ThresholdGeom()))
    with pytest.raises(AssertionError) as exp:
        gate.label_children(labels={"test1": None, "test2": None})
    assert str(exp.value) == "For a non-binary threshold gate, labels should be provided with the keys: '++', '-+', '+-' and '--'"


def test_dim_reduction():
    gate = create_gate(preprocessing=PreProcess(transform_x="logicle",
                                                transform_y="logicle",
                                                dim_reduction="UMAP"))
    data = pd.DataFrame({i: np.random.rand(1000) for i in ["x", "y", "z", "w"]})
    data = gate._dim_reduction(data)
    assert data.columns.tolist() == ["embedding1", "embedding2"]


@pytest.mark.parametrize("method, klass", [("DensityGate", DensityGate),
                                           ("ManualGate", ManualGate),
                                           ("HDBSCAN", Analyst)])
def test_init_method(method, klass):
    gate = create_gate(method=method)
    assert isinstance(gate._init_method(), klass)


def test_downsample_err():
    gate = create_gate(preprocessing=PreProcess(downsample_method="Unknown"))
    data = pd.DataFrame({i: np.random.rand(1000) for i in ["x", "y", "z", "w"]})
    with pytest.raises(ValueError) as exp:
        gate._downsample(data=data)
    assert str(exp.value) == "Invalid Gate: downsampling_method must be one of: uniform, density, or faithful"


@pytest.mark.parametrize("method,kwargs", [("uniform", {"sample_n": 100}),
                                           ("uniform", {"sample_n": 0.2}),
                                           ("density", {})])
def test_downsample(method, kwargs):
    gate = create_gate(preprocessing=PreProcess(downsample_method=method, downsample_kwargs=kwargs))
    data = pd.DataFrame({i: np.random.rand(10000) for i in ["x", "y", "z", "w"]})
    data, sample = gate._downsample(data=data)
    assert isinstance(data, pd.DataFrame)
    assert isinstance(sample, pd.DataFrame)
    assert data.shape[0] > sample.shape[0]


def test_add_child():
    gate = create_gate(shape="threshold")
    gate._add_child(population=create_population(name="test", geom=ThresholdGeom(), definition="+"))
    assert len(gate.children) == 1
    assert gate.children[0].population_name == "+"
    gate = create_gate(shape="polygon")
    gate._add_child(population=create_population(name="test", geom=ThresholdGeom(), definition="+"))
    assert len(gate.children) == 1
    assert gate.children[0].population_name == "test"


def test_label_binary_threshold():
    gate = create_gate(shape="threshold")
    gate.children.append(create_child_definition(name="pos", definition="+", geom=ThresholdGeom()))
    gate.children.append(create_child_definition(name="neg", definition="-", geom=ThresholdGeom()))
    pos = create_population(name="+", geom=ThresholdGeom(), definition="+")
    neg = create_population(name="-", geom=ThresholdGeom(), definition="-")
    new_children = gate._label_binary_threshold(new_children=[pos, neg])
    assert {p.population_name for p in new_children} == {"pos", "neg"}


def test_label_threshold():
    gate = create_gate(shape="threshold", binary=False)
    gate.children.append(create_child_definition(name="pos", definition="++", geom=ThresholdGeom()))
    gate.children.append(create_child_definition(name="neg", definition="-+,--,+-", geom=ThresholdGeom()))
    pos = create_population(name="++", geom=ThresholdGeom(), definition="++")
    neg1 = create_population(name="--", geom=ThresholdGeom(), definition="--")
    neg2 = create_population(name="-+", geom=ThresholdGeom(), definition="-+")
    neg3 = create_population(name="+-", geom=ThresholdGeom(), definition="+-")
    new_children = gate._label_threshold(new_children=[pos, neg1, neg2, neg3])
    assert {p.population_name for p in new_children} == {"pos", "neg"}


def test_label_binary_other():
    poly1, poly2, poly3 = generate_polygons()
    gate = create_gate(shape="polygon", binary=True)
    gate.children.append(create_child_definition(name="pos", definition="+", geom=poly1))
    new_populations = gate._label_binary_other(new_children=[create_population(name="poly2",
                                                                               geom=poly2),
                                                             create_population(name="poly3",
                                                                               geom=poly3)])
    assert isinstance(new_populations, list)
    assert len(new_populations) == 1
    assert new_populations[0].population_name == "pos"


def test_compare_populations():
    poly1, poly2, poly3 = generate_polygons()
    gate = create_gate(shape="polygon", binary=True)
    for name, geom in zip(["c1", "c2", "c3"], [poly1, poly2, poly3]):
        gate.children.append(create_child_definition(name=name, geom=geom))
    pops = [create_population(name=name, geom=geom) for name, geom in zip(["p1", "p2", "p3"], [poly1, poly2, poly3])]
    assignments = gate._compare_populations(new_children=pops)
    assert assignments == ["c1", "c2", "c3"]


def test_save_err():
    gate = create_gate(shape="polygon", binary=True)
    with pytest.raises(AssertionError) as exp:
        gate.save()
    assert str(exp.value) == "Gate test is newly created and has not been defined. " \
                             "Call 'label_children' to complete gating definition"
