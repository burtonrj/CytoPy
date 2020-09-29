from ...data.populations import Population, Threshold, Polygon
from ...data.gates import create_signature, _merge, ChildDefinition, population_likeness, Gate, PreProcess, PostProcess
from .test_population import generate_polygons
import pandas as pd
import numpy as np
import pytest


def create_population(name: str,
                      geom: Threshold or Polygon,
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
                            geom: Threshold or Polygon,
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
    template = create_child_definition(name="template", geom=Threshold())
    pop = create_population(name="test", geom=Threshold())
    score = population_likeness(pop, template)
    assert score == 0.
    template = create_child_definition(name="template", geom=Threshold(), signature={"x": 0.15, "y": 0.5, "z": 0.2})
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
    gate.children.append(create_child_definition(name="test", geom=Threshold()))
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
    gate.children.append(create_child_definition(name="test1", geom=Threshold()))
    gate.children.append(create_child_definition(name="test2", geom=Threshold()))
    with pytest.warns(UserWarning) as warn:
        gate.label_children({"test1": None})
    assert str(warn.list[0].message) == "The following populations are not in labels and will be dropped: ['test2']"


def test_label_children_err_binary():
    gate = create_gate(binary=True, shape="polygon")
    gate.children.append(create_child_definition(name="test1", geom=Polygon()))
    gate.children.append(create_child_definition(name="test2", geom=Polygon()))
    with pytest.raises(AssertionError) as exp:
        gate.label_children(labels={"test1": None, "test2": None})
    assert str(exp.value) == "Non-threshold binary gate's should only have a single population"


def test_label_children_err_threhsold_binary():
    gate = create_gate(binary=True, shape="threshold")
    gate.children.append(create_child_definition(name="test1", geom=Threshold()))
    gate.children.append(create_child_definition(name="test2", geom=Threshold()))
    with pytest.raises(AssertionError) as exp:
        gate.label_children(labels={"test1": None, "test2": None})
    assert str(exp.value) == "For a binary threshold gate, labels should be provided with the keys: '+' and '-'"


def test_label_children_err_threhsold_nonbinary():
    gate = create_gate(binary=False, shape="threshold")
    gate.children.append(create_child_definition(name="test1", geom=Threshold()))
    gate.children.append(create_child_definition(name="test2", geom=Threshold()))
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


def test_init_method():
    pass


def test_transform():
    pass


def test_downsample():
    pass


def test_upsample():
    pass


def test_add_child():
    pass


def test_label_binary_threshold():
    pass


def test_label_threshold():
    pass


def test_label_binary_other():
    pass


def test_match_to_children():
    pass


def test_compare_populations():
    pass


def test_save():
    pass
