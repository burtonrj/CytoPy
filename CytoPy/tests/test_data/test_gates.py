from ...data.populations import Population, Threshold, Polygon
from ...data.gates import create_signature, _merge, ChildDefinition, population_likeness, Gate
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
    return list(map(lambda i: (i - min(x))/(max(x) - min(x)), x))


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


def test_gate_init():
    pass


def test_clear_children():
    pass


def test_label_children():
    pass


def test_scale():
    pass


def test_dim_reduction():
    pass


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



