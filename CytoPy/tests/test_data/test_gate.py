from ...data.gate import Gate, ThresholdGate, PolygonGate, EllipseGate, ChildThreshold, ChildPolygon
from ...data.geometry import ThresholdGeom, PolygonGeom
import pytest


@pytest.mark.parametrize("klass", [Gate, ThresholdGate, PolygonGate, EllipseGate])
def test_gate_init(klass):
    gate = klass(gate_name="test",
                 parent="test parent",
                 x="X",
                 y="Y",
                 preprocessing=dict(method="test", kwargs={"x": 1, "y": 2}),
                 postprocessing=dict(method="test", kwargs={"x": 1, "y": 2}))
    assert gate.gate_name == "test"
    assert gate.parent == "test parent"
    assert gate.x == "X"
    assert gate.y == "Y"
    assert gate.preprocessing.get("method") == "test"
    assert gate.postprocessing.get("method") == "test"
    assert gate.preprocessing.get("kwargs").get("x") == 1
    assert gate.preprocessing.get("kwargs").get("y") == 2
    assert gate.postprocessing.get("kwargs").get("x") == 1
    assert gate.postprocessing.get("kwargs").get("y") == 2


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
