from ...data.gate import Gate, ThresholdGate, PolygonGate, EllipseGate, ChildThreshold, ChildPolygon
from ...data.geometry import ThresholdGeom, PolygonGeom
import pytest


@pytest.mark.parametrize("klass", [Gate, ThresholdGate, PolygonGate, EllipseGate])
def test_gate_init(klass):
    gate = klass(gate_name="test",
                 parent="test parent",
                 binary=False,
                 x="X",
                 y="Y",
                 preprocessing=dict(method="test", kwargs={"x": 1, "y": 2}),
                 postprocessing=dict(method="test", kwargs={"x": 1, "y": 2}))
    assert gate.gate_name == "test"
    assert gate.parent == "test parent"
    assert not gate.binary
    assert gate.x == "X"
    assert gate.y == "Y"
    assert gate.preprocessing.get("method") == "test"
    assert gate.postprocessing.get("method") == "test"
    assert gate.preprocessing.get("kwargs").get("x") == 1
    assert gate.preprocessing.get("kwargs").get("y") == 2
    assert gate.postprocessing.get("kwargs").get("x") == 1
    assert gate.postprocessing.get("kwargs").get("y") == 2


def test_gate_init_threshold_invalid():
    with pytest.raises(AssertionError) as err:
        ThresholdGate(gate_name="test",
                      parent="test parent",
                      binary=True,
                      x="X",
                      y="Y")
    assert str(err.value) == "Binary threshold gate should only receive value for x-axis not y"


@pytest.mark.parametrize("d", ["+", "-", "++", "--", "+-", "+++", "+ -"])
def test_threshold_add_child_invalid(d):
    threshold = ThresholdGate(gate_name="test",
                              parent="test parent",
                              binary=True,
                              x="X")
    child = ChildThreshold(name="test child",
                           definition=d,
                           geom=ThresholdGeom(x="X", ))
