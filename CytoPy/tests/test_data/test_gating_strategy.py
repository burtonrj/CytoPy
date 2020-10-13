from ...data.gating_strategy import GatingStrategy, Action
from ...data.gate import ThresholdGate, PolygonGate, EllipseGate, PolygonGeom
from ...data.project import Project
import matplotlib.pyplot as plt
import pandas as pd
import pytest
import os


@pytest.fixture
def example_experiment():
    test_project = Project(project_id="test")
    test_exp = test_project.add_experiment(experiment_id="test experiment",
                                           data_directory=f"{os.getcwd()}/test_data",
                                           panel_definition=f"{os.getcwd()}/CytoPy/tests/assets/test_panel.xlsx")
    test_exp.add_new_sample(sample_id="test sample",
                            primary_path=f"{os.getcwd()}/CytoPy/tests/assets/test.FCS",
                            controls_path={"test_ctrl": f"{os.getcwd()}/CytoPy/tests/assets/test.FCS"},
                            compensate=False)
    yield test_exp
    test_project.delete()


def create_gatingstrategy_and_load(example_experiment):
    gs = GatingStrategy(name="test")
    gs.load_data(experiment=example_experiment,
                 sample_id="test sample")
    return gs


def create_poly_gate():
    y = [0.2, 0.2, 0.35, 0.35, 0.2]
    x = [600, 1000, 1000, 600, 600]
    poly = PolygonGate(gate_name="test poly",
                       parent="root",
                       x="FS Lin",
                       y="IgG1-FITC",
                       transformations={"x": None,
                                        "y": "logicle"},
                       method="manual",
                       method_kwargs={"x_values": x,
                                      "y_values": y})
    return poly


def create_threshold_gate():
    threshold = ThresholdGate(gate_name="test threshold",
                              parent="root",
                              x="FS Lin",
                              y="IgG1-FITC",
                              transformations={"x": None,
                                               "y": "logicle"},
                              method="density")
    return threshold


def create_ellipse_gate():
    ellipse = EllipseGate(gate_name="test ellipse",
                          parent="root",
                          x="FS Lin",
                          y="IgG1-FITC",
                          transformations={"x": None,
                                           "y": "logicle"},
                          method="GaussianMixture",
                          method_kwargs={"n_components": 2,
                                         "random_state": 42,
                                         "conf": 0.95})
    return ellipse


def apply_some_gates(gs: GatingStrategy):
    # Apply threshold gate
    gate = create_threshold_gate()
    gs.preview_gate(gate=gate)
    gate.label_children(labels={"++": "pop1"})
    gs.apply_gate(gate)
    # Apply ellipse gate
    gate = create_ellipse_gate()
    gate.parent = "pop1"
    gate.y = "CD45-ECD"
    gs.preview_gate(gate=gate)
    gate.label_children({"A": "pop2"})
    gs.apply_gate(gate)
    # Apply another threshold gate
    gate = create_threshold_gate()
    gate.gate_name = "test threshold 2"
    gate.parent = "pop2"
    gate.x, gate.y = "IgG1-PC5", None
    gate.transformations = {"x": "logicle", "y": None}
    gs.preview_gate(gate=gate)
    gate.label_children({"+": "pop3", "-": "pop4"})
    gs.apply_gate(gate=gate)
    return gs


def test_load_data(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    assert gs.filegroup is not None
    assert isinstance(gs.filegroup.data.get("primary"), pd.DataFrame)
    assert isinstance(gs.filegroup.data.get("controls").get("test_ctrl"), pd.DataFrame)
    assert list(gs.filegroup.list_populations()) == ["root"]


@pytest.mark.parametrize("gate,child_n",
                         [(create_threshold_gate, 4),
                          (create_poly_gate, 1),
                          (create_ellipse_gate, 2)])
def test_preview_gate(example_experiment, gate, child_n):
    gs = create_gatingstrategy_and_load(example_experiment)
    gate = gate()
    gs.preview_gate(gate)
    assert len(gate.children) == child_n
    plt.show()


@pytest.mark.parametrize("gate,populations",
                         [(create_threshold_gate, ["root", "Top right", "Top left", "Bottom populations"]),
                          (create_poly_gate, ["root", "Big pop"]),
                          (create_ellipse_gate, ["root", "Big pop", "Little pop"])])
def test_apply_gate(example_experiment, gate, populations):
    gs = create_gatingstrategy_and_load(example_experiment)
    gate = gate()
    gate.fit(data=gs.filegroup.load_population_df(population=gate.parent,
                                                  transform=None,
                                                  label_downstream_affiliations=False))
    if isinstance(gate, ThresholdGate):
        gate.label_children(labels={"++": "Top right",
                                    "-+": "Top left",
                                    "--": "Bottom populations",
                                    "+-": "Bottom populations"})
    elif isinstance(gate, EllipseGate):
        pops = sorted([(c.name, c.geom.x_values) for c in gate.children], key=lambda x: x[1])
        gate.label_children({pops[0][0]: "Little pop",
                             pops[1][0]: "Big pop"})
    else:
        gate.label_children({"A": "Big pop"})
    gs.apply_gate(gate=gate,
                  plot=True,
                  print_stats=True)
    plt.show()
    assert set(gs.list_populations()) == set(populations)
    not_root = [p for p in gs.filegroup.populations if p.population_name != "root"]
    root = gs.filegroup.get_population("root")
    assert all([len(p.index) < len(root.index) for p in not_root])
    biggest_pop = [p for p in not_root
                   if p.population_name == "Top right" or p.population_name == "Big pop"][0]
    assert all([len(p.index) <= len(biggest_pop.index) for p in not_root])


def assert_expected_gated_pops(gs: GatingStrategy):
    # Test expected populations present
    expected_pops = {"root", "pop1", "pop2", "pop3", "pop4"}
    assert set(gs.list_populations()) == expected_pops
    assert all([x in gs.filegroup.tree.keys() for x in expected_pops])
    # Test population tree
    gs.filegroup.print_population_tree()
    assert gs.filegroup.get_population("pop1").parent == "root"
    assert gs.filegroup.get_population("pop2").parent == "pop1"
    assert gs.filegroup.get_population("pop3").parent == "pop2"
    assert gs.filegroup.get_population("pop4").parent == "pop2"
    # Test population indexes
    root_n = len(gs.filegroup.get_population("root").index)
    assert all([len(gs.filegroup.get_population(x).index) < root_n
                for x in ["pop1", "pop2", "pop3", "pop4"]])
    assert len(gs.filegroup.get_population("pop1").index) > len(gs.filegroup.get_population("pop2").index)
    assert gs.filegroup.get_population("pop1").n > gs.filegroup.get_population("pop2").n
    assert len(gs.filegroup.get_population("pop2").index) > len(gs.filegroup.get_population("pop3").index)
    assert gs.filegroup.get_population("pop2").n > gs.filegroup.get_population("pop3").n
    assert len(gs.filegroup.get_population("pop2").index) > len(gs.filegroup.get_population("pop4").index)
    assert gs.filegroup.get_population("pop2").n > gs.filegroup.get_population("pop4").n


def test_apply_downstream(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    assert_expected_gated_pops(gs)


@pytest.mark.parametrize("action,err",
                         [("not an action", "not an action does not exist"),
                          (Action(action_name="merge test",
                                  method="merge",
                                  left="invalid",
                                  right="pop4"), "invalid does not exist"),
                          (Action(action_name="merge test",
                                  method="subtract",
                                  left="pop4",
                                  right="invalid"), "invalid does not exist"),
                          (Action(action_name="merge test",
                                  method="merge",
                                  left="pop4",
                                  right="invalid"), "invalid does not exist"),
                          (Action(action_name="merge test",
                                  method="invalid method",
                                  left="pop4",
                                  right="invalid"), "Accepted methods are: merge, subtract")])
def test_apply_action_errors(example_experiment, action, err):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    with pytest.raises(AssertionError) as e:
        gs.apply_action(action=action)
    assert str(e.value) == err


def test_apply_action_merge(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    action = Action(action_name="test merge",
                    method="merge",
                    left="pop3",
                    right="pop4")
    gs.apply_action(action=action, add_to_strategy=True, print_stats=True)
    assert "test merge" in [a.action_name for a in gs.actions]
    assert len(gs.actions) == 1
    new_pop = gs.filegroup.get_population(population_name="merge_pop3_pop4")
    pop2 = gs.filegroup.get_population(population_name="pop2")
    pop3 = gs.filegroup.get_population(population_name="pop3")
    assert new_pop.parent == "pop2"
    assert new_pop.n == pop2.n
    assert len(new_pop.index) == len(pop2.index)
    assert isinstance(new_pop.geom, type(pop3.geom))
    assert new_pop.geom.x == pop3.geom.x
    assert new_pop.geom.y == pop3.geom.y
    assert new_pop.geom.transform_x == pop3.geom.transform_x
    assert new_pop.geom.transform_y == pop3.geom.transform_y
    assert all([x in gs.filegroup.tree.keys() for x in ["merge_pop3_pop4", "pop3", "pop4"]])


def test_apply_action_subtract(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    action = Action(action_name="test subtract",
                    method="subtract",
                    left="pop2",
                    right="pop4")
    gs.apply_action(action=action, add_to_strategy=True, print_stats=True)
    assert "test subtract" in [a.action_name for a in gs.actions]
    assert len(gs.actions) == 1
    new_pop = gs.filegroup.get_population(population_name="subtract_pop2_pop4")
    pop2 = gs.filegroup.get_population(population_name="pop2")
    assert new_pop.parent == "pop1"
    assert new_pop.n == 15995
    assert len(new_pop.index) == 15995
    assert isinstance(new_pop.geom, PolygonGeom)
    assert new_pop.geom.x == pop2.geom.x
    assert new_pop.geom.y == pop2.geom.y
    assert new_pop.geom.transform_x == pop2.geom.transform_x
    assert new_pop.geom.transform_y == pop2.geom.transform_y
    assert all([x in gs.filegroup.tree.keys() for x in ["subtract_pop2_pop4", "pop3", "pop4"]])


def test_apply_all(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    with pytest.raises(AssertionError) as err:
        gs.apply_all()
    assert str(err.value) == "No gates to apply"
    gs = apply_some_gates(gs)
    exp = Project.objects(project_id="test").get().load_experiment("test experiment")
    gs.load_data(experiment=exp,
                 sample_id="test sample")
    gs.apply_all()
    assert_expected_gated_pops(gs)
    with pytest.raises(AssertionError) as err:
        gs.apply_all()
    assert str(err.value) == "One or more of the populations generated from this gating strategy are already " \
                             "presented in the population tree"


def test_delete_gate(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    gs.delete_gate("test ellipse")
    assert "test ellipse" not in [g.gate_name for g in gs.gates]


def test_plot_gate(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    plt.close("all")
    gs.plot_gate(gate=gs.gates[0])
    plt.show()


def test_plot_gate_by_name(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    plt.close("all")
    gs.plot_gate(gate="test threshold", create_plot_kwargs={"title": "test threshold"})
    plt.show()


def test_plot_gate_invalid(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    with pytest.raises(AssertionError) as err:
        gs.plot_gate(gate="test ellipse", y="FS Lin")
    assert str(err.value) == "Can only override y-axis variable for ThresholdGate"


def test_plot_backgate(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    plt.close("all")
    gs.plot_backgate(parent="root",
                     overlay=["pop3", "pop4"],
                     x="FS Lin",
                     y="IgG1-FITC",
                     create_plot_kwargs={"transform_x": None,
                                         "transform_y": "logicle"})
    plt.show()


def test_plot_population(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    plt.close("all")
    gs.plot_population(population="pop1",
                       x="FS Lin",
                       y="IgG1-FITC")
    plt.show()


def test_population_stats(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    stats = gs.population_stats(population="root")
    assert isinstance(stats, dict)
    assert stats.get("name") == "root"
    assert stats.get("n") == 30000
    assert stats.get("prop_of_parent") == 1.0
    assert stats.get("prop_of_root") == 1.0


def test_save(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    gs.save()
    gs = GatingStrategy.objects(name="test")
    assert len(gs) == 1
    gs = gs.get()
    assert len(gs.gates) == 3


@pytest.mark.parametrize("remove_associations", [True, False])
def test_delete(example_experiment, remove_associations):
    gs = create_gatingstrategy_and_load(example_experiment)
    gs = apply_some_gates(gs)
    gs.save()
    gs = GatingStrategy.objects(name="test").get()
    populations = [[c.name for c in g.children] for g in gs.gates]
    populations = list(set([x for sl in populations for x in sl]))
    gs.delete(remove_associations=remove_associations,
              delete_gates=remove_associations)
    assert len(GatingStrategy.objects(name="test")) == 0
    n = [0, 0]
    if not remove_associations:
        n = [2, 1]
    for n_, gate in zip(n, [ThresholdGate, EllipseGate]):
        assert len(gate.objects()) == n_
    fg = (Project.objects(project_id="test")
          .get()
          .load_experiment("test experiment")
          .get_sample("test sample"))
    if remove_associations:
        assert len(fg.gating_strategy) == 0
        assert all([p not in fg.list_populations() for p in populations])
    else:
        assert len(fg.gating_strategy) == 1
        assert all([p in fg.list_populations() for p in populations])