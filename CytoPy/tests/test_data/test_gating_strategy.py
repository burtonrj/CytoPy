from ...data.gating_strategy import GatingStrategy
from ...data.gate import ThresholdGate, PolygonGate, EllipseGate
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


def test_load_data(example_experiment):
    gs = create_gatingstrategy_and_load(example_experiment)
    assert gs._filegroup is not None
    assert isinstance(gs._filegroup.data.get("primary"), pd.DataFrame)
    assert isinstance(gs._filegroup.data.get("controls").get("test_ctrl"), pd.DataFrame)
    assert list(gs._filegroup.list_populations()) == ["root"]


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
                         [(create_threshold_gate, ["root","Top right", "Top left", "Bottom populations"]),
                          (create_poly_gate, ["root", "Big pop"]),
                          (create_ellipse_gate, ["root", "Big pop", "Little pop"])])
def test_apply_gate(example_experiment, gate, populations):
    gs = create_gatingstrategy_and_load(example_experiment)
    gate = gate()
    gate.fit(data=gs._filegroup.load_population_df(population=gate.parent,
                                                   transform=None,
                                                   label_downstream_affiliations=False))
    if isinstance(gate, ThresholdGate):
        gate.label_children(labels={"++": "Top right",
                                    "-+": "Top left",
                                    "--": "Bottom populations",
                                    "+-": "Bottom populations"})
    elif isinstance(gate, PolygonGate):
        gate.label_children({"A": "Big pop"})
    else:
        pops = sorted([(c.name, c.geom.x_values) for c in gate.children], key=lambda x: x[1])
        gate.label_children({pops[0][0]: "Little pop",
                             pops[1][0]: "Big pop"})
    gs.apply_gate(gate=gate,
                  plot=True,
                  print_stats=True)
    plt.show()
    assert gs.list_populations() == populations
    not_root = [p for p in gs._filegroup.populations if p.population_name != "root"]
    root = gs._filegroup.get_population("root")
    assert all([len(p.index) < len(root.index) for p in not_root])
    biggest_pop = [p for p in not_root
                   if p.population_name == "Top right" or p.population_name == "Big pop"][0]
    assert all([len(p.index) <= len(biggest_pop.index) for p in not_root])


def test_apply_downstream():
    pass


def test_apply_all(example_experiment):
    pass


def test_delete_gate(example_experiment):
    pass


def test_edit_gate(example_experiment):
    pass


def test_delete_population(example_experiment):
    pass


def test_delete_action(example_experiment):
    pass


def test_plot_gate(example_experiment):
    pass


def test_plot_backgate(example_experiment):
    pass


def test_plot_population(example_experiment):
    pass


def test_print_population_tree(example_experiment):
    pass


def test_population_stats(example_experiment):
    pass


def test_estimate_ctrl_population(example_experiment):
    pass


def test_merge_populations(example_experiment):
    pass


def test_subtract_populations(example_experiment):
    pass


def test_save(example_experiment):
    pass


def test_delete():
    pass
