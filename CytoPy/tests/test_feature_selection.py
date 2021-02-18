from ..flow.clustering import Clustering, sklearn_clustering, sklearn_metaclustering
from ..flow import feature_selection
from ..data.experiment import experiment_subject_search
from ..data.project import Project
from .conftest import create_logicle_like
import pandas as pd
import numpy as np
import pytest
import os


def create_logicle_like_dataframe(skew_factor: float = 1.,
                                  pop_size: list or None = None):
    if pop_size is None:
        pop_size = [50000, 25000, 20000, 50000]
    assert len(pop_size) == 4, "Must be exactly 4 populations"
    x = create_logicle_like(u=[8.2, 1.8, 5.2*skew_factor, 12.2*skew_factor],
                            s=[0.8, 1.8, 2.85, 1.95],
                            size=pop_size)
    y = create_logicle_like(u=[5.2, 2.95, 4.1, 13.2],
                            s=[0.7, 2.8, 1.7, 2.85],
                            size=pop_size)
    z = create_logicle_like(u=[9.1, 1.87, 5.2, 7.2],
                            s=[1.1, 4.5, 1.25, 2.95],
                            size=pop_size)
    return pd.DataFrame({"C1": x, "C2": y, "C3": z})


def create_example_experiment():
    panel = {"channels": [{"name": f"C{i + 1}",
                           "regex": f"C{i + 1}",
                           "case": 0,
                           "permutations": ""}
                          for i in range(3)],
             "markers": [{"name": "",
                          "regex": "",
                          "case": 0,
                          "permutations": ""}
                         for i in range(3)],
             "mappings": [(f"C{i + 1}", "") for i in range(3)]}
    test_project = Project(project_id="test")
    subject_category = ["Healthy" for i in range(5)] + ["Diseased" for i in range(5)]
    for i in range(len(subject_category)):
        test_project.add_subject(subject_id=f"Subject{i + 1}", status=subject_category[i])
    exp = test_project.add_experiment(experiment_id="test experiment",
                                      data_directory=f"{os.getcwd()}/test_data",
                                      panel_definition=panel)
    skew_factors = [1.1, 1.05, 1.0, 1.2, 1.4, 2.1, 3, 1.4, 2.5, 4]
    pop_sizes = np.array([50000, 25000, 20000, 50000])
    pop_sizes = [pop_sizes,
                 pop_sizes*1.1,
                 pop_sizes*1.05,
                 pop_sizes*0.95,
                 pop_sizes,
                 pop_sizes*0.4,
                 pop_sizes*0.8,
                 pop_sizes*0.1,
                 pop_sizes*0.7,
                 pop_sizes]
    for i in range(len(subject_category)):
        exp.add_dataframes(sample_id=f"Subject{i + 1}",
                           primary_data=create_logicle_like_dataframe(skew_factor=skew_factors[i],
                                                                      pop_size=pop_sizes[i]),
                           mappings=[{"channel": f"C{i + 1}", "marker": ""} for i in range(3)],
                           subject_id=f"Subject{i + 1}")
    cluster = Clustering(experiment=exp,
                         features=["C1", "C2", "C3"],
                         transform="logicle")
    cluster.cluster(sklearn_clustering,
                    method="MiniBatchKMeans",
                    n_clusters=4,
                    batch_size=5000,
                    print_performance_metrics=False)
    cluster.meta_cluster(sklearn_metaclustering,
                         method="KMeans",
                         n_clusters=4)
    cluster.save()
    return exp


def test_create_feature_space():
    exp = create_example_experiment()
    feature_space = feature_selection.FeatureSpace(experiment=exp)
    (feature_space
     .compute_ratios(pop1="cluster_0", pop2="cluster_1")
     .compute_ratios(pop1="cluster_2")
     .channel_desc_stats(channel="C1", transform="logicle"))
    stats = ["mean", "median", "SD", "CV",
             "skew", "kurtosis", "gmean"]

    #assert all([x in feature_space.sample_id.values for x in exp.list_samples()])
    #assert all([experiment_subject_search(experiment=exp, sample_id=x).subject_id
    #            in feature_space.subject_id.values for x in exp.list_samples()])
    #for c in [f"cluster_{i}" for i in range(4)]:
    #    assert f"{c}_FOR" in feature_space.columns
    #    assert f"{c}_FOP" in feature_space.columns
    #assert "cluster_0:cluster_1" in feature_space.columns
    #assert "cluster_1:cluster_0" in feature_space.columns
    #for stat in ["mean", "median", "SD", "CV", "skew", "kurtosis", "geo_mean"]:
    #    assert f"cluster_0_C1_{stat}" in feature_space.columns
    #assert all([x in feature_space.columns for x in ["cluster_2:cluster_0",
    #                                                 "cluster_2:cluster_1",
    #                                                 "cluster_2:cluster_3"]])


def test_add_target_labels():
    pass


def test_plot_variance():
    pass


def test_filter_variance():
    pass


def test_plot_multicolinearity():
    pass


def test_pca():
    pass


def test_pca_loadings():
    pass


def test_box_swarm_plots():
    pass


def test_inference_test():
    pass


def test_mutual_information():
    pass


def test_l1_selection():
    pass


def test_permutation_feature_importance():
    pass


def test_decision_tree():
    pass


def test_shap():
    pass
