from ...flow.gating_analyst import create_convex_hull, probablistic_ellipse, inside_ellipse, find_local_minima, \
    _draw_circle, Analyst, Population
from shapely.geometry import Polygon as SPoly
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from scipy.stats import norm
from KDEpy import FFTKDE
import pandas as pd
import numpy as np
import pytest


def test_create_convex_hull():
    test_data = make_blobs(n_samples=1000,
                           n_features=2,
                           centers=1,
                           center_box=(0, 5),
                           random_state=42)[0]
    x, y = test_data[:, 0], test_data[:, 1]
    hull = create_convex_hull(x, y)
    assert isinstance(hull[0], list)
    assert isinstance(hull[1], list)
    for idx, t in enumerate([x, y]):
        lower = np.quantile(t, 0.05)
        upper = np.quantile(t, 0.95)
        t_ = [i for i in t if lower < i < upper]
        for i in range(100):
            s = np.random.choice(t_, 1)[0]
            assert s >= np.min(hull[idx])
            assert s <= np.max(hull[idx])


@pytest.mark.parametrize("conf", [0.95, 0.8, 0.5])
def test_probablistic_ellipse(conf):
    test_data = make_blobs(n_samples=1000,
                           n_features=2,
                           centers=1,
                           center_box=(1, 5),
                           random_state=42)[0]
    model = GaussianMixture(random_state=42, n_components=1)
    model.fit(test_data)
    center = model.means_[0]
    width, height, angle = probablistic_ellipse(model.covariances_[0], conf)
    mask = inside_ellipse(test_data, center=center, width=width, height=height, angle=angle)
    assert test_data[mask].shape[0] / test_data.shape[0] == pytest.approx(conf, 0.1)


@pytest.mark.parametrize("test_data,expected_mask",
                         [(np.array([[3, 4.5], [7.5, 9]]), [True, False]),
                          (np.array([[3, 4.5], [0, 0]]), [True, False]),
                          (np.array([[11, 5], [6.2, 4.3]]), [False, True])])
def test_inside_ellipse(test_data, expected_mask):
    center, width, height, angle = (5, 5), 10, 5, 15
    mask = inside_ellipse(data=test_data,
                          center=center,
                          width=width,
                          height=height,
                          angle=angle)
    assert isinstance(mask, list)
    assert np.array_equal(mask, expected_mask)


def test_find_local_minima():
    d1 = norm(loc=2, scale=1)
    d2 = norm(loc=10, scale=0.5)
    data = np.hstack([d1.rvs(1000), d2.rvs(1000)])
    x, y = FFTKDE(kernel='gaussian', bw='silverman').fit(data).evaluate()
    peak1 = np.where(y == np.max(y[np.where(x < 6)]))[0][0]
    peak2 = np.where(y == np.max(y[np.where(x > 6)]))[0][0]
    minima = x[np.where(y == np.min(y[np.where((x > 4) & (x < 7))]))[0][0]]
    assert find_local_minima(probs=y, xx=x, peaks=np.array([peak1, peak2])) == minima


def test_draw_circle():
    test_data = make_blobs(n_samples=100,
                           n_features=2,
                           centers=1,
                           center_box=(0, 5),
                           random_state=42)[0]
    circle = _draw_circle(pd.DataFrame(test_data), (2, 5))
    assert isinstance(circle, SPoly)
    assert pytest.approx(24.59, 0.1) == circle.area
    assert pytest.approx(2, 0.1) == circle.centroid.coords.xy[0][0]
    assert pytest.approx(5, 0.1) == circle.centroid.coords.xy[1][0]


def test_analyst_init_errors():
    with pytest.raises(AssertionError) as err:
        Analyst(x="X", y="Y", shape="Hexagon",
                parent="test", binary=True,
                model=None)
    assert str(err.value) == """Invalid shape, must be one of: ["threshold", "polygon", "ellipse"]"""
    with pytest.raises(AssertionError) as err:
        Analyst(x="X", y="Y", shape="polygon",
                parent="test", binary=True,
                model="RandomModel")
    assert str(err.value) == "Module RandomModel not supported. See docs for supported methods. "


@pytest.mark.parametrize("x,y,expected_pp_pop_size",
                         [(1.5, 2, 62), (.8, 3.5, 79), (5., 5., 0)])
def test_threshold_2d(x, y, expected_pp_pop_size):
    test_data = pd.DataFrame(make_blobs(n_samples=100,
                                        n_features=2,
                                        centers=1,
                                        center_box=(0, 5),
                                        random_state=42)[0], columns=["X", "Y"])
    a = Analyst(x="X", y="Y", shape="threshold",
                parent="test", binary=False,
                model=None)
    pops = a._threshold_2d(test_data, x, y)
    assert isinstance(pops, list)
    assert all([isinstance(p, Population) for p in pops])
    assert all([p.parent == a.parent for p in pops])
    assert all([p.geom.x_threshold == x for p in pops])
    assert all([p.geom.y_threshold == y for p in pops])
    assert all([d in [p.definition for p in pops] for d in ["--", "-+", "++", "+-"]])
    pospos = [p for p in pops if p.definition == "++"][0]
    assert len(pospos.index) == expected_pp_pop_size


@pytest.mark.parametrize("x,expected_pp_pop_size",
                         [(1.5, 62), (.8, 88), (5., 0)])
def test_threshold_1d(x, expected_pp_pop_size):
    test_data = pd.DataFrame(make_blobs(n_samples=100,
                                        n_features=2,
                                        centers=1,
                                        center_box=(0, 5),
                                        random_state=42)[0], columns=["X", "Y"])
    a = Analyst(x="X", y="Y", shape="threshold",
                parent="test", binary=True,
                model=None)
    pops = a._threshold_1d(test_data, x)
    assert isinstance(pops, list)
    assert all([isinstance(p, Population) for p in pops])
    assert all([p.parent == a.parent for p in pops])
    assert all([p.geom.x_threshold == x for p in pops])
    assert all([d in [p.definition for p in pops] for d in ["-", "+"]])
    pos = [p for p in pops if p.definition == "+"][0]
    assert len(pos.index) == expected_pp_pop_size


def test_ellipse():
    test_data, labels = make_blobs(n_samples=300,
                                   n_features=2,
                                   centers=3,
                                   cluster_std=0.8,
                                   random_state=42)
    test_data = pd.DataFrame(test_data, columns=["X", "Y"])
    a = Analyst(x="X", y="Y", shape="ellipse",
                parent="test", binary=False,
                model=None, conf=0.95)
    model = GaussianMixture(random_state=42, n_components=3)
    y_hat = model.fit_predict(test_data)
    pops = a._ellipse(data=test_data, labels=y_hat, centers=model.means_, covar_matrix=model.covariances_)
    assert isinstance(pops, list)
    assert len(pops) == 3
    assert all([isinstance(p, Population) for p in pops])
    assert all([p.parent == a.parent for p in pops])
    assert all([90 < len(p.index) <= 100 for p in pops])


def test_polygon():
    test_data, labels = make_blobs(n_samples=300,
                                   n_features=2,
                                   centers=3,
                                   cluster_std=0.8,
                                   random_state=42)
    test_data = pd.DataFrame(test_data, columns=["X", "Y"])
    a = Analyst(x="X", y="Y", shape="ellipse",
                parent="test", binary=False,
                model=None, conf=0.95)
    pops = a._polygon(data=test_data, labels=labels)
    assert isinstance(pops, list)
    assert len(pops) == 3
    assert all([isinstance(p, Population) for p in pops])
    assert all([p.parent == a.parent for p in pops])
    assert all([90 < len(p.index) <= 100 for p in pops])


