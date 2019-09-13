"""
Various transforms for FCS data
"""

import numpy as np

from flowutils import logicle_c


def quantile(x, n):
    """return the lower nth quantile"""
    try:
        return sorted(x)[int(n * len(x))]
    except IndexError:
        return 0


def product_log(x):
    """
    Product logarithm or LambertW function computes principal solution
    for w in f(w) = w*exp(w).
    """
    #  fast estimate with closed-form approximation
    if x <= 500:
        lxl = np.log(x + 1.0)
        return 0.665 * (1 + 0.0195 * lxl) * lxl + 0.04
    else:
        return np.log(x - 4.0) - \
               (1.0 - 1.0 / np.log(x)) * np.log(np.log(x))


def s(x, y, t, m, w):
    p = w / (2 * product_log(0.5 * np.exp(-w / 2) * w))
    sgn = np.sign(x - w)
    xw = sgn * (x - w)
    return sgn * t * np.exp(-(m - w)) * (np.exp(xw) - p ** 2 * np.exp(-xw / p) + p ** 2 - 1) - y


def _logicle(y, t=262144, m=4.5, r=None, w=0.5, a=0):
    y = np.array(y, dtype='double')
    if w is None:  # we need an r then...
        if r == 0:
            w = 1  # don't like this but it works... FIX!
        else:
            w = (m - np.log10(t / np.abs(r))) / 2.0

    # noinspection PyUnresolvedReferences
    logicle_c.logicle_scale(t, w, m, a, y)
    return y


def logicle(
        data,
        channels,
        t=262144,
        m=4.5,
        r=None,
        w=0.5,
        a=0,
        r_quant=None):
    """
    return logicle transformed points for channels listed
    """
    data_copy = data.copy()

    # run logicle scale for each channel separately
    for i in channels:
        if r_quant:
            w = None
            tmp = data_copy[i]
            r = quantile(tmp[tmp < 0], 0.05)
        if r is None and w is None:
            w = 0.5
        tmp = _logicle(data_copy[i], t, m, r, w, a)
        data_copy[i] = tmp
    return data_copy


def _hyperlog(y, t=262144, m=4.5, w=0.5, a=0):
    y = np.array(y, dtype='double')

    # noinspection PyUnresolvedReferences
    logicle_c.hyperlog_scale(t, w, m, a, y)
    return y


def hyperlog(
        data,
        channels,
        t=262144,
        m=4.5,
        w=0.5,
        a=0,
):
    """
    return hyperlog transformed points for channels listed
    """
    data_copy = data.copy()

    # run hyperlog scale for each channel separately
    for i in channels:
        tmp = _hyperlog(data_copy[i], t, m, w, a)
        data_copy[i] = tmp
    return data_copy


def asinh(data, columns, pre_scale):
    """
    return asinh transformed points (after pre-scaling) for indices listed
    """
    data_copy = data.copy()
    for c in columns:
        data_copy[c] = np.arcsinh(data_copy[c] * pre_scale)
    return data_copy


def log_transform(npy, channels):
    n_points = npy.copy()
    for i in channels:
        n_points[i] = _log_transform(n_points[i])
    return n_points


def _log_transform(npnts):
    return np.where(npnts <= 1, 0, np.log10(npnts))
