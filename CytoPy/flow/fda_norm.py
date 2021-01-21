from ..data.fcs import FileGroup
from detecta import detect_peaks
from KDEpy import FFTKDE
import pandas as pd
import numpy as np


def load_reference_data(filegroup: FileGroup,
                        parent: str,
                        x: str or None,
                        y: str or None,
                        x_transform: str or None,
                        y_transform: str or None,
                        ctrl: str or None = None):
    if ctrl is not None:
        return filegroup.load_ctrl_population_df(ctrl=ctrl,
                                                 population=parent,
                                                 transform={x: x_transform,
                                                            y: y_transform})
    return filegroup.load_population_df(population=parent,
                                        transform={x: x_transform,
                                                   y: y_transform})


def align_data(data: pd.DataFrame,
               ref_data: pd.DataFrame,
               dims: list):
    for d in dims:
        x = np.linspace(np.min([data[d].min(), ref_data[d].min()]) - 0.01,
                        np.max([data[d].max(), ref_data[d].max()]) + 0.01,
                        100)

    return data


def fda_norm(fit_predict):
    def normalise_data(gate, data: pd.DataFrame, ctrl_data: pd.DataFrame or None):
        if gate.fda_norm:
            assert gate.reference is not None, "No reference sample defined"
            ref_data = load_reference_data(filegroup=gate.reference,
                                           parent=gate.parent,
                                           x=gate.x,
                                           y=gate.y,
                                           x_transform=gate.transformations.get("x"),
                                           y_transform=gate.transformations.get("y"))
            data = align_data(data=data,
                              ref_data=ref_data,
                              dims=[d for d in [gate.x, gate.y] if d is not None])
            if ctrl_data is not None:
                ref_data = load_reference_data(filegroup=gate.reference,
                                               parent=gate.parent,
                                               x=gate.x,
                                               y=gate.y,
                                               x_transform=gate.transformations.get("x"),
                                               y_transform=gate.transformations.get("y"),
                                               ctrl=gate.ctrl)
                data = align_data(data=ctrl_data,
                                  ref_data=ref_data,
                                  dims=[d for d in [gate.x, gate.y] if d is not None])
        return fit_predict(data=data, ctrl_data=ctrl_data)
    return normalise_data
