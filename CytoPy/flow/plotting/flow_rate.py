import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


def plot_flow_rate(data: pd.DataFrame,
                   time_var: str = "time",
                   timestep: float or None = None,
                   analysis_timestep: int = 100,
                   ax: plt.Axes or None = None,
                   figsize: tuple = (8, 8),
                   clip: tuple or None = None):
    """
    Given a DataFrame of flow data plot the rate of events by identifying the
    'time' column.

    Parameters
    ----------
    data: Pandas.DataFrame
    time_var: str
        Regular expression term used to identify column name in data for column
        containing the time variable
    timestep: float (optional)
        Flow cytometer timestep; if not provided, assumed to be about 10th of a second
    analysis_timestep: float (default=100.)
        How many timesteps to use when summarising events across a time series. Defaults to 100,
        which means events are summarised into bins of 100 timesteps.
    ax: Matplotlib.Axes (optional)
    figsize: tuple (default=(8, 8))

    Returns
    -------
    Matplotlib.Axes
    """
    ax = ax or plt.subplots(figsize=figsize)[1]
    time_var_mask = data.columns.str.contains(time_var, flags=re.IGNORECASE)
    assert sum(time_var_mask) == 1, "Time variable not recognised or matches multiple columns"
    time_var = data.columns[time_var_mask][0]
    event_count = (data[time_var]
                   .value_counts()
                   .reset_index()
                   .rename({"index": "Time",
                            "Time": "Event count"}, axis=1)
                   .sort_values("Time"))
    timestep = timestep or round(event_count.diff()["Time"].mean(), 5)/10
    event_count["Time"] = event_count["Time"]*timestep
    event_count["Timestep"] = (np.array([np.repeat(1, analysis_timestep)
                                         + i for i in range(event_count.shape[0])])
                               .flatten()[0: event_count.shape[0]])
    event_count = event_count.groupby("Timestep")["Event count"].sum().reset_index()
    event_count["Time"] = event_count["Timestep"]*timestep
    ax.plot(event_count["Time"], event_count["Event count"])
    ax.xlabel("Time")
    ax.ylabel("Event count")
    return ax
