from .explore import Explorer
from dash import Dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

STYLE = ["https://github.com/plotly/dash-app-stylesheets/blob/master/dash-analytics-report.css"]
APP = Dash(name="DashExplorer", external_stylesheets=STYLE)
DATA = pd.DataFrame()


def center_controls(explorer: Explorer):
    columns = explorer.data.columns
    available_methods = [x for x in columns if any([method in x for method in ["UMAP",
                                                                               "PCA",
                                                                               "PHATE",
                                                                               "tSNE",
                                                                               "KernelPCA"]])]
    assert len(available_methods) >= 1, "No dimension reduction methods have been applied to the given explorer"
    return dcc.Dropdown(
        id="center-controls-dropdown",
        options=[{"label": i, "value": i} for i in available_methods],
        value=available_methods[0]
    )


def center_plot():
    return dcc.Graph(id="center-plot")


def right_plot(explorer: Explorer):
    backgate_plot = dcc.Graph(id="backgate-plot")
    var_available = [x for x in explorer.data.columns if x not in explorer.meta_vars]
    x_var = dcc.Dropdown(id="x-axis-var",
                         options=[{"label": i, "value": i} for i in var_available],
                         value=var_available[0])
    y_var = dcc.Dropdown(id="y-axis-var",
                         options=[{"label": i, "value": i} for i in var_available],
                         value=var_available[1])
    children = [backgate_plot,
                x_var,
                y_var]
    if "population_label" in explorer.meta_vars:
        pops = list(explorer.data.population_label.unique())
        background_pop = dcc.Dropdown(id="background-pop",
                                      options=[{"label": i, "value": i} for i in pops],
                                      value=pops[0])
        children.append(background_pop)
    return html.Div(children, style={"width": "49%", "padding": "0px 20px 20px 20px"})


def bottom_plot(explorer: Explorer):
    representation_plot = dcc.Graph(id="representation-plot")
    metavar_plot = dcc.Graph(id="metavar-plot")
    metavar_dropdown = dcc.Dropdown(id="background-pop",
                                    options=explorer.meta_vars,
                                    value=explorer.meta_vars[0])
    return html.Div(representation_plot,
                    html.Div([metavar_plot, metavar_dropdown],
                             style={"width": "25%", "padding": "0px 20px 20px 20px"}))


def explorer_dashboard(explorer: Explorer,
                       features: list,
                       summary_method: str = "median",
                       mask: pd.DataFrame or None = None,
                       sample_id: str or None = None,
                       **run_server_kwargs):
    children = [center_controls(explorer),
                center_plot(),
                right_plot(explorer)]
    if sample_id is None:
        assert "meta_label" in explorer.data.columns, "No sample_id provided; assuming meta-clustering should be " \
                                                      "displayed yet 'meta_label' is missing from explorer dataframe"
        assert not explorer.data.meta_label.isnull().all(), "No sample_id provided; assuming meta-clustering should be " \
                                                            "displayed yet 'meta_label' is empty"
        children = [html.H1("Welcome to the interactive Explorer: meta clusters")] + children
        children.append(bottom_plot(explorer))
        data = explorer.summarise_metaclusters(features=features,
                                               summary_method=summary_method,
                                               mask=mask)
        data = html.Div(explorer.data.to_json(orient="split"), id="data", style={"display": "none"})
        children.append(data)
    else:
        assert sample_id in explorer.data.sample_id, "Invalid sample ID"
        data = explorer.data[explorer.data.sample_id == sample_id]
        data = html.Div(data.to_json(orient="split"), id="data", style={"display": "none"})
        children = [html.H1(f"Welcome to the interactive Explorer: {sample_id}")] + children
        children.append(data)
    APP.layout = html.Div(children)
    APP.run_server(**run_server_kwargs)


@APP.callback(
    Output("center-plot", "figure"),
    [Input("data", "children"),
     Input("center-controls-dropdown", "value")]
)
def update_center_plot(data, center_plot_type):
    data = pd.read_json(data, orient='split')[[f"{center_plot_type}{i+1}" for i in range(2)]]
    fig = px.scatter(data,
                     x=f"{center_plot_type}1",
                     y=f"{center_plot_type}2",
                     size="cluster_size")

