from CytoPy.feedback import progress_bar
from CytoPy.flow.explore import Explorer
from sklearn import preprocessing
from dash import Dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pandas.api.types import is_numeric_dtype
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
APP = Dash(name="DashExplorer", external_stylesheets=external_stylesheets)
DATA = pd.DataFrame()


def center_controls(explorer: Explorer,
                    data: pd.DataFrame):
    methods = ["UMAP", "PCA", "PHATE", "tSNE", "KernelPCA"]
    available_methods = [x for x in methods if any([x in c for c in data.columns])]
    assert len(available_methods) >= 1, \
        "No dimension reduction methods have been applied to the given explorer"
    dim_red_method = dcc.Dropdown(
        id="method-dropdown",
        options=[{"label": i, "value": i} for i in available_methods],
        value=available_methods[0]
    )
    marker_size = dcc.Slider(
        id="marker-size-multiplier",
        min=10,
        max=1000,
        step=10,
        value=10
    )
    marker_colour = dcc.Dropdown(id="marker-colour",
                                 options=[{"label": i, "value": i}
                                          for i in [x for x in explorer.meta_vars]],
                                 value="meta_label")
    transform = dcc.Dropdown(id="transform",
                             options=[{"label": l, "value": v}
                                      for l, v in
                                      zip(["None",
                                           "Standard (min-max) scale",
                                           "Z-score",
                                           "Robust scale",
                                           "Quantile transform",
                                           "Yeo-johnson",
                                           "Box-cox"],
                                          ["none",
                                           "standard",
                                           "z_score",
                                           "robust",
                                           "quantile",
                                           "yeo_johnson",
                                           "box_cox"])],
                             value="none")
    return html.Div(
        children=[html.H6("Dimension reduction method"),
                  dim_red_method,
                  html.H6("Marker colour"),
                  marker_colour,
                  html.H6("Marker size multiplier"),
                  marker_size,
                  html.H6("Transformation"),
                  transform],
        style={"display": "inline-block", "width": "30%"}
    )


def center_plot(data, meta_vars):
    cluster_plot = dcc.Graph(id="cluster-plot")
    inspection_plot = dcc.Graph(id="inspection-plot")
    inspection_var = dcc.Dropdown(id="inspection-var",
                                  options=[{"label": i, "value": i}
                                           for i in [x for x in meta_vars]],
                                  value="meta_label")
    groups = ["None"] + categorical_columns(data)
    grouping_var = dcc.Dropdown(id="grouping-var",
                                options=[{"label": i, "value": i}
                                         for i in [x for x in groups]],
                                value="None")
    # return html.Div([
    #    html.Div([cluster_plot],
    #             className="six columns"),
    #    html.Div([inspection_plot,
    #              html.H6("Inspection variable"),
    #              inspection_var,
    #              html.H6("Grouping variable"),
    #              grouping_var],
    #             className="six columns")],
    #    className="row")
    return cluster_plot


def explorer_dashboard(explorer: Explorer,
                       features: list,
                       dim_reduction_methods: list,
                       summary_method: str = "median",
                       mask: pd.DataFrame or None = None,
                       dim_reduction_kwargs: dict or None = None,
                       port: int = 8050,
                       **run_server_kwargs):
    assert "meta_label" in explorer.data.columns, "No sample_id provided; assuming meta-clustering should be " \
                                                  "displayed yet 'meta_label' is missing from explorer dataframe"
    assert not explorer.data.meta_label.isnull().all(), "No sample_id provided; assuming meta-clustering should be " \
                                                        "displayed yet 'meta_label' is empty"
    print("Summarising meta clusters....")
    data = explorer.summarise_clusters(features=features,
                                       summary_method=summary_method,
                                       mask=mask)
    print("Dimension reduction....")
    dim_reduction_kwargs = dim_reduction_kwargs or {}
    for method in progress_bar(dim_reduction_methods, verbose=True):
        kwargs = dim_reduction_kwargs.get(method) or {}
        data = explorer.dimenionality_reduction(method=method,
                                                data=data,
                                                features=features,
                                                n_components=2,
                                                **kwargs)

    data = explorer.cluster_size(data)
    for meta_var in explorer.meta_vars:
        if meta_var not in data.columns:
            explorer.assign_labels(data=data, label=meta_var)

    g = explorer.clustered_heatmap(heatmap_var="meta clusters",
                                   features=features,
                                   cmap="inferno",
                                   standard_scale=None,
                                   z_score=1,
                                   vmin=0,
                                   vmax=1)
    heatmap = html.Div([html.Img(id="heatmap", src=fig_to_uri(g.fig))],
                       style={"display": "inline-block"})

    children = [html.H1("Welcome to the Interactive Explorer: Meta Clusters"),
                center_controls(explorer, data),
                html.Div([center_plot(data, explorer.meta_vars), heatmap])]
    print("Launching server....")
    data_div = html.Div(data.to_json(orient="split"), id="data", style={"display": "none"})
    del explorer
    children.append(data_div)
    APP.layout = html.Div(children)
    run_server_kwargs = run_server_kwargs or {}
    run_server_kwargs["port"] = port
    APP.run_server(**run_server_kwargs)


@APP.callback(
    Output("cluster-plot", "figure"),
    [Input("data", "children"),
     Input("method-dropdown", "value"),
     Input("marker-size-multiplier", "value"),
     Input("marker-colour", "value"),
     Input("transform", "value")]
)
def update_center_plot(data, center_plot_type, marker_size_multiplier, marker_colour, transform):
    data = pd.read_json(data, orient='split')
    data["marker_size"] = data["cluster_size"] * marker_size_multiplier
    data["marker_label"] = (data[["sample_id", "cluster_id", "meta_label"]]
                            .apply(lambda x: f"Sample: {x[0]}, meta-cluster: {x[2]}", axis=1))
    n = data.shape[0]
    data = data[data[marker_colour].notna()]
    if data.shape[0] < n:
        print(f"{(n-data.shape[0])/n*100}% of observations dropped due to missing values")
    if transform != "none":
        data[marker_colour] = transform_data(data[[marker_colour]], transform)
    if is_numeric_dtype(data[marker_colour]):
        if data[marker_colour].nunique() > 10:
            return px.scatter(data,
                              x=f"{center_plot_type}1",
                              y=f"{center_plot_type}2",
                              size="marker_size",
                              color=marker_colour,
                              hover_name="marker_label",
                              color_continuous_scale=px.colors.sequential.Plasma,
                              range_color=[data[marker_colour].min(),
                                           data[marker_colour].max()])
    return px.scatter(data,
                      x=f"{center_plot_type}1",
                      y=f"{center_plot_type}2",
                      size="marker_size",
                      color=marker_colour,
                      hover_name="marker_label",
                      color_discrete_sequence=px.colors.qualitative.Dark24)


"""@APP.callback(
    Output("inspection-plot", "figure"),
    [Input("data", "children"),
     Input("cluster-plot", "selectedData"),
     Input("inspection-var", "value"),
     Input("grouping-var", "value")]
)
def update_inspection_plot(data, cluster_selection, inspection_var, grouping_var):
    data = pd.read_json(data, orient='split')
    if cluster_selection:

        identifiers = [x.get("hovertext").replace("Sample: ", "").replace(" cluster: ", "")
                       for x in cluster_selection["points"]]
        sid = [x.split(",")[0] for x in identifiers]
        cid = [x.split(",")[1] for x in identifiers]

        data = data[(data["sample_id"].isin(sid)) &
                    (data["cluster_id"].isin(cid))]

        if is_categorical(data[inspection_var]):
            print("1")
            if grouping_var == "None":
                data = (data[inspection_var]
                        .value_counts()
                        .reset_index()
                        .rename({inspection_var: "Count",
                                 "index": inspection_var},
                                axis=1))
                print("2")
                return px.bar(data, x=inspection_var, y="Count")
            else:
                print("3")
                data = (pd.DataFrame(data.groupby(grouping_var)[inspection_var]
                                     .value_counts())
                        .rename({inspection_var: "Count"}, axis=1)
                        .reset_index())
                return px.bar(data, x=grouping_var, y="Count")
        else:
            print("4")
            if grouping_var == "None":
                print("5")
                return px.strip(data, x="meta_label", y=inspection_var)
            return px.strip(data, x=grouping_var, y=inspection_var)"""


def is_categorical(x: pd.Series):
    return 1.*x.nunique()/x.count() < 0.05


def categorical_columns(data: pd.DataFrame):
    categorical = list()
    for c in data.columns:
        if is_categorical(data[c]):
            categorical.append(c)
    return categorical


def transform_data(x: pd.DataFrame,
                   method: str):
    if method == "standard":
        return preprocessing.MinMaxScaler().fit_transform(x)
    if method == "z_score":
        return preprocessing.StandardScaler().fit_transform(x)
    if method == "robust":
        return preprocessing.RobustScaler().fit_transform(x)
    if method == "quantile":
        return preprocessing.QuantileTransformer(output_distribution="normal",
                                                 random_state=42).fit_transform(x)
    if method == "yeo_johnson":
        return preprocessing.PowerTransformer(method="yeo-johnson").fit_transform(x)
    if method == "box_cox":
        return preprocessing.PowerTransformer(method="box-cox").fit_transform(x)
    raise ValueError("Chosen transform method is not supported")


def fig_to_uri(in_fig, close_all=True, **save_args):
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)