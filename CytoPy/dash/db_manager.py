from ..data.setup import global_init
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash import Dash

APP = Dash(name="CytoPy Data Manager",
           external_stylesheets=[dbc.themes.SPACELAB])

APP.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col([
                    html.Img(src="https://raw.githubusercontent.com/burtonrj/CytoPy/master/docs/source/logo.png",
                             id="cytopy-logo",
                             style=dict(height="120px", width="auto", margin="8px")),
                    html.A(dbc.Button("Help",
                                      id="docs-button",
                                      color="info",
                                      className="mr-1",
                                      outline=True,
                                      style=dict(margin="12px")),
                           href="https://cytopy.readthedocs.io/en/latest/")
                ],
                    width=4,
                    align="left"),
                dbc.Col(html.H2("CytoPy Database Manager", style=dict(textAlign="center")),
                        width="auto",
                        align="center")
            ],
            align="center"
        ),
        dbc.Row(
            [dbc.Col(html.Br()),
             dbc.Col(
                 [
                     dbc.FormGroup([dbc.Label("Enter database name"),
                                    html.Br(),
                                    dbc.Input(type="text"),
                                    html.Br(),
                                    dbc.Button("Submit",
                                               id="database-load-button",
                                               className="mr-1",
                                               color="primary",
                                               outline=True)],
                                   style=dict(textAlign="center"))
                 ],
                 width=4),
             dbc.Col(html.Br())
             ])
    ]
)


