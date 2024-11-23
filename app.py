import os
from dash import Dash, html, dcc, callback, Output, Input, no_update, State, ALL, Patch
import plotly.express as px
import dash_bootstrap_components as dbc
import numpy as np
import tensorflow as tf
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="Name of decoder")
parser.add_argument("-i", type=int, nargs="+", help="Order of continuous latent dims")
args = parser.parse_args()
name = args.name
i_cont = args.i

# Load hyperparameters
with open(f"{name}.json", "r") as f:
    HP = json.load(f)

# Inputs
n_cont = HP["cont_dim"]
n_disc = HP["disc_dim"][0]  # Currently supporting only one discrete dimension
n_sliders = len(i_cont)
n_buttons = n_disc

# Load model
filepath = os.path.join(os.path.dirname(__file__), f"decoder_{name}.tflite")
interpreter = tf.lite.Interpreter(filepath)
model = interpreter.get_signature_runner()
input_details = model.get_input_details()
output_details = model.get_output_details()

sliders = [
    dcc.Slider(-3, 3, value=0, id={"type": "slider", "index": i})
    for i in range(n_sliders)
]
sliders_div = [
    html.Div(
        className="sliders_div",
        children=[f"z{i}", sliders[i]],
    )
    for i in range(n_sliders)
]

app = Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Digit Generator"
)

app.layout = html.Div(
    [
        html.Div(
            id="controls",
            children=[
                html.Div(
                    id="div_sliders",
                    children=sliders_div,
                ),
                html.Div(
                    id="div_radios",
                    children=[
                        "Discrete z",
                        dbc.RadioItems(
                            id="radios",
                            value=0,
                            options=[
                                {"label": str(i), "value": i} for i in range(n_buttons)
                            ],
                        ),
                    ],
                ),
                html.Div(id="div_button", children=[dbc.Button("Sample")]),
            ],
        ),
        html.Div(id="figure", children=dcc.Graph(id="graph")),
    ],
    id="frame",
)


@callback(
    Output("graph", "figure"),
    Input({"type": "slider", "index": ALL}, "value"),
    Input("radios", "value"),
    State("graph", "figure"),
)
def changed_disc(sliders, radio, figure):
    z_disc = np.zeros(n_buttons)
    z_disc[radio] = 1

    z_cont = np.zeros(n_cont)
    z_cont[i_cont] = sliders

    z = np.concatenate([z_cont, z_disc], axis=-1, dtype=np.float32)[None, :]

    image = model(z=z)["image"].squeeze()

    if figure is None:
        fig = px.imshow(image, color_continuous_scale=["black", "white"])
        fig.update_layout(
            coloraxis_showscale=False,
            coloraxis_cmax=1,
            coloraxis_cmin=0,
            xaxis_visible=False,
            yaxis_visible=False,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor="#1b1513",
            paper_bgcolor="#1b1513",
        )
        return fig
    else:
        patch = Patch()
        patch["data"][0]["z"] = image
        return patch


@callback(
    Output({"type": "slider", "index": ALL}, "value"),
    Output("radios", "value"),
    Input("div_button", "n_clicks"),
)
def sample(n):
    z_cont = np.random.normal(0, 1, n_sliders)
    z_disc = np.random.randint(0, n_buttons)
    return z_cont.tolist(), z_disc


if __name__ == "__main__":
    app.run(debug=False)
