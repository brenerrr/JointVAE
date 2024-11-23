# %% Modules
import plotly
import json
import os
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from utils import get_dataset
import pandas as pd
from plotly.subplots import make_subplots
import argparse

try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        # For displaying latex in vs code notebooks
        from IPython.display import display, HTML

        plotly.offline.init_notebook_mode()
        display(
            HTML(
                '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
            )
        )
    print("Executou tudo")
except:
    # Not executing as notebook
    pass

sys.exit()
# %%


class Args(argparse.Namespace):
    data = "./data/penn"
    model = "LSTM"
    emsize = 200
    nhid = 200


args = Args()

parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="Name of decoder")
parser.add_argument("-i", type=int, nargs="+", help="Order of continuous latent dims")
args = parser.parse_args()
name = args.name
i_cont = args.i

hp_file = os.path.join(os.path.dirname(__file__), f"{name}.json")
with open(hp_file, "r") as f:
    HP = json.load(f)


# %% Load model
filepath = os.path.join(os.path.dirname(__file__), f"encoder_{name}.tflite")
encoder = tf.lite.Interpreter(filepath)
encoder = encoder.get_signature_runner()

filepath = os.path.join(os.path.dirname(__file__), f"decoder_{name}.tflite")
decoder = tf.lite.Interpreter(filepath)
decoder = decoder.get_signature_runner()

filepath = os.path.join(os.path.dirname(__file__), f"{name}.tflite")
model = tf.lite.Interpreter(filepath)
model = model.get_signature_runner()


# %% Get data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.concatenate((x_train, x_test), axis=0)
y_train = np.concatenate((y_train, y_test), axis=0)

train_dataset = get_dataset(1000)
samples = list(train_dataset.take(2).as_numpy_iterator())
samples_x, samples_y = list(zip(*samples))
samples_x, samples_y = np.array(samples_x), np.array(samples_y)
samples_x = np.concatenate(samples_x, axis=0)
samples_y = np.concatenate(samples_y, axis=0)

# %% Encode samples
# z_cont_default, z_disc, *_ = model.encoder(samples_x)


output = encoder(input=samples_x)
z_cont = output["z_cont"]
z_disc = output["z_disc"]


z = np.concatenate([z_cont, z_disc], axis=1)
z = np.array(z)
y = np.array(samples_y)

# %% Distribution
# Create grid of images with plotly, in which each image is a histogram of
# the latent distribution (px.histogram)
n_cols = 2
fig = make_subplots(
    rows=5,
    cols=2,
    subplot_titles=[f"Cont. Latent $z_{i}$" for i in range(HP["cont_dim"])],
)
data = pd.DataFrame(z[:, :])
data["label"] = y
data
for i in range(HP["cont_dim"]):
    histograms = px.histogram(data, x=i, color="label")
    for histogram in histograms.data:
        histogram["showlegend"] = False
        # Add histograms to subplot
        fig.add_trace(histogram, row=(i // n_cols) + 1, col=(i % n_cols) + 1)
        fig.update_yaxes(range=[0, 70], row=(i // n_cols) + 1, col=(i % n_cols) + 1)
        fig.update_xaxes(range=[-4, 4], row=(i // n_cols) + 1, col=(i % n_cols) + 1)
fig.update_layout(
    height=1200,
    margin=dict(l=0, r=0, b=0),
    barmode="stack",
)
fig.show()


# %% # Check reconstructions
inputs = samples_x[0:10, ...]
reconstructions = model(input_1=inputs, input_2=np.array(0, dtype=np.float32))[
    "output_1"
]
# z_cont, z_disc, *_ = model.encoder(inputs)
output = encoder(input=inputs)
z_cont = output["z_cont"]
z_disc = output["z_disc"]

# Images
inputs_cat = np.concatenate(inputs, axis=0).squeeze()
reconstructions_cat = np.concatenate(reconstructions, axis=0).squeeze()
image = np.concatenate((inputs_cat, reconstructions_cat), axis=1)
fig = px.imshow(image)

# Annotations
image_height = samples_x.shape[1]
for i, (z_cont, z_disc) in enumerate(zip(z_cont, z_disc)):
    text_cont = "$" + ", ".join([f"\\tilde{{z}}_{i}={z_cont[i]:.2f}" for i in i_cont])
    text_disc = ", ".join(
        [f"\\hat{{z}}_{i}={z_disc[i]:.3f}" for i in range(HP["disc_dim"][0])]
    )
    text_disc += "$"
    text = f"{text_cont} \\\\ {text_disc}"
    text = [text]

    trace = go.Scatter(
        x=[70],
        y=[(i + 0.5) * image_height],
        mode="text",
        text=text,
        textposition="middle right",
        showlegend=False,
        # textfont_size=20,
    )

    fig.add_trace(trace)

fig.update_layout(
    height=1000,
    coloraxis={"colorscale": [[0.0, "black"], [1.0, "white"]]},
    xaxis_visible=False,
    yaxis_visible=False,
    xaxis_range=[0, 500],
    margin=dict(l=0, r=0, b=0),
    plot_bgcolor="white",
    coloraxis_showscale=False,
    coloraxis_cmax=1,
    coloraxis_cmin=0,
    hovermode=False,
)
fig.show()

# %% # Animate latent space walk

steps = [15, 30, 15]
z_x = np.linspace(0, 1.5, steps[0])
z_x = np.concatenate([z_x, np.linspace(1.5, -1.5, steps[1])])
z_x = np.concatenate([z_x, np.linspace(-1.5, 0, steps[2])])


all_images = []
for i_cont_ in i_cont:
    images = []
    for i_disc in range(HP["disc_dim"][0]):
        z = np.zeros((z_x.size, HP["cont_dim"] + HP["disc_dim"][0]))
        mask = np.zeros(HP["cont_dim"] + HP["disc_dim"][0])
        mask[[i_cont_, i_disc + HP["cont_dim"]]] = 1
        z = z + z_x[:, None] * mask
        z[:, i_disc + HP["cont_dim"]] = 1
        z = tf.constant(z, dtype=tf.float32)
        images.append(decoder(z=z)["image"].squeeze())

    all_images.append(images)

all_images = np.array(all_images)
all_images = np.concatenate(np.concatenate(all_images, axis=2), axis=2)
# %% Create frames
frames = []
for image in all_images:
    trace = px.imshow(image[::-1, :]).data[0]
    frames.append(go.Frame(data=trace))

# Create animation
layout = dict(
    xaxis_visible=False,
    yaxis_visible=False,
    xaxis_range=[0, 320],
    yaxis=dict(scaleanchor="x", scaleratio=1),
    margin=dict(l=0, r=0, b=0, t=0),
    coloraxis={"colorscale": [[0.0, "black"], [1.0, "white"]]},
    coloraxis_showscale=False,
    coloraxis_cmax=1,
    coloraxis_cmin=0,
    paper_bgcolor="white",
    plot_bgcolor="white",
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            y=1,
            x=0,
            xanchor="left",
            yanchor="top",
            pad=dict(t=0, r=10),
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        dict(
                            frame=dict(duration=60, redraw=True),
                            transition_duration=0,
                            fromcurrent=True,
                            mode="immediate",
                        ),
                    ],
                )
            ],
        )
    ],
)


fig = go.Figure(data=frames[0]["data"], layout=layout, frames=frames)
fig.show(config={"displayModeBar": False})
# Set renderer as browser
# import plotly.io as pio

# pio.renderers.default = "browser"
# pio.renderers

# %%
