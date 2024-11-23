# %% Modules
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow import keras
from tensorboardcallback import TensorBoardCallback
from jointvae import JointVAE
from utils import get_dataset
import shutil

tf.random.set_seed(10)

# Check if tensorflow is using gpu
device_name = tf.test.gpu_device_name()
if not device_name:
    raise SystemError("GPU device not found")
print("Found GPU at: {}".format(device_name))


# %% Load hyperparameters
hp_file = os.path.join(os.path.dirname(__file__), "hyperparameters.json")
with open(hp_file, "r") as f:
    HP = json.load(f)

# %% Setup logging
wandb_log = False
wandb_log = input("Should log on wandb? Y-Yes or N-no\n").lower() == "y"
if wandb_log:
    import wandb

    wandb.init(
        project="jointvae",
        config=HP,
    )
name = wandb.run.name if wandb_log else "model"
tb_dir = os.path.join(os.path.dirname(__file__), "./logs")
shutil.rmtree(tb_dir, ignore_errors=True)
# %% Callbacks
train_dataset = get_dataset(HP["batch_size"])
samples = list(train_dataset.take(200).as_numpy_iterator())
tensorboard_cb = TensorBoardCallback(
    samples, HP["n_examples"], wandb_log, log_dir=tb_dir, histogram_freq=10
)

# %% Setup model
model = JointVAE(HP["cont_dim"], HP["disc_dim"], HP["c_cont"], HP["c_disc"])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=HP["lr"]), run_eagerly=True)
# %%
model.encoder.summary()
# %%
model.decoder.summary()

# %% Train
model.fit(train_dataset, epochs=HP["epochs"], callbacks=[tensorboard_cb])
if wandb_log:
    wandb.finish()

# %% Export
# Model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
filepath = os.path.join(os.path.dirname(__file__), f"{name}.tflite")
with open(filepath, "wb") as f:
    f.write(tflite_model)
model.save_weights(os.path.join(os.path.dirname(__file__), f"{name}.h5"))

# Decoder
converter = tf.lite.TFLiteConverter.from_keras_model(model.decoder)
tflite_model = converter.convert()
filepath = os.path.join(os.path.dirname(__file__), f"decoder_{name}.tflite")
with open(filepath, "wb") as f:
    f.write(tflite_model)

# Encoder
converter = tf.lite.TFLiteConverter.from_keras_model(model.encoder)
tflite_model = converter.convert()
filepath = os.path.join(os.path.dirname(__file__), f"encoder_{name}.tflite")
with open(filepath, "wb") as f:
    f.write(tflite_model)

# Hyperparameters
hp_file = os.path.join(os.path.dirname(__file__), f"{name}.json")
with open(hp_file, "w") as f:
    json.dump(HP, f)

# %%

# filepath = os.path.join(os.path.dirname(__file__), f"encoder_model.tflite")
# encoder = tf.lite.Interpreter(filepath)

# signature = encoder.get_signature_runner()
# signature.get_input_details()["input"]["index"]
