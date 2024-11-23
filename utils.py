import tensorflow as tf
import numpy as np

def preprocess(images):
    # Normalize
    images = images.reshape(images.shape[0], 28, 28, 1) / 255
    # Make images 32x32
    images = tf.image.resize(images, [32, 32])

    return images

def get_dataset(batch):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.concatenate((x_train, x_test), axis=0)
    y_train = np.concatenate((y_train, y_test), axis=0)

    x_train = preprocess(x_train)
    train_size = x_train.shape[0]

    x_train = tf.cast(x_train, tf.float32)
    y_train = tf.cast(y_train, tf.float32)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .batch(batch, drop_remainder=True)
        .shuffle(buffer_size=train_size)
        .prefetch(2)
    )

    return train_dataset
