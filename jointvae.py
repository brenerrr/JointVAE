# %%
import tensorflow as tf
from tensorflow import keras
from keras.layers import (
    Flatten,
    Dense,
    Conv2D,
    Conv2DTranspose,
    Layer,
    Input,
    Softmax,
    Reshape,
    ReLU,
    ZeroPadding2D,
    Lambda,
)
import logging
from metrics import LastValue


class NormalSampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class GumbelSampling(Layer):
    def call(self, alpha, temp=1):
        rand_unif = tf.random.uniform(shape=tf.shape(alpha))
        gumbel = -tf.math.log(-tf.math.log(rand_unif + 1e-12) + 1e-12)
        return tf.math.softmax((tf.math.log(alpha + 1e-12) + gumbel) / temp, axis=-1)


class JointVAE(keras.Model):
    """
    Joint Variational Auto Encoder
    """

    def __init__(self, cont_dim: int, disc_dim: list, c_cont: dict, c_disc: dict):
        super(JointVAE, self).__init__()
        self.cont_dim = cont_dim  # Dimension of continuous latent space
        self.disc_dim = disc_dim  # Dimension of discrete latent space
        self.c_cont = c_cont
        self.c_disc = c_disc
        self.image_res = (32, 32, 1)
        self.image_n_pixels = tf.reduce_prod(self.image_res).numpy()

        # Logger
        self.logger = logging.getLogger(__name__)

        # Encoder
        encoder_inputs = keras.Input(shape=self.image_res, name="input")
        x = ZeroPadding2D((1, 1))(encoder_inputs)
        x = Conv2D(32, 4, activation=ReLU(), strides=2)(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(64, 4, activation=ReLU(), strides=2)(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(64, 4, activation=ReLU(), strides=2)(x)
        x = Flatten()(x)
        x = Dense(256, activation=ReLU())(x)

        z_mean = Dense(self.cont_dim, name="z_mean")(x)
        z_log_var = Dense(self.cont_dim, name="z_log_var")(x)
        z_cont = NormalSampling(name="z_cont")([z_mean, z_log_var])

        z_disc = []
        alphas = []
        for i, n_classes in enumerate(self.disc_dim):
            alpha = Dense(n_classes, activation=Softmax(), name=f"alpha_{i}")(x)
            z_disc.append(GumbelSampling()(alpha))
            alphas.append(alpha)
        z_disc = tf.concat(z_disc, axis=-1)
        naming_layer = Lambda(lambda x: x, name="z_disc")
        z_disc = naming_layer(z_disc)

        self.encoder = keras.Model(
            encoder_inputs,
            [z_cont, z_disc, (z_mean, z_log_var, alphas)],
            name="encoder",
        )

        # Decoder
        z = tf.concat([z_cont, z_disc], axis=-1)
        decoder_inputs = keras.Input(shape=z.shape[-1], name="z")
        x = Dense(256, activation=ReLU())(decoder_inputs)
        x = Dense(64 * 4 * 4, activation=ReLU())(x)
        x = Reshape((4, 4, 64))(x)
        x = Conv2DTranspose(32, 4, activation=ReLU(), strides=2, padding="same")(x)
        x = Conv2DTranspose(32, 4, activation=ReLU(), strides=2, padding="same")(x)
        reconstruction = Conv2DTranspose(
            1, 3, strides=2, padding="same", activation="sigmoid", name="image"
        )(x)
        self.decoder = keras.Model(decoder_inputs, reconstruction, name="decoder")

        self.loss_tracker = keras.metrics.Mean(name="total_loss")
        self.recon_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_disc_tracker = keras.metrics.Mean(name="kl_discrete")
        self.kl_cont_tracker = keras.metrics.Mean(name="kl_continuous")
        self.dummy_tracker = LastValue(name="iter_tracker")

        self.iter = tf.Variable(0, name="iteration", dtype=tf.float32, trainable=False)

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.recon_loss_tracker,
            self.kl_disc_tracker,
            self.kl_cont_tracker,
            self.dummy_tracker,
        ]

    def train_step(self, data):
        X, _ = data
        batch_size = tf.shape(X)[0]
        with tf.GradientTape() as tape:
            z_cont, z_disc, (z_mean, z_log_var, alphas) = self.encoder(X)
            reconstruction = self.decoder(tf.concat([z_cont, z_disc], axis=-1))

            # Reconstruction loss
            reconstruction_loss = keras.losses.binary_crossentropy(
                tf.reshape(X, (batch_size, -1)),
                tf.reshape(reconstruction, (batch_size, -1)),
            )
            reconstruction_loss = tf.reduce_mean(
                reconstruction_loss * self.image_n_pixels
            )  # Mean accross batches

            # KL Divergence of continuous variables between posterior and Gaussian
            kl_cont = tf.reduce_sum(self.kl_normal(z_mean, z_log_var))

            # KL Divergence of discrete variables between categorical latent space
            # and uniform cagetorical
            kl_disc = tf.reduce_sum(self.kl_gumbel(alphas))

            # Add capacity
            c_cont, c_disc = self.calculate_capacity()
            gamma_cont, gamma_disc = self.c_cont["gamma"], self.c_disc["gamma"]

            loss_cont = tf.abs(kl_cont - c_cont) * gamma_cont
            loss_disc = tf.abs(kl_disc - c_disc) * gamma_disc

            loss = (reconstruction_loss + loss_disc + loss_cont) / self.image_n_pixels

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.loss_tracker.update_state(loss * self.image_n_pixels)
        self.recon_loss_tracker.update_state(reconstruction_loss)
        self.kl_cont_tracker.update_state(kl_cont)
        self.kl_disc_tracker.update_state(kl_disc)
        # self.dummy_tracker.update_state(c_disc)

        self.iter.assign_add(1)

        return {
            "loss": self.loss_tracker.result(),
            "reconstruction_loss": self.recon_loss_tracker.result(),
            "kl_disc": self.kl_disc_tracker.result(),
            "kl_cont": self.kl_cont_tracker.result(),
            "c_disc": c_disc,
            "c_cont": c_cont,
            "iter": self.iter,
        }

    def kl_normal(self, mean, logvar):
        kl_cont = -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        kl_cont = tf.reduce_mean(kl_cont, axis=0)  # Mean accross batches
        return kl_cont

    def kl_gumbel(self, alphas):
        kl_disc = []
        for alpha in alphas:
            n = alpha.shape[-1]
            neg_entropy = tf.reduce_mean(  # Mean accross batches
                tf.reduce_sum(alpha * tf.math.log(alpha + 1e-12), axis=-1)
            )
            kl_disc.append(tf.math.log(float(n) + 1e-12) + neg_entropy)
        return kl_disc

    def call(self, inputs):
        z_cont, z_disc, _ = self.encoder(inputs[0])
        z = tf.concat([z_cont, z_disc], axis=-1)
        reconstruction = self.decoder(z)
        return reconstruction

    def sample(self, label):
        z = tf.random.normal(shape=(1, 2))
        label = tf.reshape(label, (1, 1))
        e = self.embedder(label)
        reconstruction = self.decoder(tf.concat([z, e], axis=-1))
        return reconstruction

    def calculate_capacity(self):
        p = tf.minimum(self.iter / self.c_cont["steps"], 1)
        c_cont = p * (self.c_cont["max"] - self.c_cont["min"]) + self.c_cont["min"]

        p = tf.minimum(self.iter / self.c_disc["steps"], 1)
        c_disc = p * (self.c_disc["max"] - self.c_disc["min"]) + self.c_disc["min"]

        return tf.cast(c_cont, tf.float32), tf.cast(c_disc, tf.float32)


if __name__ == "__main__":
    model = JointVAE(10, [10], None, None)
    model.encoder.summary()
    # %%
    model.decoder.summary()
