import io
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import wandb


class TensorBoardCallback(keras.callbacks.TensorBoard):
    def __init__(
        self,
        samples: list,
        n_examples: int,  # Number of reconstructions logged
        wandb_log=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_examples = n_examples
        self.wandb_log = wandb_log
        samples_x, samples_y = list(zip(*samples))
        samples_x, samples_y = np.array(samples_x), np.array(samples_y)
        self.samples_x = np.concatenate(samples_x, axis=0)
        self.samples_y = np.concatenate(samples_y, axis=0)
        self.cont_writers = []
        self.disc_writers = []

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)

        reconstructions = self.model(
            [self.samples_x[: self.n_examples], self.samples_y[: self.n_examples]]
        )
        z_cont, z_disc, (z_mean, z_log_var, alphas) = self.model.encoder(self.samples_x)

        z = tf.concat([z_cont, z_disc], axis=-1)

        kl_cont = self.model.kl_normal(z_mean, z_log_var)
        kl_disc = self.model.kl_gumbel(alphas)

        # Initialize writers
        if self.cont_writers == []:
            for i, _ in enumerate(kl_cont):
                writer = tf.summary.create_file_writer(f"logs/{i}")
                self.cont_writers.append(writer)

        if self.disc_writers == []:
            for i, _ in enumerate(kl_disc):
                writer = tf.summary.create_file_writer(f"logs/{i}")
                self.disc_writers.append(writer)

        # Save reconstruction images
        with self._train_writer.as_default():
            tf.summary.image(
                f"outputs", reconstructions, max_outputs=self.n_examples, step=epoch
            )
            if epoch == 0:
                tf.summary.image(
                    f"inputs", self.samples_x, max_outputs=self.n_examples, step=epoch
                )

            def create_scatter(x, y, color, name):
                fig, ax = plt.subplots(1)
                ax.scatter(x, y, c=color, alpha=0.5)
                ax.set_xlim([-5, 5])
                ax.set_ylim([-5, 5])
                ax.grid()

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                image.shape
                tf.summary.image(name, tf.expand_dims(image, 0), step=epoch)
                plt.close(fig)

            # Latent space (1st and 2nd dims)
            create_scatter(
                z[:, 0], z[:, 1], self.samples_y, "latent_space_distributions"
            )

        for i, writer in enumerate(self.cont_writers):
            with writer.as_default():
                tf.summary.scalar("KL_Continuous", kl_cont[i], epoch)

        for i, writer in enumerate(self.disc_writers):
            with writer.as_default():
                tf.summary.scalar("KL_Discrete", kl_disc[i], epoch)

        # Wandb logging
        for i, kl in enumerate(kl_cont):
            if self.wandb_log:
                wandb.log({f"KL_Continuous/{i}": kl}, step=epoch)

        for i, kl in enumerate(kl_disc):
            if self.wandb_log:
                wandb.log({f"KL_Discrete/{i}": kl}, step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.wandb_log:
            wandb.log(logs, step=epoch)
