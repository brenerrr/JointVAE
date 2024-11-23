import tensorflow as tf


class LastValue(tf.keras.metrics.Metric):
    def __init__(self, name="last value", **kwargs):
        super(LastValue, self).__init__(name=name, **kwargs)
        self.value = tf.Variable(0, name="value", trainable=False, dtype=tf.float32)

    def update_state(self, value):
        self.value.assign(value)

    def result(self):
        return self.value
