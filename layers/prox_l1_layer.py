import tensorflow as tf
from tensorflow.keras import layers


class ProximalL1Layer(layers.Layer):

    def __init__(self, A, eta):

        super(ProximalL1Layer, self).__init__()
        self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)
        self.eta = tf.Variable(initial_value=eta, dtype=tf.float32, trainable=True)

    @staticmethod
    def soft_threshold(x, threshold, name=None):
        with tf.name_scope(name or 'soft_threshold'):
            x = tf.convert_to_tensor(x, name='x')
            threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
            return tf.sign(x) * tf.maximum(tf.abs(x) - threshold, 0.)

    def call(self, inputs, **kwargs):
        [xt, y] = inputs
        nabla_f = tf.matmul(tf.transpose(self.A), tf.matmul(self.A, xt) - y[0])
        zt1 = xt - (self.eta * nabla_f)
        xt1 = ProximalL1Layer.soft_threshold(zt1, self.eta)
        return [xt1, y]
