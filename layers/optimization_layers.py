import tensorflow as tf
from tensorflow.keras import layers

from prox_operators import ProxOperators


class ProximalL1Layer(layers.Layer):
    """
    Single layer for an unrolled optimization network performing least squares minimization with l1 constraints
    solved using the Proximal Gradient Algorithm.
    """

    def __init__(self, A, eta):

        super(ProximalL1Layer, self).__init__()
        self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)
        self.eta = tf.Variable(initial_value=eta, dtype=tf.float32, trainable=True)

    def call(self, inputs, **kwargs):
        [xt, y] = inputs
        nabla_f = tf.matmul(tf.transpose(self.A), tf.matmul(self.A, xt) - y[0])
        zt1 = xt - (self.eta * nabla_f)
        xt1 = ProxOperators.soft_threshold(zt1, self.eta)
        return [xt1, y]


class ADMML1Layer(layers.Layer):
    """
    Single layer for an unrolled optimization network performing least squares minimization with l1 constraints
    solved using the Alternating Direction Method of Multipliers Algorithm.
    """

    def __init__(self, A, eta):

        super(ADMML1Layer, self).__init__()
        self.A = tf.Variable(initial_value=A, dtype=tf.float32, trainable=False)
        self.eta = tf.Variable(initial_value=eta, dtype=tf.float32, trainable=True)

    def call(self, inputs, **kwargs):
        [xt, st, y] = inputs
        zt1 = ProxOperators.least_squares_prox(self.A, xt - st, y, self.eta)
        xt1 = ProxOperators.soft_threshold(zt1 + st, self.eta)
        st1 = st + (zt1 - xt1)
        return [xt1, st1, y]
