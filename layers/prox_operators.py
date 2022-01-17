import tensorflow as tf


class ProxOperators:

    @staticmethod
    def soft_threshold(x, threshold, name=None):
        """ Proximal operator for the term ||x||_1"""
        with tf.name_scope(name or 'soft_threshold'):
            x = tf.convert_to_tensor(x, name='x')
            threshold = tf.convert_to_tensor(threshold, dtype=x.dtype, name='threshold')
            return tf.sign(x) * tf.maximum(tf.abs(x) - threshold, 0.)

    @staticmethod
    def least_squares_prox(A, x, y, eta, name=None):
        """ Proximal operator for the term ||y - Ax||^2_2 = (I + eta A'A)^-1 * (x + eta A'y) """
        with tf.name_scope(name or 'least_squares_prox'):
            A = tf.convert_to_tensor(A, name='A')
            x = tf.convert_to_tensor(x, name='x')
            y = tf.convert_to_tensor(y, name='y')
            eta = tf.convert_to_tensor(eta, name='eta')
            t1 = tf.linalg.inv(tf.eye(A.shape[1]) + eta * tf.matmul(tf.transpose(A), A))
            t2 = x + eta * tf.matmul(tf.transpose(A), y)
            return tf.matmul(t1, t2)
