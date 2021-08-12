from typing import List

import tensorflow as tf

from tensorflow_addons.activations import sparsemax


def glu(x, n_units=None):
    """Generalized linear unit nonlinear activation."""
    return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])


class GhostBatchNormalization(tf.keras.Model):
    """GhostBatchNormalization
    modified from https://github.com/ostamand/tensorflow-tabnet
    """

    def __init__(
        self, virtual_divider: int = 1, momentum: float = 0.9, epsilon: float = 1e-5
    ):
        super(GhostBatchNormalization, self).__init__()
        self.virtual_divider = virtual_divider
        self.bn = BatchNormInferenceWeighting(momentum=momentum)

    def call(self, x, training: bool = None, alpha: float = 0.0):
        if training:
            chunks = tf.split(x, self.virtual_divider)
            x = [self.bn(x, training=True) for x in chunks]
            return tf.concat(x, 0)
        return self.bn(x, training=False, alpha=alpha)

    @property
    def moving_mean(self):
        return self.bn.moving_mean

    @property
    def moving_variance(self):
        return self.bn.moving_variance


class BatchNormInferenceWeighting(tf.keras.layers.Layer):
    """BatchNormInferenceWeighting
    modified from https://github.com/ostamand/tensorflow-tabnet
    """

    def __init__(self, momentum: float = 0.9, epsilon: float = None):
        super(BatchNormInferenceWeighting, self).__init__()
        self.momentum = momentum
        self.epsilon = tf.keras.backend.epsilon() if epsilon is None else epsilon

    def build(self, input_shape):
        channels = input_shape[-1]

        self.gamma = tf.Variable(
            initial_value=tf.ones((channels,), tf.float32),
            trainable=True,
        )
        self.beta = tf.Variable(
            initial_value=tf.zeros((channels,), tf.float32),
            trainable=True,
        )

        self.moving_mean = tf.Variable(
            initial_value=tf.zeros((channels,), tf.float32),
            trainable=False,
        )
        self.moving_mean_of_squares = tf.Variable(
            initial_value=tf.zeros((channels,), tf.float32),
            trainable=False,
        )

    def __update_moving(self, var, value):
        var.assign(var * self.momentum + (1 - self.momentum) * value)

    def __apply_normalization(self, x, mean, variance):
        return self.gamma * (x - mean) / tf.sqrt(variance + self.epsilon) + self.beta

    def call(self, x, training: bool = None, alpha: float = 0.0):
        mean = tf.reduce_mean(x, axis=0)
        mean_of_squares = tf.reduce_mean(tf.pow(x, 2), axis=0)

        if training:
            # update moving stats
            self.__update_moving(self.moving_mean, mean)
            self.__update_moving(self.moving_mean_of_squares, mean_of_squares)

            variance = mean_of_squares - tf.pow(mean, 2)
            x = self.__apply_normalization(x, mean, variance)
        else:
            mean = alpha * mean + (1 - alpha) * self.moving_mean
            variance = (
                alpha * mean_of_squares + (1 - alpha) * self.moving_mean_of_squares
            ) - tf.pow(mean, 2)
            x = self.__apply_normalization(x, mean, variance)

        return x


class FeatureBlock(tf.keras.Model):
    """FeatureBlock
    modified from https://github.com/ostamand/tensorflow-tabnet
    """

    def __init__(
        self,
        feature_dim: int,
        apply_glu: bool = True,
        bn_momentum: float = 0.9,
        bn_virtual_divider: int = 32,
        fc: tf.keras.layers.Layer = None,
        epsilon: float = 1e-5,
    ):
        super(FeatureBlock, self).__init__()
        self.apply_glu = apply_glu
        self.feature_dim = feature_dim
        units = feature_dim * 2 if apply_glu else feature_dim

        self.fc = tf.keras.layers.Dense(units, use_bias=False) if fc is None else fc
        self.bn = GhostBatchNormalization(
            virtual_divider=bn_virtual_divider, momentum=bn_momentum
        )

    def call(self, x, training: bool = None, alpha: float = 0.0):
        x = self.fc(x)
        x = self.bn(x, training=training, alpha=alpha)
        if self.apply_glu:
            return glu(x, self.feature_dim)
        return x


class AttentiveTransformer(tf.keras.Model):
    """AttentiveTransformer
    modified from https://github.com/ostamand/tensorflow-tabnet
    """

    def __init__(self, feature_dim: int, bn_momentum: float, bn_virtual_divider: int):
        super(AttentiveTransformer, self).__init__()
        self.block = FeatureBlock(
            feature_dim,
            bn_momentum=bn_momentum,
            bn_virtual_divider=bn_virtual_divider,
            apply_glu=False,
        )

    def call(self, inputs, training=None, alpha: float = 0.0):
        x, prior_scales = inputs
        x = self.block(x, training=training, alpha=alpha)
        return sparsemax(x * prior_scales)


class FeatureTransformer(tf.keras.Model):
    """FeatureTransformer
    modified from https://github.com/ostamand/tensorflow-tabnet
    """

    def __init__(
        self,
        feature_dim: int,
        fcs: List[tf.keras.layers.Layer] = [],
        n_total: int = 4,
        n_shared: int = 2,
        bn_momentum: float = 0.9,
        bn_virtual_divider: int = 1,
    ):
        super(FeatureTransformer, self).__init__()
        self.n_total, self.n_shared = n_total, n_shared

        kargs = {
            "feature_dim": feature_dim,
            "bn_momentum": bn_momentum,
            "bn_virtual_divider": bn_virtual_divider,
        }

        # build blocks
        self.blocks: List[FeatureBlock] = []
        for n in range(n_total):
            # some shared blocks
            if fcs and n < len(fcs):
                self.blocks.append(FeatureBlock(**kargs, fc=fcs[n]))
            # build new blocks
            else:
                self.blocks.append(FeatureBlock(**kargs))

    def call(
        self, x: tf.Tensor, training: bool = None, alpha: float = 0.0
    ) -> tf.Tensor:
        x = self.blocks[0](x, training=training, alpha=alpha)
        for n in range(1, self.n_total):
            x = x * tf.sqrt(0.5) + self.blocks[n](x, training=training, alpha=alpha)
        return x

    @property
    def shared_fcs(self):
        return [self.blocks[i].fc for i in range(self.n_shared)]
