from typing import Tuple

import tensorflow as tf

from staker.models.base import TFInputBuilder


class MLP(TFInputBuilder):
    """MLP implementation for numerAI competition"""

    def __init__(
        self,
        config,
        vocab,
    ):
        super(MLP, self).__init__()
        self.config = config
        self.vocab = vocab

        self.tabnet_config = config.MODEL_CONFIG.type
        # Build input and preprocessing layers
        self.build_input_layers()
        # BatchNorm
        self.bn = tf.keras.layers.BatchNormalization()
        # Hidden
        self.dense1 = tf.keras.layers.Dense(1024, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dense2 = tf.keras.layers.Dense(1024, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.dense3 = tf.keras.layers.Dense(1024, activation="relu")
        self.dropout3 = tf.keras.layers.Dropout(0.1)
        self.dense4 = tf.keras.layers.Dense(1024, activation="relu")
        self.dropout4 = tf.keras.layers.Dropout(0.1)
        # Output
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(
        self, features: tf.Tensor, training: bool = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # TO DO: Improve forward pass code
        num, categ = self.encode_inputs(features)
        features = tf.keras.layers.concatenate(num + categ)
        x = self.bn(features)
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        x = self.dropout3(x)
        x = self.dense4(x)
        x = self.dropout4(x)
        # output
        yhat = self.output_layer(x)
        return yhat
