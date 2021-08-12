import tensorflow as tf
from staker.models.base import TFInputBuilder
from typing import List, Text, Tuple

from staker.models.layers import FeatureTransformer, AttentiveTransformer

class TabNet(TFInputBuilder):
    """TabNet implementation for numerAI competition
    """
    def __init__(
        self,
        config,
        vocab,
    ):
        super(TabNet, self).__init__()
        self.tabnet_config = config.MODEL_CONFIG.type
        # Define configs for saving
        self.configs = {
            "feature_dim": self.tabnet_config.feature_dim,
            "output_dim": self.tabnet_config.output_dim,
            "n_step": self.tabnet_config.n_step,
            "n_total": self.tabnet_config.n_total,
            "n_shared": self.tabnet_config.n_shared,
            "relaxation_factor": self.tabnet_config.relaxation_factor,
            "sparsity_coefficient": self.tabnet_config.sparsity_coefficient,
            "bn_epsilon": self.tabnet_config.bn_epsilon,
            "bn_momentum": self.tabnet_config.bn_momentum,
            "bn_virtual_divider": self.tabnet_config.bn_virtual_divider,
        }
        # Global config
        self.config = config
        # Vocab for preprocessing
        self.vocab = vocab
        # Counts the feature layer dimension
        self.num_features = len(vocab.keys())
        for name, value in config.FEATURES_CONFIG.__dict__.items():
            if value.__class__.__name__ == "CategFeature" and value.include:
                self.num_features = self.num_features + value.embedding_size - 1
        # Build input and preprocessing layers
        self.build_input_layers()
        self.concat = tf.keras.layers.Concatenate()
        # ? Switch to Ghost Batch Normalization
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=self.tabnet_config.bn_momentum, epsilon=self.tabnet_config.bn_epsilon
        )
        # Argument for sublayers
        kargs = {
            "feature_dim": self.tabnet_config.feature_dim + self.tabnet_config.output_dim,
            "n_total": self.tabnet_config.n_total,
            "n_shared": self.tabnet_config.n_shared,
            "bn_momentum": self.tabnet_config.bn_momentum,
            "bn_virtual_divider": self.tabnet_config.bn_virtual_divider,
        }

        # first feature transformer block is built first to get the shared blocks
        self.feature_transforms: List[FeatureTransformer] = [
            FeatureTransformer(**kargs)
        ]
        self.attentive_transforms: List[AttentiveTransformer] = []
        for i in range(self.tabnet_config.n_step):
            self.feature_transforms.append(
                FeatureTransformer(**kargs, fcs=self.feature_transforms[0].shared_fcs)
            )
            self.attentive_transforms.append(
                AttentiveTransformer(self.num_features, self.tabnet_config.bn_momentum, self.tabnet_config.bn_virtual_divider)
            )
        # Output layer
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid", use_bias=False)

    def call(
        self, features: tf.Tensor, training: bool = None, alpha: float = 0.0
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # TO DO: Improve forward pass code
        num, categ = self.encode_inputs(features)
        all_features = num + categ
        features = self.concat(all_features)

        bs = tf.shape(features)[0]
        out_agg = tf.zeros((bs, self.tabnet_config.output_dim))
        prior_scales = tf.ones((bs, self.num_features))
        masks = []

        features = self.bn(features, training=training)
        masked_features = features

        total_entropy = 0.0

        for step_i in range(self.tabnet_config.n_step + 1):
            x = self.feature_transforms[step_i](
                masked_features, training=training, alpha=alpha
            )

            if step_i > 0:
                out = tf.keras.activations.relu(x[:, : self.tabnet_config.output_dim])
                out_agg += out

            # no need to build the features mask for the last step
            if step_i < self.tabnet_config.n_step:
                x_for_mask = x[:, self.tabnet_config.output_dim :]

                mask_values = self.attentive_transforms[step_i](
                    inputs=[x_for_mask, prior_scales], training=training, alpha=alpha
                )

                # relaxation factor of 1 forces the feature to be only used once.
                prior_scales *= self.tabnet_config.relaxation_factor - mask_values

                masked_features = tf.multiply(mask_values, features)

                # entropy is used to penalize the amount of sparsity in feature selection
                total_entropy = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(mask_values, tf.math.log(mask_values + 1e-15)),
                        axis=1,
                    )
                )

                masks.append(tf.expand_dims(tf.expand_dims(mask_values, 0), 3))

        loss = total_entropy / self.tabnet_config.n_step

        # Output layers
        out = self.dropout(out_agg, training=training)
        yhat = self.output_layer(out, training=training)

        # Loss
        if training:
            self.add_loss(-self.tabnet_config.sparsity_coefficient * loss)

        return yhat
