import os
import pickle

from typing import Dict
from typing import List
from typing import NoReturn
from typing import Text
from typing import Tuple

import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing


class TFInputBuilder(tf.keras.models.Model):
    """Base class in order to build diverse model
    architecture easily. Build the input graph
    """

    def build_input_layers(self) -> NoReturn:
        """Build input layers, preprocessing layer and embedding
        which make the output of this ready to be ingested by any layer.

        :return: Inputs layers and preprocessing layers
        :rtype: Tuple[Dict,Dict]
        """
        self.inputs_layers: Dict[Text, tf.keras.layers.Input] = dict()
        self.preprocess_layers: Dict = dict()
        for name, value in self.config.FEATURES_CONFIG.__dict__.items():
            if value.include:
                if value.__class__.__name__ == "CategFeature":
                    vocab = self.vocab[name]
                    self.preprocess_layers[name] = {
                        "preprocess": preprocessing.StringLookup(
                            vocabulary=vocab,
                        ),
                        "layer": tf.keras.layers.Embedding(
                            len(vocab) + 2,
                            value.embedding_size,
                            name=f"{name}_embedding",
                        ),
                        "flat": tf.keras.layers.Flatten(),
                    }
                    self.inputs_layers[name] = tf.keras.layers.Input(
                        shape=(1,), name=name, dtype=tf.string
                    )
                elif value.__class__.__name__ == "NumFeature":
                    if not self.config.DATA_CONFIG.whiten.apply:
                        self.inputs_layers[name] = tf.keras.layers.Input(
                            shape=(1,), name=name, dtype=tf.float32
                        )
                        self.preprocess_layers[name] = {
                            "preprocess": preprocessing.Normalization(
                                mean=self.vocab[name]["mean"],
                                variance=self.vocab[name]["var"],
                            ),
                        }
                    else:
                        pass
                else:
                    pass

        if self.config.DATA_CONFIG.whiten.apply:
            for i in range(self.config.DATA_CONFIG.whiten.n_components):
                name = f"pca-{i}"
                self.inputs_layers[name] = tf.keras.layers.Input(
                    shape=(1,), name=name, dtype=tf.float32
                )
                self.preprocess_layers[name] = {
                    "preprocess": preprocessing.Normalization(
                        mean=self.vocab[name]["mean"],
                        variance=self.vocab[name]["var"],
                    ),
                }

    def encode_inputs(
        self, inputs: Dict[Text, tf.Tensor]
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
        """Make the graph for input layers

        :param inputs: Input dict
        :type inputs: Dict[Text, tf.Tensor]
        :param inputs_layers: Input layers
        :type inputs_layers: Dict
        :param preprocess_layers: Embedding/Preprocessing layers
        :type preprocess_layers: Dict

        :return: Inputs encoded
        :rtype: Tuple[List[tf.Tensor],List[tf.Tensor]]
        """
        categ: List[tf.Tensor] = []
        num: List[tf.Tensor] = []
        for name, value in self.config.FEATURES_CONFIG.__dict__.items():
            if value.include:
                if value.__class__.__name__ == "CategFeature":
                    categ.append(
                        self.preprocess_layers[name]["flat"](
                            self.preprocess_layers[name]["layer"](
                                self.preprocess_layers[name]["preprocess"](inputs[name])
                            )
                        )
                    )
                elif value.__class__.__name__ == "NumFeature":
                    if not self.config.DATA_CONFIG.whiten.apply:
                        num.append(
                            self.preprocess_layers[name]["preprocess"](inputs[name])
                        )
                    else:
                        pass
                else:
                    pass

        if self.config.DATA_CONFIG.whiten.apply:
            for i in range(self.config.DATA_CONFIG.whiten.n_components):
                name = f"pca-{i}"
                num.append(self.preprocess_layers[name]["preprocess"](inputs[name]))

        return num, categ

    def get_config(self):
        return self.configs

    def save_to_directory(self, path_to_folder: Text):
        self.save_weights(os.path.join(path_to_folder, "ckpt"), overwrite=True)
        with open(os.path.join(path_to_folder, "configs.pickle"), "wb") as f:
            pickle.dump(self.configs, f)

    @classmethod
    def load_from_directory(cls, path_to_folder: Text):
        with open(os.path.join(path_to_folder, "configs.pickle"), "rb") as f:
            configs = pickle.load(f)
        model: tf.keras.Model = cls(**configs)
        model.build((None, configs["num_features"]))
        load_status = model.load_weights(os.path.join(path_to_folder, "ckpt"))
        load_status.expect_partial()
        return model
