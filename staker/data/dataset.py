import logging
import os
import pickle

from typing import Dict
from typing import List
from typing import NoReturn
from typing import Text
from typing import Tuple

import numerapi
import numpy as np
import pandas as pd
import tensorflow as tf

from pydantic import BaseSettings
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from staker.constants import SHUFFLE_FACTOR
from staker.constants import VAL_SIZE


class NumerAIDataset:
    """NumerAI dataset for model training and inference"""

    def __init__(self, config: BaseSettings):
        self.config = config
        self.__get_env_var()
        self.napi = numerapi.NumerAPI(self.api_id, self.api_secret)
        self.categ_features: List[Text] = []
        self.num_features: List[Text] = []
        self.target_features: List[Text] = []
        self.new_num_features: List[Text] = []
        self.__get_features()

    def __get_features(self) -> NoReturn:
        """Get the feature/target uses in the model"""
        self.to_keep: List[Text] = []
        for feature, value in self.config.FEATURES_CONFIG.__dict__.items():
            if value.include:
                self.to_keep.append(feature)
            # Gather them by type
            if value.__class__.__name__ == "CategFeature":
                self.categ_features.append(feature)
            elif value.__class__.__name__ == "NumFeature":
                self.num_features.append(feature)
            elif value.__class__.__name__ == "TargetFeature":
                self.target_features.append(feature)
            else:
                pass

    def __get_env_var(self) -> NoReturn:
        """Instantiate env var"""
        self.api_id = os.environ["ID"]
        self.api_secret = os.environ["SECRET"]

    def __build_vocab_and_stats(self, data):
        """Build categorical vocab and stats

        :param data: Preprocess data
        :type data: pd.DataFrame

        :return: Vocab
        :rype: Dict
        """
        vocabs: Dict[Text, List] = dict()
        for name in self.to_keep:
            if name in self.categ_features:
                vocabs[name] = list(data[name].unique())
            elif name in self.num_features or name in self.new_num_features:
                vocabs[name] = {
                    "mean": data[name].mean(),
                    "var": data[name].var(),
                }
            else:
                pass

        return vocabs

    def download(self) -> NoReturn:
        """Download the data locally"""
        self.napi.download_current_dataset(
            unzip=True, dest_path="./data/", dest_filename="current_round"
        )

    def load_data(self, training: bool) -> pd.DataFrame:
        """Load the dataset in memory"""
        # TO DO: Avoid hardcoding
        if training:
            data = pd.read_csv("./data/current_round/numerai_training_data.csv")
        else:
            data = pd.read_csv("./data/current_round/numerai_tournament_data.csv")
        return data

    def whiten(
        self, data: pd.DataFrame, num_components: int = 80, training: bool = False
    ) -> pd.DataFrame:
        """Applies data whitening

        :param data: Data to whiten
        :type data: pd.DataFrame
        :param num_components: Number of components to reduce too
        :type num_components: int

        :return: Whiten data
        :rtype: pd.DataFrame
        """
        # **** TO DO: Make sure to only fit on TRAIN ****
        if training:
            self.pca = PCA(n_components=num_components, whiten=True)
            pca_result = self.pca.fit_transform(data[self.num_features])
        else:
            pca_result = self.pca.transform(data[self.num_features])
        # Pick transform features
        WHITE_NUM: List[Text] = []
        for i in range(num_components):
            name = f"pca-{i}"
            data.loc[:, name] = pca_result[:, i]
            WHITE_NUM.append(name)

        # Overwrite NUM with WHITE_NUM
        if training:
            self.new_num_features = WHITE_NUM
            self.to_keep = WHITE_NUM + self.categ_features + self.target_features
        logging.info(
            "::: Explained variation for all components: {}".format(
                np.sum(self.pca.explained_variance_ratio_)
            )
        )
        return data[self.to_keep]

    def split_dataset(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataset into val and train set

        :param data: Data to split
        :type data: pd.DataFrame

        :return: The val and train set
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        train, val, _, _ = train_test_split(data, data["target"], test_size=VAL_SIZE)
        return train, val

    def pandas_to_tensorflow(self, data: pd.DataFrame) -> tf.data.Dataset:
        """Transform pandas dataset to Tf.Dataset

        :param data: Data to transform
        :type data: pd.DataFrame

        :return: the tf dataset
        :type: tf.Dataset
        """
        return tf.data.Dataset.from_tensor_slices(data.to_dict("list"))

    def preprare_data(
        self, data: tf.data.Dataset, shuffle: bool = True
    ) -> tf.data.Dataset:
        """Prepare dataset with batching and shuffling...

        :param data: The data to prepare
        :type data: tf.Dataset
        :param shuffle: shuffle flag
        :type shuffle: bool

        :return: Batch and shuffle data
        :rtype: tf.Dataset
        """

        def split_label_features(data: tf.data.Dataset) -> Tuple:
            y = data.pop(self.target_features[0])
            return data, y

        data = data.map(split_label_features, tf.data.AUTOTUNE)
        if shuffle:
            data = data.cache()
            data = data.shuffle(self.config.DATA_CONFIG.batch_size * SHUFFLE_FACTOR)
        data = data.batch(self.config.DATA_CONFIG.batch_size, drop_remainder=False)
        return data

    def save(self, path: Text) -> NoReturn:
        """Save dataset object

        :param path: Path where to save
        :type path: Text
        """
        logging.info(f"::: Saving dataset object to {path}")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Text):
        """Load dataset object

        :param path: Path where to save
        :type path: Text
        """
        logging.info(f"::: Loading dataset object to {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def __call__(self, training: bool):
        """Download, split and create the TF.Dataset ready for training"""
        # 1. Download
        logging.info("::: Downloading data...")
        self.download()
        # 2. Load
        logging.info("::: Loading data...")
        data = self.load_data(training)
        if training:
            # 3. Split dataset
            logging.info("::: Split data...")
            train, val = self.split_dataset(data)
            # 4. Whiten or not
            if self.config.DATA_CONFIG.whiten.apply:
                logging.info("::: Whitening data...")
                train = self.whiten(
                    train, self.config.DATA_CONFIG.whiten.n_components, True
                )
                val = self.whiten(
                    val, self.config.DATA_CONFIG.whiten.n_components, False
                )
            else:
                train = train[self.to_keep]
                val = val[self.to_keep]
            # 5. Calcualte vocab and stats
            logging.info("::: Build stats and vocab...")
            vocab = self.__build_vocab_and_stats(train)
            # 6. Transform to tf dataset
            logging.info("::: Transforming from pandas to tensorflow...")
            train_dataset = self.pandas_to_tensorflow(train)
            val_dataset = self.pandas_to_tensorflow(val)
            # 7. Prepare tf dataset for training (batch, shuffle...)
            logging.info("::: Prepare data for training...")
            train_dataset = self.preprare_data(train_dataset, shuffle=True)
            val_dataset = self.preprare_data(val_dataset, shuffle=False)
            return train_dataset, val_dataset, vocab
        else:
            if self.config.DATA_CONFIG.whiten.apply:
                data = self.whiten(
                    data, self.config.DATA_CONFIG.whiten.n_components, False
                )
            else:
                data = data[self.to_keep]
            dataset = self.pandas_to_tensorflow(data)
            dataset = self.preprare_data(dataset, shuffle=False)
            return dataset, data
