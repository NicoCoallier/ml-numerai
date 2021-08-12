from enum import Enum
from enum import unique
import logging
from functools import partial
from typing import Dict
from typing import List
from typing import Text

import tensorflow as tf
import tensorflow_addons as tfa

from staker.models.tabnet import TabNet
from staker.models.mlp import MLP
from staker.models.loss import rmse, rmsle
from pydantic import BaseSettings


@unique
class ModelType(Enum):
    TABNET: Text = "tabnet"
    MLP: Text = "mlp"

@unique
class OptimizerType(Enum):
    RMSPROP: Text = "rmsprop"
    ADAM: Text = "adam"
    ADAMAX: Text = "adamax"
    ADAMW: Text = "adamw"
    RADAM: Text = "radam"


@unique
class LossType(Enum):
    MSLE: Text = "msle"
    MAE: Text = "mae"
    RMSE: Text = "rmse"
    RMSLE: Text = "rmsle"
    MSE: Text = "mse"
    MAPE: Text = "mape"

def get_model(config: BaseSettings) -> tf.keras.Model:
    """Returns the model type

    :param config: Configuration object
    :type config: BaseSettings

    :return: The choosen model object
    :rtype: tf.keras.model
    """
    if ModelType(config.MODEL_CONFIG.type.name) == ModelType.TABNET:
        return TabNet
    elif ModelType(config.MODEL_CONFIG.type.name) == ModelType.MLP:
        return MLP
    else:
        raise NotImplementedError(f"{config.MODEL_CONFIG.type.name} is not implemented")


def get_callbacks(config: BaseSettings) -> List:
    """Return the used callbacks

    :param config: Configuration object
    :type config: BaseSettings

    """
    callbacks: List[tf.keras.callbacks] = []
    if config.MODEL_CONFIG.tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir="logs"))
    if config.MODEL_CONFIG.early:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                verbose=0,
                mode="min",
                restore_best_weights=True,
            )
        )
    if config.MODEL_CONFIG.reduce_on_plateau:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=2,
                min_lr=0.0005,
            )
        )
    if config.MODEL_CONFIG.lr_scheduler:

        def lr_scheduler(
            epoch,
            lr,
            warmup_epochs=15.0,
            decay_epochs=100.0,
            initial_lr=1e-6,
            base_lr=1e-3,
            min_lr=5e-5,
        ):
            if epoch <= warmup_epochs:
                pct = epoch / warmup_epochs
                return ((base_lr - initial_lr) * pct) + initial_lr

            if epoch > warmup_epochs and epoch < warmup_epochs + decay_epochs:
                pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
                return ((base_lr - min_lr) * pct) + min_lr

            return min_lr

        scheduler = tf.keras.callbacks.LearningRateScheduler(
            partial(lr_scheduler), verbose=0
        )
        callbacks.append(scheduler)
    return callbacks

def get_optimizer(config):
    """Get the optimizer

    :param config: Configuration
    :type config: BaseSettings

    :return: The picked optimizer
    :rtype: tf.keras.optimizers
    """
    # TO DO: Implement proper scheduling
    if OptimizerType(config.MODEL_CONFIG.optimizer_type) == OptimizerType.RMSPROP:
        logging.info("::: Optimizer type is RMSPROP")
        return tf.keras.optimizers.RMSprop(config.MODEL_CONFIG.learning_rate)
    elif OptimizerType(config.MODEL_CONFIG.optimizer_type) == OptimizerType.ADAM:
        logging.info("::: Optimizer type is ADAM")
        return tf.keras.optimizers.Adam(config.MODEL_CONFIG.learning_rate)
    elif OptimizerType(config.MODEL_CONFIG.optimizer_type) == OptimizerType.ADAMAX:
        logging.info("::: Optimizer type is ADAMAX")
        return tf.keras.optimizers.Adamax(config.MODEL_CONFIG.learning_rate)
    elif OptimizerType(config.MODEL_CONFIG.optimizer_type) == OptimizerType.ADAMW:
        logging.info("::: Optimizer type is ADAMW")
        return tfa.optimizers.AdamW(
            learning_rate=config.MODEL_CONFIG.learning_rate, weight_decay=0.01
        )
    elif OptimizerType(config.MODEL_CONFIG.optimizer_type) == OptimizerType.RADAM:
        logging.info("::: Optimizer type is RADAM")
        return tfa.optimizers.RectifiedAdam(lr=config.MODEL_CONFIG.learning_rate)


def get_loss(config):
    """Get the loss

    :param config: Configuration
    :type config: BaseSettings

    :return: The picked loss
    :rtype: tf.keras.losses
    """
    if LossType(config.MODEL_CONFIG.loss) == LossType.MSLE:
        logging.info("::: Loss type is MSLE")
        return tf.keras.losses.MeanSquaredLogarithmicError()
    elif LossType(config.MODEL_CONFIG.loss) == LossType.RMSLE:
        logging.info("::: Loss type is RMSLE")
        return rmsle
    elif LossType(config.MODEL_CONFIG.loss) == LossType.RMSE:
        logging.info("::: Loss type is RMSE")
        return rmse
    elif LossType(config.MODEL_CONFIG.loss) == LossType.MAE:
        logging.info("::: Loss type is MAE")
        return tf.keras.losses.MeanAbsoluteError()
    elif LossType(config.MODEL_CONFIG.loss) == LossType.MSE:
        logging.info("::: Loss type is MSE")
        return tf.keras.losses.MeanSquaredError()
    elif LossType(config.MODEL_CONFIG.loss) == LossType.MAPE:
        logging.info("::: Loss type is MAPE")
        return tf.keras.losses.MeanAbsolutePercentageError()
