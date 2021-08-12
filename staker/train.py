from typing import NoReturn

from pydantic import BaseSettings
from staker.configs.main import numeraiconfig
from staker.data.dataset import NumerAIDataset
from staker.evaluation.metrics import coeff_determination
from staker.models.factory import get_callbacks
from staker.models.factory import get_loss
from staker.models.factory import get_model
from staker.models.factory import get_optimizer


def main(config: BaseSettings) -> NoReturn:
    """Main function for model training

    :param config: Configuration
    :type config: BaseSettings
    """
    # TO DO: Factory design
    # 1. Get dataset and prepare for training
    dataset_getter = NumerAIDataset(config)
    train_dataset, val_dataset, vocab = dataset_getter(training=True)
    # 2. Get model
    model = get_model(config=config)
    callbacks = get_callbacks(config=config)
    optimizer = get_optimizer(config=config)
    loss = get_loss(config=config)
    model_instance = model(config=config, vocab=vocab)
    model_instance.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["binary_crossentropy", coeff_determination, "mae", "mse"],
    )
    # 3. Fit model
    model_instance.fit(
        train_dataset,
        epochs=config.MODEL_CONFIG.max_epochs,
        callbacks=callbacks,
        validation_data=val_dataset,
        verbose=1,
    )
    # 4. Save model
    model_instance.save("./models/")


if __name__ == "__main__":
    main(config=numeraiconfig)
