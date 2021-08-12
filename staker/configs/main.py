from enum import Enum
from enum import unique
from typing import Optional
from typing import Text

from staker.configs.base import CategFeature
from staker.configs.base import DataConfig
from staker.configs.base import DomainAdaptation
from staker.configs.base import ModelConfig
from staker.configs.base import NumerAIFeaturesConfig
from staker.configs.base import NumFeature
from staker.configs.base import TargetFeature
from staker.configs.base import WhitenConfig
from staker.configs.base import TabNetConfig
from staker.configs.base import MLPConfig
from staker.models.factory import ModelType
from pydantic import BaseSettings
from pydantic import Field
from pydantic import BaseModel

@unique
class EnvState(Enum):
    PROD: Text = "prod"
    DEV: Text = "dev"

def get_model_type_config(type: Text)->BaseModel:
    """Get the model configuration based on
    the model type picked

    :param type: Model type (architecture )
    :type type: Text

    :return: The pick model config
    :rtype: BaseModel
    """
    if ModelType(type) == ModelType.TABNET:
        return TabNetConfig(feature_dim=128, output_dim=64)
    elif ModelType(type) == ModelType.MLP:
        return MLPConfig()
    else:
        raise NotImplementedError(f"{type} is not implemented")

class NumerAIPredictor(BaseSettings):
    """ NumeraAI Predictor main config object"""

    TYPE = "tabnet"
    model_type_config = get_model_type_config(TYPE)
    DATA_CONFIG = DataConfig(type=TYPE, whiten=WhitenConfig(apply=True, n_components=150), batch_size=1024)
    MODEL_CONFIG = ModelConfig(
        type=model_type_config,
        max_epochs=250,
        learning_rate=1e-3,
        da=DomainAdaptation(
            active=False, target="era"
        ),
        # WARNING: Changing the domain adaptation target require change in the feature set
        # TO DO: Validator that automatically check assumptions are meet
    )
    FEATURES_CONFIG = NumerAIFeaturesConfig()
    ENV_STATE: Optional[str] = Field(None, env="ENV_STATE")
    MODEL_OUTPUT: Optional[Text] = None

    class Config:
        """ Loads the dotenv file"""

        env_file: str = "numerai/configs/.environnement"


class DevConfig(NumerAIPredictor):
    """ Development configurations"""

    class Config:
        env_prefix: str = "DEV_"


class ProdConfig(NumerAIPredictor):
    """ Production configurations"""

    class Config:
        env_prefix: str = "PROD_"


class FactoryConfig:
    """ Returns a config instance dependeing on the ENV_STATE"""

    def __init__(self, env_state: Optional[str]):
        self.env_state = env_state

    def __call__(self):
        if EnvState(self.env_state) == EnvState.DEV:
            return DevConfig()
        elif EnvState(self.env_state) == EnvState.PROD:
            return ProdConfig()


numeraiconfig = FactoryConfig("dev")()
