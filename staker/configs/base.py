from typing import Any
from typing import Text

import tensorflow as tf

from pydantic import BaseModel
from pydantic import PositiveInt


class DomainAdaptation(BaseModel):
    """Target variable

    :param active: If active or not
    :param target: Name of the target domain
    """

    active: bool = True
    target: Text = "era"


class TargetFeature(BaseModel):
    """Target variable

    :param include: Include the feature or not
    :param dtype: Type of the feature
    :param default: Default value
    :param sparse: to make sparse or not
    """

    include: bool = True
    dtype: Text = "float32"
    default: Any = 0.0


class CategFeature(BaseModel):
    """Categorical feature settings

    :param include: Include the feature or not
    :param embedding_size: Size of the embedding layer
    :param dtype: Type of the feature
    :param default: default value
    """

    include: bool = True
    dtype: Text = "string"
    embedding_size: PositiveInt
    default: Any = "[UKN]"


class NumFeature(BaseModel):
    """Numerical feature settings

    :param include: Include the feature or not
    :param scale: To scale or not the feature
    :param dtype: Type of the feature
    :param default: default value BatchMeanFill method
    """

    include: bool = True
    scale: bool = True
    dtype: Text = "float32"
    default: Any = -1

class MLPConfig(BaseModel):
    """MLP Config
    """
    name: Text = "mlp"
    dropout: float = 0.15

class TabNetConfig(BaseModel):
    """TabNet config

    :param feature_dim: Embedding feature dimention to use.
    :param output_dim: Output dimension.
    :param n_stes: Total number of steps. Defaults to 1.
    :param n_total: Total number of feature transformer blocks. Defaults to 4.
    :param n_shared: Number of shared feature transformer blocks. Defaults to 2.
    :param relaxation_factor: >1 will allow features to be used more than once. Defaults to 1.5.
    :param bn_epsilon: Batch normalization, epsilon. Defaults to 1e-5.
    :param bn_momentum: Batch normalization, momentum. Defaults to 0.7.
    :param bn_virtual_divider: Batch normalization. Full batch will be divided by this.
    """
    name: Text = "tabnet"
    feature_dim: int
    output_dim: int
    n_step: int = 1
    n_total: int = 4
    n_shared: int = 2
    sparsity_coefficient: float = 1e-5
    relaxation_factor: float = 1.5
    bn_epsilon: float = 1e-5
    bn_momentum: float = 0.7
    bn_virtual_divider: int = 1

class ModelConfig(BaseModel):
    """Model configuration

    :param type: Type of model to use
    :param reduce_on_plateau: To use reduce_on_plateau callbacks or not
    :param early: To use EarlyStopping callbacks or not
    :param max_epochs: Maximum number of epochs
    """

    type: BaseModel
    reduce_on_plateau: bool = False
    lr_scheduler: bool = True
    early: bool = True
    tensorboard: bool = False
    max_epochs: int = 500
    learning_rate: float = 1e-3
    optimizer_type: Text = "adam"
    loss: Text = "rmse"
    da: DomainAdaptation = DomainAdaptation()

class WhitenConfig(BaseModel):
    """Whiten configuration
    """
    apply: bool = True
    n_components:int = 220

class DataConfig(BaseModel):
    """Data configuration

    :param type: Type of data to use
    """

    type: Text
    batch_size: int = 32
    whiten:WhitenConfig = WhitenConfig()


class FeaturesConfig(BaseModel):
    @staticmethod
    def get_dtype_from_string(string: Text):
        """Return tf dtype from a string

        :param string: type
        :type string: Text

        :return: TF dtypes
        :rtype: tf.dtype
        """
        if string == "string":
            return tf.string
        elif string == "int32":
            return tf.int32
        elif string == "int64":
            return tf.int64
        elif string == "float32":
            return tf.float32
        else:
            raise NotImplementedError(f"{string} is not a valid dtype")


class NumerAIFeaturesConfig(FeaturesConfig):
    """Feature configuration

    The features signification is not explained on NumerAI beside era which related to the
    trading period (time)
    """

    target: TargetFeature = TargetFeature()
    feature_intelligence1: NumFeature = NumFeature()
    feature_intelligence2: NumFeature = NumFeature()
    feature_intelligence3: NumFeature = NumFeature()
    feature_intelligence4: NumFeature = NumFeature()
    feature_intelligence5: NumFeature = NumFeature()
    feature_intelligence6: NumFeature = NumFeature()
    feature_intelligence7: NumFeature = NumFeature()
    feature_intelligence8: NumFeature = NumFeature()
    feature_intelligence9: NumFeature = NumFeature()
    feature_intelligence10: NumFeature = NumFeature()
    feature_intelligence11: NumFeature = NumFeature()
    feature_intelligence12: NumFeature = NumFeature()
    feature_charisma1: NumFeature = NumFeature()
    feature_charisma2: NumFeature = NumFeature()
    feature_charisma3: NumFeature = NumFeature()
    feature_charisma4: NumFeature = NumFeature()
    feature_charisma5: NumFeature = NumFeature()
    feature_charisma6: NumFeature = NumFeature()
    feature_charisma7: NumFeature = NumFeature()
    feature_charisma8: NumFeature = NumFeature()
    feature_charisma9: NumFeature = NumFeature()
    feature_charisma10: NumFeature = NumFeature()
    feature_charisma11: NumFeature = NumFeature()
    feature_charisma12: NumFeature = NumFeature()
    feature_charisma13: NumFeature = NumFeature()
    feature_charisma14: NumFeature = NumFeature()
    feature_charisma15: NumFeature = NumFeature()
    feature_charisma16: NumFeature = NumFeature()
    feature_charisma17: NumFeature = NumFeature()
    feature_charisma18: NumFeature = NumFeature()
    feature_charisma19: NumFeature = NumFeature()
    feature_charisma20: NumFeature = NumFeature()
    feature_charisma21: NumFeature = NumFeature()
    feature_charisma22: NumFeature = NumFeature()
    feature_charisma23: NumFeature = NumFeature()
    feature_charisma24: NumFeature = NumFeature()
    feature_charisma25: NumFeature = NumFeature()
    feature_charisma26: NumFeature = NumFeature()
    feature_charisma27: NumFeature = NumFeature()
    feature_charisma28: NumFeature = NumFeature()
    feature_charisma29: NumFeature = NumFeature()
    feature_charisma30: NumFeature = NumFeature()
    feature_charisma31: NumFeature = NumFeature()
    feature_charisma32: NumFeature = NumFeature()
    feature_charisma33: NumFeature = NumFeature()
    feature_charisma34: NumFeature = NumFeature()
    feature_charisma35: NumFeature = NumFeature()
    feature_charisma36: NumFeature = NumFeature()
    feature_charisma37: NumFeature = NumFeature()
    feature_charisma38: NumFeature = NumFeature()
    feature_charisma39: NumFeature = NumFeature()
    feature_charisma40: NumFeature = NumFeature()
    feature_charisma41: NumFeature = NumFeature()
    feature_charisma42: NumFeature = NumFeature()
    feature_charisma43: NumFeature = NumFeature()
    feature_charisma44: NumFeature = NumFeature()
    feature_charisma45: NumFeature = NumFeature()
    feature_charisma46: NumFeature = NumFeature()
    feature_charisma47: NumFeature = NumFeature()
    feature_charisma48: NumFeature = NumFeature()
    feature_charisma49: NumFeature = NumFeature()
    feature_charisma50: NumFeature = NumFeature()
    feature_charisma51: NumFeature = NumFeature()
    feature_charisma52: NumFeature = NumFeature()
    feature_charisma53: NumFeature = NumFeature()
    feature_charisma54: NumFeature = NumFeature()
    feature_charisma55: NumFeature = NumFeature()
    feature_charisma56: NumFeature = NumFeature()
    feature_charisma57: NumFeature = NumFeature()
    feature_charisma58: NumFeature = NumFeature()
    feature_charisma59: NumFeature = NumFeature()
    feature_charisma60: NumFeature = NumFeature()
    feature_charisma61: NumFeature = NumFeature()
    feature_charisma62: NumFeature = NumFeature()
    feature_charisma63: NumFeature = NumFeature()
    feature_charisma64: NumFeature = NumFeature()
    feature_charisma65: NumFeature = NumFeature()
    feature_charisma66: NumFeature = NumFeature()
    feature_charisma67: NumFeature = NumFeature()
    feature_charisma68: NumFeature = NumFeature()
    feature_charisma69: NumFeature = NumFeature()
    feature_charisma70: NumFeature = NumFeature()
    feature_charisma71: NumFeature = NumFeature()
    feature_charisma72: NumFeature = NumFeature()
    feature_charisma73: NumFeature = NumFeature()
    feature_charisma74: NumFeature = NumFeature()
    feature_charisma75: NumFeature = NumFeature()
    feature_charisma76: NumFeature = NumFeature()
    feature_charisma77: NumFeature = NumFeature()
    feature_charisma78: NumFeature = NumFeature()
    feature_charisma79: NumFeature = NumFeature()
    feature_charisma80: NumFeature = NumFeature()
    feature_charisma81: NumFeature = NumFeature()
    feature_charisma82: NumFeature = NumFeature()
    feature_charisma83: NumFeature = NumFeature()
    feature_charisma84: NumFeature = NumFeature()
    feature_charisma85: NumFeature = NumFeature()
    feature_charisma86: NumFeature = NumFeature()
    feature_strength1: NumFeature = NumFeature()
    feature_strength2: NumFeature = NumFeature()
    feature_strength3: NumFeature = NumFeature()
    feature_strength4: NumFeature = NumFeature()
    feature_strength5: NumFeature = NumFeature()
    feature_strength6: NumFeature = NumFeature()
    feature_strength7: NumFeature = NumFeature()
    feature_strength8: NumFeature = NumFeature()
    feature_strength9: NumFeature = NumFeature()
    feature_strength10: NumFeature = NumFeature()
    feature_strength11: NumFeature = NumFeature()
    feature_strength12: NumFeature = NumFeature()
    feature_strength13: NumFeature = NumFeature()
    feature_strength14: NumFeature = NumFeature()
    feature_strength15: NumFeature = NumFeature()
    feature_strength16: NumFeature = NumFeature()
    feature_strength17: NumFeature = NumFeature()
    feature_strength18: NumFeature = NumFeature()
    feature_strength19: NumFeature = NumFeature()
    feature_strength20: NumFeature = NumFeature()
    feature_strength21: NumFeature = NumFeature()
    feature_strength22: NumFeature = NumFeature()
    feature_strength23: NumFeature = NumFeature()
    feature_strength24: NumFeature = NumFeature()
    feature_strength25: NumFeature = NumFeature()
    feature_strength26: NumFeature = NumFeature()
    feature_strength27: NumFeature = NumFeature()
    feature_strength28: NumFeature = NumFeature()
    feature_strength29: NumFeature = NumFeature()
    feature_strength30: NumFeature = NumFeature()
    feature_strength31: NumFeature = NumFeature()
    feature_strength32: NumFeature = NumFeature()
    feature_strength33: NumFeature = NumFeature()
    feature_strength34: NumFeature = NumFeature()
    feature_strength35: NumFeature = NumFeature()
    feature_strength36: NumFeature = NumFeature()
    feature_strength37: NumFeature = NumFeature()
    feature_strength38: NumFeature = NumFeature()
    feature_dexterity1: NumFeature = NumFeature()
    feature_dexterity2: NumFeature = NumFeature()
    feature_dexterity3: NumFeature = NumFeature()
    feature_dexterity4: NumFeature = NumFeature()
    feature_dexterity5: NumFeature = NumFeature()
    feature_dexterity6: NumFeature = NumFeature()
    feature_dexterity7: NumFeature = NumFeature()
    feature_dexterity8: NumFeature = NumFeature()
    feature_dexterity9: NumFeature = NumFeature()
    feature_dexterity10: NumFeature = NumFeature()
    feature_dexterity11: NumFeature = NumFeature()
    feature_dexterity12: NumFeature = NumFeature()
    feature_dexterity13: NumFeature = NumFeature()
    feature_dexterity14: NumFeature = NumFeature()
    feature_constitution1: NumFeature = NumFeature()
    feature_constitution2: NumFeature = NumFeature()
    feature_constitution3: NumFeature = NumFeature()
    feature_constitution4: NumFeature = NumFeature()
    feature_constitution5: NumFeature = NumFeature()
    feature_constitution6: NumFeature = NumFeature()
    feature_constitution7: NumFeature = NumFeature()
    feature_constitution8: NumFeature = NumFeature()
    feature_constitution9: NumFeature = NumFeature()
    feature_constitution10: NumFeature = NumFeature()
    feature_constitution11: NumFeature = NumFeature()
    feature_constitution12: NumFeature = NumFeature()
    feature_constitution13: NumFeature = NumFeature()
    feature_constitution14: NumFeature = NumFeature()
    feature_constitution15: NumFeature = NumFeature()
    feature_constitution16: NumFeature = NumFeature()
    feature_constitution17: NumFeature = NumFeature()
    feature_constitution18: NumFeature = NumFeature()
    feature_constitution19: NumFeature = NumFeature()
    feature_constitution20: NumFeature = NumFeature()
    feature_constitution21: NumFeature = NumFeature()
    feature_constitution22: NumFeature = NumFeature()
    feature_constitution23: NumFeature = NumFeature()
    feature_constitution24: NumFeature = NumFeature()
    feature_constitution25: NumFeature = NumFeature()
    feature_constitution26: NumFeature = NumFeature()
    feature_constitution27: NumFeature = NumFeature()
    feature_constitution28: NumFeature = NumFeature()
    feature_constitution29: NumFeature = NumFeature()
    feature_constitution30: NumFeature = NumFeature()
    feature_constitution31: NumFeature = NumFeature()
    feature_constitution32: NumFeature = NumFeature()
    feature_constitution33: NumFeature = NumFeature()
    feature_constitution34: NumFeature = NumFeature()
    feature_constitution35: NumFeature = NumFeature()
    feature_constitution36: NumFeature = NumFeature()
    feature_constitution37: NumFeature = NumFeature()
    feature_constitution38: NumFeature = NumFeature()
    feature_constitution39: NumFeature = NumFeature()
    feature_constitution40: NumFeature = NumFeature()
    feature_constitution41: NumFeature = NumFeature()
    feature_constitution42: NumFeature = NumFeature()
    feature_constitution43: NumFeature = NumFeature()
    feature_constitution44: NumFeature = NumFeature()
    feature_constitution45: NumFeature = NumFeature()
    feature_constitution46: NumFeature = NumFeature()
    feature_constitution47: NumFeature = NumFeature()
    feature_constitution48: NumFeature = NumFeature()
    feature_constitution49: NumFeature = NumFeature()
    feature_constitution50: NumFeature = NumFeature()
    feature_constitution51: NumFeature = NumFeature()
    feature_constitution52: NumFeature = NumFeature()
    feature_constitution53: NumFeature = NumFeature()
    feature_constitution54: NumFeature = NumFeature()
    feature_constitution55: NumFeature = NumFeature()
    feature_constitution56: NumFeature = NumFeature()
    feature_constitution57: NumFeature = NumFeature()
    feature_constitution58: NumFeature = NumFeature()
    feature_constitution59: NumFeature = NumFeature()
    feature_constitution60: NumFeature = NumFeature()
    feature_constitution61: NumFeature = NumFeature()
    feature_constitution62: NumFeature = NumFeature()
    feature_constitution63: NumFeature = NumFeature()
    feature_constitution64: NumFeature = NumFeature()
    feature_constitution65: NumFeature = NumFeature()
    feature_constitution66: NumFeature = NumFeature()
    feature_constitution67: NumFeature = NumFeature()
    feature_constitution68: NumFeature = NumFeature()
    feature_constitution69: NumFeature = NumFeature()
    feature_constitution70: NumFeature = NumFeature()
    feature_constitution71: NumFeature = NumFeature()
    feature_constitution72: NumFeature = NumFeature()
    feature_constitution73: NumFeature = NumFeature()
    feature_constitution74: NumFeature = NumFeature()
    feature_constitution75: NumFeature = NumFeature()
    feature_constitution76: NumFeature = NumFeature()
    feature_constitution77: NumFeature = NumFeature()
    feature_constitution78: NumFeature = NumFeature()
    feature_constitution79: NumFeature = NumFeature()
    feature_constitution80: NumFeature = NumFeature()
    feature_constitution81: NumFeature = NumFeature()
    feature_constitution82: NumFeature = NumFeature()
    feature_constitution83: NumFeature = NumFeature()
    feature_constitution84: NumFeature = NumFeature()
    feature_constitution85: NumFeature = NumFeature()
    feature_constitution86: NumFeature = NumFeature()
    feature_constitution87: NumFeature = NumFeature()
    feature_constitution88: NumFeature = NumFeature()
    feature_constitution89: NumFeature = NumFeature()
    feature_constitution90: NumFeature = NumFeature()
    feature_constitution91: NumFeature = NumFeature()
    feature_constitution92: NumFeature = NumFeature()
    feature_constitution93: NumFeature = NumFeature()
    feature_constitution94: NumFeature = NumFeature()
    feature_constitution95: NumFeature = NumFeature()
    feature_constitution96: NumFeature = NumFeature()
    feature_constitution97: NumFeature = NumFeature()
    feature_constitution98: NumFeature = NumFeature()
    feature_constitution99: NumFeature = NumFeature()
    feature_constitution100: NumFeature = NumFeature()
    feature_constitution101: NumFeature = NumFeature()
    feature_constitution102: NumFeature = NumFeature()
    feature_constitution103: NumFeature = NumFeature()
    feature_constitution104: NumFeature = NumFeature()
    feature_constitution105: NumFeature = NumFeature()
    feature_constitution106: NumFeature = NumFeature()
    feature_constitution107: NumFeature = NumFeature()
    feature_constitution108: NumFeature = NumFeature()
    feature_constitution109: NumFeature = NumFeature()
    feature_constitution110: NumFeature = NumFeature()
    feature_constitution111: NumFeature = NumFeature()
    feature_constitution112: NumFeature = NumFeature()
    feature_constitution113: NumFeature = NumFeature()
    feature_constitution114: NumFeature = NumFeature()
    feature_wisdom1: NumFeature = NumFeature()
    feature_wisdom2: NumFeature = NumFeature()
    feature_wisdom3: NumFeature = NumFeature()
    feature_wisdom4: NumFeature = NumFeature()
    feature_wisdom5: NumFeature = NumFeature()
    feature_wisdom6: NumFeature = NumFeature()
    feature_wisdom7: NumFeature = NumFeature()
    feature_wisdom8: NumFeature = NumFeature()
    feature_wisdom9: NumFeature = NumFeature()
    feature_wisdom10: NumFeature = NumFeature()
    feature_wisdom11: NumFeature = NumFeature()
    feature_wisdom12: NumFeature = NumFeature()
    feature_wisdom13: NumFeature = NumFeature()
    feature_wisdom14: NumFeature = NumFeature()
    feature_wisdom15: NumFeature = NumFeature()
    feature_wisdom16: NumFeature = NumFeature()
    feature_wisdom17: NumFeature = NumFeature()
    feature_wisdom18: NumFeature = NumFeature()
    feature_wisdom19: NumFeature = NumFeature()
    feature_wisdom20: NumFeature = NumFeature()
    feature_wisdom21: NumFeature = NumFeature()
    feature_wisdom22: NumFeature = NumFeature()
    feature_wisdom23: NumFeature = NumFeature()
    feature_wisdom24: NumFeature = NumFeature()
    feature_wisdom25: NumFeature = NumFeature()
    feature_wisdom26: NumFeature = NumFeature()
    feature_wisdom27: NumFeature = NumFeature()
    feature_wisdom28: NumFeature = NumFeature()
    feature_wisdom29: NumFeature = NumFeature()
    feature_wisdom30: NumFeature = NumFeature()
    feature_wisdom31: NumFeature = NumFeature()
    feature_wisdom32: NumFeature = NumFeature()
    feature_wisdom33: NumFeature = NumFeature()
    feature_wisdom34: NumFeature = NumFeature()
    feature_wisdom35: NumFeature = NumFeature()
    feature_wisdom36: NumFeature = NumFeature()
    feature_wisdom37: NumFeature = NumFeature()
    feature_wisdom38: NumFeature = NumFeature()
    feature_wisdom39: NumFeature = NumFeature()
    feature_wisdom40: NumFeature = NumFeature()
    feature_wisdom41: NumFeature = NumFeature()
    feature_wisdom42: NumFeature = NumFeature()
    feature_wisdom43: NumFeature = NumFeature()
    feature_wisdom44: NumFeature = NumFeature()
    feature_wisdom45: NumFeature = NumFeature()
    feature_wisdom46: NumFeature = NumFeature()
    era: CategFeature = CategFeature(embedding_size=12)
