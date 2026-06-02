# imports of third party packages
import dataclasses
from dataclasses import field
from sklearn.metrics import accuracy_score


@dataclasses.dataclass
class ModelConfig:
    """
    Dataclass to hold information about the model to be used/trained
    """
    model_id: str
    model_parameters: dict = field(default_factory=lambda : {})
    wrap_calibrated: bool = False

@dataclasses.dataclass
class EncodingConfig:
    """
    Dataclass to hold information about the variable encoding to be used
     the columns in the lists are treated with respect to their names
    """
    one_hot_cols: list[str]
    passthrough_cols: list[str]
    target_enc_cols: list[str]
    std_scale_cols: list[str]
    str_cat_cols: list[str]


@dataclasses.dataclass
class RunConfig:
    """
    dataclass to define all the variables of an "app run" (can be a training with a following evaluation)
    """
    model_config: ModelConfig
    encoding_config: EncodingConfig
    metric_string: str = "neg_brier_score"
    metric_function = accuracy_score
    context_name: str = "default"
    use_only_field_goals: bool = False
    return_probabilities: bool = False
    use_action_type_fix: bool = False
    decision_boundary: float = 0.5


def build_default_run_config():
    """
    :return: a fixed baseline RunConfig
    """
    return RunConfig(
        model_config=ModelConfig(model_id=MODEL_ID_RANDOM_FOREST),
        encoding_config=EncodingConfig(
            one_hot_cols=[],
            passthrough_cols=[
                "SHOT_DISTANCE",
                "IS_HOME",
                "is_playoffs",
            ],
            target_enc_cols=[],
            std_scale_cols=[],
            # model type should decide,
            # whether this is a passed through cat or a OneHot (see encode_for_model function)
            str_cat_cols=["MAIN_ACTION_TYPE", "PLAYER_NAME", "SHOT_TYPE", 'ANGLE_SECTOR']
        )
    )

def build_best_run_config():
    """
    :return: the currently best performing RunConfig
    """
    return RunConfig(
        model_config=ModelConfig(model_id=MODEL_ID_LIGHT_GBM),
        encoding_config=EncodingConfig(
            one_hot_cols=[],
            passthrough_cols=[
                "SHOT_DISTANCE",
                "IS_HOME",
                "is_playoffs",
                "TimeRemainingInGame",
                "TimeRemainingInPeriod",
                "IsClutchTime",
                "year",
                "player_age",
                "best_age",
                "ABS_ANGLE"
            ],
            target_enc_cols=[],
            std_scale_cols=[],
            # model type should decide,
            # whether this is a passed through cat or a OneHot (see encode_for_model function)
            str_cat_cols=["ACTION_TYPE", "PLAYER_NAME", "SHOT_TYPE", 'ANGLE_SECTOR', "MAIN_ACTION_TYPE"]
        ),
        use_action_type_fix=True
    )

def build_interpretability_run_config():
    """
    Deprecated
    :return: a RunConfig that should deliver an easily interpretable.
    """
    return RunConfig(
        model_config=ModelConfig(model_id=MODEL_ID_LIGHT_GBM),
        encoding_config=EncodingConfig(
            one_hot_cols=[],
            passthrough_cols=[
                "SHOT_DISTANCE",
                "IS_HOME",
                "is_playoffs",
                "TimeRemainingInGame",
                "IsClutchTime",
                "ABS_ANGLE",
                "TimeRemainingInPeriod",
                "year",
                "player_age",
                "best_age"

            ],
            target_enc_cols=[],
            std_scale_cols=[],
            # model type should decide,
            # whether this is a passed through cat or a OneHot (see encode_for_model function)
            str_cat_cols=["MAIN_ACTION_TYPE", "PLAYER_NAME"]
        ),
        use_only_field_goals=True
    )

# Fixed Model-Algorithm Identifiers
MODEL_ID_RANDOM_FOREST = "RandomForest"
# MODEL_ID_SVM = "SVM" # too slow for this amount of data
MODEL_ID_LOGISTIC_REGRESSION = "LogisticRegression"
MODEL_ID_LIGHT_GBM = "LightGBM"
MODEL_ID_DECISION_TREE = "DecisionTree"
MODEL_ID_SIMPLE_LOOKUP = "SimpleLookup"
MODEL_ID_DEEP_LEARNING = "DeepLearning"


def get_active_columns(config: EncodingConfig):
    """
    Convenience function to compute all active columns inside an Encoding config
    :param config:
    :return:
    """
    return (config.std_scale_cols + config.passthrough_cols +
            config.one_hot_cols + config.target_enc_cols + config.str_cat_cols)

def is_tree_based(model_id: str) -> bool:
    """
    Whether or not a Model is tree based, e.g. used to know if special SHAP explainers can be used
    :param model_id:
    :return:
    """
    return model_id in {
        MODEL_ID_DECISION_TREE,
        MODEL_ID_LIGHT_GBM,
        MODEL_ID_RANDOM_FOREST,
    }

def can_handle_categories(model_id: str) -> bool:
    """
    Decide if a model can handle Pandas Category Column type. Mainly used to decide if they must be OneHot encoded or not
    :param model_id:
    :return:
    """
    return model_id in {
        MODEL_ID_LIGHT_GBM,
        MODEL_ID_DEEP_LEARNING
    }

