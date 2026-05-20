import dataclasses
from dataclasses import field

from sklearn.metrics import accuracy_score, brier_score_loss


@dataclasses.dataclass
class ModelConfig:
    model_id: str
    model_parameters: dict = field(default_factory=lambda : {})
    wrap_calibrated: bool = False

@dataclasses.dataclass
class EncodingConfig:
    one_hot_cols: list[str]
    passthrough_cols: list[str]
    target_enc_cols: list[str]
    std_scale_cols: list[str]
    str_cat_cols: list[str]


@dataclasses.dataclass
class RunConfig:
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
            str_cat_cols=["MAIN_ACTION_TYPE", "PLAYER_ID", "SHOT_TYPE", 'ANGLE_SECTOR']
        )
    )

def build_best_run_config():
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
            ],
            target_enc_cols=[],
            std_scale_cols=[],
            # model type should decide,
            # whether this is a passed through cat or a OneHot (see encode_for_model function)
            str_cat_cols=["ACTION_TYPE", "PLAYER_ID", "SHOT_TYPE", 'ANGLE_SECTOR', "MAIN_ACTION_TYPE"]
        ),
        use_action_type_fix=True
    )

def build_interpretability_run_config():
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
                "ABS_ANGLE"
            ],
            target_enc_cols=[],
            std_scale_cols=[],
            # model type should decide,
            # whether this is a passed through cat or a OneHot (see encode_for_model function)
            str_cat_cols=["MAIN_ACTION_TYPE", "PLAYER_ID", "SHOT_TYPE"]
        ),
        use_only_field_goals=True
    )


MODEL_ID_RANDOM_FOREST = "RandomForest"
# MODEL_ID_SVM = "SVM" # too slow for this amount of data
MODEL_ID_LOGISTIC_REGRESSION = "LogisticRegression"
MODEL_ID_LIGHT_GBM = "LightGBM"
MODEL_ID_DECISION_TREE = "DecisionTree"


def get_active_columns(config: EncodingConfig):
    return config.std_scale_cols + config.passthrough_cols + config.one_hot_cols + config.target_enc_cols

def is_tree_based(model_id: str) -> bool:
    return model_id in {
        MODEL_ID_DECISION_TREE,
        MODEL_ID_LIGHT_GBM,
        MODEL_ID_RANDOM_FOREST,
    }

def can_handle_categories(model_id: str) -> bool:
    return model_id in {
        MODEL_ID_LIGHT_GBM,
    }

