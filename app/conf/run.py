import dataclasses
from dataclasses import field


@dataclasses.dataclass
class ModelConfig:
    model_id: str
    model_parameters: dict = field(default_factory=lambda : {})

@dataclasses.dataclass
class EncodingConfig:
    one_hot_cols: list[str]
    passthrough_cols: list[str]


@dataclasses.dataclass
class RunConfig:
    model_config: ModelConfig
    encoding_config: EncodingConfig


def build_default_run_config():
    return RunConfig(
        model_config=ModelConfig(model_id=MODEL_ID_RANDOM_FOREST),
        encoding_config=EncodingConfig(
            one_hot_cols=["MAIN_ACTION_TYPE", "PLAYER_ID", "SHOT_TYPE", 'ANGLE_SECTOR'],
            passthrough_cols=[
                "SHOT_DISTANCE",
                "ANGLE_SECTOR",
                "IS_HOME",
                "is_playoffs",
            ]
        )
    )


MODEL_ID_RANDOM_FOREST = "RandomForest"
