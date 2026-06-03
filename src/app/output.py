import os
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss, roc_auc_score

from app.conf.run import RunConfig
from evaluation.custom_scores import expected_calibration_error, brier_decomposition


def get_next_run_id(output_dir, prefix="run"):
    """
    Finds the next run id for the given directory
    """
    if not os.path.exists(output_dir):
        return 1

    existing = [
        f for f in os.listdir(output_dir)
        if f.startswith(prefix) and f.endswith(".json")
    ]

    if not existing:
        return 1

    ids = []
    for f in existing:
        try:
            ids.append(int(f.split("_")[1].split(".")[0]))
        except:
            continue

    return max(ids) + 1 if ids else 1

def numeric_dirs_in_path(path:Path) -> list[Path]:
    """
    find all numeric direcotries in a given path
    :param path:
    :return:
    """
    numeric_dirs = sorted(
        (
            p for p in path.iterdir()
            if p.is_dir() and p.name.isdigit()
        ),
        key=lambda p: int(p.name)
    )
    return numeric_dirs

def current_path_version_for_dirs(dirs: list[Path]) -> int|None:
    """
    get the current version directory in a directory
    /experiments/model_comparison/1
    /experiments/model_comparison/2
    => return 2
    :param dirs:
    :return:
    """
    max_number = max(
        (int(p.name) for p in dirs),
        default=None
    )
    return max_number

def create_new_output_version_dir(path: Path):
    """
    Before an experiment run (or what ever use case) create a new version dir according to the last versions
    existing:
    /experiments/model_comparison/1
    /experiments/model_comparison/2
    ...will create...
    /experiments/model_comparison/3
    :param path: the containing directory path
    :return: the path created
    """
    dirs_in_path = numeric_dirs_in_path(path)
    current = current_path_version_for_dirs(dirs_in_path)
    next_version = current + 1 if current is not None else 1
    new_path = path / str(next_version)
    new_path.mkdir(exist_ok=False, parents=False)
    return new_path


def save_classification_run(
    y_true,
    y_pred,
    config: RunConfig,
    output_dir="runs",
    prefix="run",
    additional_infos = None,
    y_proba = None,
    y_true_train = None,
    y_pred_train = None,
    y_proba_train = None
):
    """
    Save/Log the results of a training + evaluation run
    :param y_true:
    :param y_pred:
    :param config: the RunConfig that produced the result
    :param output_dir:
    :param prefix: first part of the logfile
    :param additional_infos: dict, like GridSearch best parameters
    :param y_proba: to compute also brier_score
    :param y_true_train:
    :param y_pred_train: to estimate overfitting
    :param y_proba_train:
    :return: the filepath where the logs have been written to
    """
    os.makedirs(output_dir, exist_ok=True)

    run_id = get_next_run_id(output_dir, prefix)

    # sklearn outputs
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # JSON-kompatibel machen
    cm_list = cm.tolist()
    if additional_infos is None:
        additional_infos = {}
    ece_10 = None
    ece_20 = None
    brier_decomposition_10 = None
    brier_decomposition_20 = None
    if y_proba is not None:
        # how good do we match the calibration curve diagonal line?
        ece_10, calibration_table = expected_calibration_error(
            y_true,
            y_proba[:,1],
            n_bins=10,
        )
        ece_20, calibration_table = expected_calibration_error(
            y_true,
            y_proba[:,1],
            n_bins=20,
        )
        # murphy decomposition of brier_score in resolution, reliability and uncertainty
        brier_decomposition_10 = brier_decomposition(
            y_true,
            y_proba[:,1],
            n_bins=10,
        )
        brier_decomposition_20 = brier_decomposition(
            y_true,
            y_proba[:,1],
            n_bins=20,
        )

    result = {
        "run_id": run_id,
        "context_name": config.context_name,
        "input": {
            "model": {
                "name": config.model_config.model_id,
                "parameters": config.model_config.model_parameters,
            },
            "features": {
                "encoding": {
                    "one_hot": config.encoding_config.one_hot_cols,
                    "target_enc": config.encoding_config.target_enc_cols,
                    "passthrough": config.encoding_config.passthrough_cols,
                    "std_scale": config.encoding_config.std_scale_cols,
                    "str_cat_cols": config.encoding_config.str_cat_cols,
                }
            }
        },
        "result": {
            "metrics": {
                "accuracy": report.get("accuracy"),
                "roc_auc": "n/a" if y_proba is None else roc_auc_score(y_true, y_proba[:,1]),
                "macro_avg": report.get("macro avg"),
                "weighted_avg": report.get("weighted avg"),
                "brier_score": "n/a" if y_proba is None else brier_score_loss(y_true, y_proba),
                "expected_calibration_error_10": "n/a" if ece_10 is None else ece_10,
                "expected_calibration_error_20": "n/a" if ece_20 is None else ece_20,
                "brier_decomposition_10": {} if brier_decomposition_10 is None else
                {
                    "brier_score": brier_decomposition_10['brier_score'],
                    "reliability": brier_decomposition_10['reliability'],
                    "resolution": brier_decomposition_10['resolution'],
                    "uncertainty": brier_decomposition_10['uncertainty'],
                },
                "brier_decomposition_20": {} if brier_decomposition_20 is None else
                {
                    "brier_score": brier_decomposition_20['brier_score'],
                    "reliability": brier_decomposition_20['reliability'],
                    "resolution": brier_decomposition_20['resolution'],
                    "uncertainty": brier_decomposition_20['uncertainty'],
                },
            },
            "classification_report": report,
            "confusion_matrix": cm_list,
            "decomposition_table_10": {} if brier_decomposition_10 is None else brier_decomposition_10['table'].to_dict(),
            "decomposition_table_20": {} if brier_decomposition_20 is None else brier_decomposition_20['table'].to_dict(),
        },
        "timestamp": datetime.now().isoformat(),
        "additional_infos": additional_infos
    }

    # add training results if given
    if y_pred_train is not None and y_true_train is not None:
        train_report = classification_report(y_true_train, y_pred_train, output_dict=True)
        result["train_result"] = {
            "metrics": {
                "accuracy": train_report.get("accuracy"),
                "brier_score": "n/a" if y_proba_train is None else brier_score_loss(y_true_train, y_proba_train)
            }
        }

    # now some file io related stuff: filepath + save
    filename = f"{prefix}_{run_id:04d}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return filepath

