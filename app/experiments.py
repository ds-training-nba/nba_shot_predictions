import os
import json
import pandas as pd

from app.modeling import run_grid_search
from app.conf.run import RunConfig
from app.config import EXPERIMENTS_PATH
from pathlib import Path
from app.modeling import model_prediction
from app.output import save_classification_run, create_new_output_version_dir, current_path_version_for_dirs, \
    numeric_dirs_in_path


def experiment_base_path(experiment_id, auto_create=True):
    path = Path('./' + EXPERIMENTS_PATH + "/" + experiment_id)
    if (not path.exists()) and auto_create:
        path.mkdir(parents=True)
    return path

def experiment_current_path(experiment_id):
    base = experiment_base_path(experiment_id)
    current = current_path_version_for_dirs(
        numeric_dirs_in_path(base)
    )
    return base / str(current)

def run_experiment_part(config: RunConfig, path):
    config.return_probabilities = True
    y_proba, y_test, y_proba_train, y_train = model_prediction(config)

    save_classification_run(y_test, y_proba[:,1] > config.decision_boundary, config, path,
                            y_proba=y_proba, y_proba_train=y_proba_train, y_true_train=y_train,y_pred_train=y_proba_train[:,1] > config.decision_boundary)

def run_grid_search_experiment_part(config: RunConfig, path):
    y_pred, y_test, grid_search_results = run_grid_search(config, 3)
    save_classification_run(y_test, y_pred, config, path,"grid_search", {"grid_search_results": grid_search_results})


def run_experiment(configs: list[RunConfig], experiment_id:str):
    exp_path = experiment_base_path(experiment_id)
    run_path = create_new_output_version_dir(exp_path)
    for config in configs:
        run_experiment_part(config,run_path)


def run_grid_search_experiment(configs: list[RunConfig], experiment_id:str):
    exp_path = experiment_base_path(experiment_id)
    run_path = create_new_output_version_dir(exp_path)
    for config in configs:
        run_grid_search_experiment_part(config,run_path)

def load_runs_to_dataframe(directory="runs"):
    records = []

    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(directory, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        flat = {}

        # Basisfelder
        flat["run_id"] = data.get("run_id")
        flat["model"] = data.get("input").get('model').get('name')
        flat["timestamp"] = data.get("timestamp")
        flat["context_name"] = data.get("context_name")

        # Parameter flatten
        for k, v in data.get("parameters", {}).items():
            flat[f"param_{k}"] = v

        # Metriken flatten
        for k, v in data.get("result").get("metrics", {}).items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict):
                        for sub_sub_k, sub_sub_v in sub_v.items():
                            flat[f"{k}_{sub_k}_{sub_sub_k}"] = sub_sub_v
                    else:
                        flat[f"{k}_{sub_k}"] = sub_v
            else:
                flat[f"metric_{k}"] = v
        for k, v in data.get("train_result").get("metrics", {}).items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, dict):
                        for sub_sub_k, sub_sub_v in sub_v.items():
                            flat[f"{k}_{sub_k}_{sub_sub_k}"] = sub_sub_v
                    else:
                        flat[f"{k}_{sub_k}"] = sub_v
            else:
                flat[f"metric_{k}"] = v


        # Optional: wichtigste Scores direkt ziehen
        report = data.get("result").get("classification_report", {})
        if "macro avg" in report:
            flat["f1_macro"] = report["macro avg"]["f1-score"]
        if "weighted avg" in report:
            flat["f1_weighted"] = report["weighted avg"]["f1-score"]

        records.append(flat)

    df = pd.DataFrame(records)

    # Optional: nach run_id sortieren
    df = df.sort_values("run_id").reset_index(drop=True)

    return df

