from app.conf.run import RunConfig
from app.config import EXPERIMENTS_PATH
from pathlib import Path
from app.modeling import model_prediction
from app.output import save_classification_run, create_new_output_version_dir, current_path_version_for_dirs, \
    numeric_dirs_in_path
import os
import json
import pandas as pd

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
    y_pred, y_test = model_prediction(config)
    save_classification_run(y_test, y_pred, config, path)

def run_experiment(configs: list[RunConfig], experiment_id:str):
    exp_path = experiment_base_path(experiment_id)
    run_path = create_new_output_version_dir(exp_path)
    for config in configs:
        run_experiment_part(config,run_path)

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

