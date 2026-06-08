#3rd party
import os
import json
import pandas as pd
from pathlib import Path

# import own code
from app.modeling import run_grid_search
from app.conf.run import RunConfig
from app.config import EXPERIMENTS_PATH
from app.modeling import model_prediction
from app.output import save_classification_run, create_new_output_version_dir, current_path_version_for_dirs, \
    numeric_dirs_in_path


def experiment_base_path(experiment_id, auto_create=True):
    """
    Get the base path for the result logs of an experiment with the given identifier
    :param experiment_id:
    :param auto_create: whether or not to create a non-existing directory
    :return: Path
    """
    path = Path('./' + EXPERIMENTS_PATH + "/" + experiment_id)
    if (not path.exists()) and auto_create:
        path.mkdir(parents=True)
    return path

def experiment_current_path(experiment_id):
    """
    Get the actual sub-path for the result logs of an experiment with the given identifier
    /experiment_base_path/current_run_id_directory
    :param experiment_id:
    :return: Path
    """
    base = experiment_base_path(experiment_id)
    current = current_path_version_for_dirs(
        numeric_dirs_in_path(base)
    )
    return base / str(current)

def run_experiment_part(config: RunConfig, path):
    """
    Runs the part of an experiment that consists of a single config:
    Use the whole pipeline until the model prediction, then save the result log in the according path
    :param config:
    :param path:
    :return: None
    """
    config.return_probabilities = True
    y_proba, y_test, y_proba_train, y_train = model_prediction(config)

    save_classification_run(y_test, y_proba[:,1] > config.decision_boundary, config, path,
                            y_proba=y_proba, y_proba_train=y_proba_train, y_true_train=y_train,y_pred_train=y_proba_train[:,1] > config.decision_boundary)

def run_grid_search_experiment_part(config: RunConfig, path):
    """
    Same as experiment, but with a grid search instead of a simple training, logging also the gridsearch results
    :param config:
    :param path:
    :return:
    """
    y_pred, y_test, grid_search_results = run_grid_search(config, 3)
    save_classification_run(y_test, y_pred, config, path,"grid_search", {"grid_search_results": grid_search_results})


def run_experiment(configs: list[RunConfig], experiment_id:str):
    """
    Run the whole experiment (a list of configs, and an identifier to group
     the results and find them under the same path)
     With the use of experiments, we can compare different configs (models/encodings/choice of explanatory variables)
     directly and conveniently. Only the difference in parameters needs to be defined and the whole pipeline and logging
     is done automatically.
    :param configs:
    :param experiment_id:
    :return:
    """
    exp_path = experiment_base_path(experiment_id)
    run_path = create_new_output_version_dir(exp_path)
    for config in configs:
        run_experiment_part(config,run_path)


def run_grid_search_experiment(configs: list[RunConfig], experiment_id:str):
    """
    same as run_experiment, but with a grid search involved in the training process
    :param configs: list of configs to compare
    :param experiment_id: str to define the experiments name
    :return: None
    """
    exp_path = experiment_base_path(experiment_id)
    run_path = create_new_output_version_dir(exp_path)
    for config in configs:
        run_grid_search_experiment_part(config,run_path)

def load_runs_to_dataframe(directory="runs"):
    """
    Load/Parse experiment result logs into a dataframe. I.e. to display in a streamlit app
    :param directory: str the actual log directory of the experiment
    :return: pd.DataFrame
    """
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

