import os
from pathlib import Path
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

from app.conf.run import RunConfig


def get_next_run_id(output_dir, prefix="run"):
    """
    Findet die nächste freie run_id basierend auf vorhandenen Dateien.
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
    numeric_dirs = sorted(
        (
            p for p in path.iterdir()
            if p.is_dir() and p.name.isdigit()
        ),
        key=lambda p: int(p.name)
    )
    return numeric_dirs

def current_path_version_for_dirs(dirs: list[Path]) -> int|None:
    max_number = max(
        (int(p.name) for p in dirs),
        default=None
    )
    return max_number

def create_new_output_version_dir(path: Path):
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
    prefix="run"
):
    os.makedirs(output_dir, exist_ok=True)

    run_id = get_next_run_id(output_dir, prefix)

    # sklearn outputs
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    # JSON-kompatibel machen
    cm_list = cm.tolist()

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
                }
            }
        },
        "result": {
            "metrics": {
                "accuracy": report.get("accuracy"),
                "macro_avg": report.get("macro avg"),
                "weighted_avg": report.get("weighted avg"),
            },
            "classification_report": report,
            "confusion_matrix": cm_list,
        },
        "timestamp": datetime.now().isoformat()
    }

    filename = f"{prefix}_{run_id:04d}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return filepath

