import os
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix


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



def save_classification_run(
    y_true,
    y_pred,
    model_name,
    parameters,
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
        "input": {
            "model": {
                "name": model_name,
                "parameters": parameters,
            },
            "features": {

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

