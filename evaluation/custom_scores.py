from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    log_loss,
)

import numpy as np
import pandas as pd

def expected_calibration_error(
    y_true,
    y_prob,
    n_bins=10,
):
    """
    Expected Calibration Error (ECE)

    Returns:
        ece
        calibration_table
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    rows = []

    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):

        mask = bin_ids == i

        if np.sum(mask) == 0:
            continue

        bin_size = np.sum(mask)

        avg_confidence = np.mean(y_prob[mask])
        accuracy = np.mean(y_true[mask])

        calibration_gap = abs(avg_confidence - accuracy)

        weighted_gap = (bin_size / n) * calibration_gap

        ece += weighted_gap

        rows.append({
            "bin": i,
            "count": bin_size,
            "avg_confidence": avg_confidence,
            "empirical_accuracy": accuracy,
            "gap": calibration_gap,
        })

    calibration_table = pd.DataFrame(rows)

    return ece, calibration_table

def brier_decomposition(
    y_true,
    y_prob,
    n_bins=10,
):
    """
    Murphy decomposition:
    Brier = Reliability - Resolution + Uncertainty
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    overall_event_rate = np.mean(y_true)

    reliability = 0.0
    resolution = 0.0

    n = len(y_true)

    rows = []

    for i in range(n_bins):

        mask = bin_ids == i

        if np.sum(mask) == 0:
            continue

        n_bin = np.sum(mask)

        prob_mean = np.mean(y_prob[mask])
        outcome_mean = np.mean(y_true[mask])

        weight = n_bin / n

        # Reliability
        reliability += weight * (prob_mean - outcome_mean) ** 2

        # Resolution
        resolution += weight * (
            outcome_mean - overall_event_rate
        ) ** 2

        rows.append({
            "bin": i,
            "count": n_bin,
            "pred_mean": prob_mean,
            "actual_mean": outcome_mean,
        })

    uncertainty = overall_event_rate * (1 - overall_event_rate)

    brier = reliability - resolution + uncertainty

    decomposition_table = pd.DataFrame(rows)

    return {
        "brier_score": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "table": decomposition_table,
    }