"""Frame-level and video-level classification metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def _class_names(labels, label_names=None):
    if label_names is None:
        return [str(label) for label in labels]
    if isinstance(label_names, dict):
        return [label_names[label] for label in labels]
    return list(label_names)


def compute_classification_metrics(y_true, y_pred, labels, label_names=None):
    """Return accuracy, macro/weighted F1, and per-class metrics."""
    names = _class_names(labels, label_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(names)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
    }


def compute_confusion_matrix(y_true, y_pred, labels, label_names=None, normalize=None):
    """Return a labeled confusion matrix; normalize by true, predicted, or all."""
    names = _class_names(labels, label_names)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize == "true":
        totals = matrix.sum(axis=1, keepdims=True)
    elif normalize in ("pred", "predicted"):
        totals = matrix.sum(axis=0, keepdims=True)
    elif normalize == "all":
        totals = np.asarray([[matrix.sum()]])
    elif normalize is None:
        totals = None
    else:
        raise ValueError("normalize must be None, 'true', 'pred', 'predicted', or 'all'.")

    if totals is not None:
        matrix = np.divide(
            matrix,
            totals,
            out=np.zeros(matrix.shape, dtype=float),
            where=totals != 0,
        )
    return pd.DataFrame(matrix, index=names, columns=names)


def compute_per_video_metrics(y_true, y_pred, video_ids, labels, label_names=None):
    """Return frame counts and classification metrics separately per video."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    video_ids = np.asarray(video_ids)
    names = _class_names(labels, label_names)
    rows = []

    for video_id in dict.fromkeys(video_ids.tolist()):
        mask = video_ids == video_id
        metrics = compute_classification_metrics(
            y_true[mask], y_pred[mask], labels, label_names
        )
        row = {
            "video_id": video_id,
            "frames": int(mask.sum()),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }
        for name in names:
            row[f"f1_{name}"] = metrics["per_class"][name]["f1"]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_predictions(y_true, y_pred, labels, label_names=None, video_ids=None):
    """Bundle aggregate metrics, confusion matrix, and optional per-video metrics."""
    result = {
        "metrics": compute_classification_metrics(y_true, y_pred, labels, label_names),
        "confusion_matrix": compute_confusion_matrix(
            y_true, y_pred, labels, label_names
        ),
        "per_video": None,
    }
    if video_ids is not None:
        result["per_video"] = compute_per_video_metrics(
            y_true, y_pred, video_ids, labels, label_names
        )
    return result


def compare_model_runs(run_results, metrics=("macro_f1", "weighted_f1", "accuracy")):
    """Build a comparison table for model/feature-set runs using the same split."""
    rows = []
    for run in run_results:
        values = run.get("metrics", run)
        row = {key: run[key] for key in ("model", "feature_set") if key in run}
        for metric in metrics:
            row[metric] = values[metric]
        rows.append(row)
    return pd.DataFrame(rows)
