"""Plots for comparing CalMS21 model runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import compare_model_runs


def _save_figure(fig, output_path):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    return fig


def plot_confusion_matrix(matrix, title="Confusion matrix", output_path=None):
    """Plot a labeled confusion matrix."""
    if isinstance(matrix, pd.DataFrame):
        values = matrix.to_numpy()
        labels = list(matrix.index)
    else:
        values = np.asarray(matrix)
        labels = [str(index) for index in range(values.shape[0])]

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted behavior")
    ax.set_ylabel("True behavior")
    ax.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = f"{value:.2f}" if np.issubdtype(values.dtype, np.floating) else f"{int(value):,}"
            ax.text(column, row, text, ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_per_class_metrics(metrics, metric="f1", title=None, output_path=None):
    """Plot one score for each behavior class."""
    per_class = metrics.get("per_class", metrics)
    names = list(per_class)
    values = [per_class[name][metric] for name in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, values, color="#5470a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Per-class {metric.replace('_', ' ')}")
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_model_comparison(run_results, metric="macro_f1", title=None, output_path=None):
    """Compare one metric across model and feature-set combinations."""
    if isinstance(run_results, pd.DataFrame):
        table = run_results
    else:
        table = compare_model_runs(run_results, metrics=(metric,))
    feature_labels = {
        "raw": "Raw",
        "spatial": "Spatial",
        "spatial_motion": "Spatial +\nmotion",
        "spatial_motion_treba": "Spatial + motion\n+ TREBA",
        "spatial_motion_treba_pca": "Spatial + motion\n+ TREBA PCA",
    }
    labels = [
        f"{row.model.upper()}\n{feature_labels.get(row.feature_set, row.feature_set.replace('_', ' '))}"
        for row in table.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.9), 5))
    bars = ax.bar(range(len(labels)), table[metric], color="#5470a8")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Model comparison: {metric.replace('_', ' ')}")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.tick_params(axis="x", labelrotation=0, labelsize=9)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_per_video_metrics(video_results, metric="macro_f1", title=None, output_path=None):
    """Show metric variation across held-out videos."""
    table = video_results if isinstance(video_results, pd.DataFrame) else pd.DataFrame(video_results)
    table = table.sort_values(metric, ascending=False).reset_index(drop=True)
    labels = table["video_id"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.55), 4.8))
    ax.bar(range(len(labels)), table[metric], color="#6a9f85")
    ax.set_xticks(range(len(labels)), labels, rotation=75, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Per-video {metric.replace('_', ' ')}")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def plot_feature_importance(importance, feature_names, title="Feature importance", output_path=None):
    """Plot model-provided feature scores."""
    importance = np.asarray(importance)
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.24)))
    ax.barh(np.asarray(feature_names)[order], importance[order], color="#c88c24")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    return _save_figure(fig, output_path)
