"""Function interfaces for figures used in the CalMS21 model comparison."""


def plot_confusion_matrix(matrix, title="Confusion matrix", output_path=None):
    """Plot a labeled confusion matrix for one model and feature set."""
    pass


def plot_per_class_metrics(metrics, metric="f1", title=None, output_path=None):
    """Plot per-behavior scores so rare-class performance is visible."""
    pass


def plot_model_comparison(run_results, metric="macro_f1", title=None, output_path=None):
    """Compare one metric across selected model and feature-set combinations."""
    pass


def plot_per_video_metrics(video_results, metric="macro_f1", title=None, output_path=None):
    """Show score variation across held-out videos, not only the pooled frame score."""
    pass


def plot_feature_importance(importance, feature_names, title="Feature importance", output_path=None):
    """Plot model-provided feature scores when the selected estimator exposes them."""
    pass
