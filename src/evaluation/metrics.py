"""Function interfaces for frame-level and video-level model evaluation."""


def compute_classification_metrics(y_true, y_pred, labels, label_names=None):
    """Return accuracy, macro/weighted F1, and per-class precision, recall, F1, support.

    Keep the supplied label order so metrics remain comparable across runs, including
    classes that are absent from a particular validation subset.
    """
    pass


def compute_confusion_matrix(y_true, y_pred, labels, label_names=None, normalize=None):
    """Return a labeled confusion matrix; optionally normalize by true, predicted, or all."""
    pass


def compute_per_video_metrics(y_true, y_pred, video_ids, labels, label_names=None):
    """Return frame counts and classification metrics separately for each held-out video."""
    pass


def evaluate_predictions(y_true, y_pred, labels, label_names=None, video_ids=None):
    """Bundle aggregate metrics, confusion matrix, and optional per-video metrics."""
    pass


def compare_model_runs(run_results, metrics=("macro_f1", "weighted_f1", "accuracy")):
    """Build a comparison table for model/feature-set runs using the same split."""
    pass
