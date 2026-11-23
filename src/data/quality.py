"""Per-video data-quality summaries used during CalMS21 preprocessing."""

import numpy as np

from src.data.preprocessing import (
    LABEL_NAMES,
    get_video_arrays,
    iter_behavior_bouts,
    iter_videos,
    valid_keypoints,
)


def iter_video_quality(training_json, confidence_threshold=0.5):
    """Yield a compact quality summary for each video in the training JSON."""
    for video_id, payload in iter_videos(training_json):
        yield summarize_video_quality(video_id, payload, confidence_threshold)


def summarize_video_quality(video_id, payload, confidence_threshold=0.5):
    """Summarize labels, tracking confidence, invalid points, and label runs.

    The returned values are plain Python scalars and can be collected into a
    DataFrame by a notebook or written directly to a CSV by a script.
    """
    keypoints, scores, labels = get_video_arrays(payload, video_id=video_id)
    valid = valid_keypoints(keypoints, scores, threshold=confidence_threshold)
    counts = np.bincount(labels, minlength=len(LABEL_NAMES))
    finite_scores = scores[np.isfinite(scores)]

    result = {
        "video_id": video_id,
        "n_frames": int(len(labels)),
        "flagged_keypoint_pct": float(100 * (~valid).mean()) if valid.size else 0.0,
        "nonfinite_coordinates": int((~np.isfinite(keypoints)).sum()),
        "mean_keypoint_confidence": (
            float(finite_scores.mean()) if finite_scores.size else float("nan")
        ),
        "fraction_keypoints_below_threshold": (
            float((scores < confidence_threshold).mean()) if scores.size else 0.0
        ),
        "n_label_runs": sum(1 for _ in iter_behavior_bouts(labels)),
    }
    for label_id, name in LABEL_NAMES.items():
        result[f"frames_{name}"] = int(counts[label_id])
    return result
