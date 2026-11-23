"""Reusable readers and validators for CalMS21 video records."""

from pathlib import Path

import ijson
import numpy as np


ANNOTATOR_KEY = "annotator-id_0"
IMAGE_SIZE = (1024, 570)  # width, height in pixels
BODY_PARTS = (
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "left_hip",
    "right_hip",
    "tail_base",
)
LABEL_NAMES = {
    0: "attack",
    1: "investigation",
    2: "mount",
    3: "other",
}


def iter_videos(path, prefix=ANNOTATOR_KEY):
    """Yield ``(video_id, record)`` pairs from a Task 1 JSON file, one at a time."""
    with Path(path).open("rb") as source:
        yield from ijson.kvitems(source, prefix, use_float=True)


def get_video_arrays(video, video_id="<unknown>"):
    """Convert and validate a pose record's keypoints, scores, and annotations."""
    try:
        keypoints = np.asarray(video["keypoints"], dtype=np.float32)
        scores = np.asarray(video["scores"], dtype=np.float32)
        labels = np.asarray(video["annotations"], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"Missing {error.args[0]!r} in video {video_id}.") from error

    if keypoints.ndim != 4 or keypoints.shape[1:] != (2, 2, 7):
        raise ValueError(f"Unexpected keypoint shape in {video_id}: {keypoints.shape}")
    if scores.shape != (len(keypoints), 2, 7):
        raise ValueError(f"Unexpected score shape in {video_id}: {scores.shape}")
    if labels.shape != (len(keypoints),):
        raise ValueError(f"Keypoints, scores, and labels do not align in {video_id}.")
    unknown_labels = set(np.unique(labels)) - set(LABEL_NAMES)
    if unknown_labels:
        raise ValueError(f"Unknown labels in {video_id}: {sorted(unknown_labels)}")
    return keypoints, scores, labels


def valid_keypoints(keypoints, scores, threshold=None, image_size=IMAGE_SIZE):
    """Return a ``(frames, mice, body_parts)`` mask for usable tracked points.

    A point is valid when its coordinates and confidence are finite, its
    coordinates lie within the image bounds, and (when supplied) its score is
    at least ``threshold``. ``image_size`` is given as ``(width, height)``.
    """
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)
    if keypoints.ndim != 4 or keypoints.shape[1:] != (2, 2, 7):
        raise ValueError(f"Expected keypoints shaped (frames, 2, 2, 7), got {keypoints.shape}")
    if scores.shape != (len(keypoints), 2, 7):
        raise ValueError(f"Expected scores shaped {(len(keypoints), 2, 7)}, got {scores.shape}")
    if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size must contain positive (width, height) values.")

    width, height = image_size
    x = keypoints[:, :, 0, :]
    y = keypoints[:, :, 1, :]
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(scores)
        & (x >= 0)
        & (x <= width)
        & (y >= 0)
        & (y <= height)
    )
    if threshold is not None:
        valid &= scores >= threshold
    return valid


def get_video_example(path, video_id=None, max_frames=None, prefix=ANNOTATOR_KEY):
    """Read the first or selected video, optionally truncating frame arrays."""
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative or None.")

    videos = iter_videos(path, prefix=prefix)
    try:
        for current_id, video in videos:
            if video_id is not None and current_id != video_id:
                continue
            if max_frames is not None:
                frame_fields = {"keypoints", "scores", "annotations", "features"}
                video = {
                    field: values[:max_frames] if field in frame_fields else values
                    for field, values in video.items()
                }
            return current_id, video
    finally:
        videos.close()

    raise ValueError(f"Video {video_id!r} was not found in {Path(path).name}")


def iter_behavior_bouts(labels):
    """Yield ``(label_id, frame_count)`` for each contiguous label run."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"Expected one-dimensional labels, got {labels.shape}")
    if len(labels) == 0:
        return

    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    boundaries = np.r_[0, changes, len(labels)]
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        yield int(labels[start]), int(end - start)
