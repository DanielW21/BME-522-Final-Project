"""Load CalMS21 videos using one shared identity-grouped development split."""

import csv
import re
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.data.preprocessing import LABEL_NAMES, get_video_arrays, iter_videos
from src.features import FEATURE_SETS, TREBA_FEATURES, extract_pose_features


def iter_training_videos(training_json):
    """Yield one video payload at a time from the annotated Task 1 training file."""
    yield from iter_videos(training_json)


def read_split_manifest(manifest_path):
    """Read and validate a saved CSV video split."""
    assignments = {}
    with Path(manifest_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not {"video_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("The split CSV must have video_id and split columns.")
        for row in reader:
            video_id = row["video_id"].strip()
            split = row["split"].strip()
            if not video_id or split not in ("train", "validation"):
                raise ValueError(f"Invalid split row: {row}")
            if video_id in assignments:
                raise ValueError(f"Duplicate video ID in split CSV: {video_id}")
            assignments[video_id] = split
    if not assignments:
        raise ValueError(f"The split CSV is empty: {manifest_path}")
    if set(assignments.values()) != {"train", "validation"}:
        raise ValueError("The split must contain both train and validation videos.")
    return assignments


def create_validation_assignments(
    training_json, random_seed, validation_split, split_candidates
):
    """Choose a fixed identity-grouped split with representative class and frame shares."""
    video_ids = []
    identity_ids = []
    frame_counts = []
    class_counts = []
    for video_id, payload in iter_training_videos(training_json):
        labels = np.asarray(payload["annotations"], dtype=np.int64)
        video_ids.append(video_id)
        identity_ids.append(re.search(r"mouse\d+", video_id).group())
        frame_counts.append(len(labels))
        class_counts.append(np.bincount(labels, minlength=len(LABEL_NAMES)))

    order = np.argsort(video_ids)
    video_ids = np.asarray(video_ids, dtype=object)[order]
    identity_ids = np.asarray(identity_ids, dtype=object)[order]
    frame_counts = np.asarray(frame_counts)[order]
    class_counts = np.stack(class_counts)[order]
    overall_share = class_counts.sum(axis=0) / class_counts.sum()

    splitter = GroupShuffleSplit(
        n_splits=split_candidates,
        test_size=validation_split,
        random_state=random_seed,
    )
    best_score = np.inf
    best_validation_indices = None
    for train_indices, validation_indices in splitter.split(
        np.arange(len(video_ids)), groups=identity_ids
    ):
        train_counts = class_counts[train_indices].sum(axis=0)
        validation_counts = class_counts[validation_indices].sum(axis=0)
        if (train_counts == 0).any() or (validation_counts == 0).any():
            continue

        train_share = train_counts / train_counts.sum()
        validation_share = validation_counts / validation_counts.sum()
        class_error = (
            np.abs(train_share - overall_share) / overall_share
        ).mean()
        class_error += (
            np.abs(validation_share - overall_share) / overall_share
        ).mean()
        frame_error = abs(
            frame_counts[validation_indices].sum() / frame_counts.sum()
            - validation_split
        ) / validation_split
        score = class_error + frame_error
        if score < best_score:
            best_score = score
            best_validation_indices = validation_indices

    assignments = {video_id: "train" for video_id in video_ids}
    for index in best_validation_indices:
        assignments[video_ids[index]] = "validation"

    return assignments


def write_split_manifest(assignments, manifest_path):
    """Write one video ID and split assignment per CSV row."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["video_id", "split"])
        writer.writeheader()
        for video_id in sorted(assignments):
            writer.writerow({"video_id": video_id, "split": assignments[video_id]})
    return assignments


def ensure_split_manifest(
    training_json, manifest_path, random_seed, validation_split, split_candidates
):
    """Reuse the saved split or create an identity-grouped, class-balanced split."""
    manifest_path = Path(manifest_path)
    if manifest_path.is_file():
        return read_split_manifest(manifest_path)

    assignments = create_validation_assignments(
        training_json, random_seed, validation_split, split_candidates
    )
    return write_split_manifest(assignments, manifest_path)


def _video_arrays(video_id, payload):
    keypoints, _, labels = get_video_arrays(payload, video_id=video_id)
    return keypoints, labels


def _assemble_split_parts(parts, feature_names):
    """Combine per-video arrays while preserving video-to-frame group IDs."""
    result = {}
    for split_name, videos in parts.items():
        if not videos:
            raise ValueError(f"No videos assigned to {split_name}.")
        result[f"X_{split_name}"] = np.concatenate([video[1] for video in videos])
        result[f"y_{split_name}"] = np.concatenate([video[2] for video in videos])
        result[f"{split_name}_video_ids"] = [video[0] for video in videos]
        result[f"{split_name}_groups"] = np.concatenate(
            [np.full(len(video[2]), index, dtype=np.int16) for index, video in enumerate(videos)]
        )
    result["feature_names"] = list(feature_names)
    return result


def _features_in_video_order(data, split_name, video_ids):
    """Return one split's feature rows in a requested video ID order."""
    source_video_ids = data[f"{split_name}_video_ids"]
    if source_video_ids == video_ids:
        return data[f"X_{split_name}"]

    source_groups = {video_id: index for index, video_id in enumerate(source_video_ids)}
    group_sizes = np.bincount(data[f"{split_name}_groups"], minlength=len(source_video_ids))
    offsets = np.r_[0, np.cumsum(group_sizes)]
    features = data[f"X_{split_name}"]
    ordered_features = []
    for video_id in video_ids:
        group = source_groups[video_id]
        ordered_features.append(features[offsets[group]:offsets[group + 1]])
    return np.concatenate(ordered_features)


def load_development_data(
    training_json,
    manifest_path,
    feature_set="spatial_motion",
    feature_json=None,
):
    """Return train/validation arrays, labels, and per-frame video groups.

    ``feature_set="treba"`` loads the precomputed 32-dimensional feature file
    and aligns each video to its annotations by video ID and frame count.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
    if feature_set in ("treba", "spatial_motion_treba"):
        if feature_json is None:
            raise ValueError(f"feature_json is required when feature_set={feature_set!r}.")
        treba_data = load_treba_development_data(
            training_json, feature_json, manifest_path
        )
        if feature_set == "treba":
            return treba_data

        pose_data = load_development_data(
            training_json, manifest_path, "spatial_motion"
        )
        result = dict(pose_data)
        for split_name in ("train", "validation"):
            treba_features = _features_in_video_order(
                treba_data,
                split_name,
                pose_data[f"{split_name}_video_ids"],
            )
            result[f"X_{split_name}"] = np.concatenate(
                [pose_data[f"X_{split_name}"], treba_features],
                axis=1,
            )
        result["feature_names"] = list(FEATURE_SETS[feature_set])
        return result

    assignments = read_split_manifest(manifest_path)
    feature_names = FEATURE_SETS[feature_set]
    parts = {"train": [], "validation": []}
    all_video_ids = set()

    for video_id, payload in iter_training_videos(training_json):
        if video_id in all_video_ids:
            raise ValueError(f"Duplicate video ID in training JSON: {video_id}")
        all_video_ids.add(video_id)
        if video_id not in assignments:
            continue

        keypoints, labels = _video_arrays(video_id, payload)
        features = extract_pose_features(keypoints)
        X = np.column_stack([features[name] for name in feature_names]).astype(np.float32)
        parts[assignments[video_id]].append((video_id, X, labels))

    missing_ids = set(assignments) - all_video_ids
    extra_ids = all_video_ids - set(assignments)
    if missing_ids or extra_ids:
        raise ValueError(
            "Split manifest does not match the training JSON: "
            f"{len(missing_ids)} missing IDs, {len(extra_ids)} unassigned IDs. "
            "Commit and reuse one manifest for the project dataset."
        )

    return _assemble_split_parts(parts, feature_names)


def load_treba_development_data(training_json, feature_json, manifest_path):
    """Load and align precomputed TREBA features with labels for the saved split."""
    assignments = read_split_manifest(manifest_path)
    labels_by_video = {}
    training_video_ids = set()

    for video_id, payload in iter_training_videos(training_json):
        if video_id in training_video_ids:
            raise ValueError(f"Duplicate video ID in training JSON: {video_id}")
        training_video_ids.add(video_id)
        if video_id in assignments:
            _, labels = _video_arrays(video_id, payload)
            labels_by_video[video_id] = labels

    missing_labels = set(assignments) - training_video_ids
    unassigned_labels = training_video_ids - set(assignments)
    if missing_labels or unassigned_labels:
        raise ValueError(
            "Split manifest does not match the training JSON: "
            f"{len(missing_labels)} missing IDs, {len(unassigned_labels)} unassigned IDs."
        )

    parts = {"train": [], "validation": []}
    found_feature_ids = set()
    expected_width = len(TREBA_FEATURES)
    for video_id, payload in iter_videos(feature_json):
        if video_id not in assignments:
            continue
        if video_id in found_feature_ids:
            raise ValueError(f"Duplicate video ID in TREBA feature JSON: {video_id}")
        found_feature_ids.add(video_id)

        if "features" not in payload:
            raise ValueError(f"TREBA record has no 'features' field: {video_id}")
        features = np.asarray(payload["features"], dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != expected_width:
            raise ValueError(
                f"Expected {expected_width} TREBA features per frame in {video_id}, "
                f"got {features.shape}."
            )
        labels = labels_by_video[video_id]
        if len(features) != len(labels):
            raise ValueError(
                f"TREBA features and annotations do not align in {video_id}: "
                f"{len(features)} feature rows, {len(labels)} labels."
            )
        parts[assignments[video_id]].append((video_id, features, labels))

    missing_features = set(assignments) - found_feature_ids
    if missing_features:
        sample_ids = sorted(missing_features)[:5]
        raise ValueError(
            f"TREBA feature JSON is missing {len(missing_features)} split videos; "
            f"examples: {sample_ids}"
        )

    return _assemble_split_parts(parts, TREBA_FEATURES)
