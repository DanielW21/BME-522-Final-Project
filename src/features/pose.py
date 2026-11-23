import numpy as np


SPATIAL_FEATURES = (
    "center_distance_px",
    "nose_distance_px",
    "resident_body_length_px",
    "intruder_body_length_px",
    "resident_orientation_to_intruder_cos",
    "intruder_orientation_to_resident_cos",
)
MOTION_FEATURES = (
    "resident_speed_px_per_frame",
    "intruder_speed_px_per_frame",
    "approach_rate_px_per_frame",
)
FEATURE_SETS = {
    "spatial": SPATIAL_FEATURES,
    "spatial_motion": SPATIAL_FEATURES + MOTION_FEATURES,
}


def extract_pose_features(keypoints):
    """Create pose features for one video shaped (frames, 2 mice, 2 coordinates, 7 parts)."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.ndim != 4 or keypoints.shape[1:] != (2, 2, 7):
        raise ValueError(f"Expected (frames, 2, 2, 7) keypoints, got {keypoints.shape}")

    nose = keypoints[:, :, :, 0]
    tail_base = keypoints[:, :, :, 6]
    centers = keypoints[:, :, :, 3:6].mean(axis=3)
    body_axis = nose - tail_base
    body_length = np.linalg.norm(body_axis, axis=2)

    between = centers[:, 1] - centers[:, 0]
    center_distance = np.linalg.norm(between, axis=1)
    nose_distance = np.linalg.norm(nose[:, 1] - nose[:, 0], axis=1)
    unit_between = between / np.maximum(center_distance[:, None], 1e-6)
    unit_body = body_axis / np.maximum(body_length[:, :, None], 1e-6)
    resident_facing = np.sum(unit_body[:, 0] * unit_between, axis=1)
    intruder_facing = np.sum(unit_body[:, 1] * -unit_between, axis=1)

    center_displacement = np.vstack(
        [np.full((1, 2, 2), np.nan, dtype=np.float32), np.diff(centers, axis=0)]
    )
    speed = np.linalg.norm(center_displacement, axis=2)
    approach_rate = np.r_[np.nan, -np.diff(center_distance)]

    return {
        "center_distance_px": center_distance,
        "nose_distance_px": nose_distance,
        "resident_body_length_px": body_length[:, 0],
        "intruder_body_length_px": body_length[:, 1],
        "resident_orientation_to_intruder_cos": resident_facing,
        "intruder_orientation_to_resident_cos": intruder_facing,
        "resident_speed_px_per_frame": speed[:, 0],
        "intruder_speed_px_per_frame": speed[:, 1],
        "approach_rate_px_per_frame": approach_rate,
    }
