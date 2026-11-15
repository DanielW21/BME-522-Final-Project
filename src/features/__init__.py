"""Feature catalog used by data loaders and experiment runners."""

from src.features.pose import (
    FEATURE_SETS as POSE_FEATURE_SETS,
    MOTION_FEATURES,
    SPATIAL_FEATURES,
    extract_pose_features,
)


TREBA_FEATURES = tuple(f"F{index:02d}" for index in range(1, 33))
FEATURE_SETS = {**POSE_FEATURE_SETS, "treba": TREBA_FEATURES}


__all__ = [
    "FEATURE_SETS",
    "MOTION_FEATURES",
    "POSE_FEATURE_SETS",
    "SPATIAL_FEATURES",
    "TREBA_FEATURES",
    "extract_pose_features",
]
