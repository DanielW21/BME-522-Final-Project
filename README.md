# BME 522 Final Project

Compare k-nearest neighbours, random forest, support vector machine, and a small
neural network for frame-level mouse social behaviour classification using
CalMS21 Task 1. Labels: attack, investigation, mount, and other.

## Set up the dataset

```sh
python3 scripts/setup_data.py
```

## Planned modules

Add these files when implementing their corresponding steps:

| Location                           | Responsibility                                              |
| ---------------------------------- | ----------------------------------------------------------- |
| `src/data/load_dataset.py`       | Read videos, keypoints, labels, and video IDs.              |
| `src/data/split_dataset.py`      | Define reproducible train/validation/test video splits.     |
| `src/features/spatial.py`        | Inter-mouse distances, relative orientation, and posture.   |
| `src/features/motion.py`         | Speeds and short-term changes computed within each video.   |
| `src/features/build_features.py` | Produce aligned features and labels for either feature set. |
| `src/models/knn.py`              | k-nearest neighbours.                                       |
| `src/models/random_forest.py`    | Random forest.                                              |
| `src/models/svm.py`              | Support vector machine.                                     |
| `src/models/mlp.py`              | Small neural network.                                       |
