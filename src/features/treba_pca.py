"""Train-only PCA transform for the TREBA block in combined features."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.features.pose import FEATURE_SETS as POSE_FEATURE_SETS


def pca_treba_block(data, n_features, whiten, imputation):
    """Reduce only TREBA columns, fitting imputation, scaling, and PCA on train."""
    pose_width = len(POSE_FEATURE_SETS["spatial_motion"])
    X_train = data["X_train"]
    X_validation = data["X_validation"]

    imputer = SimpleImputer(strategy=imputation)
    train_treba = imputer.fit_transform(X_train[:, pose_width:])
    validation_treba = imputer.transform(X_validation[:, pose_width:])

    scaler = StandardScaler().fit(train_treba)
    train_treba = scaler.transform(train_treba)
    validation_treba = scaler.transform(validation_treba)

    pca = PCA(n_components=n_features, whiten=whiten, svd_solver="full")
    train_components = pca.fit_transform(train_treba)
    validation_components = pca.transform(validation_treba)

    transformed = dict(data)
    transformed["X_train"] = np.column_stack(
        [X_train[:, :pose_width], train_components]
    )
    transformed["X_validation"] = np.column_stack(
        [X_validation[:, :pose_width], validation_components]
    )
    transformed["feature_names"] = data["feature_names"][:pose_width] + [
        f"treba_pc_{index + 1:02d}" for index in range(n_features)
    ]

    metrics = {
        "pca_components": int(pca.n_components_),
        "pca_variance_retained": float(pca.explained_variance_ratio_.sum()),
    }
    return transformed, metrics
