"""Build baseline sklearn pipelines from model settings in config.yaml."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.models.majority import MajorityClassModel
from src.models.mlp import MLPModel


def build_model(name, parameters, preprocessing, random_seed):
    """Create one configured classifier with train-only preprocessing."""
    parameters = dict(parameters)
    if name == "majority":
        return MajorityClassModel()
    if name == "mlp":
        return MLPModel(parameters, preprocessing, random_seed)

    steps = [
        ("imputer", SimpleImputer(strategy=preprocessing["imputation"]))
    ]

    if name == "knn":
        classifier = KNeighborsClassifier(**parameters)
    elif name == "random_forest":
        parameters["random_state"] = random_seed
        classifier = RandomForestClassifier(**parameters)
    elif name == "svm":
        parameters["random_state"] = random_seed
        classifier = LinearSVC(**parameters)

    if preprocessing["scaling"][name] == "standard":
        steps.append(("scale", StandardScaler()))

    steps.append(("classifier", classifier))
    return Pipeline(steps)
