"""Build baseline sklearn pipelines from model settings in config.yaml."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.models.majority import MajorityClassModel


def build_model(name, parameters, random_seed):
    """Create one configured classifier with train-only imputation and scaling."""
    parameters = dict(parameters or {})
    if name == "majority":
        return MajorityClassModel()

    steps = [("imputer", SimpleImputer(strategy="median"))]

    if name == "knn":
        steps.append(("scale", StandardScaler()))
        classifier = KNeighborsClassifier(**parameters)
    elif name == "random_forest":
        parameters.setdefault("random_state", random_seed)
        classifier = RandomForestClassifier(**parameters)
    elif name == "svm":
        steps.append(("scale", StandardScaler()))
        parameters.setdefault("random_state", random_seed)
        classifier = LinearSVC(**parameters)
    elif name == "mlp":
        steps.append(("scale", StandardScaler()))
        parameters.setdefault("random_state", random_seed)
        if "hidden_layer_sizes" in parameters:
            parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
        classifier = MLPClassifier(**parameters)
    else:
        raise ValueError(
            f"Unknown model {name!r}; choose majority, knn, random_forest, svm, or mlp."
        )

    steps.append(("classifier", classifier))
    return Pipeline(steps)
