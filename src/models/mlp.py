"""Small feed-forward classifier for frame-level pose features."""

from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel


class MLPModel(BaseModel):
    def __init__(self, parameters, preprocessing, random_seed):
        parameters = dict(parameters)
        parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
        parameters["random_state"] = random_seed

        self.imputer = SimpleImputer(strategy=preprocessing["imputation"])
        self.scaler = (
            StandardScaler()
            if preprocessing["scaling"]["mlp"] == "standard"
            else None
        )
        self.classifier = MLPClassifier(**parameters)

    def fit(self, X, y):
        X = self.imputer.fit_transform(X)
        if self.scaler:
            X = self.scaler.fit_transform(X)
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        X = self.imputer.transform(X)
        if self.scaler:
            X = self.scaler.transform(X)
        return self.classifier.predict(X)
