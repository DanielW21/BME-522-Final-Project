"""A simple majority-class baseline."""

import numpy as np

from src.models.base_model import BaseModel


class MajorityClassModel(BaseModel):
    """Always predict the most common label seen during fitting."""

    def fit(self, X, y):
        labels, counts = np.unique(y, return_counts=True)
        self.majority_class_ = labels[counts.argmax()]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_class_)


def main():
    X_train = np.array([[0], [1], [2], [3], [4]])
    y_train = np.array([0, 1, 1, 2, 1])
    X_test = np.array([[10], [11], [12]])

    model = MajorityClassModel().fit(X_train, y_train)
    print(f"Majority label: {model.majority_class_}")
    print(f"Predictions: {model.predict(X_test).tolist()}")


if __name__ == "__main__":
    main()
