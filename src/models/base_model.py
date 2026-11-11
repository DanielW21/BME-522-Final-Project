from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Learn from the training features and labels."""

    @abstractmethod
    def predict(self, X):
        """Return one predicted label per input row."""
