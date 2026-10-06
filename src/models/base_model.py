from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    base class for all prediction models
    """

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass