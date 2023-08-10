from abc import ABC, abstractmethod


class AbstractTabPFNClassifier(ABC):

    @abstractmethod
    def remove_models_from_memory(self):
        pass

    @abstractmethod
    def fit(self, X, y, overwrite_warning=False):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X, return_winning_probability=False, normalize_with_test=False):
        pass

    @abstractmethod
    def try_root(self):
        pass

