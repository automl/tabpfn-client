from abc import ABC, abstractmethod


class TabPFNClassifierInterface(ABC):

    @abstractmethod
    def remove_models_from_memory(self):
        pass

    @abstractmethod
    def load_result_minimal(self, path, i, e):
        pass

    @abstractmethod
    def fit(self, X, y):
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

