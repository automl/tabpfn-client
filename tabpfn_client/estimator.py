#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from typing import Optional, Literal, Dict, Union
import logging
import numpy as np
import pandas as pd
from tabpfn_client.config import init
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

from tabpfn_client.config import Config
from tabpfn_client.service_wrapper import InferenceClient

logger = logging.getLogger(__name__)

MAX_ROWS = 10000
MAX_COLS = 500
MAX_NUMBER_OF_CLASSES = 10


class TabPFNModelSelection:
    """Base class for TabPFN model selection and path handling."""

    _AVAILABLE_MODELS: list[str] = []
    _VALID_TASKS = {"classification", "regression"}

    @classmethod
    def list_available_models(cls) -> list[str]:
        return cls._AVAILABLE_MODELS

    @classmethod
    def _validate_model_name(cls, model_name: str) -> None:
        if model_name != "default" and model_name not in cls._AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Available models are: {cls.list_available_models()}"
            )

    @classmethod
    def _model_name_to_path(
        cls, task: Literal["classification", "regression"], model_name: str
    ) -> str:
        cls._validate_model_name(model_name)
        model_name_task = "classifier" if task == "classification" else "regressor"
        if model_name == "default":
            return f"tabpfn-v2-{model_name_task}.ckpt"
        return f"tabpfn-v2-{model_name_task}-{model_name}.ckpt"


class TabPFNClassifier(BaseEstimator, ClassifierMixin, TabPFNModelSelection):
    _AVAILABLE_MODELS = [
        "default",
        "gn2p4bpt",
        "llderlii",
        "od3j1g5m",
        "vutqq28w",
        "znskzxi4",
    ]

    def __init__(
        self,
        model_path: str = "default",
        n_estimators: int = 4,
        softmax_temperature: float = 0.9,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        ignore_pretraining_limits: bool = False,
        inference_precision: Literal["autocast", "auto"] = "auto",
        random_state: Optional[
            Union[int, np.random.RandomState, np.random.Generator]
        ] = None,
        inference_config: Optional[Dict] = None,
        paper_version: bool = False,
    ):
        """Initialize TabPFNClassifier.

        Parameters
        ----------
        model_path: str, default="default"
            The name of the model to use.
        n_estimators: int, default=4
            The number of estimators in the TabPFN ensemble. We aggregate the
             predictions of `n_estimators`-many forward passes of TabPFN. Each forward
             pass has (slightly) different input data. Think of this as an ensemble of
             `n_estimators`-many "prompts" of the input data.
        softmax_temperature: float, default=0.9
            The temperature for the softmax function. This is used to control the
            confidence of the model's predictions. Lower values make the model's
            predictions more confident. This is only applied when predicting during a
            post-processing step. Set `softmax_temperature=1.0` for no effect.
        balance_probabilities: bool, default=False
            Whether to balance the probabilities based on the class distribution
            in the training data. This can help to improve predictive performance
            when the classes are highly imbalanced. This is only applied when predicting
            during a post-processing step.
        average_before_softmax: bool, default=False
             Only used if `n_estimators > 1`. Whether to average the predictions of the
             estimators before applying the softmax function. This can help to improve
             predictive performance when there are many classes or when calibrating the
             model's confidence. This is only applied when predicting during a
             post-processing.
        ignore_pretraining_limits: bool, default=False
            Whether to ignore the pre-training limits of the model. The TabPFN models
            have been pre-trained on a specific range of input data. If the input data
            is outside of this range, the model may not perform well. You may ignore
            our limits to use the model on data outside the pre-training range.
        inference_precision: "autocast" or "auto", default="auto"
            The precision to use for inference. This can dramatically affect the
            speed and reproducibility of the inference.
        random_state: int or RandomState or RandomGenerator or None, default=None
            Controls the randomness of the model. Pass an int for reproducible results.
        inference_config: dict or None, default=None
            Additional advanced arguments for model interface. See the doc of ModelInterfaceConfig
            in the tabpfn package for more details. For the client, the inference_config and the
            preprocess transforms need to be dictionaries.
        paper_version: bool, default=False
            If True, will use the model described in the paper, instead of the newest
            version available on the API, which e.g handles text features better.
        """
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision = inference_precision
        self.random_state = random_state
        self.inference_config = inference_config
        self.paper_version = paper_version
        self.last_train_set_uid = None
        self.last_train_X = None
        self.last_train_y = None

    def fit(self, X, y):
        # assert init() is called
        init()

        validate_data_size(X, y)
        X = _clean_text_features(X)
        self._validate_targets_and_classes(y)
        _check_paper_version(self.paper_version, X)

        estimator_param = self.get_params()
        estimator_param["model_path"] = TabPFNClassifier._model_name_to_path(
            "classification", self.model_path
        )
        if Config.use_server:
            self.last_train_set_uid = InferenceClient.fit(X, y, config=estimator_param)
            self.last_train_X = X
            self.last_train_y = y
            self.fitted_ = True
        else:
            raise NotImplementedError(
                "Only server mode is supported at the moment for init(use_server=False)"
            )
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: The input samples.

        Returns:
            The predicted class labels.
        """
        return self._predict(X, output_type="preds")

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Args:
            X: The input samples.

        Returns:
            The class probabilities of the input samples.
        """
        return self._predict(X, output_type="probas")

    def _predict(self, X, output_type):
        check_is_fitted(self)
        validate_data_size(X)
        _check_paper_version(self.paper_version, X)
        X = _clean_text_features(X)

        estimator_param = self.get_params()
        estimator_param["model_path"] = TabPFNClassifier._model_name_to_path(
            "classification", self.model_path
        )

        res = InferenceClient.predict(
            X,
            task="classification",
            train_set_uid=self.last_train_set_uid,
            config=estimator_param,
            predict_params={"output_type": output_type},
            X_train=self.last_train_X,
            y_train=self.last_train_y,
        )
        return res

    def _validate_targets_and_classes(self, y) -> np.ndarray:
        y_ = column_or_1d(y, warn=True)
        if sum(pd.isnull(y_)) > 0:
            raise ValueError("Input y contains NaN.")
        check_classification_targets(y)
        # Get classes and encode before type conversion to guarantee correct class labels.
        # TODO: should pass this from the server
        self.classes_ = np.unique(y_)
        # TODO: these things should ideally be shared with the local package
        if len(self.classes_) > MAX_NUMBER_OF_CLASSES:
            raise ValueError(
                f"Number of classes {len(self.classes_)} exceeds the maximal number of "
                f"{MAX_NUMBER_OF_CLASSES} classes supported by TabPFN. Consider using "
                "a strategy to reduce the number of classes. For code see "
                "https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/"
                "tabpfn_extensions/many_class/many_class_classifier.py"
            )


class TabPFNRegressor(BaseEstimator, RegressorMixin, TabPFNModelSelection):
    _AVAILABLE_MODELS = [
        "default",
        "2noar4o2",
        "5wof9ojf",
        "09gpqh39",
        "wyl4o83o",
    ]

    def __init__(
        self,
        model_path: str = "default",
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        average_before_softmax: bool = False,
        ignore_pretraining_limits: bool = False,
        inference_precision: Literal["autocast", "auto"] = "auto",
        random_state: Optional[
            Union[int, np.random.RandomState, np.random.Generator]
        ] = None,
        inference_config: Optional[Dict] = None,
        paper_version: bool = False,
    ):
        """Initialize TabPFNRegressor.

        Parameters
        ----------
        model_path: str, default="default"
            The name to the model to use.
        n_estimators: int, default=8
            The number of estimators in the TabPFN ensemble. We aggregate the
             predictions of `n_estimators`-many forward passes of TabPFN. Each forward
             pass has (slightly) different input data. Think of this as an ensemble of
             `n_estimators`-many "prompts" of the input data.
        softmax_temperature: float, default=0.9
            The temperature for the softmax function. This is used to control the
            confidence of the model's predictions. Lower values make the model's
            predictions more confident. This is only applied when predicting during a
            post-processing step. Set `softmax_temperature=1.0` for no effect.
        average_before_softmax: bool, default=False
            Only used if `n_estimators > 1`. Whether to average the predictions of the
            estimators before applying the softmax function. This can help to improve
            predictive performance when calibrating the model's confidence. This is only
            applied when predicting during a post-processing step.
        ignore_pretraining_limits: bool, default=False
            Whether to ignore the pre-training limits of the model. The TabPFN models
            have been pre-trained on a specific range of input data. If the input data
            is outside of this range, the model may not perform well. You may ignore
            our limits to use the model on data outside the pre-training range.
        inference_precision: "autocast" or "auto", default="auto"
            The precision to use for inference. This can dramatically affect the
            speed and reproducibility of the inference.
        random_state: int or RandomState or RandomGenerator or None, default=None
            Controls the randomness of the model. Pass an int for reproducible results.
        inference_config: dict or None, default=None
            Additional advanced arguments for model interface. See the doc of ModelInterfaceConfig
            in the tabpfn package for more details. For the client, the inference_config and the
            preprocess transforms need to be dictionaries.
        paper_version: bool, default=False
            If True, will use the model described in the paper, instead of the newest
            version available on the API, which e.g handles text features better.
        """
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision = inference_precision
        self.random_state = random_state
        self.inference_config = inference_config
        self.paper_version = paper_version
        self.last_train_set_uid = None
        self.last_train_X = None
        self.last_train_y = None

    def fit(self, X, y):
        # assert init() is called
        init()

        validate_data_size(X, y)
        self._validate_targets(y)
        X = _clean_text_features(X)
        _check_paper_version(self.paper_version, X)

        estimator_param = self.get_params()
        estimator_param["model_path"] = TabPFNRegressor._model_name_to_path(
            "regression", self.model_path
        )
        if Config.use_server:
            self.last_train_set_uid = InferenceClient.fit(X, y, config=estimator_param)
            self.last_train_X = X
            self.last_train_y = y
            self.fitted_ = True
        else:
            raise NotImplementedError(
                "Only server mode is supported at the moment for init(use_server=False)"
            )

        return self

    def predict(
        self,
        X: np.ndarray,
        output_type: Literal[
            "mean", "median", "mode", "quantiles", "full", "main"
        ] = "mean",
        quantiles: Optional[list[float]] = None,
    ) -> Union[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        output_type : str, default="mean"
            The type of prediction to return:
            - "mean": Return mean prediction
            - "median": Return median prediction
            - "mode": Return mode prediction
            - "quantiles": Return predictions for specified quantiles
            - "full": Return full prediction details
            - "main": Return main prediction metrics
        quantiles : list[float] or None, default=None
            Quantiles to compute when output_type="quantiles".
            Default is [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        Returns
        -------
        array-like or dict
            The predicted values.
        """
        check_is_fitted(self)
        validate_data_size(X)
        X = _clean_text_features(X)
        _check_paper_version(self.paper_version, X)

        # Add new parameters
        predict_params = {
            "output_type": output_type,
            "quantiles": quantiles,
        }

        estimator_param = self.get_params()
        estimator_param["model_path"] = TabPFNRegressor._model_name_to_path(
            "regression", self.model_path
        )

        return InferenceClient.predict(
            X,
            task="regression",
            train_set_uid=self.last_train_set_uid,
            config=estimator_param,
            predict_params=predict_params,
            X_train=self.last_train_X,
            y_train=self.last_train_y,
        )

    def _validate_targets(self, y) -> np.ndarray:
        y_ = column_or_1d(y, warn=True)
        if sum(pd.isnull(y_)) > 0:
            raise ValueError("Input y contains NaN.")


def validate_data_size(X: np.ndarray, y: Union[np.ndarray, None] = None):
    """
    Check the integrity of the training data.
    - check if the number of rows between X and y is consistent
        if y is not None (ValueError)
    - check if the number of rows is less than MAX_ROWS (ValueError)
    - check if the number of columns is less than MAX_COLS (ValueError)
    """

    # check if the number of samples is consistent (ValueError)
    if y is not None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

    # length and feature assertions
    if X.shape[0] > MAX_ROWS:
        raise ValueError(f"The number of rows cannot be more than {MAX_ROWS}.")
    if X.shape[1] > MAX_COLS:
        raise ValueError(f"The number of columns cannot be more than {MAX_COLS}.")


def _check_paper_version(paper_version, X):
    pass


def _clean_text_features(X):
    """
    Clean text features in the input data. This is used to avoid
    serialization errors, which happens when the input data contains
    commas or weird spaces, and to limit the length of the text features.
    """
    # Convert numpy array to pandas DataFrame if necessary
    # not necessary if numpy array of numbers
    if isinstance(X, np.ndarray):
        if np.issubdtype(X.dtype, np.number):
            return X
        else:
            X_ = pd.DataFrame(X.copy())
    else:
        X_ = X.copy()

    # limit to 2500 chars and remove commas for text features
    for col in X_.columns:
        # check if we can't convert to float
        try:
            pd.to_numeric(X_[col])
        except ValueError:
            if X_[col].dtype == object:  # only process string/object columns
                X_[col] = (
                    X_[col]
                    .str.replace(",", "")
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                    .str.slice(0, 2500)
                )

    # Convert back to numpy if input was numpy
    if isinstance(X, np.ndarray):
        return X_.to_numpy()
    return X_
