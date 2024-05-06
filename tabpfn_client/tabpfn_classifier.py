from typing import Optional, Tuple, Literal
import logging
import shutil
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from tabpfn import TabPFNClassifier as LocalTabPFNClassifier
from tabpfn_client.service_wrapper import UserAuthenticationClient, InferenceClient
from tabpfn_client.client import ServiceClient
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client.prompt_agent import PromptAgent


logger = logging.getLogger(__name__)


class TabPFNConfig:
    is_initialized = None
    use_server = None
    user_auth_handler = None
    inference_handler = None


g_tabpfn_config = TabPFNConfig()


def init(use_server=True):
    # initialize config
    use_server = use_server
    global g_tabpfn_config

    if use_server:
        PromptAgent.prompt_welcome()

        service_client = ServiceClient()
        user_auth_handler = UserAuthenticationClient(service_client)

        # check connection to server
        if not user_auth_handler.is_accessible_connection():
            raise RuntimeError("TabPFN is inaccessible at the moment, please try again later.")

        is_valid_token_set = user_auth_handler.try_reuse_existing_token()

        if is_valid_token_set:
            PromptAgent.prompt_reusing_existing_token()
        else:
            if not PromptAgent.prompt_terms_and_cond():
                raise RuntimeError("You must agree to the terms and conditions to use TabPFN")

            # prompt for login / register
            PromptAgent.prompt_and_set_token(user_auth_handler)

        # Print new greeting messages. If there are no new messages, nothing will be printed.
        PromptAgent.prompt_retrieved_greeting_messages(user_auth_handler.retrieve_greeting_messages())

        g_tabpfn_config.use_server = True
        g_tabpfn_config.user_auth_handler = user_auth_handler
        g_tabpfn_config.inference_handler = InferenceClient(service_client)

    else:
        g_tabpfn_config.use_server = False

    g_tabpfn_config.is_initialized = True


def reset():
    # reset config
    global g_tabpfn_config
    g_tabpfn_config = TabPFNConfig()

    # reset user auth handler
    if g_tabpfn_config.use_server:
        g_tabpfn_config.user_auth_handler.reset_cache()

    # remove cache dir
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


@dataclass(eq=True, frozen=True)
class PreprocessorConfig:
    """
    Configuration for data preprocessors.

    Attributes:
        name (Literal): Name of the preprocessor.
        categorical_name (Literal): Name of the categorical encoding method. Valid options are "none", "numeric",
                                "onehot", "ordinal", "ordinal_shuffled". Default is "none".
        append_original (bool): Whether to append the original features to the transformed features. Default is False.
        subsample_features (float): Fraction of features to subsample. -1 means no subsampling. Default is -1.
        global_transformer_name (str): Name of the global transformer to use. Default is None.
    """

    name: Literal[
        "per_feature",  # a different transformation for each feature
        "power",  # a standard sklearn power transformer
        "safepower",  # a power transformer that prevents some numerical issues
        "power_box",
        "safepower_box",
        "quantile_uni_coarse",  # different quantile transformations with few quantiles up to a lot
        "quantile_norm_coarse",
        "quantile_uni",
        "quantile_norm",
        "quantile_uni_fine",
        "quantile_norm_fine",
        "robust",  # a standard sklearn robust scaler
        "kdi",
        "none",  # no transformation (inside the transformer we anyways do a standardization)
        "kdi_random_alpha",
        "kdi_uni",
        "kdi_random_alpha_uni",
        "adaptive",
        "norm_and_kdi",
        # KDI with alpha collection
        "kdi_alpha_0.3_uni",
        "kdi_alpha_0.5_uni",
        "kdi_alpha_0.8_uni",
        "kdi_alpha_1.0_uni",
        "kdi_alpha_1.2_uni",
        "kdi_alpha_1.5_uni",
        "kdi_alpha_2.0_uni",
        "kdi_alpha_3.0_uni",
        "kdi_alpha_5.0_uni",
        "kdi_alpha_0.3",
        "kdi_alpha_0.5",
        "kdi_alpha_0.8",
        "kdi_alpha_1.0",
        "kdi_alpha_1.2",
        "kdi_alpha_1.5",
        "kdi_alpha_2.0",
        "kdi_alpha_3.0",
        "kdi_alpha_5.0",
    ]
    categorical_name: Literal[
        "none",
        "numeric",
        "onehot",
        "ordinal",
        "ordinal_shuffled",
        "ordinal_very_common_categories_shuffled",
    ] = "none"
    # categorical_name meanings:
    # "none": categorical features are pretty much treated as ordinal, just not resorted
    # "numeric": categorical features are treated as numeric, that means they are also power transformed for example
    # "onehot": categorical features are onehot encoded
    # "ordinal": categorical features are sorted and encoded as integers from 0 to n_categories - 1
    # "ordinal_shuffled": categorical features are encoded as integers from 0 to n_categories - 1 in a random order
    append_original: bool = False
    subsample_features: Optional[float] = -1
    global_transformer_name: Optional[str] = None
    # if True, the transformed features (e.g. power transformed) are appended to the original features

    def __str__(self):
        return (
            f"{self.name}_cat:{self.categorical_name}"
            + ("_and_none" if self.append_original else "")
            + (
                "_subsample_feats_" + str(self.subsample_features)
                if self.subsample_features > 0
                else ""
            )
            + (
                f"_global_transformer_{self.global_transformer_name}"
                if self.global_transformer_name is not None
                else ""
            )
        )

    def can_be_cached(self):
        return not self.subsample_features > 0

    def to_dict(self):
        return {k: str(v) if not isinstance(v, (str, int, float, list, dict)) else v for k, v in asdict(self).items()}


ClassificationOptimizationMetricType = Literal[
    "auroc", "roc", "auroc_ovo", "balanced_acc", "acc", "log_loss", None
]


class TabPFNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model="latest_tabpfn_hosted",
        n_estimators: int = 4,
        preprocess_transforms: Tuple[PreprocessorConfig, ...] = (
            PreprocessorConfig(
                "quantile_uni_coarse",
                append_original=True,
                categorical_name="ordinal_very_common_categories_shuffled",
                global_transformer_name="svd",
                subsample_features=-1,
            ),
            PreprocessorConfig(
                "none", categorical_name="numeric", subsample_features=-1
            ),
        ),
        feature_shift_decoder: str = "shuffle",
        normalize_with_test: bool = False,
        average_logits: bool = False,
        optimize_metric: ClassificationOptimizationMetricType = "roc",
        transformer_predict_kwargs: Optional[dict] = None,
        multiclass_decoder="shuffle",
        softmax_temperature: Optional[float] = -0.1,
        use_poly_features=False,
        max_poly_features=50,
        remove_outliers=12.0,
        add_fingerprint_features=True,
        subsample_samples=-1,
    ):
        self.model = model
        self.n_estimators = n_estimators
        self.preprocess_transforms = preprocess_transforms
        self.feature_shift_decoder = feature_shift_decoder
        self.normalize_with_test = normalize_with_test
        self.average_logits = average_logits
        self.optimize_metric = optimize_metric
        self.transformer_predict_kwargs = transformer_predict_kwargs
        self.multiclass_decoder = multiclass_decoder
        self.softmax_temperature = softmax_temperature
        self.use_poly_features = use_poly_features
        self.max_poly_features = max_poly_features
        self.remove_outliers = remove_outliers
        self.add_fingerprint_features = add_fingerprint_features
        self.subsample_samples = subsample_samples

    def fit(self, X, y):
        # assert init() is called
        if not g_tabpfn_config.is_initialized:
            raise RuntimeError("tabpfn_client.init() must be called before using TabPFNClassifier")

        if g_tabpfn_config.use_server:
            try:
                assert self.model == "latest_tabpfn_hosted", "Only 'latest_tabpfn_hosted' model is supported at the moment for tabpfn_classifier.init(use_server=True)"
            except AssertionError as e:
                print(e)
            g_tabpfn_config.inference_handler.fit(X, y)
            self.fitted_ = True
        else:
            raise NotImplementedError("Only server mode is supported at the moment for tabpfn_classifier.init(use_server=False)")
        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def predict_proba(self, X):
        check_is_fitted(self)
        return g_tabpfn_config.inference_handler.predict(X, config=self.get_params())


