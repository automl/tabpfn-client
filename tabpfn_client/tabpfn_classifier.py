import logging
from pathlib import Path
import shutil

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from tabpfn import TabPFNClassifier as LocalTabPFNClassifier
from tabpfn_client.remote_tabpfn_classifier import RemoteTabPFNClassifier
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


class TabPFNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            model=None,
            device="cpu",
            base_path=Path(__file__).parent.parent.resolve(),
            model_string="",
            batch_size_inference=4,
            fp16_inference=False,
            inference_mode=True,
            c=None,
            N_ensemble_configurations=10,
            preprocess_transforms=("none", "power_all"),
            feature_shift_decoder=False,
            normalize_with_test=False,
            average_logits=False,
            categorical_features=tuple(),
            optimize_metric=None,
            seed=None,
            transformer_predict_kwargs_init=None,
            multiclass_decoder="permutation",
    ):
        # config for tabpfn
        self.model = model
        self.device = device
        self.base_path = base_path
        self.model_string = model_string
        self.batch_size_inference = batch_size_inference
        self.fp16_inference = fp16_inference
        self.inference_mode = inference_mode
        self.c = c
        self.N_ensemble_configurations = N_ensemble_configurations
        self.preprocess_transforms = preprocess_transforms
        self.feature_shift_decoder = feature_shift_decoder
        self.normalize_with_test = normalize_with_test
        self.average_logits = average_logits
        self.categorical_features = categorical_features
        self.optimize_metric = optimize_metric
        self.seed = seed
        self.transformer_predict_kwargs_init = transformer_predict_kwargs_init
        self.multiclass_decoder = multiclass_decoder

    def fit(self, X, y):
        # assert init() is called
        if not g_tabpfn_config.is_initialized:
            raise RuntimeError("TabPFNClassifier.init() must be called before using TabPFNClassifier")

        # create classifier if not created yet
        if not hasattr(self, "classifier"):
            # arguments that are commented out are not used at the moment
            # (not supported until new TabPFN interface is released)
            classifier_cfg = {
                # "model": self.model,
                "device": self.device,
                "base_path": self.base_path,
                "model_string": self.model_string,
                "batch_size_inference": self.batch_size_inference,
                # "fp16_inference": self.fp16_inference,
                # "inference_mode": self.inference_mode,
                # "c": self.c,
                "N_ensemble_configurations": self.N_ensemble_configurations,
                # "preprocess_transforms": self.preprocess_transforms,
                "feature_shift_decoder": self.feature_shift_decoder,
                # "normalize_with_test": self.normalize_with_test,
                # "average_logits": self.average_logits,
                # "categorical_features": self.categorical_features,
                # "optimize_metric": self.optimize_metric,
                "seed": self.seed,
                # "transformer_predict_kwargs_init": self.transformer_predict_kwargs_init,
                "multiclass_decoder": self.multiclass_decoder
            }

            if g_tabpfn_config.use_server:
                self.classifier_ = RemoteTabPFNClassifier(
                    **classifier_cfg,
                    inference_handler=g_tabpfn_config.inference_handler
                )
            else:
                self.classifier_ = LocalTabPFNClassifier(**classifier_cfg)

        self.classifier_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.classifier_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.classifier_.predict_proba(X)


