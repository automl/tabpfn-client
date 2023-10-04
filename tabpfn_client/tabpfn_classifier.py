import logging
from typing import Union
from pathlib import Path
import getpass
import textwrap

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from tabpfn import TabPFNClassifier as TabPFNClassifierLocal
from tabpfn_client import tabpfn_service_client
from tabpfn_client.tabpfn_service_client import TabPFNServiceClient

ACCESS_TOKEN_FILENAME = "config"
DEFAULT_CACHE_FOLDER = ".tabpfn"

logger = logging.getLogger(__name__)


class TabPFNConfig:
    is_initialized = None
    use_server = None


g_tabpfn_config = TabPFNConfig()


def init(
        use_server=True,
        cache_dir: Union[Path, str] = Path.cwd() / DEFAULT_CACHE_FOLDER,
):
    global g_tabpfn_config

    if use_server:
        # check connection to server
        if not TabPFNServiceClient.try_connection():
            raise RuntimeError("TabPFN is unaccessible at the moment, please try again later.")

        token_file = Path(cache_dir) / ACCESS_TOKEN_FILENAME
        token = None

        # check previously saved token file (if exists)
        if Path.exists(token_file):
            print(f"Using previously saved access token from {str(token_file)}")
            token = Path(token_file).read_text()
            if not TabPFNServiceClient.try_authenticate(token):
                # invalidate token and delete token file
                logger.debug("Previously saved access token is invalid, deleting token file")
                token = None
                Path.unlink(token_file, missing_ok=True)

        if token is None:
            # prompt for terms and conditions
            if not prompt_for_terms_and_cond():
                raise RuntimeError("You must agree to the terms and conditions to use TabPFN")

            # prompt for token
            token = prompt_for_token()
            if not TabPFNServiceClient.try_authenticate(token):
                raise RuntimeError("Invalid access token")
            print(f"API key is saved to {str(token_file)} for future use.")
            token_file.parent.mkdir(parents=True, exist_ok=True)
            token_file.write_text(token)

        assert token is not None

        g_tabpfn_config.use_server = True
        tabpfn_service_client.init(token)

    else:
        g_tabpfn_config.use_server = False

    g_tabpfn_config.is_initialized = True


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
                self.classifier_ = TabPFNServiceClient(**classifier_cfg)
            else:
                self.classifier_ = TabPFNClassifierLocal(**classifier_cfg)

        self.classifier_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self.classifier_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.classifier_.predict_proba(X)


def indent(text: str) -> str:
    indent_factor = 2
    indent_str = " " * indent_factor
    return textwrap.indent(text, indent_str)


def prompt_for_token():
    prompt = "\n".join([
        "",
        "Welcome to TabPFN!",
        "",
        "You are not logged in yet.",
        "",
        "Please choose one of the following options:",
        "(1) Create a TabPFN account",
        "(2) Login to your TabPFN account",
        "",
        "Please enter your choice: ",
    ])

    choice = input(indent(prompt))

    if choice == "1":
        # create account
        email = input(indent("Please enter your email: "))

        password_req = TabPFNServiceClient.get_password_policy()["requirements"]
        password_req_prompt = "\n".join([
            "",
            "Password requirements (minimum):",
            "\n".join([f". {req}" for req in password_req]),
            "",
            "Please enter your password: ",
        ])

        password = getpass.getpass(indent(password_req_prompt))
        password_confirm = getpass.getpass(indent("Please confirm your password: "))

        if password != password_confirm:
            raise RuntimeError("Fail to register account, mismatched password")

        success, message = TabPFNServiceClient.register(email, password, password_confirm)
        if not success:
            raise RuntimeError(f"Fail to register account, {message}")

    elif choice == "2":
        # login to account
        email = input(indent("Please enter your email: "))
        password = getpass.getpass(indent("Please enter your password: "))

    else:
        raise RuntimeError("Invalid choice")

    token = TabPFNServiceClient.login(email, password)
    if token is None:
        raise RuntimeError(f"Fail to login with the given email and password")

    return token


def prompt_for_terms_and_cond():
    t_and_c = "\n".join([
        "",
        "By using TabPFN, you agree to the following terms and conditions:",
        "",
        "...",
        "",
        "Do you agree to the above terms and conditions? (y/n): ",
    ])

    choice = input(indent(t_and_c))

    # retry for 3 attempts until valid choice is made
    is_valid_choice = False
    for _ in range(3):
        if choice.lower() not in ["y", "n"]:
            choice = input(indent("Invalid choice, please enter 'y' or 'n': "))
        else:
            is_valid_choice = True
            break

    if not is_valid_choice:
        raise RuntimeError("Invalid choice")

    return choice.lower() == "y"


