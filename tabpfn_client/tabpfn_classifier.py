import logging
from pathlib import Path
import textwrap

from sklearn.base import BaseEstimator, ClassifierMixin

from tabpfn import TabPFNClassifier as TabPFNClassifierLocal
from tabpfn_client import tabpfn_service_client
from tabpfn_client.tabpfn_service_client import TabPFNServiceClient

ACCESS_TOKEN_FILENAME = "access_token.txt"

logger = logging.getLogger(__name__)


class TabPFNConfig:
    is_initialized = None
    use_server = None


g_tabpfn_config = TabPFNConfig()


def init(
        use_server=True,
        cache_dir: Path = Path.cwd()
):
    global g_tabpfn_config

    if use_server:
        # check connection to server
        if not TabPFNServiceClient.try_connection():
            raise RuntimeError("TabPFN is unaccessible at the moment, please try again later.")

        token_file = cache_dir / ACCESS_TOKEN_FILENAME
        token = None

        # check previously saved token file (if exists)
        if Path.exists(token_file):
            token = Path(token_file).read_text()
            if not TabPFNServiceClient.try_authenticate(token):
                # invalidate token and delete token file
                logger.debug("Previously saved access token is invalid, deleting token file")
                token = None
                Path.unlink(token_file, missing_ok=True)

        if token is None:
            # prompt for token
            token = prompt_for_token()
            if not TabPFNServiceClient.try_authenticate(token):
                raise RuntimeError("Invalid access token")
            print(f"API key is saved to {str(token_file)} for future use.")
            Path(token_file).write_text(token)

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
            base_path=".",
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
        # assert init() is called
        if not g_tabpfn_config.is_initialized:
            raise RuntimeError("TabPFNClassifier.init() must be called before using TabPFNClassifier")

        # config for tabpfn
        # self.model = model
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

        self.classifier = None

    def fit(self, X, y):
        if self.classifier is None:
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
                self.classifier = TabPFNServiceClient(**classifier_cfg)
            else:
                self.classifier = TabPFNClassifierLocal(**classifier_cfg)

        return self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)


REGISTER_LINK = "http://0.0.0.0/docs#/default/register_auth_register__post"     # TODO: add link
LOGIN_LINK = "http://0.0.0.0/docs#/default/login_auth_login__post"              # TODO: add link


def prompt_for_token():

    indent = 2

    def ask_input_with_indent(prompt: str) -> str:
        indent_str = " " * indent
        return input(textwrap.indent(prompt, indent_str))

    def print_with_indent(text: str):
        indent_str = " " * indent
        print(textwrap.indent(text, indent_str))

    prompt = "\n".join([
        "",
        "Welcome to TabPFN!",
        "",
        "Sadly you are not logged in yet.",
        "Please choose one of the following options:",
        "(1) Create a TabPFN account",
        "(2) Login to your TabPFN account",
        ""
    ])

    print_with_indent(prompt)
    choice = ask_input_with_indent("Please enter your choice: ")

    if choice == "1":
        print_with_indent(f"\nYou could create an account at {REGISTER_LINK}")
        token = ask_input_with_indent("After you are done, paste your API key here and hit enter: ")

    elif choice == "2":
        print_with_indent(f"Retrieve your API key here: {LOGIN_LINK}")
        token = ask_input_with_indent("Please enter your API key and hit enter: ")

    else:
        raise RuntimeError("Invalid choice")

    return token