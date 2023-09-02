import httpx
import logging
from pathlib import Path
from omegaconf import OmegaConf

from tabpfn_client.tabpfn_common_utils import utils as common_utils

g_access_token = None

SERVER_SPEC_FILE = Path(__file__).parent.resolve() / "server_spec.yaml"
SERVER_SPEC = OmegaConf.load(SERVER_SPEC_FILE)


def init(access_token: str):
    if access_token is None or access_token == "":
        raise RuntimeError("access_token must be provided")
    TabPFNServiceClient.access_token = access_token


class TabPFNServiceClient:

    SERVER_ENDPOINTS = SERVER_SPEC["endpoints"]

    httpx_client = httpx.Client(base_url=f"http://{SERVER_SPEC['host']}:{SERVER_SPEC['port']}")     # TODO: use https
    access_token = None

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
        if self.access_token is None or self.access_token == "":
            raise RuntimeError("tabpfn_service_client.init() must be called before instantiating TabPFNServiceClient")

        self.last_per_user_train_set_id = None

        # TODO: (in the coming version)
        #  will be used as the reference to the per-client TabPFN on the server
        self.tabpfn_id = None

        # TODO:
        #  These configs are ignored at the moment -> all clients share the same (default) on-server TabPFNClassifier.
        #  In the future version, these configs will be used to create per-user TabPFNClassifier,
        #    allowing the user to setup the desired TabPFNClassifier on the server.
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
        # TODO: (in the coming version)
        #  create a per-client TabPFN on the server (referred by self.tabpfn_id) if it doesn't exist yet

        X = common_utils.serialize_to_csv_formatted_bytes(X)
        y = common_utils.serialize_to_csv_formatted_bytes(y)

        response = self.httpx_client.post(
            url=self.SERVER_ENDPOINTS["upload_train_set"]["path"],
            headers={"Authorization": f"Bearer {self.access_token}"},
            files=common_utils.to_httpx_post_file_format([
                ("x_file", X),
                ("y_file", y)
            ])
        )

        if response.status_code != 200:
            logging.error(f"Fail to call upload_train_set(), response status: {response.status_code}")
            # TODO: error probably doesn't have json() method, check in unit test
            logging.error(f"Fail to call fit(), server response: {response.json()}")
            raise RuntimeError(f"Fail to call fit(), server response: {response.json()}")

        self.last_per_user_train_set_id = response.json()["per_user_train_set_id"]

        return self

    def predict(self, X):
        # check if user has already called fit() before
        if self.last_per_user_train_set_id is None:
            raise RuntimeError("You must call fit() before calling predict()")

        X = common_utils.serialize_to_csv_formatted_bytes(X)

        response = self.httpx_client.post(
            url=self.SERVER_ENDPOINTS["predict"]["path"],
            headers={"Authorization": f"Bearer {self.access_token}"},
            params={"per_user_train_set_id": self.last_per_user_train_set_id},
            files=common_utils.to_httpx_post_file_format([
                ("x_file", X)
            ])
        )

        if response.status_code != 200:
            logging.error(f"Fail to call predict(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call predict(), server response: {response.json()}")

        return response.json()

    def predict_proba(self, X):
        raise NotImplementedError

    @classmethod
    def try_connection(cls) -> bool:
        found_valid_connection = False
        try:
            response = cls.httpx_client.get(cls.SERVER_ENDPOINTS["root"]["path"])
            if response.status_code == 200:
                found_valid_connection = True

        except httpx.ConnectError:
            found_valid_connection = False

        return found_valid_connection

    @classmethod
    def try_authenticate(cls, access_token) -> bool:
        is_authenticated = False
        try:
            response = cls.httpx_client.get(
                cls.SERVER_ENDPOINTS["protected_root"]["path"],
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if response.status_code == 200:
                is_authenticated = True

        except httpx.ConnectError:
            is_authenticated = False

        return is_authenticated
