import httpx
import logging
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin

from tabpfn_client.tabpfn_common_utils import utils as common_utils

g_access_token = None

SERVER_CONFIG_FILE = Path(__file__).parent.resolve() / "server_config.yaml"
SERVER_CONFIG = OmegaConf.load(SERVER_CONFIG_FILE)


def init(access_token: str):
    if access_token is None or access_token == "":
        raise RuntimeError("access_token must be provided")
    TabPFNServiceClient.access_token = access_token


class TabPFNServiceClient(BaseEstimator, ClassifierMixin):

    server_config = SERVER_CONFIG
    server_endpoints = SERVER_CONFIG["endpoints"]

    httpx_client = httpx.Client(base_url=f"http://{server_config.host}:{server_config.port}")     # TODO:
    # use https
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
        self.last_per_user_train_set_id_ = None

        # TODO: (in the coming version)
        #  create a per-client TabPFN on the server (referred by self.tabpfn_id) if it doesn't exist yet
        self.tabpfn_id_ = None

        X = common_utils.serialize_to_csv_formatted_bytes(X)
        y = common_utils.serialize_to_csv_formatted_bytes(y)

        response = self.httpx_client.post(
            url=self.server_endpoints.upload_train_set.path,
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

        self.last_per_user_train_set_id_ = response.json()["per_user_train_set_id"]

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = common_utils.serialize_to_csv_formatted_bytes(X)

        response = self.httpx_client.post(
            url=self.server_endpoints.predict.path,
            headers={"Authorization": f"Bearer {self.access_token}"},
            params={"per_user_train_set_id": self.last_per_user_train_set_id_},
            files=common_utils.to_httpx_post_file_format([
                ("x_file", X)
            ])
        )

        if response.status_code != 200:
            logging.error(f"Fail to call predict(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call predict(), server response: {response.json()}")

        return np.array(response.json()["y_pred"])

    def predict_proba(self, X):
        check_is_fitted(self)

        X = common_utils.serialize_to_csv_formatted_bytes(X)

        response = self.httpx_client.post(
            url=self.server_endpoints.predict_proba.path,
            headers={"Authorization": f"Bearer {self.access_token}"},
            params={"per_user_train_set_id": self.last_per_user_train_set_id_},
            files=common_utils.to_httpx_post_file_format([
                ("x_file", X)
            ])
        )

        if response.status_code != 200:
            logging.error(f"Fail to call predict_proba(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call predict_proba(), server response: {response.json()}")

        return np.array(response.json()["y_pred_proba"])

    @classmethod
    def try_connection(cls) -> bool:
        found_valid_connection = False
        try:
            response = cls.httpx_client.get(cls.server_endpoints.root.path)
            if response.status_code == 200:
                found_valid_connection = True

        except httpx.ConnectError:
            found_valid_connection = False

        return found_valid_connection

    @classmethod
    def try_authenticate(cls, access_token) -> bool:
        is_authenticated = False
        response = cls.httpx_client.get(
            cls.server_endpoints.protected_root.path,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if response.status_code == 200:
            is_authenticated = True

        return is_authenticated

    @classmethod
    def register(cls, email, password) -> (bool, str):
        is_created = False
        response = cls.httpx_client.post(
            cls.server_endpoints.register.path,
            params={"email": email, "password": password}
        )
        if response.status_code == 200:
            is_created = True
            message = response.json()["message"]

        else:
            message = response.json()["detail"]

        return is_created, message

    @classmethod
    def login(cls, email, password) -> str:
        access_token = None
        response = cls.httpx_client.post(
            cls.server_endpoints.login.path,
            data=common_utils.to_oauth_request_form(email, password)
        )
        if response.status_code == 200:
            access_token = response.json()["access_token"]

        return access_token
