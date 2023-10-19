from pathlib import Path
import httpx
import logging

import numpy as np
from omegaconf import OmegaConf

from tabpfn_client.tabpfn_common_utils import utils as common_utils


logger = logging.getLogger(__name__)

SERVER_CONFIG_FILE = Path(__file__).parent.resolve() / "server_config.yaml"
SERVER_CONFIG = OmegaConf.load(SERVER_CONFIG_FILE)


@common_utils.singleton
class ServiceClient:
    """
    Singleton class for handling communication with the server.
    It encapsulates all the API calls to the server.
    """

    def __init__(self):
        self.server_config = SERVER_CONFIG
        self.server_endpoints = SERVER_CONFIG["endpoints"]
        self.httpx_timeout_s = 30   # temporary workaround for slow computation on server side
        self.httpx_client = httpx.Client(
            base_url=f"https://{self.server_config.host}:{self.server_config.port}",
            timeout=self.httpx_timeout_s
        )

        self._access_token = None

    @property
    def access_token(self):
        return self._access_token

    def set_access_token(self, access_token: str):
        self._access_token = access_token

    def reset_access_token(self):
        self._access_token = None

    @property
    def is_initialized(self):
        return self.access_token is not None \
            and self.access_token != ""

    def upload_train_set(self, X, y) -> str:
        """
        Upload a train set to server and return the train set UID if successful.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.

        Returns
        -------
        train_set_uid : str
            The unique ID of the train set in the server.

        """
        X = common_utils.serialize_to_csv_formatted_bytes(X)
        y = common_utils.serialize_to_csv_formatted_bytes(y)

        response = self.httpx_client.post(
            url=self.server_endpoints.upload_train_set.path,
            headers={"Authorization": f"Bearer {self.access_token}"},
            files=common_utils.to_httpx_post_file_format([
                ("x_file", "x_train_filename", X),
                ("y_file", "y_train_filename", y)
            ])
        )

        if response.status_code != 200:
            logger.error(f"Fail to call upload_train_set(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call upload_train_set(), server response: {response.json()}")

        train_set_uid = response.json()["train_set_uid"]
        return train_set_uid

    def predict(self, train_set_uid: str, x_test):
        """
        Predict the class labels for the provided data (test set).

        Parameters
        ----------
        train_set_uid : str
            The unique ID of the train set in the server.
        x_test : array-like of shape (n_samples, n_features)
            The test input.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """

        x_test = common_utils.serialize_to_csv_formatted_bytes(x_test)

        response = self.httpx_client.post(
            url=self.server_endpoints.predict.path,
            headers={"Authorization": f"Bearer {self.access_token}"},
            params={"train_set_uid": train_set_uid},
            files=common_utils.to_httpx_post_file_format([
                ("x_file", "x_test_filename", x_test)
            ])
        )

        if response.status_code != 200:
            logger.error(f"Fail to call predict(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call predict(), server response: {response.json()}")

        return np.array(response.json()["y_pred"])

    def predict_proba(self, train_set_uid: str, x_test):
        """
        Predict the class probabilities for the provided data (test set).

        Parameters
        ----------
        train_set_uid : str
            The unique ID of the train set in the server.
        x_test : array-like of shape (n_samples, n_features)
            The test input.

        Returns
        -------

        """
        x_test = common_utils.serialize_to_csv_formatted_bytes(x_test)

        response = self.httpx_client.post(
            url=self.server_endpoints.predict_proba.path,
            headers={"Authorization": f"Bearer {self.access_token}"},
            params={"train_set_uid": train_set_uid},
            files=common_utils.to_httpx_post_file_format([
                ("x_file", "x_test_filename", x_test)
            ])
        )

        if response.status_code != 200:
            logger.error(f"Fail to call predict_proba(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call predict_proba(), server response: {response.json()}")

        return np.array(response.json()["y_pred_proba"])

    def try_connection(self) -> bool:
        """
        Check if server is reachable and return True if successful.
        """
        found_valid_connection = False
        try:
            response = self.httpx_client.get(self.server_endpoints.root.path)
            if response.status_code == 200:
                found_valid_connection = True

        except httpx.ConnectError:
            found_valid_connection = False

        return found_valid_connection

    def try_authenticate(self, access_token) -> bool:
        """
        Check if the provided access token is valid and return True if successful.
        """
        is_authenticated = False
        response = self.httpx_client.get(
            self.server_endpoints.protected_root.path,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        if response.status_code == 200:
            is_authenticated = True

        return is_authenticated

    def register(
            self,
            email: str,
            password: str,
            password_confirm: str
    ) -> (bool, str):
        """
        Register a new user with the provided credentials.

        Parameters
        ----------
        email : str
        password : str
        password_confirm : str

        Returns
        -------
        is_created : bool
            True if the user is created successfully.
        message : str
            The message returned from the server.
        """

        response = self.httpx_client.post(
            self.server_endpoints.register.path,
            params={"email": email, "password": password, "password_confirm": password_confirm}
        )

        if response.status_code == 200:
            is_created = True
            message = response.json()["message"]
        else:
            is_created = False
            message = response.json()["detail"]

        return is_created, message

    def login(self, email: str, password: str) -> str | None:
        """
        Login with the provided credentials and return the access token if successful.

        Parameters
        ----------
        email : str
        password : str

        Returns
        -------
        access_token : str | None
            The access token returned from the server. Return None if login fails.
        """

        access_token = None
        response = self.httpx_client.post(
            self.server_endpoints.login.path,
            data=common_utils.to_oauth_request_form(email, password)
        )

        if response.status_code == 200:
            access_token = response.json()["access_token"]

        return access_token

    def get_password_policy(self) -> {}:
        """
        Get the password policy from the server.

        Returns
        -------
        password_policy : {}
            The password policy returned from the server.
        """

        response = self.httpx_client.get(
            self.server_endpoints.password_policy.path,
        )
        if response.status_code != 200:
            logger.error(f"Fail to call get_password_policy(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call get_password_policy(), server response: {response.json()}")

        return response.json()["requirements"]
