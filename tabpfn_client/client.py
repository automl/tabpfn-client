from pathlib import Path
import httpx
import logging
import json

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
        self.base_url = f"{self.server_config.protocol}://{self.server_config.host}:{self.server_config.port}"
        self.httpx_timeout_s = 30  # temporary workaround for slow computation on server side
        self.httpx_client = httpx.Client(
            base_url=self.base_url,
            timeout=self.httpx_timeout_s
        )

        self._access_token = None

    @property
    def access_token(self):
        return self._access_token

    def authorize(self, access_token: str):
        self._access_token = access_token
        self.httpx_client.headers.update(
            {"Authorization": f"Bearer {self.access_token}"}
        )

    def reset_authorization(self):
        self._access_token = None
        self.httpx_client.headers.pop("Authorization", None)

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
            files=common_utils.to_httpx_post_file_format([
                ("x_file", "x_train_filename", X),
                ("y_file", "y_train_filename", y)
            ])
        )

        if response.status_code != 200:
            logger.error(f"Fail to call upload_train_set(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call upload_train_set()")

        train_set_uid = response.json()["train_set_uid"]
        return train_set_uid

    def predict(self, train_set_uid: str, x_test, with_proba: bool = True, tabpfn_config: dict = None):
        """
        Predict the class labels (and the class proba.) for the provided data (i.e. test set).

        Parameters
        ----------
        train_set_uid : str
            The unique ID of the train set in the server.
        x_test : array-like of shape (n_samples, n_features)
            The test input.
        with_proba : bool
            Whether to return the class probabilities.
        tabpfn_config : dict
            The configuration of TabPFN model.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """

        x_test = common_utils.serialize_to_csv_formatted_bytes(x_test)

        self.validate_tabpfn_config(tabpfn_config)

        response = self.httpx_client.post(
            url=self.server_endpoints.predict.path,
            params={
                "train_set_uid": train_set_uid,
                "with_proba": with_proba,
                "serialized_tabpfn_config": json.dumps(tabpfn_config),
            },
            files=common_utils.to_httpx_post_file_format([("x_file", "x_test_filename", x_test)])
        )

        if response.status_code != 200:
            logger.error(f"Fail to call predict(), response status: {response.status_code}, response: {response}")
            raise RuntimeError(f"Fail to call predict()")

        return np.array(response.json()["res"])

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
            raise RuntimeError(f"Fail to call get_password_policy()")

        return response.json()["requirements"]

    def get_data_summary(self) -> {}:
        """
        Get the data summary of the user from the server.

        Returns
        -------
        data_summary : {}
            The data summary returned from the server.
        """
        response = self.httpx_client.get(
            self.server_endpoints.get_data_summary.path,
        )
        if response.status_code != 200:
            logger.error(f"Fail to call get_data_summary(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call get_data_summary()")

        return response.json()

    def download_all_data(self, save_dir: Path) -> Path | None:
        """
        Download all data uploaded by the user from the server.

        Returns
        -------
        save_path : Path | None
            The path to the downloaded file. Return None if download fails.

        """

        save_path = None

        full_url = self.base_url + self.server_endpoints.download_all_data.path
        with httpx.stream("GET", full_url, headers={"Authorization": f"Bearer {self.access_token}"}) as response:
            if response.status_code != 200:
                logger.error(f"Fail to call download_all_data(), response status: {response.status_code}")
                raise RuntimeError(f"Fail to call download_all_data()")

            filename = response.headers["Content-Disposition"].split("filename=")[1]
            save_path = Path(save_dir) / filename
            with open(save_path, "wb") as f:
                for data in response.iter_bytes():
                    f.write(data)

        return save_path

    def delete_dataset(self, dataset_uid: str) -> [str]:
        """
        Delete the dataset with the provided UID from the server.
        Note that deleting a train set with lead to deleting all associated test sets.

        Parameters
        ----------
        dataset_uid : str
            The UID of the dataset to be deleted.

        Returns
        -------
        deleted_dataset_uids : [str]
            The list of deleted dataset UIDs.

        """
        response = self.httpx_client.delete(
            self.server_endpoints.delete_dataset.path,
            params={"dataset_uid": dataset_uid}
        )

        if response.status_code != 200:
            logger.error(f"Fail to call delete_dataset(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call delete_dataset()")

        return response.json()["deleted_dataset_uids"]

    def delete_all_datasets(self) -> [str]:
        """
        Delete all datasets uploaded by the user from the server.

        Returns
        -------
        deleted_dataset_uids : [str]
            The list of deleted dataset UIDs.
        """
        response = self.httpx_client.delete(
            self.server_endpoints.delete_all_datasets.path,
        )

        if response.status_code != 200:
            logger.error(f"Fail to call delete_all_datasets(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call delete_all_datasets()")

        return response.json()["deleted_dataset_uids"]

    def delete_user_account(self, confirm_pass: str) -> None:
        response = self.httpx_client.delete(
            self.server_endpoints.delete_user_account.path,
            params={"confirm_password": confirm_pass}
        )

        if response.status_code != 200:
            logger.error(f"Fail to call delete_user_account(), response status: {response.status_code}")
            raise RuntimeError(f"Fail to call delete_user_account()")

    @staticmethod
    def validate_tabpfn_config(tabpfn_config: dict | None):
        if tabpfn_config is not None and not isinstance(tabpfn_config, dict):
            raise ValueError("Invalid TabPFN config - must be a dict or None")
