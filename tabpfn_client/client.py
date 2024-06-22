from __future__ import annotations

import traceback
from pathlib import Path
import httpx
import logging
from importlib.metadata import version, PackageNotFoundError
import numpy as np
from enum import Enum
from omegaconf import OmegaConf
import json
from typing import Literal

from tabpfn_client.tabpfn_common_utils import utils as common_utils


logger = logging.getLogger(__name__)

SERVER_CONFIG_FILE = Path(__file__).parent.resolve() / "server_config.yaml"
SERVER_CONFIG = OmegaConf.load(SERVER_CONFIG_FILE)


def get_client_version() -> str:
    try:
        return version("tabpfn_client")
    except PackageNotFoundError:
        # Package not found, should only happen during development. Execute 'pip install -e .' to use the actual
        # version number during development. Otherwise, simply return a version number that is large enough.
        return "5.5.5"


@common_utils.singleton
class ServiceClient:
    """
    Singleton class for handling communication with the server.
    It encapsulates all the API calls to the server.
    """
    class Status(Enum):
        OKAY = 0
        USER_NOT_VERIFIED = 1

    def __init__(self):
        self.server_config = SERVER_CONFIG
        self.server_endpoints = SERVER_CONFIG["endpoints"]
        self.base_url = f"{self.server_config.protocol}://{self.server_config.host}:{self.server_config.port}"
        self.httpx_timeout_s = (
            30  # temporary workaround for slow computation on server side
        )
        self.httpx_client = httpx.Client(
            base_url=self.base_url,
            timeout=self.httpx_timeout_s,
            headers={"client-version": get_client_version()},
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
        return self.access_token is not None and self.access_token != ""

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
            files=common_utils.to_httpx_post_file_format(
                [("x_file", "x_train_filename", X), ("y_file", "y_train_filename", y)]
            ),
        )

        self._validate_response(response, "upload_train_set")

        train_set_uid = response.json()["train_set_uid"]
        return train_set_uid

    def predict(
        self,
        train_set_uid: str,
        x_test,
        task: Literal["classification", "regression"],
        tabpfn_config: dict | None = None,
    ) -> dict[str, np.ndarray]:
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

        params = {"train_set_uid": train_set_uid, "task": task}

        if tabpfn_config is not None:
            params["tabpfn_config"] = json.dumps(
                tabpfn_config, default=lambda x: x.to_dict()
            )

        response = self.httpx_client.post(
            url=self.server_endpoints.predict.path,
            params=params,
            files=common_utils.to_httpx_post_file_format(
                [("x_file", "x_test_filename", x_test)]
            ),
        )

        self._validate_response(response, "predict")

        # The response from the predict API always returns a dictionary with the task as the key.
        # This is just s.t. we do not confuse the tasks, as they both use the same API endpoint.
        # That is why below we use the task as the key to access the response.
        result = response.json()[task]

        # The results contain different things for the different tasks
        # - classification: probas_array
        # - regression: {"mean": mean_array, "median": median_array, "mode": mode_array, ...}
        # So, if the result is not a dictionary, we add a "probas" key to it.
        if not isinstance(result, dict):
            result = {"probas": result}

        for k in result:
            result[k] = np.array(result[k])

        return result

    @staticmethod
    def _validate_response(response, method_name, only_version_check=False):
        # If status code is 200, no errors occurred on the server side.
        if response.status_code == 200:
            return

        # Read response.
        load = None
        try:
            load = response.json()
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from response in {method_name}: {e}")

        # Check if the server requires a newer client version.
        if response.status_code == 426:
            logger.error(
                f"Fail to call {method_name}, response status: {response.status_code}"
            )
            raise RuntimeError(load.get("detail"))

        # If we not only want to check the version compatibility, also raise other errors.
        if not only_version_check:
            if load is not None:
                raise RuntimeError(f"Fail to call {method_name} with error: {load}")
            logger.error(
                f"Fail to call {method_name}, response status: {response.status_code}"
            )
            if (
                len(
                    reponse_split_up := response.text.split(
                        "The following exception has occurred:"
                    )
                )
                > 1
            ):
                raise RuntimeError(
                    f"Fail to call {method_name} with error: {reponse_split_up[1]}"
                )
            raise RuntimeError(
                f"Fail to call {method_name} with error: {response.status_code} and reason: "
                f"{response.reason_phrase}"
            )

    def try_connection(self) -> bool:
        """
        Check if server is reachable and accepts the connection.
        """
        found_valid_connection = False
        try:
            response = self.httpx_client.get(self.server_endpoints.root.path)
            self._validate_response(response, "try_connection", only_version_check=True)
            if response.status_code == 200:
                found_valid_connection = True

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to the server with error: {e}")
            traceback.print_exc()
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

        self._validate_response(response, "try_authenticate", only_version_check=True)
        if response.status_code == 200:
            is_authenticated = True
        elif response.status_code == 403:
            is_authenticated = (False, self.Status.USER_NOT_VERIFIED)
        return is_authenticated

    def validate_email(self, email: str) -> tuple[bool, str]:
        """
        Send entered email to server that checks if it is valid and not already in use.

        Parameters
        ----------
        email : str

        Returns
        -------
        is_valid : bool
            True if the email is valid.
        message : str
            The message returned from the server.
        """
        response = self.httpx_client.post(
            self.server_endpoints.validate_email.path, params={"email": email}
        )

        self._validate_response(response, "validate_email", only_version_check=True)
        if response.status_code == 200:
            is_valid = True
            message = ""
        else:
            is_valid = False
            message = response.json()["detail"]

        return is_valid, message

    def register(
        self,
        email: str,
        password: str,
        password_confirm: str,
        validation_link: str,
        additional_info: dict,
    ) -> tuple[bool, str]:
        """
        Register a new user with the provided credentials.

        Parameters
        ----------
        email : str
        password : str
        password_confirm : str
        validation_link: str
        additional_info : dict

        Returns
        -------
        is_created : bool
            True if the user is created successfully.
        message : str
            The message returned from the server.
        """

        response = self.httpx_client.post(
            self.server_endpoints.register.path,
            params={
                "email": email,
                "password": password,
                "password_confirm": password_confirm,
                "validation_link": validation_link,
                **additional_info,
            },
        )

        self._validate_response(response, "register", only_version_check=True)
        if response.status_code == 200:
            is_created = True
            message = response.json()["message"]
        else:
            is_created = False
            message = response.json()["detail"]

        return is_created, message

    def login(self, email: str, password: str) -> tuple[str, str]:
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
        message : str
            The message returned from the server.
        """

        access_token = None
        response = self.httpx_client.post(
            self.server_endpoints.login.path,
            data=common_utils.to_oauth_request_form(email, password),
        )

        self._validate_response(response, "login", only_version_check=True)
        if response.status_code == 200:
            access_token = response.json()["access_token"]
            message = ""
        else:
            message = response.json()["detail"]

        return access_token, message

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
        self._validate_response(
            response, "get_password_policy", only_version_check=True
        )

        return response.json()["requirements"]

    def send_reset_password_email(self, email: str) -> tuple[bool, str]:
        """
        Let the server send an email for resetting the password.
        """
        response = self.httpx_client.post(
            self.server_endpoints.send_reset_password_email.path,
            params={"email": email},
        )
        if response.status_code == 200:
            sent = True
            message = response.json()["message"]
        else:
            sent = False
            message = response.json()["detail"]
        return sent, message

    def retrieve_greeting_messages(self) -> list[str]:
        """
        Retrieve greeting messages that are new for the user.
        """
        response = self.httpx_client.get(
            self.server_endpoints.retrieve_greeting_messages.path
        )

        self._validate_response(
            response, "retrieve_greeting_messages", only_version_check=True
        )
        if response.status_code != 200:
            return []

        greeting_messages = response.json()["messages"]
        return greeting_messages
    
    # bool optional parameter is accesstoken required
    def get_user_email_verification_status(self, email: str, access_token_required: bool) -> tuple[bool, str]:
        """
        Check if the user's email is verified.
        """
        response = self.httpx_client.post(
            self.server_endpoints.get_user_verification_status_via_email.path,
            params={"email": email, "access_token_required": access_token_required},
        )
        return response.json()

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
        self._validate_response(response, "get_data_summary")

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
        with httpx.stream(
            "GET", full_url, headers={"Authorization": f"Bearer {self.access_token}"}
        ) as response:
            self._validate_response(response, "download_all_data")

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
            params={"dataset_uid": dataset_uid},
        )

        self._validate_response(response, "delete_dataset")

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

        self._validate_response(response, "delete_all_datasets")

        return response.json()["deleted_dataset_uids"]

    def delete_user_account(self, confirm_pass: str) -> None:
        response = self.httpx_client.delete(
            self.server_endpoints.delete_user_account.path,
            params={"confirm_password": confirm_pass},
        )

        self._validate_response(response, "delete_user_account")
