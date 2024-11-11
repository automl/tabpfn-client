from __future__ import annotations

import traceback
import re
from pathlib import Path
import httpx
import logging
from importlib.metadata import version, PackageNotFoundError
import numpy as np
from omegaconf import OmegaConf
import json
from typing import Literal, Optional
from cityhash import CityHash128
import os
from collections import OrderedDict

from tabpfn_client.tabpfn_common_utils import utils as common_utils
from tabpfn_client.constants import CACHE_DIR


logger = logging.getLogger(__name__)

# avoid logging of httpx and httpcore on client side
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)


class SensitiveDataFilter(logging.Filter):
    def filter(self, record):
        if "password" in record.getMessage():
            original_query = str(record.args[1])
            filtered_query = re.sub(
                r"(password|password_confirm)=[^&]*", r"\1=[FILTERED]", original_query
            )
            record.args = (record.args[0], filtered_query, *record.args[2:])
        return True


class DatasetUIDCacheManager:
    """
    Manages a cache of the last 50 uploaded datasets, tracking dataset hashes and their UIDs.
    """

    def __init__(self):
        self.file_path = CACHE_DIR / "dataset_cache"
        self.cache = self.load_cache()
        self.cache_limit = 50

    def load_cache(self):
        """
        Loads the cache from disk if it exists, otherwise initializes an empty cache.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                data = json.load(file)
                return OrderedDict(data)
        return OrderedDict()

    def _compute_hash(self, *args):
        combined_bytes = b"".join(
            item if isinstance(item, bytes) else str.encode(item) for item in args
        )
        return str(CityHash128(combined_bytes))

    def get_dataset_uid(self, *args):
        """
        Generates hash by all received arguments and returns cached dataset uid if in cache, otherwise None.
        """
        dataset_hash = self._compute_hash(*args)
        if str(dataset_hash) in self.cache:
            self.cache.move_to_end(dataset_hash)
            return self.cache[dataset_hash], dataset_hash
        else:
            return None, dataset_hash

    def add_dataset_uid(self, hash: str, dataset_uid: str):
        """
        Adds a new dataset to the cache, removing the oldest item if the cache exceeds 50 entries.
        Assumes the dataset is not already in the cache.
        """
        self.cache[hash] = dataset_uid
        # Move to end for the case that hash already was stored in the cache
        self.cache.move_to_end(hash)
        if len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)
        self.save_cache()

    def save_cache(self):
        """
        Saves the current cache to disk.
        """
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w") as file:
            json.dump(self.cache, file)

    def delete_uid(self, dataset_uid: str) -> Optional[str]:
        """
        Deletes an entry from the cache based on the dataset UID.
        """
        hash_to_delete = None
        for hash, uid in self.cache.items():
            if uid == dataset_uid:
                hash_to_delete = hash
                break

        if hash_to_delete:
            del self.cache[hash_to_delete]
            self.save_cache()
            return hash_to_delete
        return None


# Apply the custom filter to the httpx logger
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
httpx_logger.addFilter(SensitiveDataFilter())

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

    def __init__(self):
        self.server_config = SERVER_CONFIG
        self.server_endpoints = SERVER_CONFIG["endpoints"]
        self.base_url = f"{self.server_config.protocol}://{self.server_config.host}:{self.server_config.port}"
        self.httpx_timeout_s = (
            4 * 5 * 60 + 15  # temporary workaround for slow computation on server side
        )
        self.httpx_client = httpx.Client(
            base_url=self.base_url,
            timeout=self.httpx_timeout_s,
            headers={"client-version": get_client_version()},
        )

        self._access_token = None
        self.dataset_uid_cache_manager = DatasetUIDCacheManager()

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
        # Save until prediction for retrying train set upload for the case that anything went wrong with cache.
        X_serialized = common_utils.serialize_to_csv_formatted_bytes(X)
        y_serialized = common_utils.serialize_to_csv_formatted_bytes(y)

        # Get hash for dataset. Include access token for the case that one user uses different accounts.
        cached_dataset_uid, dataset_hash = (
            self.dataset_uid_cache_manager.get_dataset_uid(
                X_serialized, y_serialized, self._access_token
            )
        )
        if cached_dataset_uid:
            return cached_dataset_uid

        response = self.httpx_client.post(
            url=self.server_endpoints.upload_train_set.path,
            files=common_utils.to_httpx_post_file_format(
                [
                    ("x_file", "x_train_filename", X_serialized),
                    ("y_file", "y_train_filename", y_serialized),
                ]
            ),
        )

        self._validate_response(response, "upload_train_set")

        train_set_uid = response.json()["train_set_uid"]
        self.dataset_uid_cache_manager.add_dataset_uid(dataset_hash, train_set_uid)
        return train_set_uid

    def predict(
        self,
        train_set_uid: str,
        x_test,
        task: Literal["classification", "regression"],
        tabpfn_config: dict | None = None,
        X_train=None,
        y_train=None,
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

        x_test_serialized = common_utils.serialize_to_csv_formatted_bytes(x_test)

        # In the arguments for hashing, include train_set_uid for the case that the same test set was previously used
        # with different train set. Include access token for the case that a user uses different accounts.
        cached_test_set_uid, dataset_hash = (
            self.dataset_uid_cache_manager.get_dataset_uid(
                x_test_serialized, train_set_uid, self._access_token
            )
        )

        params = {"train_set_uid": train_set_uid, "task": task}
        if tabpfn_config is not None:
            params["tabpfn_config"] = json.dumps(
                tabpfn_config, default=lambda x: x.to_dict()
            )

        # Send prediction request. Loop two times, such that if anything cached is not correct
        # anymore, there is a second iteration where the datasets are uploaded.
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                response = self._make_prediction_request(
                    cached_test_set_uid, x_test_serialized, params
                )
                self._validate_response(response, "predict")
                break  # Successful response, exit the retry loop
            except RuntimeError as e:
                error_message = str(e)
                if (
                    "Invalid train or test set uid" in error_message
                    and attempt < max_attempts - 1
                ):
                    # Retry by re-uploading the train set
                    self.dataset_uid_cache_manager.delete_uid(train_set_uid)
                    if X_train is None or y_train is None:
                        raise RuntimeError(
                            "Train set data is required to re-upload but was not provided."
                        )
                    train_set_uid = self.upload_train_set(X_train, y_train)
                    params["train_set_uid"] = train_set_uid
                    cached_test_set_uid = None
                else:
                    raise  # Re-raise the exception if it's not recoverable

        else:
            raise RuntimeError(
                f"Failed to get prediction after {max_attempts} attempts."
            )

        # The response from the predict API always returns a dictionary with the task as the key.
        # This is just s.t. we do not confuse the tasks, as they both use the same API endpoint.
        # That is why below we use the task as the key to access the response.
        response = response.json()
        result = response[task]
        test_set_uid = response["test_set_uid"]
        if cached_test_set_uid is None:
            self.dataset_uid_cache_manager.add_dataset_uid(dataset_hash, test_set_uid)

        # The results contain different things for the different tasks
        # - classification: probas_array
        # - regression: {"mean": mean_array, "median": median_array, "mode": mode_array, ...}
        # So, if the result is not a dictionary, we add a "probas" key to it.
        if not isinstance(result, dict):
            result = {"probas": result}

        for k in result:
            result[k] = np.array(result[k])

        return result

    def _make_prediction_request(self, test_set_uid, x_test_serialized, params):
        """
        Helper function to make the prediction request to the server.
        """
        if test_set_uid:
            params["test_set_uid"] = test_set_uid
            response = self.httpx_client.post(
                url=self.server_endpoints.predict.path, params=params
            )
        else:
            response = self.httpx_client.post(
                url=self.server_endpoints.predict.path,
                params=params,
                files=common_utils.to_httpx_post_file_format(
                    [("x_file", "x_test_filename", x_test_serialized)]
                ),
            )
        return response

    @staticmethod
    def _validate_response(
        response: httpx.Response, method_name, only_version_check=False
    ):
        # If status code is 200, no errors occurred on the server side.
        if response.status_code == 200:
            return

        # Read response.
        load = None
        try:
            # This if clause is necessary for streaming responses (e.g. download) to
            # prevent httpx.ResponseNotRead error.
            if not response.is_closed:
                response.read()
            load = response.json()
        except json.JSONDecodeError as e:
            logging.info(f"Failed to parse JSON from response in {method_name}: {e}")

        # Check if the server requires a newer client version.
        if response.status_code == 426:
            logger.info(
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
            try:
                if (
                    len(
                        reponse_split_up := response.text.split(
                            "The following exception has occurred:"
                        )
                    )
                    > 1
                ):
                    relevant_reponse_text = reponse_split_up[1].split(
                        "debug_error_string"
                    )[0]
                    if "ValueError" in relevant_reponse_text:
                        # Extract the ValueError message
                        value_error_msg = relevant_reponse_text.split(
                            "ValueError. Arguments: ("
                        )[1].split(",)")[0]
                        # Remove extra quotes and spaces
                        value_error_msg = value_error_msg.strip("'")
                        # Raise the ValueError with the extracted message
                        raise ValueError(value_error_msg)
                    raise RuntimeError(relevant_reponse_text)
            except Exception as e:
                if isinstance(e, (ValueError, RuntimeError)):
                    raise e
            raise RuntimeError(
                f"Fail to call {method_name} with error: {response.status_code}, reason: "
                f"{response.reason_phrase} and text: {response.text}"
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

    def is_auth_token_outdated(self, access_token) -> bool | None:
        """
        Check if the provided access token is valid and return True if successful.
        """
        is_authenticated = False
        response = self.httpx_client.get(
            self.server_endpoints.protected_root.path,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        self._validate_response(
            response, "is_auth_token_outdated", only_version_check=True
        )
        if response.status_code == 200:
            is_authenticated = True
        elif response.status_code == 403:
            is_authenticated = None
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
    ):
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
            json={
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

        access_token = response.json()["token"] if is_created else None
        return is_created, message, access_token

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

    def send_verification_email(self, access_token: str) -> tuple[bool, str]:
        """
        Let the server send an email for verifying the email.
        """
        response = self.httpx_client.post(
            self.server_endpoints.send_verification_email.path,
            headers={"Authorization": f"Bearer {access_token}"},
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
            "GET",
            full_url,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "client-version": get_client_version(),
            },
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
