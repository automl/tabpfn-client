from pathlib import Path
import httpx
import logging
from importlib.metadata import version, PackageNotFoundError
import numpy as np
from omegaconf import OmegaConf

from tabpfn_client.tabpfn_common_utils import utils as common_utils


logger = logging.getLogger(__name__)

SERVER_CONFIG_FILE = Path(__file__).parent.resolve() / "server_config.yaml"
SERVER_CONFIG = OmegaConf.load(SERVER_CONFIG_FILE)


def get_client_version() -> str:
    try:
        return version('tabpfn_client')
    except PackageNotFoundError:
        # Package not found, should only happen during development. Execute 'pip install -e .' to use the actual
        # version number during development. Otherwise, simply return a version number that is large enough.
        return '5.5.5'


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
        self.httpx_timeout_s = 30   # temporary workaround for slow computation on server side
        self.httpx_client = httpx.Client(
            base_url=self.base_url,
            timeout=self.httpx_timeout_s,
            headers={"client-version": get_client_version()}
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

        self._validate_response(response, "upload_train_set")

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
            params={"train_set_uid": train_set_uid},
            files=common_utils.to_httpx_post_file_format([
                ("x_file", "x_test_filename", x_test)
            ])
        )

        self._validate_response(response, "predict")

        return np.array(response.json()["y_pred"])

    @staticmethod
    def _validate_response(response, method_name, only_version_check=False):
        # If status code is 200, no errors occurred on the server side.
        if response.status_code == 200:
            return

        # Read response.
        load = None
        try:
            load = response.json()
        except Exception:
            pass

        # Check if the server requires a newer client version.
        if response.status_code == 426:
            logger.error(f"Fail to call {method_name}, response status: {response.status_code}")
            raise RuntimeError(load.get("detail"))

        # If we not only want to check the version compatibility, also raise other errors.
        if not only_version_check:
            if load is not None:
                raise RuntimeError(f"Fail to call {method_name} with error: {load}")
            logger.error(f"Fail to call {method_name}, response status: {response.status_code}")
            if len(reponse_split_up:=response.text.split("The following exception has occurred:")) > 1:
                raise RuntimeError(f"Fail to call {method_name} with error: {reponse_split_up[1]}")
            raise RuntimeError(f"Fail to call {method_name} with error: {response.status_code} and reason: "
                               f"{response.reason_phrase}")


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
            params={"train_set_uid": train_set_uid},
            files=common_utils.to_httpx_post_file_format([
                ("x_file", "x_test_filename", x_test)
            ])
        )

        self._validate_response(response, "predict_proba")

        return np.array(response.json()["y_pred_proba"])

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

        self._validate_response(response, "try_authenticate", only_version_check=True)

        if response.status_code == 200:
            is_authenticated = True

        return is_authenticated

    def register(
            self,
            email: str,
            password: str,
            password_confirm: str,
            validation_link: str
    ) -> (bool, str):
        """
        Register a new user with the provided credentials.

        Parameters
        ----------
        email : str
        password : str
        password_confirm : str
        validation_link: str

        Returns
        -------
        is_created : bool
            True if the user is created successfully.
        message : str
            The message returned from the server.
        """

        response = self.httpx_client.post(
            self.server_endpoints.register.path,
            params={"email": email, "password": password, "password_confirm": password_confirm, "validation_link": validation_link}
        )

        self._validate_response(response, "register", only_version_check=True)
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

        self._validate_response(response, "login", only_version_check=False)
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
        self._validate_response(response, "get_password_policy", only_version_check=True)

        return response.json()["requirements"]

    def retrieve_greeting_messages(self) -> list[str]:
        """
        Retrieve greeting messages that are new for the user.
        """
        response = self.httpx_client.get(self.server_endpoints.retrieve_greeting_messages.path)

        self._validate_response(response, "retrieve_greeting_messages", only_version_check=True)
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
        with httpx.stream("GET", full_url, headers={"Authorization": f"Bearer {self.access_token}"}) as response:
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
            params={"dataset_uid": dataset_uid}
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
            params={"confirm_password": confirm_pass}
        )

        self._validate_response(response, "delete_user_account")
