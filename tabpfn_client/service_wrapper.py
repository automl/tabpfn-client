import logging
from pathlib import Path
from typing import Literal

from tabpfn_client.client import ServiceClient
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client.prompt_agent import PromptAgent

logger = logging.getLogger(__name__)


class ServiceClientWrapper:
    def __init__(self, service_client: ServiceClient):
        self.service_client = service_client


class UserAuthenticationClient(ServiceClientWrapper):
    """
    Wrapper of ServiceClient to handle user authentication, including:
    - user registration and login
    - access token caching

    """

    CACHED_TOKEN_FILE = CACHE_DIR / "config"

    def is_accessible_connection(self) -> bool:
        return self.service_client.try_connection()

    def set_token(self, access_token: str):
        self.service_client.authorize(access_token)
        self.CACHED_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.CACHED_TOKEN_FILE.write_text(access_token)

    def validate_email(self, email: str) -> tuple[bool, str]:
        is_valid, message = self.service_client.validate_email(email)
        return is_valid, message

    def set_token_by_registration(
        self,
        email: str,
        password: str,
        password_confirm: str,
        validation_link: str,
        additional_info: dict,
    ) -> tuple[bool, str]:
        is_created, message, access_token = self.service_client.register(
            email, password, password_confirm, validation_link, additional_info
        )
        if access_token is not None:
            self.set_token(access_token)
        return is_created, message

    def set_token_by_login(self, email: str, password: str) -> tuple[bool, str]:
        access_token, message = self.service_client.login(email, password)

        if access_token is None:
            return False, message

        self.set_token(access_token)
        return True, message

    def try_reuse_existing_token(self) -> bool | tuple[bool, str]:
        if self.service_client.access_token is None:
            if not self.CACHED_TOKEN_FILE.exists():
                return False

            access_token = self.CACHED_TOKEN_FILE.read_text()

        else:
            access_token = self.service_client.access_token

        is_valid = self.service_client.is_auth_token_outdated(access_token)
        if is_valid is False:
            self._reset_token()
            return False
        elif is_valid is None:
            return False, access_token

        logger.debug(f"Reusing existing access token? {is_valid}")
        self.set_token(access_token)

        return True

    def get_password_policy(self):
        return self.service_client.get_password_policy()

    def reset_cache(self):
        self._reset_token()

    def _reset_token(self):
        self.service_client.reset_authorization()
        self.CACHED_TOKEN_FILE.unlink(missing_ok=True)

    def retrieve_greeting_messages(self):
        return self.service_client.retrieve_greeting_messages()

    def send_reset_password_email(self, email: str) -> tuple[bool, str]:
        sent, message = self.service_client.send_reset_password_email(email)
        return sent, message

    def send_verification_email(self, access_token: str) -> tuple[bool, str]:
        sent, message = self.service_client.send_verification_email(access_token)
        return sent, message


class UserDataClient(ServiceClientWrapper):
    """
    Wrapper of ServiceClient to handle user data, including:
    - query, or delete user account data
    - query, download, or delete uploaded data
    """

    def __init__(self, service_client=ServiceClient()):
        super().__init__(service_client)

    def get_data_summary(self) -> {}:
        try:
            summary = self.service_client.get_data_summary()
        except RuntimeError as e:
            logging.error(f"Failed to get data summary: {e}")
            raise e

        return summary

    def download_all_data(self, save_dir: Path = Path(".")) -> Path:
        try:
            saved_path = self.service_client.download_all_data(save_dir)
        except RuntimeError as e:
            logging.error(f"Failed to download data: {e}")
            raise e

        if saved_path is None:
            raise RuntimeError("Failed to download data.")

        logging.info(f"Data saved to {saved_path}")
        return saved_path

    def delete_dataset(self, dataset_uid: str) -> [str]:
        try:
            deleted_datasets = self.service_client.delete_dataset(dataset_uid)
        except RuntimeError as e:
            logging.error(f"Failed to delete dataset: {e}")
            raise e

        logging.info(f"Deleted datasets: {deleted_datasets}")

        return deleted_datasets

    def delete_all_datasets(self) -> [str]:
        try:
            deleted_datasets = self.service_client.delete_all_datasets()
        except RuntimeError as e:
            logging.error(f"Failed to delete all datasets: {e}")
            raise e

        logging.info(f"Deleted datasets: {deleted_datasets}")

        return deleted_datasets

    def delete_user_account(self):
        confirm_pass = PromptAgent.prompt_confirm_password_for_user_account_deletion()
        try:
            self.service_client.delete_user_account(confirm_pass)
        except RuntimeError as e:
            logging.error(f"Failed to delete user account: {e}")
            raise e

        PromptAgent.prompt_account_deleted()


class InferenceClient(ServiceClientWrapper):
    """
    Wrapper of ServiceClient to handle inference, including:
    - fitting
    - prediction
    """

    def __init__(self, service_client=ServiceClient()):
        super().__init__(service_client)

    def fit(self, X, y) -> str:
        if not self.service_client.is_initialized:
            raise RuntimeError(
                "Dear TabPFN User, please initialize the client first by verifying your E-mail address sent to your registered E-mail account."
                "Please Note: The email verification token expires in 30 minutes."
            )

        return self.service_client.upload_train_set(X, y)

    def predict(
        self,
        X,
        task: Literal["classification", "regression"],
        train_set_uid: str,
        config=None,
        X_train=None,
        y_train=None,
    ):
        return self.service_client.predict(
            train_set_uid=train_set_uid,
            x_test=X,
            tabpfn_config=config,
            task=task,
            X_train=X_train,
            y_train=y_train,
        )
