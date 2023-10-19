import logging

from tabpfn_client.client import ServiceClient
from tabpfn_client.constants import CACHE_DIR

logger = logging.getLogger(__name__)


class ServiceClientWrapper:
    def __init__(self, service_client: ServiceClient):
        self.service_client = service_client


class UserAuthenticationClient(ServiceClientWrapper):
    """
    Singleton class for handling user authentication, including:
    - user registration and login
    - access token caching

    """
    CACHED_TOKEN_FILE = CACHE_DIR / "config"

    def is_accessible_connection(self) -> bool:
        return self.service_client.try_connection()

    def set_token(self, access_token: str):
        self.service_client.set_access_token(access_token)
        self.CACHED_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.CACHED_TOKEN_FILE.write_text(access_token)

    def set_token_by_registration(
            self,
            email: str,
            password: str,
            password_confirm: str
    ) -> None:
        if password != password_confirm:
            raise ValueError("Password and password_confirm must be the same.")

        is_created, message = self.service_client.register(email, password, password_confirm)
        if not is_created:
            raise RuntimeError(f"Failed to register user: {message}")

        # login after registration
        self.set_token_by_login(email, password)

    def set_token_by_login(self, email: str, password: str) -> None:
        access_token = self.service_client.login(email, password)

        if access_token is None:
            raise RuntimeError("Failed to login, please check your email and password.")

        self.set_token(access_token)

    def try_reuse_existing_token(self) -> bool:
        if self.service_client.access_token is None:
            if not self.CACHED_TOKEN_FILE.exists():
                return False

            access_token = self.CACHED_TOKEN_FILE.read_text()

        else:
            access_token = self.service_client.access_token

        is_valid = self.service_client.try_authenticate(access_token)
        if not is_valid:
            self._reset_token()
            return False

        logger.debug(f"Reusing existing access token? {is_valid}")
        self.set_token(access_token)

        return True

    def get_password_policy(self):
        return self.service_client.get_password_policy()

    def reset_cache(self):
        self._reset_token()

    def _reset_token(self):
        self.service_client.reset_access_token()
        self.CACHED_TOKEN_FILE.unlink()


class UserDataClient(ServiceClientWrapper):
    """
    Singleton class for handling user data, including:
    - query, or delete user account data
    - query, download, or delete uploaded data
    """
    pass


class InferenceClient(ServiceClientWrapper):
    """
    Singleton class for handling inference, including:
    - fitting
    - prediction
    """

    def __init__(self, service_client = ServiceClient()):
        super().__init__(service_client)
        self.last_train_set_uid = None

    def fit(self, X, y) -> None:
        if not self.service_client.is_initialized:
            raise RuntimeError("Service client is not initialized.")

        self.last_train_set_uid = self.service_client.upload_train_set(X, y)

    def predict(self, X):
        return self.service_client.predict(
            train_set_uid=self.last_train_set_uid,
            x_test=X
        )

    def predict_proba(self, X):
        return self.service_client.predict_proba(
            train_set_uid=self.last_train_set_uid,
            x_test=X
        )


