import os
import httpx
from typing import Any
import logging

from tabpfn_client.tabpfn_classifier_interface import AbstractTabPFNClassifier
from tabpfn_client.tabpfn_common_utils import utils as common_utils

SERVER_ENDPOINTS_YAML = os.path.join(os.path.dirname(__file__), "server_endpoints.yaml")


class TabPFNServiceClient(AbstractTabPFNClassifier):
    def __init__(self, server_spec: dict, access_token: str):
        self.host = server_spec["host"]
        self.port = server_spec["port"]
        self.client = httpx.Client(
            base_url=f"http://{self.host}:{self.port}",     # TODO: change to https
        )
        self.access_token = access_token
        self.server_endpoints = server_spec["endpoints"]

        self.last_per_user_train_set_id = None

    def remove_models_from_memory(self):
        raise NotImplementedError

    def fit(self, X: Any, y: Any):
        X = common_utils.serialize_to_csv_formatted_bytes(X)
        y = common_utils.serialize_to_csv_formatted_bytes(y)

        response = self.client.post(
            url=self.server_endpoints["upload_train_set"]["path"],
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

    def predict(self, X, return_winning_class=False, normalize_with_test=False):

        # TODO: handle return_winning_class and normalize_with_test

        # check if user has already called fit() before
        if self.last_per_user_train_set_id is None:
            raise RuntimeError("You must call fit() before calling predict()")

        X = common_utils.serialize_to_csv_formatted_bytes(X)

        response = self.client.post(
            url=self.server_endpoints["predict"]["path"],
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

    def predict_proba(self, X, return_winning_probability=False, normalize_with_test=False):
        pass

    def try_root(self):
        response = self.client.get(
            self.server_endpoints["protected_root"]["path"],
            headers={"Authorization": f"Bearer {self.access_token}"},
        )
        print("response:", response.json())
        return response
