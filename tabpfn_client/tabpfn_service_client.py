import os
import httpx

from tabpfn_client.tabpfn_classifier_interface import TabPFNClassifierInterface

SERVER_ENDPOINTS_YAML = os.path.join(os.path.dirname(__file__), "server_endpoints.yaml")


class TabPFNServiceClient(TabPFNClassifierInterface):
    def __init__(self, server_spec: dict, access_token: str):
        self.host = server_spec["host"]
        self.port = server_spec["port"]
        self.client = httpx.Client(
            base_url=f"http://{self.host}:{self.port}",     # TODO: change to https
        )
        self.access_token = access_token
        self.server_endpoints = server_spec["endpoints"]

    def remove_models_from_memory(self):
        raise NotImplementedError

    def load_result_minimal(self, path, i, e):
        raise NotImplementedError

    def fit(self, X, y):
        pass

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X, return_winning_probability=False, normalize_with_test=False):
        pass

    def try_root(self):
        response = self.client.get("/")
        print("response:", response.json())
        return response
