import unittest
from unittest.mock import patch
import tempfile

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier as TabPFNClassifierLocal

from tabpfn_client.tabpfn_classifier import TabPFNClassifier
from tabpfn_client import tabpfn_classifier
from tabpfn_client.tests.mock_tabpfn_server import with_mock_server


class TestTabPFNClassifier(unittest.TestCase):
    def setUp(self):
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    def test_use_local_tabpfn_classifier(self):
        tabpfn_classifier.init(use_server=False)
        tabpfn = TabPFNClassifier(device="cpu")
        tabpfn.fit(self.X_train, self.y_train)

        self.assertTrue(isinstance(tabpfn.classifier_, TabPFNClassifierLocal))
        pred = tabpfn.predict(self.X_test)
        self.assertEqual(pred.shape[0], self.X_test.shape[0])

    @with_mock_server()
    @patch("tabpfn_client.tabpfn_classifier.prompt_for_token", side_effect=["dummy_token"])
    def test_use_remote_tabpfn_classifier(self, mock_server, mock_prompt_for_token):
        with tempfile.TemporaryDirectory() as temp_cache_dir:
            # mock connection and authentication
            mock_server.router.get(mock_server.endpoints.root.path).respond(200)
            mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
            tabpfn_classifier.init(use_server=True, cache_dir=temp_cache_dir)

            tabpfn = TabPFNClassifier()

            # mock fitting
            mock_server.router.post(mock_server.endpoints.upload_train_set.path).respond(
                200, json={"per_user_train_set_id": 5})
            tabpfn.fit(self.X_train, self.y_train)

            # mock prediction
            mock_server.router.post(mock_server.endpoints.predict.path).respond(
                200,
                json={"y_pred": TabPFNClassifierLocal().fit(self.X_train, self.y_train).predict(self.X_test).tolist()}
            )
            pred = tabpfn.predict(self.X_test)
            self.assertEqual(pred.shape[0], self.X_test.shape[0])
