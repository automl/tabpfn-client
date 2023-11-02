import unittest
from unittest.mock import MagicMock, patch
import shutil

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from tabpfn_client.remote_tabpfn_classifier import RemoteTabPFNClassifier
from tabpfn_client.client import ServiceClient
from tabpfn_client.service_wrapper import InferenceClient
from tabpfn_client.constants import CACHE_DIR


class TestRemoteTabPFNClassifier(unittest.TestCase):

    def setUp(self):
        self.dummy_token = "dummy_token"
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.33)

        # mock service client
        self.mock_client = MagicMock(spec=ServiceClient)
        self.mock_client.is_initialized.return_value = True
        inference_handler = InferenceClient(service_client=self.mock_client)

        self.remote_tabpfn = RemoteTabPFNClassifier(
            device="cpu",
            base_path=".",
            model_string="",
            batch_size_inference=4,
            N_ensemble_configurations=4,
            feature_shift_decoder=False,
            seed=None,
            multiclass_decoder="permutation",
            inference_handler=inference_handler
        )

    def tearDown(self):
        patch.stopall()
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    def test_fit_and_predict_with_valid_datasets(self):
        # mock responses
        self.mock_client.upload_train_set.return_value = "dummy_train_set_uid"

        mock_predict_response = [1, 1, 0]
        self.mock_client.predict.return_value = mock_predict_response

        self.remote_tabpfn.fit(self.X_train, self.y_train)
        y_pred = self.remote_tabpfn.predict(self.X_test)

        self.assertEqual(mock_predict_response, y_pred)
        self.mock_client.upload_train_set.called_once_with(self.X_train, self.y_train)
        self.mock_client.predict.called_once_with(self.X_test)

    def test_call_predict_without_calling_fit_before(self):
        self.assertRaises(
            NotFittedError,
            self.remote_tabpfn.predict,
            self.X_test
        )

    def test_call_predict_proba_without_calling_fit_before(self):
        self.assertRaises(
            NotFittedError,
            self.remote_tabpfn.predict_proba,
            self.X_test
        )

    def test_predict_with_conflicting_test_set(self):
        # TODO: implement this
        pass
