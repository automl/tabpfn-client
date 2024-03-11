import unittest
from unittest.mock import Mock

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

from tabpfn_client.client import ServiceClient
from tabpfn_client.tests.mock_tabpfn_server import with_mock_server


class TestServiceClient(unittest.TestCase):
    def setUp(self):
        # setup data
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.33, random_state=42)

        self.client = ServiceClient()

    @with_mock_server()
    def test_try_connection(self, mock_server):
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        self.assertTrue(self.client.try_connection())

    @with_mock_server()
    def test_try_connection_with_invalid_server(self, mock_server):
        mock_server.router.get(mock_server.endpoints.root.path).respond(404)
        self.assertFalse(self.client.try_connection())

    @with_mock_server()
    def test_try_connection_with_outdated_client(self, mock_server):
        mock_server.router.get(mock_server.endpoints.root.path).respond(
            426, json={"detail": "Client version too old. ..."})
        with self.assertRaises(RuntimeError) as cm:
            self.client.try_connection()
        self.assertTrue(str(cm.exception).startswith("Client version too old."))

    @with_mock_server()
    def test_register_user(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(200, json={"message": "dummy_message"})
        self.assertTrue(self.client.register("dummy_email", "dummy_password", "dummy_password", "dummy_validation")[0])

    @with_mock_server()
    def test_register_user_with_invalid_email(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(401, json={"detail": "dummy_message"})
        self.assertFalse(self.client.register("dummy_email", "dummy_password", "dummy_password", "dummy_validation")[0])

    @with_mock_server()
    def test_register_user_with_invalid_validation_link(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(401, json={"detail": "dummy_message"})
        self.assertFalse(self.client.register("dummy_email", "dummy_password", "dummy_password", "dummy_validation")[0])

    @with_mock_server()
    def test_register_user_with_limit_reached(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(401, json={"detail": "dummy_message"})
        self.assertFalse(self.client.register("dummy_email", "dummy_password", "dummy_password", "dummy_validation")[0])

    @with_mock_server()
    def test_invalid_auth_token(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(401)
        self.assertFalse(self.client.try_authenticate("fake_token"))

    @with_mock_server()
    def test_valid_auth_token(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        self.assertTrue(self.client.try_authenticate("true_token"))

    @with_mock_server()
    def test_retrieve_greeting_messages(self, mock_server):
        mock_server.router.get(mock_server.endpoints.retrieve_greeting_messages.path).respond(
            200, json={"messages": ["message_1", "message_2"]})
        self.assertEqual(self.client.retrieve_greeting_messages(), ["message_1", "message_2"])

    @with_mock_server()
    def test_predict_with_valid_train_set_and_test_set(self, mock_server):
        dummy_json = {"train_set_uid": 5}
        mock_server.router.post(mock_server.endpoints.upload_train_set.path).respond(
            200, json=dummy_json)

        self.client.upload_train_set(self.X_train, self.y_train)

        dummy_result = {"y_pred": [1, 2, 3]}
        mock_server.router.post(mock_server.endpoints.predict.path).respond(
            200, json=dummy_result)

        pred = self.client.predict(
            train_set_uid=dummy_json["train_set_uid"],
            x_test=self.X_test
        )
        self.assertTrue(np.array_equal(pred, dummy_result["y_pred"]))

    def test_validate_response_no_error(self):
        response = Mock()
        response.status_code = 200
        r = self.client._validate_response(response, "test")
        self.assertIsNone(r)

    def test_validate_response(self):
        response = Mock()
        # Test for "Client version too old." error
        response.status_code = 426
        response.json.return_value = {"detail": "Client version too old."}
        with self.assertRaises(RuntimeError) as cm:
            self.client._validate_response(response, "test")
        self.assertEqual(str(cm.exception), "Client version too old.")

        # Test for "Some other error" which is translated to a generic failure message
        response.status_code = 400
        response.json.return_value = {"detail": "Some other error"}
        with self.assertRaises(RuntimeError) as cm:
            self.client._validate_response(response, "test")
        self.assertTrue(str(cm.exception).startswith("Fail to call test"))

    def test_validate_response_only_version_check(self):
        response = Mock()
        response.status_code = 426
        response.json.return_value = {"detail": "Client version too old."}
        with self.assertRaises(RuntimeError) as cm:
            self.client._validate_response(response, "test", only_version_check=True)
        self.assertEqual(str(cm.exception), "Client version too old.")

        # Errors that have nothing to do with client version should be skipped.
        response = Mock()
        response.status_code = 400
        response.json.return_value = {"detail": "Some other error"}
        r = self.client._validate_response(response, "test", only_version_check=True)
        self.assertIsNone(r)

