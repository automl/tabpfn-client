import unittest
from unittest.mock import patch
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_client import tabpfn_classifier
from tabpfn_client.tabpfn_classifier import TabPFNClassifier
from tabpfn_client.tabpfn_service_client import TabPFNServiceClient
from tabpfn import TabPFNClassifier as TabPFNClassifierLocal
from tabpfn_client.tests.mock_tabpfn_server import with_mock_server


class TestTabPFNClassifierInit(unittest.TestCase):

    dummy_token = "dummy_token"

    def setUp(self):
        # set up dummy data
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33)

    def tearDown(self):
        # remove token file if exists
        Path.unlink(tabpfn_classifier.TOKEN_FILE, missing_ok=True)

    def test_init_local_classifier(self):
        tabpfn_classifier.init(use_server=False)
        tabpfn = TabPFNClassifier().fit(self.X_train, self.y_train)
        self.assertTrue(isinstance(tabpfn.classifier_, TabPFNClassifierLocal))

    @with_mock_server()
    @patch("tabpfn_client.tabpfn_classifier.prompt_for_token", side_effect=[dummy_token])
    @patch("tabpfn_client.tabpfn_classifier.prompt_for_terms_and_cond", side_effect=[True])
    def test_init_remote_classifier(self, mock_server, mock_prompt_for_token, mock_prompt_for_terms_and_cond):
        # mock connection, authentication, and fitting
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        mock_server.router.post(mock_server.endpoints.upload_train_set.path).respond(
            200, json={"per_user_train_set_id": 5}
        )

        tabpfn_classifier.init(use_server=True)
        tabpfn = TabPFNClassifier().fit(self.X_train, self.y_train)
        self.assertTrue(isinstance(tabpfn.classifier_, TabPFNServiceClient))

        # check if access token is saved
        token_file = tabpfn_classifier.TOKEN_FILE
        self.assertTrue(token_file.exists())
        self.assertEqual(token_file.read_text(), self.dummy_token)

    @with_mock_server()
    @patch("tabpfn_client.tabpfn_classifier.prompt_for_token", side_effect=[dummy_token])
    @patch("tabpfn_client.tabpfn_classifier.prompt_for_terms_and_cond", side_effect=[True])
    def test_init_remote_classifier_with_invalid_token(
            self, mock_server, mock_prompt_for_token, mock_prompt_for_terms_and_cond
    ):
        # mock connection and invalid authentication
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(401)

        self.assertRaises(RuntimeError, tabpfn_classifier.init, use_server=True)

        # check if access token is not saved
        token_file = tabpfn_classifier.TOKEN_FILE
        self.assertFalse(token_file.exists())

    @with_mock_server()
    def test_reuse_saved_access_token(self, mock_server):
        # mock connection and authentication
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)

        # create dummy token file
        token_file = tabpfn_classifier.TOKEN_FILE
        token_file.write_text(self.dummy_token)

        # init is called without error
        tabpfn_classifier.init(use_server=True)

    @with_mock_server()
    @patch("tabpfn_client.tabpfn_classifier.prompt_for_token", side_effect=[RuntimeError("Invalid token")])
    @patch("tabpfn_client.tabpfn_classifier.prompt_for_terms_and_cond", side_effect=[True])
    def test_invalid_saved_access_token(self, mock_server, mock_prompt_for_token, mock_prompt_for_terms_and_cond):
        # mock connection and invalid authentication
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(401)

        # create dummy token file
        token_file = tabpfn_classifier.TOKEN_FILE
        token_file.write_text("invalid_token")

        self.assertRaises(RuntimeError, tabpfn_classifier.init, use_server=True)
        self.assertTrue(mock_prompt_for_token.called)

    def test_reset_on_local_classifier(self):
        tabpfn_classifier.init(use_server=False)
        tabpfn_classifier.reset()
        self.assertFalse(tabpfn_classifier.g_tabpfn_config.is_initialized)

    @with_mock_server()
    @patch("tabpfn_client.tabpfn_classifier.prompt_for_token", side_effect=[dummy_token])
    @patch("tabpfn_client.tabpfn_classifier.prompt_for_terms_and_cond", side_effect=[True])
    def test_reset_on_remote_classifier(self, mock_server, mock_prompt_for_token, mock_prompt_for_terms_and_cond):
        # init classifier as usual
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        tabpfn_classifier.init(use_server=True)

        # check if access token is saved
        token_file = tabpfn_classifier.TOKEN_FILE
        self.assertTrue(token_file.exists())

        # reset
        tabpfn_classifier.reset()

        # check if access token is deleted
        self.assertFalse(token_file.exists())

        # check if config is reset
        self.assertFalse(tabpfn_classifier.g_tabpfn_config.is_initialized)

    @with_mock_server()
    @patch("tabpfn_client.tabpfn_classifier.prompt_for_terms_and_cond", side_effect=[False])
    def test_decline_terms_and_cond(self, mock_server, mock_prompt_for_terms_and_cond):
        # mock connection
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)

        self.assertRaises(RuntimeError, tabpfn_classifier.init, use_server=True)
        self.assertTrue(mock_prompt_for_terms_and_cond.called)
