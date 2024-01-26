import unittest
from unittest.mock import patch
import shutil

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier as LocalTabPFNClassifier

from tabpfn_client import tabpfn_classifier
from tabpfn_client.tabpfn_classifier import TabPFNClassifier
from tabpfn_client.remote_tabpfn_classifier import RemoteTabPFNClassifier
from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.client import ServiceClient
from tabpfn_client.tests.mock_tabpfn_server import with_mock_server
from tabpfn_client.constants import CACHE_DIR


class TestTabPFNClassifierInit(unittest.TestCase):

    dummy_token = "dummy_token"

    def setUp(self):
        # set up dummy data
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.33)

    def tearDown(self):
        tabpfn_classifier.reset()

        # remove singleton instance of ServiceClient
        ServiceClient().delete_instance()

        # remove cache dir
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    def test_init_local_classifier(self):
        tabpfn_classifier.init(use_server=False)
<<<<<<< HEAD
        tabpfn = TabPFNClassifier(model="tabpfn_1_local").fit(self.X_train, self.y_train)
=======
        tabpfn = TabPFNClassifier(model="public_tabpfn_hosted").fit(self.X_train, self.y_train)
>>>>>>> ae59bf5 (Fix: Test Cases in Client and Add Model to TabPFN Classifier)
        self.assertTrue(isinstance(tabpfn.classifier_, LocalTabPFNClassifier))

    @with_mock_server()
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_and_set_token")
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_terms_and_cond",
           return_value=True)
    def test_init_remote_classifier(self, mock_server, mock_prompt_for_terms_and_cond, mock_prompt_and_set_token):
        mock_prompt_and_set_token.side_effect = \
            lambda user_auth_handler: user_auth_handler.set_token(self.dummy_token)

        # mock server connection
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.post(mock_server.endpoints.upload_train_set.path).respond(
            200, json={"train_set_uid": 5}
        )

        tabpfn_classifier.init(use_server=True)
        tabpfn = TabPFNClassifier().fit(self.X_train, self.y_train)
        self.assertTrue(isinstance(tabpfn.classifier_, RemoteTabPFNClassifier))
        self.assertTrue(mock_prompt_and_set_token.called)
        self.assertTrue(mock_prompt_for_terms_and_cond.called)

    @with_mock_server()
    def test_reuse_saved_access_token(self, mock_server):
        # mock connection and authentication
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)

        # create dummy token file
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(self.dummy_token)

        # init is called without error
        tabpfn_classifier.init(use_server=True)

        # check if access token still exists
        self.assertTrue(UserAuthenticationClient.CACHED_TOKEN_FILE.exists())

    @with_mock_server()
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_and_set_token")
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_terms_and_cond",
           return_value=True)
    def test_invalid_saved_access_token(self, mock_server, mock_prompt_for_terms_and_cond, mock_prompt_and_set_token):
        mock_prompt_and_set_token.side_effect = [RuntimeError]

        # mock connection and invalid authentication
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(401)

        # create dummy token file
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("invalid_token")

        self.assertRaises(RuntimeError, tabpfn_classifier.init, use_server=True)
        self.assertTrue(mock_prompt_and_set_token.called)

    def test_reset_on_local_classifier(self):
        tabpfn_classifier.init(use_server=False)
        tabpfn_classifier.reset()
        self.assertFalse(tabpfn_classifier.g_tabpfn_config.is_initialized)

    @with_mock_server()
    def test_reset_on_remote_classifier(self, mock_server):
        # create dummy token file
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(self.dummy_token)

        # init classifier as usual
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        tabpfn_classifier.init(use_server=True)

        # check if access token is saved
        self.assertTrue(UserAuthenticationClient.CACHED_TOKEN_FILE.exists())

        # reset
        tabpfn_classifier.reset()

        # check if access token is deleted
        self.assertFalse(UserAuthenticationClient.CACHED_TOKEN_FILE.exists())

        # check if config is reset
        self.assertFalse(tabpfn_classifier.g_tabpfn_config.is_initialized)

    @with_mock_server()
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_terms_and_cond",
           return_value=False)
    def test_decline_terms_and_cond(self, mock_server, mock_prompt_for_terms_and_cond):
        # mock connection
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)

        self.assertRaises(RuntimeError, tabpfn_classifier.init, use_server=True)
        self.assertTrue(mock_prompt_for_terms_and_cond.called)
