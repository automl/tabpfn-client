import unittest
from unittest.mock import patch
import shutil
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from tabpfn_client import init, reset
from tabpfn_client import estimator
from tabpfn_client.estimator import TabPFNRegressor
from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.client import ServiceClient
from tabpfn_client.tests.mock_tabpfn_server import with_mock_server
from tabpfn_client.constants import CACHE_DIR


class TestTabPFNRegressorInit(unittest.TestCase):
    dummy_token = "dummy_token"

    def setUp(self):
        # set up dummy data
        X, y = load_diabetes(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33
        )

    def tearDown(self):
        reset()

        # remove singleton instance of ServiceClient
        ServiceClient().delete_instance()

        # remove cache dir
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    @with_mock_server()
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_and_set_token")
    @patch(
        "tabpfn_client.prompt_agent.PromptAgent.prompt_terms_and_cond",
        return_value=True,
    )
    def test_init_remote_regressor(
        self, mock_server, mock_prompt_for_terms_and_cond, mock_prompt_and_set_token
    ):
        mock_prompt_and_set_token.side_effect = (
            lambda user_auth_handler: user_auth_handler.set_token(self.dummy_token)
        )

        # mock server connection
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.post(mock_server.endpoints.upload_train_set.path).respond(
            200, json={"train_set_uid": 5}
        )
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})
        mock_predict_response = [100, 200, 300]
        predict_route = mock_server.router.post(mock_server.endpoints.predict.path)
        predict_route.respond(200, json={"regression": {"mean": mock_predict_response}})

        init(use_server=True)

        tabpfn = TabPFNRegressor(n_estimators=10)
        self.assertRaises(NotFittedError, tabpfn.predict, self.X_test)
        tabpfn.fit(self.X_train, self.y_train)
        self.assertTrue(mock_prompt_and_set_token.called)
        self.assertTrue(mock_prompt_for_terms_and_cond.called)

        y_pred = tabpfn.predict(self.X_test)
        self.assertTrue(np.all(np.array(mock_predict_response) == y_pred))

        self.assertIn(
            "n_estimators%22%3A%2010",
            str(predict_route.calls.last.request.url),
            "check that n_estimators is passed to the server",
        )

    @with_mock_server()
    def test_reuse_saved_access_token(self, mock_server):
        # mock connection and authentication
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})

        # create dummy token file
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(self.dummy_token)

        # init is called without error
        init(use_server=True)

        # check if access token still exists
        self.assertTrue(UserAuthenticationClient.CACHED_TOKEN_FILE.exists())

    @with_mock_server()
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_and_set_token")
    @patch(
        "tabpfn_client.prompt_agent.PromptAgent.prompt_terms_and_cond",
        return_value=True,
    )
    def test_invalid_saved_access_token(
        self, mock_server, mock_prompt_for_terms_and_cond, mock_prompt_and_set_token
    ):
        mock_prompt_and_set_token.side_effect = [RuntimeError]

        # mock connection and invalid authentication
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(401)

        # create dummy token file
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("invalid_token")

        self.assertRaises(RuntimeError, init, use_server=True)
        self.assertTrue(mock_prompt_and_set_token.called)

    @with_mock_server()
    def test_reset_on_remote_regressor(self, mock_server):
        # create dummy token file
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(self.dummy_token)

        # init classifier as usual
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})
        init(use_server=True)

        # check if access token is saved
        self.assertTrue(UserAuthenticationClient.CACHED_TOKEN_FILE.exists())

        # reset
        reset()

        # check if access token is deleted
        self.assertFalse(UserAuthenticationClient.CACHED_TOKEN_FILE.exists())

        # check if config is reset
        self.assertFalse(estimator.config.g_tabpfn_config.is_initialized)

    @with_mock_server()
    @patch(
        "tabpfn_client.prompt_agent.PromptAgent.prompt_terms_and_cond",
        return_value=False,
    )
    def test_decline_terms_and_cond(self, mock_server, mock_prompt_for_terms_and_cond):
        # mock connection
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)

        self.assertRaises(RuntimeError, init, use_server=True)
        self.assertTrue(mock_prompt_for_terms_and_cond.called)
