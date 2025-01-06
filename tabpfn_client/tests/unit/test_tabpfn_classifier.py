import unittest
from unittest.mock import patch
import shutil
import json

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

from tabpfn_client import init, reset
from tabpfn_client.estimator import TabPFNClassifier
from tabpfn_client.service_wrapper import UserAuthenticationClient, InferenceClient
from tabpfn_client.tests.mock_tabpfn_server import with_mock_server
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client import config


class TestTabPFNClassifierInit(unittest.TestCase):
    dummy_token = "dummy_token"

    def setUp(self):
        # set up dummy data
        reset()
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33
        )

    def tearDown(self):
        # remove cache dir
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    @patch("tabpfn_client.browser_auth.webbrowser.open", return_value=False)
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_and_set_token")
    @with_mock_server()
    def test_init_remote_classifier(
        self,
        mock_server,
        mock_prompt_and_set_token,
        mock_webbrowser_open,
    ):
        mock_prompt_and_set_token.side_effect = (
            lambda: UserAuthenticationClient.set_token(self.dummy_token)
        )

        # mock server connection
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.post(mock_server.endpoints.fit.path).respond(
            200, json={"train_set_uid": "5"}
        )

        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})

        mock_predict_response = [1, 0, 1]
        predict_route = mock_server.router.post(mock_server.endpoints.predict.path)
        predict_route.respond(
            200,
            content=f'data: {json.dumps({"event": "result", "data": {"classification": mock_predict_response, "test_set_uid": "6"}})}\n\n',
            headers={"Content-Type": "text/event-stream"},
        )

        init(use_server=True)
        self.assertTrue(mock_prompt_and_set_token.called)

        tabpfn = TabPFNClassifier(n_estimators=10)
        self.assertRaises(NotFittedError, tabpfn.predict, self.X_test)
        tabpfn.fit(self.X_train, self.y_train)
        self.assertTrue(mock_prompt_and_set_token.called)
        y_pred = tabpfn.predict(self.X_test)
        self.assertTrue(np.all(mock_predict_response == y_pred))

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
    def test_invalid_saved_access_token(self, mock_server, mock_prompt_and_set_token):
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
    def test_reset_on_remote_classifier(self, mock_server):
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
        self.assertFalse(config.Config.is_initialized)

    @patch(
        "tabpfn_client.prompt_agent.PromptAgent.prompt_terms_and_cond",
        return_value=False,
    )
    @patch("tabpfn_client.browser_auth.webbrowser.open", return_value=False)
    @patch("builtins.input", side_effect=["1"])
    @with_mock_server()  # TODO (leo): investigate why this needs to be the last decorator
    def test_decline_terms_and_cond(
        self,
        mock_server,
        mock_input,
        mock_webbrowser_open,
        mock_prompt_for_terms_and_cond,
    ):
        # mock connection
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)

        self.assertRaises(RuntimeError, init, use_server=True)
        self.assertTrue(mock_prompt_for_terms_and_cond.called)

    @with_mock_server()
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_and_set_token")
    @patch(
        "tabpfn_client.prompt_agent.PromptAgent.prompt_terms_and_cond",
        return_value=True,
    )
    def test_cache_based_on_paper_version(
        self, mock_server, mock_prompt_for_terms_and_cond, mock_prompt_and_set_token
    ):
        """
        Check that the dataset cached is different for different paper_version
        and similar for the same paper_version
        """
        # a bit out of place but we don't want to skip init for this test
        mock_prompt_and_set_token.side_effect = (
            lambda: UserAuthenticationClient.set_token(self.dummy_token)
        )

        # mock server connection
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        fit_route = mock_server.router.post(mock_server.endpoints.fit.path)
        fit_route.respond(200, json={"train_set_uid": "5"})

        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})

        mock_predict_response = [[1, 0.0], [0.9, 0.1], [0.01, 0.99]]
        predict_route = mock_server.router.post(mock_server.endpoints.predict.path)
        predict_route.respond(
            200,
            content=f'data: {json.dumps({"event": "result", "data": {"classification": mock_predict_response, "test_set_uid": "6"}})}\n\n',
            headers={"Content-Type": "text/event-stream"},
        )

        init(use_server=True)

        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        test_X = np.random.rand(5, 5)

        # Initialize with paper_version=True
        tabpfn_true = TabPFNClassifier(paper_version=True)

        tabpfn_true.fit(X, y)
        tabpfn_true.predict(test_X)

        # Call fit and predict again with the same paper_version
        tabpfn_true.fit(X, y)
        tabpfn_true.predict(test_X)

        # Ensure fit endpoint is not called again
        self.assertEqual(
            fit_route.call_count,
            1,
            "Fit endpoint should not be called again with the same paper_version",
        )

        # Initialize with paper_version=False
        tabpfn_false = TabPFNClassifier(paper_version=False)

        tabpfn_false.fit(X, y)
        tabpfn_false.predict(test_X)

        # check fit is called
        self.assertEqual(
            fit_route.call_count,
            2,
            "Fit endpoint should be called again with a different paper_version",
        )

        # Call fit and predict again with the same paper_version
        tabpfn_false.fit(X, y)
        tabpfn_false.predict(test_X)

        # Ensure fit endpoint is not called again
        self.assertEqual(
            fit_route.call_count,
            2,
            "Fit endpoint should not be called again with the same paper_version",
        )

        # TODO: fix this
        # # Check that different cache entries are created for training set
        # cache_manager = ServiceClient.dataset_uid_cache_manager
        # X_serialized = common_utils.serialize_to_csv_formatted_bytes(X)
        # y_serialized = common_utils.serialize_to_csv_formatted_bytes(y)
        # uid_true_train, hash_true_train = cache_manager.get_dataset_uid(
        #     X_serialized, y_serialized, self.dummy_token, "_".join([])
        # )
        # uid_false_train, hash_false_train = cache_manager.get_dataset_uid(
        #     X_serialized,
        #     y_serialized,
        #     self.dummy_token,
        #     "_".join(["preprocessing", "text"]),
        # )

        # self.assertNotEqual(
        #     hash_true_train,
        #     hash_false_train,
        #     "Cache hash should differ based on paper_version for training set",
        # )

        # # Check that different cache entries are created for test set
        # test_X_serialized = common_utils.serialize_to_csv_formatted_bytes(test_X)
        # uid_true_test, hash_true_test = cache_manager.get_dataset_uid(
        #     test_X_serialized, uid_true_train, self.dummy_token, "_".join([])
        # )
        # uid_false_test, hash_false_test = cache_manager.get_dataset_uid(
        #     test_X_serialized,
        #     uid_false_train,
        #     self.dummy_token,
        #     "_".join(["preprocessing", "text"]),
        # )

        # self.assertNotEqual(
        #     hash_true_test,
        #     hash_false_test,
        #     "Cache hash should differ based on paper_version for test set",
        # )

        # # Verify that the cache entries are used correctly
        # self.assertIsNotNone(
        #     uid_true_train, "Training set cache should be used for paper_version=True"
        # )
        # self.assertIsNotNone(
        #     uid_false_train, "Training set cache should be used for paper_version=False"
        # )
        # self.assertIsNotNone(
        #     uid_true_test, "Test set cache should be used for paper_version=True"
        # )
        # self.assertIsNotNone(
        #     uid_false_test, "Test set cache should be used for paper_version=False"
        # )


class TestTabPFNClassifierInference(unittest.TestCase):
    def setUp(self):
        # skip init
        config.Config.is_initialized = True

    def tearDown(self):
        # undo setUp
        config.reset()

    def test_data_size_check_on_train_with_inconsistent_number_of_samples_raise_error(
        self,
    ):
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 11)
        tabpfn = TabPFNClassifier()

        with self.assertRaises(ValueError):
            tabpfn.fit(X, y)

    def test_data_size_check_on_train_with_oversized_data_raise_error(self):
        X = np.random.randn(10001, 501)
        y = np.random.randint(0, 2, 10001)

        tabpfn = TabPFNClassifier()

        # test oversized columns
        with self.assertRaises(ValueError):
            tabpfn.fit(X[:10], y[:10])

        # test oversized rows
        with self.assertRaises(ValueError):
            tabpfn.fit(X[:, :10], y)

    def test_data_size_check_on_predict_with_oversized_data_raise_error(self):
        test_X = np.random.randn(10001, 5)
        tabpfn = TabPFNClassifier()

        # skip fitting
        tabpfn.fitted_ = True

        # test oversized rows
        with self.assertRaises(ValueError):
            tabpfn.predict(test_X)

    def test_data_check_on_predict_with_valid_data_pass(self):
        test_X = np.random.randn(10, 5)
        tabpfn = TabPFNClassifier()

        # skip fitting
        tabpfn.fitted_ = True
        tabpfn.classes_ = np.array([0, 1])

        # mock prediction
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = {"probas": np.random.rand(10, 2)}
            tabpfn.predict(test_X)

    def test_only_allowed_parameters_passed_to_config(self):
        """Test that only allowed parameters are passed to the config."""
        ALLOWED_PARAMS = {
            "n_estimators",
            # TODO: put it back
            # "categorical_features_indices",
            "softmax_temperature",
            "average_before_softmax",
            "ignore_pretraining_limits",
            "inference_precision",
            "random_state",
            "inference_config",
            "model_path",
            "balance_probabilities",
            "paper_version",
        }

        # Create classifier with various parameters
        classifier = TabPFNClassifier(
            n_estimators=5,
            softmax_temperature=0.8,
            paper_version=True,
            random_state=42,
            balance_probabilities=True,
        )

        # Skip fitting
        classifier.fitted_ = True
        classifier.last_train_set_uid = "dummy_uid"

        test_X = np.random.randn(10, 5)

        # Mock predict and capture config
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = np.random.rand(10, 2)
            classifier.predict(test_X)

            # Get the config that was passed to predict
            actual_config = mock_predict.call_args[1]["config"]

            # Check that only allowed parameters are present
            config_params = set(actual_config.keys())
            unexpected_params = config_params - ALLOWED_PARAMS
            missing_params = ALLOWED_PARAMS - config_params

            self.assertEqual(
                unexpected_params,
                set(),
                f"Found unexpected parameters in config: {unexpected_params}",
            )
            self.assertEqual(
                missing_params,
                set(),
                f"Missing required parameters in config: {missing_params}",
            )

    def test_predict_params_output_type(self):
        """Test that predict_params contains correct output_type."""
        classifier = TabPFNClassifier()
        classifier.fitted_ = True  # Skip fitting
        test_X = np.random.randn(10, 5)

        # Test predict() sets output_type to "preds"
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = np.random.rand(10)
            classifier.predict(test_X)

            predict_params = mock_predict.call_args[1]["predict_params"]
            self.assertEqual(predict_params, {"output_type": "preds"})

        # Test predict_proba() sets output_type to "probas"
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = np.random.rand(10, 2)
            classifier.predict_proba(test_X)

            predict_params = mock_predict.call_args[1]["predict_params"]
            self.assertEqual(predict_params, {"output_type": "probas"})


class TestTabPFNModelSelection(unittest.TestCase):
    def setUp(self):
        # skip init
        config.Config.is_initialized = True
        config.Config.use_server = True

    def tearDown(self):
        # undo setUp
        config.reset()

    def test_list_available_models_returns_expected_models(self):
        expected_models = [
            "default",
            "gn2p4bpt",
            "llderlii",
            "od3j1g5m",
            "vutqq28w",
            "znskzxi4",
        ]
        self.assertEqual(TabPFNClassifier.list_available_models(), expected_models)

    def test_validate_model_name_with_valid_model_passes(self):
        # Should not raise any exception
        TabPFNClassifier._validate_model_name("default")
        TabPFNClassifier._validate_model_name("gn2p4bpt")

    def test_validate_model_name_with_invalid_model_raises_error(self):
        with self.assertRaises(ValueError):
            TabPFNClassifier._validate_model_name("invalid_model")

    def test_model_name_to_path_returns_expected_path(self):
        # Test default model path
        expected_default_path = "tabpfn-v2-classifier.ckpt"
        self.assertEqual(
            TabPFNClassifier._model_name_to_path("classification", "default"),
            expected_default_path,
        )

        # Test specific model path
        expected_specific_path = "tabpfn-v2-classifier-gn2p4bpt.ckpt"
        self.assertEqual(
            TabPFNClassifier._model_name_to_path("classification", "gn2p4bpt"),
            expected_specific_path,
        )

    def test_model_name_to_path_with_invalid_model_raises_error(self):
        with self.assertRaises(ValueError):
            TabPFNClassifier._model_name_to_path("classification", "invalid_model")

    def test_predict_proba_uses_correct_model_path(self):
        # Setup
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)

        tabpfn = TabPFNClassifier(model_path="gn2p4bpt")

        # Mock the inference client
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = {"probas": np.random.rand(10, 2)}

            with patch.object(InferenceClient, "fit") as mock_fit:
                mock_fit.return_value = "dummy_uid"

                # Fit and predict
                tabpfn.fit(X, y)
                tabpfn.predict_proba(X)

                # Verify the model path was correctly passed to predict
                predict_kwargs = mock_predict.call_args[1]
                expected_model_path = "tabpfn-v2-classifier-gn2p4bpt.ckpt"

                self.assertEqual(
                    predict_kwargs["config"]["model_path"], expected_model_path
                )

    @patch.object(InferenceClient, "fit", return_value="dummy_uid")
    @patch.object(
        InferenceClient, "predict", return_value={"probas": np.random.rand(10, 2)}
    )
    def test_paper_version_behavior(self, mock_predict, mock_fit):
        # this just test that it doesn't break,
        # but the actual behavior is easier to test
        # on the server side
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        test_X = np.random.rand(5, 5)

        # Test with paper_version=True
        tabpfn_true = TabPFNClassifier(paper_version=True)
        tabpfn_true.fit(X, y)
        y_pred_true = tabpfn_true.predict(test_X)
        self.assertIsNotNone(y_pred_true)

        # Test with paper_version=False
        tabpfn_false = TabPFNClassifier(paper_version=False)
        tabpfn_false.fit(X, y)
        y_pred_false = tabpfn_false.predict(test_X)
        self.assertIsNotNone(y_pred_false)
