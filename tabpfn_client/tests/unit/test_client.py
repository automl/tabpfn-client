import os
import unittest
from unittest.mock import Mock, patch

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

from tabpfn_client.client import ServiceClient
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client.tests.mock_tabpfn_server import with_mock_server


class TestServiceClient(unittest.TestCase):
    def setUp(self):
        # setup data
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        self.client = ServiceClient()
        self.client.dataset_uid_cache_manager.file_path = (
            CACHE_DIR / "test_dataset_cache"
        )
        self.client.dataset_uid_cache_manager.cache = (
            self.client.dataset_uid_cache_manager.load_cache()
        )

    def tearDown(self):
        try:
            os.remove(CACHE_DIR / "test_dataset_cache")
        except OSError:
            pass

    @with_mock_server()
    def test_try_connection(self, mock_server):
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        self.assertTrue(self.client.try_connection())

    @with_mock_server()
    def test_try_connection_with_invalid_server(self, mock_server):
        mock_server.router.get(mock_server.endpoints.root.path).respond(404)
        self.assertFalse(self.client.try_connection())

    @with_mock_server()
    def test_try_connection_with_outdated_client_raises_runtime_error(
        self, mock_server
    ):
        mock_server.router.get(mock_server.endpoints.root.path).respond(
            426, json={"detail": "Client version too old. ..."}
        )
        with self.assertRaises(RuntimeError) as cm:
            self.client.try_connection()
        self.assertTrue(str(cm.exception).startswith("Client version too old."))

    @with_mock_server()
    def test_validate_email(self, mock_server):
        mock_server.router.post(mock_server.endpoints.validate_email.path).respond(
            200, json={"message": "dummy_message"}
        )
        self.assertTrue(self.client.validate_email("dummy_email")[0])

    @with_mock_server()
    def test_validate_email_invalid(self, mock_server):
        mock_server.router.post(mock_server.endpoints.validate_email.path).respond(
            401, json={"detail": "dummy_message"}
        )
        self.assertFalse(self.client.validate_email("dummy_email")[0])
        self.assertEqual("dummy_message", self.client.validate_email("dummy_email")[1])

    @with_mock_server()
    def test_register_user(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(
            200, json={"message": "dummy_message", "token": "DUMMY_TOKEN"}
        )
        self.assertTrue(
            self.client.register(
                "dummy_email",
                "dummy_password",
                "dummy_password",
                "dummy_validation",
                {
                    "company": "dummy_company",
                    "use_case": "dummy_usecase",
                    "role": "dummy_role",
                    "contact_via_email": False,
                },
            )[0]
        )

    @with_mock_server()
    def test_register_user_with_invalid_email(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(
            401, json={"detail": "dummy_message", "token": None}
        )
        self.assertFalse(
            self.client.register(
                "dummy_email",
                "dummy_password",
                "dummy_password",
                "dummy_validation",
                {
                    "company": "dummy_company",
                    "use_case": "dummy_usecase",
                    "role": "dummy_role",
                    "contact_via_email": False,
                },
            )[0]
        )

    @with_mock_server()
    def test_register_user_with_invalid_validation_link(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(
            401, json={"detail": "dummy_message", "token": None}
        )
        self.assertFalse(
            self.client.register(
                "dummy_email",
                "dummy_password",
                "dummy_password",
                "dummy_validation",
                {
                    "company": "dummy_company",
                    "use_case": "dummy_usecase",
                    "role": "dummy_role",
                    "contact_via_email": False,
                },
            )[0]
        )

    @with_mock_server()
    def test_register_user_with_limit_reached(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(
            401, json={"detail": "dummy_message", "token": "DUMMY_TOKEN"}
        )
        self.assertFalse(
            self.client.register(
                "dummy_email",
                "dummy_password",
                "dummy_password",
                "dummy_validation",
                {
                    "company": "dummy_company",
                    "use_case": "dummy_usecase",
                    "role": "dummy_role",
                    "contact_via_email": False,
                },
            )[0]
        )

    @with_mock_server()
    def test_invalid_auth_token(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(401)
        self.assertFalse(self.client.is_auth_token_outdated("fake_token"))

    @with_mock_server()
    def test_valid_auth_token(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        self.assertTrue(self.client.is_auth_token_outdated("true_token"))

    @with_mock_server()
    def test_send_reset_password_email(self, mock_server):
        mock_server.router.post(
            mock_server.endpoints.send_reset_password_email.path
        ).respond(200, json={"message": "Password reset email sent!"})
        self.assertEqual(
            self.client.send_reset_password_email("test"),
            (True, "Password reset email sent!"),
        )

    @with_mock_server()
    def test_send_verification_email(self, mock_server):
        mock_server.router.post(
            mock_server.endpoints.send_verification_email.path
        ).respond(200, json={"message": "Verification Email sent!"})
        self.assertEqual(
            self.client.send_verification_email("test"),
            (True, "Verification Email sent!"),
        )

    @with_mock_server()
    def test_retrieve_greeting_messages(self, mock_server):
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": ["message_1", "message_2"]})
        self.assertEqual(
            self.client.retrieve_greeting_messages(), ["message_1", "message_2"]
        )

    @with_mock_server()
    def test_predict_with_valid_train_set_and_test_set(self, mock_server):
        dummy_json = {"train_set_uid": "5"}
        mock_server.router.post(mock_server.endpoints.upload_train_set.path).respond(
            200, json=dummy_json
        )
        self.client.authorize("dummy_token")
        self.client.upload_train_set(self.X_train, self.y_train)

        dummy_result = {"test_set_uid": "dummy_uid", "classification": [1, 2, 3]}
        mock_server.router.post(mock_server.endpoints.predict.path).respond(
            200, json=dummy_result
        )

        pred = self.client.predict(
            train_set_uid=dummy_json["train_set_uid"],
            x_test=self.X_test,
            task="classification",
        )
        self.assertTrue(np.array_equal(pred["probas"], dummy_result["classification"]))

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

    @with_mock_server()
    def test_upload_train_set_with_caching(self, mock_server):
        """
        Test that uploading the same training set multiple times uses the cache and
        only calls the upload_train_set endpoint once.
        """
        self.client.authorize("dummy_access_token")

        # Mock the upload_train_set endpoint to return a fixed train_set_uid
        with patch.object(
            self.client.httpx_client, "post", wraps=self.client.httpx_client.post
        ) as mock_post:
            # Set up the mock response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"train_set_uid": "dummy_train_set_uid"}
            mock_post.return_value = mock_response

            # First upload
            train_set_uid1 = self.client.upload_train_set(self.X_train, self.y_train)

            # Second upload with the same data
            train_set_uid2 = self.client.upload_train_set(self.X_train, self.y_train)

            # The train_set_uid should be the same due to caching
            self.assertEqual(train_set_uid1, train_set_uid2)

            # The upload_train_set endpoint should have been called only once
            mock_post.assert_called_once()

    def test_predict_with_caching(self):
        """
        Test that making predictions with the same test set uses the cache and
        avoids re-uploading the test set.
        """
        self.client.authorize("dummy_access_token")

        # Mock the upload_train_set and predict endpoints
        with patch.object(
            self.client.httpx_client, "post", wraps=self.client.httpx_client.post
        ) as mock_post:
            # Mock responses
            def side_effect(*args, **kwargs):
                if (
                    kwargs.get("url")
                    == self.client.server_endpoints.upload_train_set.path
                ):
                    response = Mock()
                    response.status_code = 200
                    response.json.return_value = {
                        "train_set_uid": "dummy_train_set_uid"
                    }
                    return response
                elif kwargs.get("url") == self.client.server_endpoints.predict.path:
                    response = Mock()
                    response.status_code = 200
                    response.json.return_value = {
                        "classification": [1, 2, 3],
                        "test_set_uid": "dummy_test_set_uid",
                    }
                    return response
                else:
                    return Mock(status_code=404)

            mock_post.side_effect = side_effect

            # Upload train set
            train_set_uid = self.client.upload_train_set(self.X_train, self.y_train)

            # First prediction
            pred1 = self.client.predict(
                train_set_uid=train_set_uid, x_test=self.X_test, task="classification"
            )

            # Second prediction with the same test set
            pred2 = self.client.predict(
                train_set_uid=train_set_uid, x_test=self.X_test, task="classification"
            )

            # The predictions should be the same
            self.assertTrue(np.array_equal(pred1["probas"], pred2["probas"]))

            # The predict endpoint should have been called twice
            self.assertEqual(
                mock_post.call_count, 3
            )  # 1 for upload_train_set, 2 for predict

            # Check that the test set was uploaded only once (first predict call)
            upload_calls = [
                call for call in mock_post.call_args_list if "files" in call[1]
            ]
            self.assertEqual(
                len(upload_calls), 2
            )  # 1 for train set, 1 for test set upload

    def test_predict_with_invalid_cached_uids(self):
        """
        Test that when the cached UIDs are invalid, the client re-uploads the datasets
        and retries the prediction.
        """
        self.client.authorize("dummy_access_token")

        # Mock the upload_train_set and predict endpoints
        with patch.object(
            self.client.httpx_client, "post", wraps=self.client.httpx_client.post
        ) as mock_post:
            # Mock responses with side effects to simulate invalid cached UIDs
            def side_effect(*args, **kwargs):
                if (
                    kwargs.get("url")
                    == self.client.server_endpoints.upload_train_set.path
                ):
                    response = Mock()
                    response.status_code = 200
                    response.json.return_value = {
                        "train_set_uid": "dummy_train_set_uid"
                    }
                    return response
                elif kwargs.get("url") == self.client.server_endpoints.predict.path:
                    # Simulate invalid UID on first call, success on second
                    if side_effect.call_count == 2:
                        response = Mock()
                        response.status_code = 400
                        response.json.return_value = {
                            "detail": "Invalid train or test set uid"
                        }
                        return response
                    else:
                        response = Mock()
                        response.status_code = 200
                        response.json.return_value = {
                            "classification": [1, 2, 3],
                            "test_set_uid": "new_dummy_test_set_uid",
                        }
                        return response
                else:
                    return Mock(status_code=404)

            side_effect.call_count = 0

            def side_effect_counter(*args, **kwargs):
                side_effect.call_count += 1
                return side_effect(*args, **kwargs)

            mock_post.side_effect = side_effect_counter

            # Upload train set
            train_set_uid = self.client.upload_train_set(self.X_train, self.y_train)

            # Attempt prediction, which should fail and trigger retry
            pred = self.client.predict(
                train_set_uid=train_set_uid,
                x_test=self.X_test,
                task="classification",
                X_train=self.X_train,
                y_train=self.y_train,
            )

            # The predictions should be as expected
            self.assertTrue(np.array_equal(pred["probas"], [1, 2, 3]))

            # The predict endpoint should have been called twice due to retry
            self.assertEqual(
                mock_post.call_count, 4
            )  # 1 upload_train_set + 2 predict + 1 re-upload

            # Ensure that upload_train_set was called again (re-upload)
            upload_calls = [
                call
                for call in mock_post.call_args_list
                if call.kwargs.get("url")
                == self.client.server_endpoints.upload_train_set.path
            ]
            self.assertEqual(len(upload_calls), 2)

    def test_dataset_cache_manager(self):
        """
        Test the DatasetCacheManager's basic functionality: adding, retrieving,
        and deleting dataset UIDs based on hashes.
        """
        # Create a fresh cache manager
        cache_manager = self.client.dataset_uid_cache_manager

        # Mock dataset hashes and UIDs
        dataset_1 = "data1"
        dataset_uid_1 = "uid1"
        dataset_2 = "data2"
        dataset_uid_2 = "uid2"

        # Get hash by trying to get dataset_uid from cache
        _, dataset_hash_1 = cache_manager.get_dataset_uid(dataset_1)
        _, dataset_hash_2 = cache_manager.get_dataset_uid(dataset_2)

        # Add datasets to cache
        cache_manager.add_dataset_uid(dataset_hash_1, dataset_uid_1)
        cache_manager.add_dataset_uid(dataset_hash_2, dataset_uid_2)

        # Retrieve datasets from cache
        retrieved_uid_1, _ = cache_manager.get_dataset_uid(dataset_1)
        retrieved_uid_2, _ = cache_manager.get_dataset_uid(dataset_2)
        self.assertEqual(retrieved_uid_1, dataset_uid_1)
        self.assertEqual(retrieved_uid_2, dataset_uid_2)

        # Delete a dataset by UID
        deleted_hash = cache_manager.delete_uid(dataset_uid_1)
        self.assertEqual(deleted_hash, dataset_hash_1)

        # Ensure the deleted dataset is no longer in the cache
        self.assertIsNone(cache_manager.get_dataset_uid(dataset_1)[0])

    def test_cache_limit(self):
        """
        Test that the cache does not exceed its limit and evicts the oldest entries.
        """
        cache_manager = self.client.dataset_uid_cache_manager
        cache_manager.cache_limit = 3  # Set a small limit for testing

        # Add more datasets than the cache limit
        for i in range(5):
            cache_manager.add_dataset_uid(f"hash{i}", f"uid{i}")

        # The cache should only contain the last 3 added datasets
        expected_hashes = ["hash2", "hash3", "hash4"]
        actual_hashes = list(cache_manager.cache.keys())

        self.assertEqual(actual_hashes, expected_hashes)
        self.assertEqual(len(cache_manager.cache), 3)
