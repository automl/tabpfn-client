import unittest
from unittest.mock import patch, MagicMock
import respx
from httpx import Response
from tabpfn_client.prompt_agent import PromptAgent
from tabpfn_client.tests.mock_tabpfn_server import with_mock_server
from tabpfn_client.service_wrapper import UserAuthenticationClient, ServiceClient


class TestPromptAgent(unittest.TestCase):
    @with_mock_server()
    @patch('getpass.getpass', side_effect=['password123', 'password123'])
    @patch('builtins.input', side_effect=['1', 'user@example.com', 'test', 'test', 'test', 'y'])
    def test_prompt_and_set_token_registration(self, mock_input, mock_getpass, mock_server):
        mock_auth_client = MagicMock()
        mock_auth_client.get_password_policy.return_value = ['At least 8 characters']
        mock_auth_client.set_token_by_registration.return_value = (True, 'Registration successful')
        PromptAgent.prompt_and_set_token(user_auth_handler=mock_auth_client)
        mock_auth_client.set_token_by_registration.assert_called_once()

    @patch('getpass.getpass', side_effect = ['password123'])
    @patch('builtins.input', side_effect=['2', 'user@example.com'])
    def test_prompt_and_set_token_login(self, mock_input, mock_getpass):
        mock_auth_client = MagicMock()
        mock_auth_client.set_token_by_login.return_value = (True, 'Login successful')
        PromptAgent.prompt_and_set_token(user_auth_handler=mock_auth_client)
        mock_auth_client.set_token_by_login.assert_called_once()

    @patch('builtins.input', return_value='y')
    def test_prompt_terms_and_cond_returns_true(self, mock_input):
        result = PromptAgent.prompt_terms_and_cond()
        self.assertTrue(result)

    @patch('builtins.input', return_value='n')
    def test_prompt_terms_and_cond_returns_false(self, mock_input):
        result = PromptAgent.prompt_terms_and_cond()
        self.assertFalse(result)

    @patch('builtins.input', return_value='1')
    def test_choice_with_retries_valid_first_try(self, mock_input):
        result = PromptAgent._choice_with_retries("Please enter your choice: ", ["1", "2"])
        self.assertEqual(result, '1')

    @patch('builtins.input', side_effect=['3', '3', '1'])
    def test_choice_with_retries_valid_last_try(self, mock_input):
        result = PromptAgent._choice_with_retries("Please enter your choice: ", ["1", "2"], max_attempts=3)
        self.assertEqual(result, '1')

    # Test handling all inputs invalid within the allowed attempts
    @patch('builtins.input', side_effect=['3', '4', '5'])
    def test_choice_with_retries_raises_runtime_error(self, mock_input):
        with self.assertRaises(RuntimeError):
            PromptAgent._choice_with_retries("Please enter your choice: ", ["1", "2"], max_attempts=3)


if __name__ == '__main__':
    unittest.main()
