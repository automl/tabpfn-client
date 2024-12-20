import shutil

from tabpfn_client.client import ServiceClient
from tabpfn_client.service_wrapper import UserAuthenticationClient
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client.prompt_agent import PromptAgent


class Config:
    def __new__(cls, *args, **kwargs):
        """
        This class is a singleton and should not be instantiated directly.
        Only use class methods.
        """
        raise TypeError("Cannot instantiate this class")

    is_initialized = False
    use_server = False


def init(use_server=True):
    # initialize config
    Config.use_server = use_server

    if Config.is_initialized:
        # Only do the following if the initialization has not been done yet
        return

    if use_server:
        # check connection to server
        if not UserAuthenticationClient.is_accessible_connection():
            raise RuntimeError(
                "TabPFN is inaccessible at the moment, please try again later."
            )

        is_valid_token_set = UserAuthenticationClient.try_reuse_existing_token()


        if isinstance(is_valid_token_set, bool) and is_valid_token_set:
            PromptAgent.prompt_reusing_existing_token()
        elif (
            isinstance(is_valid_token_set, tuple) and is_valid_token_set[1] is not None
        ):
            print("Your email is not verified. Please verify your email to continue...")
            PromptAgent.reverify_email(is_valid_token_set[1])
        else:
            PromptAgent.prompt_welcome()
            if not PromptAgent.prompt_terms_and_cond():
                raise RuntimeError(
                    "You must agree to the terms and conditions to use TabPFN"
                )

            # prompt for login / register
            PromptAgent.prompt_and_set_token()

        # Print new greeting messages. If there are no new messages, nothing will be printed.
        PromptAgent.prompt_retrieved_greeting_messages(
            UserAuthenticationClient.retrieve_greeting_messages()
        )

        Config.use_server = True
        Config.is_initialized = True
    else:
        raise RuntimeError("Local inference is not supported yet.")


def reset():
    Config.is_initialized = False
    # reset user auth handler
    if Config.use_server:
        UserAuthenticationClient.reset_cache()

    # remove cache dir
    shutil.rmtree(CACHE_DIR, ignore_errors=True)


def get_access_token() -> str:
    init()
    return ServiceClient.get_access_token()


def set_access_token(access_token: str):
    UserAuthenticationClient.set_token(access_token)
    Config.is_initialized = True
