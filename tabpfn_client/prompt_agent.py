import textwrap
import getpass
from password_strength import PasswordPolicy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tabpfn_client.estimator import UserAuthenticationClient


class PromptAgent:
    @staticmethod
    def indent(text: str):
        indent_factor = 2
        indent_str = " " * indent_factor
        return textwrap.indent(text, indent_str)

    @staticmethod
    def password_req_to_policy(password_req: list[str]):
        """
        Small function that receives password requirements as a list of
        strings like "Length(8)" and returns a corresponding
        PasswordPolicy object.
        """
        requirements = {}
        for req in password_req:
            word_part, number_part = req.split("(")
            number = int(number_part[:-1])
            requirements[word_part.lower()] = number
        return PasswordPolicy.from_names(**requirements)

    @classmethod
    def prompt_welcome(cls):
        prompt = "\n".join(
            [
                "Welcome to TabPFN!",
                "",
                "TabPFN is still under active development, and we are working hard to make it better.",
                "Please bear with us if you encounter any issues.",
                "",
            ]
        )

        print(cls.indent(prompt))

    @classmethod
    def prompt_and_set_token(cls, user_auth_handler: "UserAuthenticationClient"):
        # Choose between registration and login
        prompt = "\n".join(
            [
                "Please choose one of the following options:",
                "(1) Create a TabPFN account",
                "(2) Login to your TabPFN account",
                "",
                "Please enter your choice: ",
            ]
        )
        choice = cls._choice_with_retries(prompt, ["1", "2"])
        email = ""

        # Registration
        if choice == "1":
            # validation_link = input(cls.indent("Please enter your secret code: "))
            validation_link = "tabpfn-2023"
            while True:
                email = input(cls.indent("Please enter your email: "))
                # Send request to server to check if email is valid and not already taken.
                is_valid, message = user_auth_handler.validate_email(email)
                if is_valid:
                    break
                else:
                    print(cls.indent(message + "\n"))

            password_req = user_auth_handler.get_password_policy()
            password_policy = cls.password_req_to_policy(password_req)
            password_req_prompt = "\n".join(
                [
                    "",
                    "Password requirements (minimum):",
                    "\n".join([f". {req}" for req in password_req]),
                    "",
                    "Please enter your password: ",
                ]
            )
            while True:
                password = getpass.getpass(cls.indent(password_req_prompt))
                password_req_prompt = "Please enter your password: "
                if len(password_policy.test(password)) != 0:
                    print(cls.indent("Password requirements not satisfied.\n"))
                    continue

                password_confirm = getpass.getpass(
                    cls.indent("Please confirm your password: ")
                )
                if password == password_confirm:
                    break
                else:
                    print(
                        cls.indent(
                            "Entered password and confirmation password do not match, please try again.\n"
                        )
                    )
            additional_info = cls.prompt_add_user_information()
            is_created, message = user_auth_handler.set_token_by_registration(
                email, password, password_confirm, validation_link, additional_info
            )
            if not is_created:
                raise RuntimeError("User registration failed: " + str(message) + "\n")

            print(
                cls.indent(
                    "Account created successfully! To start using TabPFN please click on the link in the verification email we sent you."
                )
                + "\n"
            )

        # Login
        elif choice == "2":
            # login to account
            while True:
                email = input(cls.indent("Please enter your email: "))
                password = getpass.getpass(cls.indent("Please enter your password: "))

                successful, message = user_auth_handler.set_token_by_login(
                    email, password
                )
                if successful:
                    break
                print(cls.indent("Login failed: " + message) + "\n")

                prompt = "\n".join(
                    [
                        "Please choose one of the following options:",
                        "(1) Retry login",
                        "(2) Reset your password",
                        "",
                        "Please enter your choice: ",
                    ]
                )
                choice = cls._choice_with_retries(prompt, ["1", "2"])

                if choice == "1":
                    continue
                elif choice == "2":
                    sent = False
                    print(
                        cls.indent(
                            "We will send you an email with a link "
                            "that allows you to reset your password. \n"
                        )
                    )
                    while not sent:
                        email = input(cls.indent("Please enter your email address: "))

                        sent, message = user_auth_handler.send_reset_password_email(
                            email
                        )
                        print("\n" + cls.indent(message))
                    print(
                        cls.indent(
                            "Once you have reset your password, you will be able to login here: "
                        )
                    )

            print(cls.indent("Login successful!") + "\n")

    @classmethod
    def prompt_terms_and_cond(cls) -> bool:
        t_and_c = "\n".join(
            [
                "Please refer to our terms and conditions at: https://www.priorlabs.ai/terms-eu-en "
                "By using TabPFN, you agree to the following terms and conditions:",
                "Do you agree to the above terms and conditions? (y/n): ",
            ]
        )
        choice = cls._choice_with_retries(t_and_c, ["y", "n"])
        return choice == "y"

    @classmethod
    def prompt_add_user_information(cls) -> dict:
        print(
            cls.indent(
                "To help us tailor our support and services to your needs, we have a few optional questions. "
                "Feel free to skip any question by leaving it blank."
            )
            + "\n"
        )
        company = input(cls.indent("Where do you work? "))
        role = input(cls.indent("What is your role? "))
        use_case = input(cls.indent("What do you want to use TabPFN for? "))

        choice_contact = cls._choice_with_retries(
            "Can we reach out to you via email to support you? (y/n):", ["y", "n"]
        )
        contact_via_email = True if choice_contact == "y" else False

        return {
            "company": company,
            "role": role,
            "use_case": use_case,
            "contact_via_email": contact_via_email,
        }

    @classmethod
    def prompt_reusing_existing_token(cls):
        prompt = "\n".join(
            [
                "Welcome Back! Found existing access token, reusing it for authentication."
            ]
        )

        print(cls.indent(prompt))

    @classmethod
    def reverify_email(
        cls, access_token, user_auth_handler: "UserAuthenticationClient"
    ):
        prompt = "\n".join(
            [
                "Please check your inbox for the verification email.",
                "Note: The email might be in your spam folder or could have expired.",
            ]
        )
        print(cls.indent(prompt))
        retry_verification = "\n".join(
            [
                "Do you want to resend email verification link? (y/n): ",
            ]
        )
        choice = cls._choice_with_retries(retry_verification, ["y", "n"])
        if choice == "y":
            # get user email from user_auth_handler and resend verification email
            sent, message = user_auth_handler.send_verification_email(access_token)
            if not sent:
                print(cls.indent("Failed to send verification email: " + message))
            else:
                print(
                    cls.indent(
                        "A verification email has been sent, provided the details are correct!"
                    )
                    + "\n"
                )
        return

    @classmethod
    def prompt_retrieved_greeting_messages(cls, greeting_messages: list[str]):
        for message in greeting_messages:
            print(cls.indent(message))

    @classmethod
    def prompt_confirm_password_for_user_account_deletion(cls) -> str:
        print(cls.indent("You are about to delete your account."))
        confirm_pass = getpass.getpass(
            cls.indent("Please confirm by entering your password: ")
        )

        return confirm_pass

    @classmethod
    def prompt_account_deleted(cls):
        print(cls.indent("Your account has been deleted."))

    @classmethod
    def _choice_with_retries(cls, prompt: str, choices: list) -> str:
        """
        Prompt text and give user infinitely many attempts to select one of the possible choices. If valid choice
        is selected, return choice in lowercase.
        """
        assert all(c.lower() == c for c in choices), "Choices need to be lower case."
        choice = input(cls.indent(prompt))

        # retry until valid choice is made
        while True:
            if choice.lower() not in choices:
                choices_str = (
                    ", ".join(f"'{item}'" for item in choices[:-1])
                    + f" or '{choices[-1]}'"
                )
                choice = input(
                    cls.indent(f"Invalid choice, please enter {choices_str}: ")
                )
            else:
                break

        return choice.lower()
