import textwrap
import getpass


class PromptAgent:
    @staticmethod
    def indent(text: str):
        indent_factor = 2
        indent_str = " " * indent_factor
        return textwrap.indent(text, indent_str)

    @classmethod
    def prompt_welcome(cls):
        prompt = "\n".join([
            "Welcome to TabPFN!",
            "",
            "TabPFN is still under active development, and we are working hard to make it better.",
            "Please bear with us if you encounter any issues.",
            ""
        ])

        print(cls.indent(prompt))

    @classmethod
    def prompt_and_set_token(cls, user_auth_handler: "UserAuthenticationClient"):
        # Choose between registration and login
        prompt = "\n".join([
            "Please choose one of the following options:",
            "(1) Create a TabPFN account",
            "(2) Login to your TabPFN account",
            "",
            "Please enter your choice: ",
        ])
        choice = cls._choice_with_retries(prompt, ["1", "2"])

        # Registration
        if choice == "1":
            #validation_link = input(cls.indent("Please enter your secret code: "))
            validation_link = "tabpfn-2023"
            # create account
            while True:
                email = input(cls.indent("Please enter your email: "))

                password_req = user_auth_handler.get_password_policy()
                password_req_prompt = "\n".join([
                    "",
                    "Password requirements (minimum):",
                    "\n".join([f". {req}" for req in password_req]),
                    "",
                    "Please enter your password: ",
                ])

                password = getpass.getpass(cls.indent(password_req_prompt))
                password_confirm = getpass.getpass(cls.indent("Please confirm your password: "))

                is_created, message = user_auth_handler.set_token_by_registration(
                    email, password, password_confirm, validation_link)
                if is_created:
                    break
                print(cls.indent("User registration failed: " + message) + "\n")
            cls.prompt_add_user_information(user_auth_handler)
            print(cls.indent("Account created successfully!") + "\n")

        # Login
        elif choice == "2":
            # login to account
            while True:
                email = input(cls.indent("Please enter your email: "))
                password = getpass.getpass(cls.indent("Please enter your password: "))

                successful, message = user_auth_handler.set_token_by_login(email, password)
                if successful:
                    break
                print(cls.indent("Login failed: " + message) + "\n")
            print(cls.indent("Login successful!") + "\n")

        else:
            raise RuntimeError("Invalid choice")

    @classmethod
    def prompt_terms_and_cond(cls) -> bool:
        t_and_c = "\n".join([
            "Please refer to our terms and conditions at: https://www.priorlabs.ai/terms-eu-en "
            "By using TabPFN, you agree to the following terms and conditions:",
            "Do you agree to the above terms and conditions? (y/n): ",
        ])
        choice = cls._choice_with_retries(t_and_c, ["y", "n"])
        return choice.lower() == "y"

    @classmethod
    def prompt_add_user_information(cls, user_auth_handler: "UserAuthenticationClient"):
        print(cls.indent("To help us tailor our support and services to your needs, we have a few optional questions. "
                         "Feel free to skip any question by leaving it blank.") + "\n")
        company = input(cls.indent("Where do you work?: "))
        role = input(cls.indent("What is your role?: "))
        use_case = input(cls.indent("What do you want to use TabPFN for?: "))

        choice_contact = cls._choice_with_retries(
            "Can we reach out to you via email to support you? (y/n):", ["y", "n"]
        )
        contact_via_email = True if choice_contact == "y" else False

        user_auth_handler.add_user_information(company, role, use_case, contact_via_email)

    @classmethod
    def prompt_reusing_existing_token(cls):
        prompt = "\n".join([
            "Found existing access token, reusing it for authentication."
        ])

        print(cls.indent(prompt))

    @classmethod
    def prompt_retrieved_greeting_messages(cls, greeting_messages: list[str]):
        for message in greeting_messages:
            print(cls.indent(message))


    @classmethod
    def prompt_confirm_password_for_user_account_deletion(cls) -> str:
        print(cls.indent("You are about to delete your account."))
        confirm_pass = getpass.getpass(cls.indent("Please confirm by entering your password: "))

        return confirm_pass

    @classmethod
    def prompt_account_deleted(cls):
        print(cls.indent("Your account has been deleted."))

    @classmethod
    def _choice_with_retries(cls, prompt: str, choices: list, max_attempts: int = 3) -> str:
        """
        Prompt text and give user predefined number of attempts to select one of the possible choices. If valid choice
        is selected, return choice in lowercase, otherwise raise RuntimeError.
        """
        choice = input(cls.indent(prompt))

        # retry for 3 attempts until valid choice is made
        is_valid_choice = False
        for _ in range(max_attempts):
            if choice.lower() not in choices:
                choices_str = ", ".join(f"'{item}'" for item in choices[:-1]) + f" or '{choices[-1]}'"
                choice = input(cls.indent(f"Invalid choice, please enter {choices_str}: "))
            else:
                is_valid_choice = True
                break

        if not is_valid_choice:
            raise RuntimeError("Invalid choice")

        return choice.lower()
