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
        prompt = "\n".join([
            "Please choose one of the following options:",
            "(1) Create a TabPFN account",
            "(2) Login to your TabPFN account",
            "",
            "Please enter your choice: ",
        ])

        choice = input(cls.indent(prompt))

        if choice == "1":
            #validation_link = input(cls.indent("Please enter your secret code: "))
            validation_link = "tabpfn-2023"
            # create account
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

            user_auth_handler.set_token_by_registration(email, password, password_confirm, validation_link)

            print(cls.indent("Account created successfully!") + "\n")

        elif choice == "2":
            # login to account
            email = input(cls.indent("Please enter your email: "))
            password = getpass.getpass(cls.indent("Please enter your password: "))

            user_auth_handler.set_token_by_login(email, password)

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

        choice = input(cls.indent(t_and_c))

        # retry for 3 attempts until valid choice is made
        is_valid_choice = False
        for _ in range(3):
            if choice.lower() not in ["y", "n"]:
                choice = input(cls.indent("Invalid choice, please enter 'y' or 'n': "))
            else:
                is_valid_choice = True
                break

        if not is_valid_choice:
            raise RuntimeError("Invalid choice")

        return choice.lower() == "y"

    @classmethod
    def prompt_reusing_existing_token(cls):
        prompt = "\n".join([
            "Found existing access token, reusing it for authentication."
        ])

        print(cls.indent(prompt))

    @classmethod
    def prompt_retrieved_messages(cls, messages: list[str]):
        for message in messages:
            print(cls.indent(message))


    @classmethod
    def prompt_confirm_password_for_user_account_deletion(cls) -> str:
        print(cls.indent("You are about to delete your account."))
        confirm_pass = getpass.getpass(cls.indent("Please confirm by entering your password: "))

        return confirm_pass

    @classmethod
    def prompt_account_deleted(cls):
        print(cls.indent("Your account has been deleted."))
