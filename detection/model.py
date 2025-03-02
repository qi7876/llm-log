"""model.py

This module contains the LogLLM class, which is used to build prompt,
maintain the conversation history and chat with the LLM model.
@author: qi7876
"""

from llama_cpp import Llama


class LogLLM:
    def __init__(
        self,
        model_path,
        num_max_content_tokens,
        template,
        num_gpu_layers=0,
        verbose=False,
    ) -> None:
        # Create an instance of LLM.
        self.llm = Llama(
            model_path=model_path,
            n_ctx=num_max_content_tokens,
            n_gpu_layers=num_gpu_layers,
            verbose=verbose,
        )

        self.verbose = verbose

        # Use prompt to limit the content return by the LLM.
        self.template = template

    def chat(self, message) -> str:
        """Chat with the LLM model.

        Args:
            message (dict): A dictionary containing the message, like {"log": "log content"}.

        Returns:
            str: The response from the LLM model.
        """
        # Build the prompt for the LLM model.
        input = self.buildPrompt(message)

        if self.verbose:
            print("\n=========================================================")
            print(f"[Input]: {input}")
            print("\n=========================================================")
            print(f"[Input Token Num]: {self.getTokenNum(input)}")
            print("=========================================================\n")

        # Get the response from the LLM model.
        output = self.llm(input)
        response = output["choices"][0]["text"].strip()

        if self.verbose:
            print("\n=========================================================")
            print(f"Response: {response}")
            print("=========================================================\n")

        return response

    def buildPrompt(self, message):
        """Build the prompt for the LLM model.

        Args:
            message (str): The message from the user.

        Returns:
            str: The completed prompt for the LLM model.
        """
        input = self.template

        # Replace the placeholder in the template with the message dict.
        for key, value in message.items():
            input = input.replace("{" + key + "}", value)

        return input

    def getTokenNum(self, message) -> int:
        """Get the number of tokens from a string.

        Args:
            message (str): Words, sentences or other strings.

        Returns:
            int: The number of tokens.
        """
        num_tokens = len(self.llm.tokenize(message.encode("utf-8")))
        return num_tokens
