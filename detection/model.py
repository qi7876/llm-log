"""model.py

This module contains the LogLLM class, which is used to build prompt,
maintain the conversation history and chat with the LLM model.
@author: qi7876
"""

from llama_cpp import Llama


def build_prompt(origin_prompt: str, message: dict) -> str:
    """Build the prompt for the LLM model.

    Args:
        message (dict): The message from the user.
        origin_prompt(str): The original prompt from the user.

    Returns:
        str: The completed prompt for the LLM model.
    """
    prompt = origin_prompt

    # Replace the placeholder in the template with the message dict.
    for key, value in message.items():
        prompt = prompt.replace("{" + key + "}", value)

    return prompt


class LogLLM:
    def __init__(
        self,
        model_path,
        num_max_content_tokens,
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

    def chat(self, prompt_template: str, message: dict) -> str:
        """Chat with the LLM model.

        Args:
            message (dict): A dictionary containing the message, like {"new_log": "new_log content"}.
            prompt_template (str): The prompt template.

        Returns:
            str: The response from the LLM model.
        """
        # Build the prompt for the LLM model.
        complete_prompt = build_prompt(prompt_template, message)

        if self.verbose:
            print("\n=========================================================")
            print(f"[Input]: {complete_prompt}")
            print("\n=========================================================")
            print(f"[Input Token Num]: {self.get_token_num(complete_prompt)}")
            print("=========================================================\n")

        # Get the response from the LLM model.
        output = self.llm(complete_prompt)
        response = output["choices"][0]["text"].strip()

        if self.verbose:
            print("\n=========================================================")
            print(f"Response: {response}")
            print("=========================================================\n")

        return response

    def get_token_num(self, message: str) -> int:
        """Get the number of tokens from a string.

        Args:
            message (str): Words, sentences or other strings.

        Returns:
            int: The number of tokens.
        """
        num_tokens = len(self.llm.tokenize(message.encode("utf-8")))
        return num_tokens
