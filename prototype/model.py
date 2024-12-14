from llama_cpp import Llama
from collections import deque


class LogLLM:
    def __init__(
        self,
        model_path,
        num_max_content_tokens,
        template,
        num_upper_content_tokens,
        num_upper_output_tokens,
        num_gpu_layers=0,
        num_keep_rounds=20,
        verbose=False,
    ) -> None:
        self.llm = Llama(
            model_path=model_path,
            n_ctx=num_max_content_tokens,
            n_gpu_layers=num_gpu_layers,
            verbose=verbose,
        )

        self.verbose = verbose
        # Use prompt to limit the content return by LLM.
        self.template = template
        self.num_upper_content_tokens = num_upper_content_tokens
        self.num_fixed_tokens = num_upper_output_tokens + self.getTokenNum(
            self.template
        )
        self.num_max_content_tokens = num_max_content_tokens
        # Record the history of conversation.
        self.num_keep_rounds = num_keep_rounds
        self.conversation_history = deque(maxlen=num_keep_rounds * 2)
        self.history_tokens_count = 0

    def chat(self, message) -> str:
        user_message = "User: "
        for _, value in message.items():
            user_message += value
        num_user_message_token = self.getTokenNum(user_message)

        # Check if the total token number exceeds the limit.
        while (
            self.num_fixed_tokens + num_user_message_token + self.history_tokens_count
            > self.num_upper_content_tokens
        ):
            # Clean the history if the total token number exceeds the limit.
            if not self.conversation_history:
                break
            self.deletHistory()

        input = self.buildPrompt(message)

        if self.verbose:
            print("\n=========================================================")
            print(f"[Input]: {input}")
            print("\n=========================================================")
            print(f"[Input Token Num]: {self.getTokenNum(input)}")
            print("=========================================================\n")

        output = self.llm(input)
        response = output["choices"][0]["text"].strip()

        if self.verbose:
            print("\n=========================================================")
            print(f"Response: {response}")
            print("=========================================================\n")

        # Add new conversation to the history.
        assistant_response = "Assistant: " + response
        num_response_tokens = self.getTokenNum(assistant_response)
        self.conversation_history.append(user_message)
        self.conversation_history.append(assistant_response)
        self.history_tokens_count += num_user_message_token + num_response_tokens + 2

        return response

    def buildPrompt(self, message):
        input = self.template

        # Replace the placeholder in the template with the message.
        for key, value in message.items():
            input = input.replace("{" + key + "}", value)

        # Add history to input.
        if self.num_keep_rounds != 0:
            history = ""
            for msg in self.conversation_history:
                history += msg + "\n"
            input = input.replace("{history}", history)

        return input

    def deletHistory(self) -> None:
        if self.conversation_history:
            oldest_message = self.conversation_history.popleft()
            self.history_tokens_count -= self.getTokenNum(oldest_message) + 1

    def getTokenNum(self, message) -> int:
        num_tokens = len(self.llm.tokenize(message.encode("utf-8")))
        return num_tokens
