from llama_cpp import Llama
from collections import deque


class LogLLM:
    def __init__(
        self,
        modelPath,
        maxContentWindowSize,
        template,
        upperContentWindowSize,
        upperOutputTokenNum,
        gpuLayerNum=0,
        roundKeepNum=20,
        verbose=False,
    ) -> None:
        self.llm = Llama(
            model_path=modelPath,
            n_ctx=maxContentWindowSize,
            n_gpu_layers=gpuLayerNum,
            verbose=verbose,
        )

        self.verbose = verbose
        # Use prompt to limit the content return by LLM.
        self.template = template
        self.upperContentWindowSize = upperContentWindowSize
        self.fixedTokenNum = upperOutputTokenNum + self.getTokenNum(self.template)
        self.maxContentWindowSize = maxContentWindowSize
        # Record the history of conversation.
        self.roundKeepNum = roundKeepNum
        self.conversationHistory = deque(maxlen=roundKeepNum * 2)
        self.historyTokenNumCount = 0

    def chat(self, message) -> str:
        userMessage = "User: "
        for _, value in message.items():
            userMessage += value
        userMessageTokenNum = self.getTokenNum(userMessage)

        # Check if the total token number exceeds the limit.
        while (
            self.fixedTokenNum + userMessageTokenNum + self.historyTokenNumCount
            > self.upperContentWindowSize
        ):
            # Clean the history if the total token number exceeds the limit.
            if not self.conversationHistory:
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
        assistantResponse = "Assistant: " + response
        responseTokenNum = self.getTokenNum(assistantResponse)
        self.conversationHistory.append(userMessage)
        self.conversationHistory.append(assistantResponse)
        self.historyTokenNumCount += userMessageTokenNum + responseTokenNum + 2

        return response

    def buildPrompt(self, message):
        input = self.template

        # Replace the placeholder in the template with the message.
        for key, value in message.items():
            input = input.replace("{" + key + "}", value)

        # Add history to input.
        if self.roundKeepNum != 0:
            history = ""
            for msg in self.conversationHistory:
                history += msg + "\n"
            input = input.replace("{history}", history)

        return input

    def deletHistory(self) -> None:
        if self.conversationHistory:
            oldestMessage = self.conversationHistory.popleft()
            self.historyTokenNumCount -= self.getTokenNum(oldestMessage) + 1

    def getTokenNum(self, message) -> int:
        tokenNum = len(self.llm.tokenize(message.encode("utf-8")))
        return tokenNum
