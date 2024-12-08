from llama_cpp import Llama
from collections import deque


class LogLLM:
    def __init__(
        self,
        modelPath,
        maxContentWindowSize,
        maxInputTokenNum,
        maxOutputTokenNum,
        systemPrompt,
        upperContentWindowSize,
        upperOutputTokenNum,
        gpuLayerNum=0,
        roundKeepNum=20,
    ) -> None:
        self.llm = Llama(
            model_path=modelPath,
            n_ctx=maxContentWindowSize,
            max_context_tokens=maxInputTokenNum,
            max_tokens=maxOutputTokenNum,
            n_gpu_layers=gpuLayerNum,
        )

        # Use prompt to limit the content return by LLM.
        self.systemPrompt = systemPrompt
        self.upperContentWindowSize = upperContentWindowSize
        self.fixedTokenNum = upperOutputTokenNum + self.getTokenNum(self.systemPrompt)
        self.maxContentWindowSize = maxContentWindowSize
        # Record the history of conversation.
        self.conversationHistory = deque(maxlen=roundKeepNum * 2)
        self.historyTokenNumCount = 0

    def chat(self, message) -> str:
        userMessage = "User: " + message
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

        # Build the final input for LLM.
        finalInput = (
            "**System Prompot**\n" + self.systemPrompt + "\n" + "**History Contents**\n"
        )
        for msg in self.conversationHistory:
            finalInput += msg + "\n"
        finalInput += "**New Logs**\n" + userMessage

        output = self.llm(finalInput, stop=["User:", "\n"])
        response = output["choices"][0]["text"].strip()
        print(f"Response: {response}")

        # Add new conversation to the history.
        assistantResponse = "Assistant: " + response
        responseTokenNum = self.getTokenNum(assistantResponse)
        self.conversationHistory.append(userMessage)
        self.conversationHistory.append(assistantResponse)
        self.historyTokenNumCount += userMessageTokenNum + responseTokenNum + 2

        return response

    def deletHistory(self) -> None:
        if self.conversationHistory:
            oldestMessage = self.conversationHistory.popleft()
            self.historyTokenNumCount -= self.getTokenNum(oldestMessage) + 1

    def getTokenNum(self, message) -> int:
        tokenNum = len(self.llm.tokenize(message.encode("utf-8")))
        return tokenNum
