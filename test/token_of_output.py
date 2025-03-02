# NOTE:Need to be rebuilt.
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest

tokenizerPath = "../models/tokenizer.model.v3"
output1 = "--++--++--++--+++---"
output2 = "++++++++++++++++++++"
output3 = "--------------------"
output4 = "-+-+-+-+-+-+-+-+-+-+"
output5 = "+-+-+-+-+-+-+-+-+-+-"
output6 = "---------+++++++++++"
output7 = "--+-++--+-++---++-++"


def getTokenNum(message):
    tokenizer = MistralTokenizer.from_file(tokenizerPath)
    completionRequest = ChatCompletionRequest(messages=[UserMessage(content=message)])
    encodedMessage = tokenizer.encode_chat_completion(completionRequest)
    tokens = encodedMessage.tokens
    tokenNum = len(tokens)
    return tokenNum


print(getTokenNum(output1))
print(getTokenNum(output2))
print(getTokenNum(output3))
print(getTokenNum(output4))
print(getTokenNum(output5))
print(getTokenNum(output6))
print(getTokenNum(output7))
