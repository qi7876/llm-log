from inspect import currentframe
from model import LogLLM
from dataset import loadDataset
import tomli
from queue import deque
import time
import random
import threading

with open("config.toml", "rb") as file:
    config = tomli.load(file)

# Model related
MODEL_PATH = config["MODEL"]["MODEL_PATH"]
MAX_CONTENT_WINDOW_SIZE = config["MODEL"]["MAX_CONTENT_WINDOW_SIZE"]
MAX_INPUT_TOKEN_NUM = config["MODEL"]["MAX_INPUT_TOKEN_NUM"]
MAX_OUTPUT_TOKEN_NUM = config["MODEL"]["MAX_OUTPUT_TOKEN_NUM"]
SYSTEM_PROMPT = config["MODEL"]["SYSTEM_PROMPT"]
UPPER_CONTENT_WINDOW_SIZE = config["MODEL"]["UPPER_CONTENT_WINDOW_SIZE"]
UPPER_OUTPUT_TOKEN_NUM = config["MODEL"]["UPPER_OUTPUT_TOKEN_NUM"]
GPU_LAYER_NUM = config["MODEL"]["GPU_LAYER_NUM"]
CONVERSATION_ROUND_KEEP_NUM = config["MODEL"]["CONVERSATION_ROUND_KEEP_NUM"]
BUFFER_SIZE = config["MODEL"]["BUFFER_SIZE"]

# Dataset related
DATASET_PATH = config["DATASET"]["DATASET_PATH"]
MAX_LOG_NUM = config["DATASET"]["MAX_LOG_NUM"]

buffer = deque(maxlen=BUFFER_SIZE)
totalLogCount = 0
bufferLogCount = 0
lastCheckTime = time.time()
currentTime = time.time()

logList = loadDataset(DATASET_PATH, MAX_LOG_NUM)

logLLM = LogLLM(
    modelPath=MODEL_PATH,
    maxContentWindowSize=MAX_CONTENT_WINDOW_SIZE,
    maxInputTokenNum=MAX_INPUT_TOKEN_NUM,
    maxOutputTokenNum=MAX_OUTPUT_TOKEN_NUM,
    systemPrompt=SYSTEM_PROMPT,
    upperContentWindowSize=UPPER_CONTENT_WINDOW_SIZE,
    upperOutputTokenNum=UPPER_OUTPUT_TOKEN_NUM,
    gpuLayerNum=GPU_LAYER_NUM,
    roundKeepNum=CONVERSATION_ROUND_KEEP_NUM,
)


def addLogToBuffer():
    global buffer, logList, totalLogCount, bufferLogCount

    if totalLogCount < MAX_LOG_NUM:
        buffer.append(logList[totalLogCount])
        print("[Log Info]: ", logList[totalLogCount])
        bufferLogCount += 1
        totalLogCount += 1
    else:
        print("No more logs to add.")


def chatToLLM():
    global buffer, bufferLogCount, lastCheckTime, currentTime

    while True:
        currentTime = time.time()

        if bufferLogCount >= BUFFER_SIZE or (
            currentTime - lastCheckTime >= 10 and bufferLogCount > 0
        ):
            print("Start chatting to LLM.")
            input = "\n".join([f"{log}" for log in list(buffer)])
            output = logLLM.chat(input)

            with open("output.txt", "a") as f:
                f.write(output + "\n")

            bufferLogCount = 0
            lastCheckTime = currentTime


llmThread = threading.Thread(target=chatToLLM, daemon=True)
llmThread.start()
print("LLM thread started.")


try:
    while True:
        addLogToBuffer()
        time.sleep(random.uniform(1, 5))
except KeyboardInterrupt:
    print("STOP")
