"""test.py

This module is used to test the LogLLM.
@date: 2025-01-12
@author: qi7876
"""

from model import LogLLM
from dataset import loadDataset
import tomli
from queue import deque
import time
import random
import threading

with open("config.toml", "rb") as file:
    config = tomli.load(file)

# Work mode: offline or online.
mode = "offline"

# Debug related
VERBOSE = config["DEBUG"].get("VERBOSE", False)

# Model related
MODEL_PATH = config["MODEL"].get("MODEL_PATH", "")
NUM_MAX_CONTENT_TOKENS = config["MODEL"].get("NUM_MAX_CONTENT_TOKENS", 512)
TEMPLATE = config["MODEL"].get("TEMPLATE", "")
NUM_UPPER_CONTENT_TOKENS = config["MODEL"].get("NUM_UPPER_CONTENT_TOKENS", 128)
NUM_UPPER_OUTPUT_TOKENS = config["MODEL"].get("NUM_UPPER_OUTPUT_TOKENS", 128)
NUM_GPU_LAYERS = config["MODEL"].get("NUM_GPU_LAYERS", 0)
NUM_CONVERSATION_KEEP_ROUNDS = config["MODEL"].get("NUM_CONVERSATION_KEEP_ROUNDS", 1)
NUM_BUFFER_SIZE = config["MODEL"].get("NUM_BUFFER_SIZE", 100)

# Dataset related
DATASET_PATH = config["DATASET"].get("DATASET_PATH", "")
NUM_MAX_LOGS = config["DATASET"].get("NUM_MAX_LOGS", 1000)

# Maintain the buffer of logs.
buffer = deque(maxlen=NUM_BUFFER_SIZE)
# Count the number of logs we total loaded.
total_log_count = 0
buffer_log_count = 0

# Load the dataset.
log_list = loadDataset(DATASET_PATH, NUM_MAX_LOGS)

# Create an instance of LogLLM.
logLLM = LogLLM(
    model_path=MODEL_PATH,
    num_max_content_tokens=NUM_MAX_CONTENT_TOKENS,
    template=TEMPLATE,
    num_upper_content_tokens=NUM_UPPER_CONTENT_TOKENS,
    num_upper_output_tokens=NUM_UPPER_OUTPUT_TOKENS,
    num_gpu_layers=NUM_GPU_LAYERS,
    num_keep_rounds=NUM_CONVERSATION_KEEP_ROUNDS,
    verbose=VERBOSE,
)


def addLogToBuffer(log_number) -> bool:
    """Add logs to the buffer.

    Args:
        log_number (str): The number of logs to add.

    Returns:
        bool: True if we successfully add logs to the buffer
    """
    global buffer, log_list, total_log_count, buffer_log_count

    # Process the normal part of logs.
    if total_log_count < NUM_MAX_LOGS and total_log_count + log_number <= NUM_MAX_LOGS:
        for _ in range(log_number):
            buffer.append(log_list[total_log_count])
            print("[Log Info]: ", log_list[total_log_count])
            buffer_log_count += 1
            total_log_count += 1
            return True

    # Process the tail part of logs.
    elif total_log_count < NUM_MAX_LOGS and total_log_count + log_number > NUM_MAX_LOGS:
        for _ in range(total_log_count + log_number - NUM_MAX_LOGS):
            buffer.append(log_list[total_log_count])
            print("[Log Info]: ", log_list[total_log_count])
            buffer_log_count += 1
            total_log_count += 1
            return True

    else:
        print("No more logs to add.")
        return False


def onceChatToLLM():
    global buffer

    log = "\n".join([f"{log}" for log in list(buffer)])

    input = {"log": log}
    logLLM.template = TEMPLATE
    output = logLLM.chat(input)

    with open("output.txt", "a") as f:
        f.write(str(total_log_count) + output + "\n")


# Thread function to chat to LLM.
def threadChatToLLM():
    global buffer, buffer_log_count, last_check_time, current_time

    while True:
        current_time = time.time()

        if buffer_log_count >= NUM_BUFFER_SIZE or (
            current_time - last_check_time >= 10 and buffer_log_count > 0
        ):
            print("Start chatting to LLM.")
            log = "\n".join([f"{log}" for log in list(buffer)])
            input = {"log": log}
            output = logLLM.chat(input)

            with open("output.txt", "a") as f:
                f.write(output + "\n")

            buffer_log_count = 0
            last_check_time = current_time


start_time = time.time()

# Start the main loop.
if mode == "offline":
    try:
        ret = True
        while ret:
            ret = addLogToBuffer(NUM_BUFFER_SIZE)
            onceChatToLLM()
    except KeyboardInterrupt:
        print("STOP")
elif mode == "online":
    llmThread = threading.Thread(target=threadChatToLLM, daemon=True)
    llmThread.start()
    print("LLM thread started.")

    try:
        ret = True
        while ret:
            time.sleep(random.randrange(0.1, 0.3))
            ret = addLogToBuffer(1)
    except KeyboardInterrupt:
        print("STOP")

end_time = time.time()

print("Time cost: ", end_time - start_time)
