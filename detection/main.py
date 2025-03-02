"""main.py

This module is used to test the LogLLM.
@author: qi7876
"""

from model import LogLLM
from dataset import loadDataset
import tomli
import time

with open("config.toml", "rb") as file:
    config = tomli.load(file)

# Work mode: offline or online.
mode = "offline"

# Debug related
VERBOSE = config["DEBUG"].get("VERBOSE", False)

# Model related
MODEL_PATH = config["MODEL"].get("MODEL_PATH", "")
NUM_MAX_CONTENT_TOKENS = config["MODEL"].get("NUM_MAX_CONTENT_TOKENS", 8192)
TEMPLATE = config["MODEL"].get("TEMPLATE", "")
NUM_GPU_LAYERS = config["MODEL"].get("NUM_GPU_LAYERS", 0)
NUM_BUFFER_SIZE = config["MODEL"].get("NUM_BUFFER_SIZE", 1)

# Dataset related
DATASET_PATH = config["DATASET"].get("DATASET_PATH", "")
NUM_MAX_LOGS = config["DATASET"].get("NUM_MAX_LOGS", 1000)

# Count the number of logs we total loaded.
total_log_count = 1

# Load the dataset.
log_list = loadDataset(DATASET_PATH, NUM_MAX_LOGS)

# Create an instance of LogLLM.
logLLM = LogLLM(
    model_path=MODEL_PATH,
    num_max_content_tokens=NUM_MAX_CONTENT_TOKENS,
    template=TEMPLATE,
    num_gpu_layers=NUM_GPU_LAYERS,
    verbose=VERBOSE,
)


def onceChatToLLM(log):
    global total_log_count

    input = {"log": log}
    output = logLLM.chat(input)

    with open("output.txt", "a") as f:
        f.write(str(total_log_count) + output + "\n")
    
    total_log_count += 1


start_time = time.time()

# Start the main loop.
try:
    for log in log_list:
        onceChatToLLM(log)
except KeyboardInterrupt:
    print("STOP")

end_time = time.time()

print("Time cost: ", end_time - start_time)
