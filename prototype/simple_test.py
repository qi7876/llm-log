from model import LogLLM
from dataset import loadDataset
import tomli
from queue import deque

with open("config.toml", "rb") as file:
    config = tomli.load(file)

# Debug related
VERBOSE = config["DEBUG"]["VERBOSE"]

# Model related
MODEL_PATH = config["MODEL"]["MODEL_PATH"]
NUM_MAX_CONTENT_TOKENS = config["MODEL"]["NUM_MAX_CONTENT_TOKENS"]
TEMPLATE = config["MODEL"]["TEMPLATE"]
NUM_UPPER_CONTENT_TOKENS = config["MODEL"]["NUM_UPPER_CONTENT_TOKENS"]
NUM_UPPER_OUTPUT_TOKENS = config["MODEL"]["NUM_UPPER_OUTPUT_TOKENS"]
NUM_GPU_LAYERS = config["MODEL"]["NUM_GPU_LAYERS"]
NUM_CONVERSATION_KEEP_ROUNDS = config["MODEL"]["NUM_CONVERSATION_KEEP_ROUNDS"]
NUM_BUFFER_SIZE = config["MODEL"]["NUM_BUFFER_SIZE"]

# Dataset related
DATASET_PATH = config["DATASET"]["DATASET_PATH"]
NUM_MAX_LOGS = config["DATASET"]["NUM_MAX_LOGS"]

buffer = deque(maxlen=NUM_BUFFER_SIZE)
total_log_count = 0
buffer_log_count = 0


log_list = loadDataset(DATASET_PATH, NUM_MAX_LOGS)

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


def addLogToBuffer(log_number):
    global buffer, log_list, total_log_count, buffer_log_count

    if total_log_count < NUM_MAX_LOGS and total_log_count + log_number <= NUM_MAX_LOGS:
        for _ in range(log_number):
            buffer.append(log_list[total_log_count])
            print("[Log Info]: ", log_list[total_log_count])
            buffer_log_count += 1
            total_log_count += 1
            return True
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


try:
    ret = True
    while ret:
        ret = addLogToBuffer(NUM_BUFFER_SIZE)
        onceChatToLLM()
except KeyboardInterrupt:
    print("STOP")