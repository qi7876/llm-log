"""main.py

This module is used to test the LogLLM.
@author: qi7876
"""

from model import LogLLM
from database import VectorDatabase
from dataset_utils import load_dataset
import tomli
import time

with open("config.toml", "rb") as file:
    config = tomli.load(file)

# Debug related
VERBOSE = config["debug"].get("verbose", False)

# Model related
MODEL_PATH = config["model"].get("model_path", "")
NUM_MAX_CONTENT_TOKENS = config["model"].get("num_max_content_tokens", 8192)
NUM_GPU_LAYERS = config["model"].get("num_gpu_layers", 0)
NUM_BUFFER_SIZE = config["model"].get("num_buffer_size", 1)

# RAG related
CHROMA_DB_DIR = config["rag"].get("chroma_db_dir", "chroma_db")
COLLECTION_NAME = config["rag"].get("collection_name", "documents")

# Dataset related
DATASET_PATH = config["dataset"].get("dataset_path", "")
NUM_MAX_LOGS = config["dataset"].get("num_max_logs", 1000)

prompt_without_rag = """<|system|>
You are an anomaly detector in a log system. Analyze each log entry as normal or abnormal. You must choose one word in 'yes' or 'no' as the result. 'yes' means something failed or interrupted now. 'no' describes a process work normally or a normal state.
<|end|>
<|user|>
New logs are as follows:
{log}
<|end|>
<|assistant|>"""
prompt_with_rag = """<|system|>
You are an anomaly detector in a log system. Analyze each log entry as normal or abnormal. You must choose one word in 'yes' or 'no' as the result. 'yes' means something failed or interrupted now. 'no' describes a process work normally or a normal state.
<|end|>
<|user|>
New logs are as follows:
{log}
You can refer to the following information for judgment:
{db_response}
<|end|>
<|assistant|>"""

# Count the number of logs we total loaded.
total_log_count = 1

# Load the dataset.
log_list = load_dataset(DATASET_PATH, NUM_MAX_LOGS)

# Create an instance of LogLLM.
logLLM = LogLLM(
    model_path=MODEL_PATH,
    num_max_content_tokens=NUM_MAX_CONTENT_TOKENS,
    num_gpu_layers=NUM_GPU_LAYERS,
    verbose=VERBOSE,
)

vectorDatabase = VectorDatabase(db_dir=CHROMA_DB_DIR, collection_name=COLLECTION_NAME)


def once_chat_to_llm(log):
    global total_log_count

    db_response = vectorDatabase.query(log)

    if db_response != "":
        user_input = {"log": log, "db_response": db_response}
        output = logLLM.chat(prompt_with_rag ,user_input)
    else:
        user_input = {"log": log}
        output = logLLM.chat(prompt_without_rag, user_input)

    with open("output.txt", "a") as f:
        f.write(str(total_log_count) + output + "\n")

    total_log_count += 1


start_time = time.time()

# Start the main loop.
try:
    for new_log in log_list:
        once_chat_to_llm(new_log)
except KeyboardInterrupt:
    print("STOP")

end_time = time.time()

print("Time cost: ", end_time - start_time)
