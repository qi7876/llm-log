"""main.py

This module is used to test the LogLLM.
@author: qi7876
"""

from model import LogLLM
from database import VectorDatabase
from get_logs import get_logs
import tomli
import time

with open("./configs/config_phi-4_BGL.toml", "rb") as file:
    config = tomli.load(file)

# Debug related
VERBOSE = config["debug"].get("verbose", False)

# Model related
MODEL_PATH = config["model"].get("model_path", "")
NUM_MAX_CONTENT_TOKENS = config["model"].get("num_max_content_tokens", 8192)
NUM_GPU_LAYERS = config["model"].get("num_gpu_layers", 0)
NUM_BUFFER_SIZE = config["model"].get("num_buffer_size", 1)
PROMPT_WITHOUT_RAG = config["model"].get("prompt_without_rag", "")
PROMPT_WITH_RAG = config["model"].get("prompt_with_rag", "")

# RAG related
CHROMA_DB_DIR = config["rag"].get("chroma_db_dir", "chroma_db")
COLLECTION_NAME = config["rag"].get("collection_name", "documents")

# Dataset related
DATASET_NAME = config["dataset"].get("dataset_name", "")
DATASET_PATH = config["dataset"].get("dataset_path", "")
NUM_MAX_LOGS = config["dataset"].get("num_max_logs", 1000)


# Load the dataset.
log_list = get_logs(DATASET_NAME)(DATASET_PATH, NUM_MAX_LOGS)

# Create an instance of LogLLM.
logLLM = LogLLM(
    model_path=MODEL_PATH,
    num_max_content_tokens=NUM_MAX_CONTENT_TOKENS,
    num_gpu_layers=NUM_GPU_LAYERS,
    verbose=VERBOSE,
)

# Load the vector database for RAG.
vectorDatabase = VectorDatabase(db_dir=CHROMA_DB_DIR, collection_name=COLLECTION_NAME)

counter = 1

def chat_to_llm(log):
    global counter

    db_response = vectorDatabase.query(log, n_results=1)

    if db_response != "":
        user_input = {"log": log, "db_response": db_response}
        output = logLLM.chat(PROMPT_WITH_RAG ,user_input)
    else:
        user_input = {"log": log}
        output = logLLM.chat(PROMPT_WITHOUT_RAG, user_input)

    with open("./tools/output.txt", "a") as f:
        f.write(str(counter) + output + "\n")

    counter += 1


if __name__ == "__main__":
    start_time = time.time()

    # Start the main loop.
    try:
        for new_log in log_list:
            chat_to_llm(new_log)
    except KeyboardInterrupt:
        print("STOP")

    end_time = time.time()

    print("Time cost: ", end_time - start_time)