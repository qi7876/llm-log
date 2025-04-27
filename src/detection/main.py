"""main.py

This module is used to test the LogLLM.
@author: qi7876
"""

from model import LogLLM
from database import VectorDatabase
import tomli
import time

with open("./configs/config_phi-4_liberty2.toml", "rb") as file:
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

def get_logs(dataset_name: str, dataset_path: str):
    logs = []
    with open(dataset_path, "r") as dataset:
        for line in dataset:
            parts = line.split(" ")
            if dataset_name == "BGL":
                new_line = " ".join(parts[6:])
            elif dataset_name == "Thunderbird":
                new_line = " ".join(parts[7:])
            elif dataset_name == "liberty2":
                new_line = " ".join(parts[8:])
            logs.append(new_line)
    return logs

def pretty_db_response(response):
    if response.startswith("- "):
        response = "This is a normal log: " + response[2:]
    else:
        first_space_index = response.find(' ')
        response = "This is a anomaly log: " + response[first_space_index + 1:]
    return response

# Load the dataset.
log_list = get_logs(DATASET_NAME, DATASET_PATH)

# Create an instance of LogLLM.
logLLM = LogLLM(
    model_path=MODEL_PATH,
    num_max_content_tokens=NUM_MAX_CONTENT_TOKENS,
    num_gpu_layers=NUM_GPU_LAYERS,
    verbose=VERBOSE,
)

# Load the vector database for RAG.
vectorDatabase = VectorDatabase(db_dir=CHROMA_DB_DIR, collection_name=COLLECTION_NAME)


if __name__ == "__main__":
    start_time = time.time()

    # Start the main loop.
    try:
        for log in log_list:
            counter = 1
            db_response = vectorDatabase.query(log, n_results=1)
            db_response = pretty_db_response(db_response)

            if db_response != "":
                user_input = {"log": log, "db_response": db_response}
                output = logLLM.chat(PROMPT_WITH_RAG, user_input)
            else:
                user_input = {"log": log}
                output = logLLM.chat(PROMPT_WITHOUT_RAG, user_input)

            with open("./tools/output.txt", "a") as f:
                f.write(str(counter) + output + "\n")

            counter += 1
    except KeyboardInterrupt:
        print("STOP")
    except Exception as e:
        print("Unexpected error: ", e)

    end_time = time.time()

    print("Time cost: ", end_time - start_time)