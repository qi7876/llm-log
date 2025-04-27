"""main.py

This module is used to test the LogLLM.
@author: qi7876
"""

from model import LogLLM
from vector_database import VectorDatabase
import my_utils
import tomli
import time

with open("configs/config_phi-4_BGL_test.toml", "rb") as file:
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

# misc
PREDICTION_FILE_PATH = "results/output.txt"
LINE_NUM_FILE_PATH = "results/line_num.txt"
EXTRACTED_DATASET_PATH = "results/extracted_dataset.txt"
RAG_DATASET_PATH = "rag_dataset/phi-4_BGL_test.txt"


def get_logs(dataset_name: str, dataset_path: str):
    logs = []
    with open(dataset_path, "r") as raw_dataset:
        for line in raw_dataset:
            log_parts = line.split(" ")
            if dataset_name == "BGL":
                processed_log = " ".join(log_parts[6:])
            elif dataset_name == "Thunderbird":
                processed_log = " ".join(log_parts[7:])
            elif dataset_name == "liberty2":
                processed_log = " ".join(log_parts[8:])
            logs.append(processed_log)
    return logs


def pretty_db_response(response):
    if response.startswith("- "):
        response = "This is a normal log: " + response[2:]
    else:
        first_space_index = response.find(' ')
        response = "This is a anomaly log: " + response[first_space_index + 1:]
    return response


if __name__ == "__main__":
    my_utils.empty_directory_pathlib(CHROMA_DB_DIR)
    my_utils.create_database(RAG_DATASET_PATH, CHROMA_DB_DIR, COLLECTION_NAME)

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

    my_utils.empty_directory_pathlib("../results")
    start_time = time.time()

    # Start the main loop.
    try:
        counter = 1
        for log in log_list:
            db_response = vectorDatabase.query(log, n_results=1)
            db_response = pretty_db_response(db_response)

            if db_response != "":
                user_input = {"log": log, "db_response": db_response}
                output = logLLM.chat(PROMPT_WITH_RAG, user_input)
            else:
                user_input = {"log": log}
                output = logLLM.chat(PROMPT_WITHOUT_RAG, user_input)

            with open(PREDICTION_FILE_PATH, "a") as f:
                f.write(str(counter) + output + "\n")

            counter += 1
    except KeyboardInterrupt:
        print("STOP")
    except Exception as e:
        print("Unexpected error: ", e)

    end_time = time.time()

    print("Time cost: ", end_time - start_time)

    results = my_utils.calculate_accuracy_and_recall(DATASET_PATH, PREDICTION_FILE_PATH, LINE_NUM_FILE_PATH)

    if results:
        print("\nResults:")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print("\nDetailed Statistics:")
        print(f"True Positives: {results['true_positives']}")
        print(f"True Negatives: {results['true_negatives']}")
        print(f"False Positives: {results['false_positives']}")
        print(f"False Negatives: {results['false_negatives']}")
        if results['invalid_lines'] > 0:
            print(f"Invalid Lines: {results['invalid_lines']}")
    else:
        print("Evaluation failed. Please check the errors above.")

    with open(LINE_NUM_FILE_PATH, 'r') as line_num, open(DATASET_PATH, 'r') as dataset:
        # Read all line numbers into a list and convert to integers
        line_numbers = [int(line.strip()) for line in line_num if line.strip()]

        # Read all lines from the original dataset
        all_lines = dataset.readlines()

        # Extract the lines corresponding to the line numbers
        # Note: Line numbers typically start from 1, while list indices start from 0
        extracted_lines = [all_lines[num - 1] for num in line_numbers if 0 < num <= len(all_lines)]

    # Write the extracted lines to a new dataset file
    with open(EXTRACTED_DATASET_PATH, 'w') as output:
        new_lines = []
        for line in extracted_lines:
            parts = line.split(" ")
            if DATASET_NAME == "BGL":
                new_line = parts[0] + " " + " ".join(parts[6:])
            elif DATASET_NAME == "Thunderbird":
                new_line = parts[0] + " " + " ".join(parts[7:])
            elif DATASET_NAME == "liberty2":
                new_line = parts[0] + " " + " ".join(parts[8:])
            else:
                new_line = line

            # Deduplicate
            if new_line not in new_lines:
                output.write(new_line)
                new_lines.append(new_line)

    print(f"Extraction complete. {len(new_lines)} lines extracted to {EXTRACTED_DATASET_PATH}")
