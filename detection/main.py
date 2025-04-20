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

prompt_without_rag = """<|im_start|>system<|im_sep|>You are an anomaly log detector. Please analyze logs according to the following refined criteria:

1. Log Level Filtering:
  - INFO level logs or logs with "debug" marked are normal.
  - WARNING/ERROR/FATAL level logs maybe abnormal. You should analyze the log according to the third part.
   
2. Special cases:
  **The Cases Below should not be detected to anomaly log.**
  - RAS APP FATAL ciod: Error loading PATH: invalid or missing program image, Permission denied.
  - RAS APP FATAL ciod: Error loading PATH: invalid or missing program image, No such file or directory

3. Module Context Analysis:
  **Anomaly Log Must Satisfy One Of The Following Classes:**
  1. Application Errors(RAS APP FATAL):
    - Application Child Process Error: There is no child processes when creating node map.
    - Application I/O Operation Error: Input/output error in chdir.
    - Application Stream Read Error: Failed to read message prefix.
    - Application Connection Reset Error: Connection reset by peer when reading message prefix.
    - Application Link Severance Error: Link has been severed when reading message prefix.
    - Application Connection Timeout Error: Connection timed out when reading message prefix.
  2. Kernel Errors(RAS KERNEL FATAL):
    - Kernel Data TLB Error: data TLB error interrupt.
    - Kernel Storage Error: data storage interrupt.
    - Kernel Filesystem Mount Error: Lustre mount FAILED.
    - Kernel Packet Reception Error: Error receiving packet on tree network, type mismatch.
    - Kernel Real-Time System Panic: rts panic.
    - Kernel Termination Error: Kernel terminated for some reason.

After completing the analysis, you only need to output Yes/No to indicate whether the log is abnormal. **DO NOT OUTPUT THE ANALYSIS.**<|im_end|><|im_start|>user<|im_sep|>New logs are as follows:
{log}<|im_end|><|im_start|>assistant<|im_sep|>"""
prompt_with_rag = """<|im_start|>system<|im_sep|>You are an anomaly log detector. Please analyze logs according to the following refined criteria:

1. Log Level Filtering:
  - INFO level logs or logs with "debug" marked are normal.
  - WARNING/ERROR/FATAL level logs maybe abnormal. You should analyze the log according to the third part.
   
2. Special cases:
  **The Cases Below should not be detected to anomaly log.**
  - RAS APP FATAL ciod: Error loading PATH: invalid or missing program image, Permission denied.
  - RAS APP FATAL ciod: Error loading PATH: invalid or missing program image, No such file or directory

3. Module Context Analysis:
  **Anomaly Log Must Satisfy One Of The Following Classes:**
  1. Application Errors(RAS APP FATAL):
    - Application Child Process Error: There is no child processes when creating node map.
    - Application I/O Operation Error: Input/output error in chdir.
    - Application Stream Read Error: Failed to read message prefix.
    - Application Connection Reset Error: Connection reset by peer when reading message prefix.
    - Application Link Severance Error: Link has been severed when reading message prefix.
    - Application Connection Timeout Error: Connection timed out when reading message prefix.
  2. Kernel Errors(RAS KERNEL FATAL):
    - Kernel Data TLB Error: data TLB error interrupt.
    - Kernel Storage Error: data storage interrupt.
    - Kernel Filesystem Mount Error: Lustre mount FAILED.
    - Kernel Packet Reception Error: Error receiving packet on tree network, type mismatch.
    - Kernel Real-Time System Panic: rts panic.
    - Kernel Termination Error: Kernel terminated for some reason.

After completing the analysis, you only need to output Yes/No to indicate whether the log is abnormal. **DO NOT OUTPUT THE ANALYSIS.**<|im_end|><|im_start|>user<|im_sep|>New logs are as follows:
{log}
You must refer to the following ground-truth information for judgment:
{db_response}<|im_end|><|im_start|>assistant<|im_sep|>"""

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

    db_response = vectorDatabase.query(log, n_results=1)

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
