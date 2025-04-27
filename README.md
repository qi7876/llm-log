# 🤖llm-log

> Still WIP. Be careful of breaking changes.

## 📄Introduction

We want to apply LLM to anomaly detection in logs.

## 📇Project Structure

```
.
├── LICENSE
├── README.md
├── dataset
│         └── ...
├── misc
│         └── ...
├── models
│         ├── phi-4-Q6_K.gguf
│         └── ...
└── src
    ├── chroma_db
    │         └── ...
    ├── configs
    │         ├── config_phi-4_BGL_test.toml
    │         ├── config_phi-4_liberty2_test.toml
    │         └── ...
    ├── rag_dataset
    │         ├── phi-4_BGL_test.txt
    │         ├── phi-4_liberty2_test.txt
    │         └── ...
    ├── results
    │         ├── extracted_dataset.txt
    │         ├── line_num.txt
    │         └── output.txt
    ├── main.py
    ├── model.py
    ├── my_utils.py
    └── vector_database.py
```

## ⚙️Config Example

```toml
[debug]
verbose = false

[model]
model_path = "path/to/model.gguf"
num_max_content_tokens = 8192
num_gpu_layers = 41
num_buffer_size = 1
prompt_without_rag = """..."""
prompt_with_rag = """..."""

[rag]
chroma_db_dir = "chroma_db"
collection_name = "documents"

[dataset]
dataset_name = "liberty2/BGL/Thunderbird"
dataset_path = "path/to/dataset"
```

