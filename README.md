# 🤖llm-log

> [!IMPORTANT]
> This project is still in development and is not yet ready for use.

## 📄Introduction

We want to apply LLM to anomaly detection in logs.

## ⚙️Config

```toml
[DEBUG]
VERBOSE = false

[MODEL]
MODEL_PATH = "/path/to/your/gguf_model"
NUM_MAX_CONTENT_TOKENS = 8192
TEMPLATE = "According to the detailed prompt template provided by GGUF model. Use {history} in your history prompt part to load conversation history."
NUM_GPU_LAYERS = 0
NUM_BUFFER_SIZE = 1

[DATASET]
DATASET_PATH = "/path/to/your/dataset"
NUM_MAX_LOGS = 2000
```

