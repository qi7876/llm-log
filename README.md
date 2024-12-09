# 🤖llm-log

> [!IMPORTANT]
> This project is still in development and is not yet ready for use.

## 📄Introduction

We want to apply LLM to anomaly detection in logs.

## ⚙️Config

```toml
[MODEL]
MODEL_PATH = "path/to/your/gguf_model"
MAX_CONTENT_WINDOW_SIZE = 4096
TEMPLATE = """
According to the detailed prompt template provided by GGUF model.
Use {history} in your history prompt part to load conversation history.
"""
UPPER_CONTENT_WINDOW_SIZE = 2048
UPPER_OUTPUT_TOKEN_NUM = 1024
GPU_LAYER_NUM = 0
CONVERSATION_ROUND_KEEP_NUM = 1
BUFFER_SIZE = 1

[DATASET]
DATASET_PATH = "path/to/your/dataset"
MAX_LOG_NUM = 2000
```

