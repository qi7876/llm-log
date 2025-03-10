"""r1.py

This module is used to talk with Deepseek R1 via API.
@author: qi7876
"""

from openai import OpenAI
import os


def api_request(content):
    api_key = os.environ.get("DEEPSEEK_API_KEY")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {
                "role": "system",
                "content": """You are now an experienced system operations engineer working as an anomaly log detector. Please analyze logs according to the following refined criteria:

Core Judgment Rules (executed in descending order of priority):
1. **Final Layer Validation** Any of the following characteristics will be considered as normal logs:
   - Contains positive keywords such as /_COMPLETED|SUCCESS|by design|normal operation|scheduled maintenance|alignment_/
   - The log content is part of the system's designed fault-tolerant mechanism (e.g., automatic retries, forced alignment, or other recovery operations)

2. Log Level Filtering:
   - INFO level logs are automatically excluded (logs containing /debug/i keywords are automatically excluded)
   - WARNING/ERROR level logs that meet the positive characteristics in the first rule are still considered normal

3. Module Context Analysis:
   ▸ Special rules for the RAS_KERNEL module: The following are considered expected behaviors:
   ▸ Debug operations related to address alignment
   ▸ Preventive logs triggered by hardware fault-tolerant mechanisms
   ▸ Standardized error codes with numbers (e.g., EC 0x0000)

4. Anomaly Judgment Must Satisfy:
   - Causes substantial system/service interruption (leading to service stoppage or degradation)
   - Contains a clear description of the error result (e.g., aborted/crash/out of memory)
   - Does not fall under the aforementioned positive characteristics

5. Semantic Ambiguity Handling:
   - Continuous punctuation marks (e.g., .......) without critical error descriptions are considered normal by default
   - Numeric error codes without accompanying explanatory text should be marked as ignored

After completing the analysis, you only need to output Yes/No to indicate whether the log is abnormal.""",
            },
            {"role": "user", "content": content},
        ],
        stream=False,
        temperature=0,
    )

    return response
