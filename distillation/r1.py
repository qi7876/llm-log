"""r1.py

This module is used to talk with Deepseek R1 via API.
@author: qi7876
"""

from openai import OpenAI
import os

def api_request(content):

    api_key = os.environ.get("DEEPSEEK_API_KEY")

    client = OpenAI(
        api_key=api_key, base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {
                "role": "system",
                "content": "You are an anomaly detector in a log system. Analyze each log entry as normal or abnormal. You must choose one word in 'yes' or 'no' as the result. 'yes' means something failed or interrupted now. 'no' describes a process work normally or a normal state. New logs are as follows:",
            },
            {"role": "user", "content": content},
        ],
        stream=False,
    )

    return response