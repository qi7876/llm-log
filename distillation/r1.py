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
                "content": """You are now an anomaly log detector. Next, I will input a log to you each time, and you need to analyze it from multiple aspects to determine whether this log describes an abnormal event.

Your analysis process must follow the following rules:
1. Logs at the INFO level definitely do not describe an abnormal event, while logs at other levels may describe an abnormal event;
2. The module name in the system will appear in the log header, indicating that it is a log from the corresponding module. You can analyze the log in conjunction with the module;
3. Abnormal events include but are not limited to the following categories: storage errors, mount failures, network errors, service unavailability, kernel termination, system crashes, insufficient memory, CPU overload, disk I/O bottlenecks, application crashes, service interruptions, configuration errors, permission issues, time anomalies, container/virtualization anomalies;
4. Logs that are extremely semantically ambiguous and cannot be further analyzed can be ignored.

After completing the analysis, you only need to output Yes/No to indicate whether the log is abnormal.""",
            },
            {"role": "user", "content": content},
        ],
        stream=False,
        temperature=0,
    )

    return response
