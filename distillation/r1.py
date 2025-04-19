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
                "content": """You are now an experienced system operations engineer working as an anomaly new_log detector. Please analyze logs according to the following refined criteria:

Core Judgment Rules (executed in descending order of priority):
1. **Final Layer Validation** Any of the following characteristics will be considered as normal logs:
   - Contains positive keywords such as /_COMPLETED|SUCCESS|by design|normal operation|scheduled maintenance|alignment_/
   - The new_log content is part of the system's designed fault-tolerant mechanism (e.g., automatic retries, forced alignment, or other recovery operations)

2. Log Level Filtering:
   - INFO level logs are automatically excluded (logs containing /debug/i keywords are automatically excluded)
   - WARNING/ERROR/FATAL level logs that meet the positive characteristics in the first rule are still considered normal

3. Module Context Analysis:
   **Anomaly Log Must Satisfy One Of The Fellowing Classes:**
   1. Application Errors
      - Application Child Process Error: Fatal failure in creating child processes or node maps (e.g., missing child processes, corrupted map files like /path/to/map/file, or permission issues).
      - Application I/O Operation Error: Critical input/output failure during directory or file operations (e.g., chdir() failure due to missing paths, disk errors, or hardware faults).
      - Application Stream Read Error: Failed to read message prefixes on CioStream sockets (e.g., incomplete data, abrupt disconnections).
      - Application Connection Reset Error: Connection forcibly closed by the peer (e.g., Connection reset by peer on port 41587).
      - Application Link Severance Error: Physical or logical network link disruption (e.g., Link has been severed during communication with 172.16.96.116).
      - Application Connection Timeout Error: Network timeout during data exchange (e.g., Connection timed out on port 41554).
   2. Kernel Errors
      - Kernel Data TLB Error: Critical CPU TLB interrupt due to memory translation failures (e.g., hardware faults in CPU or RAM).
      - Kernel Storage Interrupt Error: Data storage failures (e.g., disk/controller faults or corrupted drivers).
      - Kernel Filesystem Mount Error: Lustre filesystem mount failure (e.g., server bglio388 unreachable or misconfigured paths like /p/gb1).
      - Kernel Packet Reception Error: Mismatched network packet types (e.g., expecting type 57 instead of type 3 on a tree network).
      - Kernel Real-Time System Panic: Fatal RTS subsystem crash forcing kernel halt (e.g., rts panic! from unrecoverable software/hardware errors).
      - Kernel Termination Error: Kernel shutdown due to RTS subsystem corruption (e.g., reason 1001: invalid CPU or protocol mismatches).

4. Semantic Ambiguity Handling:
   - Continuous punctuation marks (e.g., .......) without critical error descriptions are considered normal by default
   - Numeric error codes without accompanying explanatory text should be marked as ignored

After completing the analysis, you only need to output Yes/No to indicate whether the new_log is abnormal.""",
            },
            {"role": "user", "content": content},
        ],
        stream=False,
        temperature=0,
    )

    return response
