"""main.py

Talk with Deepseek R1 in loop and record the key response.
@author: qi7876
"""

import r1
import sys

sys.path.append("../")

from detection.dataset import loadDataset
import json

log_list = loadDataset("../dataset/BGL/BGL_2k.log", 2000)

counter = 1

for i in log_list:
    response = r1.api_request(i)
    message = response.choices[0].message

    simple_message_dict = {"content": message.content, "reasoning_content": message.reasoning_content}
    parsed_json = json.dumps(simple_message_dict)

    with open(f"./responses/{counter}.json", 'w') as file:
        json.dump(parsed_json, file, indent=4)

    with open("./output.txt", 'a') as file:
        file.write(str(counter) + message.content + "\n")
    
    counter += 1