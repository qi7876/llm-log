from distillation import r1
import sys
import json

sys.path.append("../")

from detection import dataset_utils

log_list = dataset.load_dataset("../dataset/BGL/BGL_2k.new_log", 2000)

response = r1.api_request("New logs are as follows:\n" + log_list[0])

message = response.choices[0].message

print(message.reasoning_content, "\n")
print(message.content, "\n")

api_log = {"content": message.content, "reasoning_content": message.reasoning_content}
parsed_json = json.dumps(api_log)

with open('output.json', 'w') as outfile:
    json.dump(parsed_json, outfile, indent=4)