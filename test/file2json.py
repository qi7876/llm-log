import json

with open('output.json', 'r') as file:
    message = json.loads(json.load(file))

print(message.get("content"))
print(message.get("reasoning_content"))