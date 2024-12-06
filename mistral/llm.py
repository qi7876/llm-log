import requests
import json


def interactive(url: str, messages: list[dict[str]], options: dict, model: str = 'mistral', raw: bool = False,
                stream: bool = False) -> str:
    data = {'model': model, 'messages': messages, 'raw': raw, 'stream': stream, 'options': options}
    data_json = json.dumps(data)
    print(f"senddata:{data_json}")
    resp_json = requests.post(url, data=data_json, headers={'Content-Type': 'application/json'})
    if resp_json.status_code == 200:
        print('Respond successfully!The answer is below:')
        resp = json.loads(resp_json.text)
        print(resp["message"]["content"])
        return resp["message"]["content"]
    else:
        raise Exception("Respond failed.")

