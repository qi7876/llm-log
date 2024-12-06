import re
import json
import random

def read_log_file(path: str) -> list[str]:
    logs = []
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            logs.append(line.strip())
    return logs


class DataSet:
    def __init__(self):
        self._train_data_list = []
        self._test_data_list = []

    def put_train_data(self, label, message_content):
        self._train_data_list.append([label, message_content])

    def put_test_data(self, label, message_content):
        self._test_data_list.append([label, message_content])

    def get_random_test_data(self):
        return random.choice(self._test_data_list)




def convert_log_list_to_dataset(logs: list[str], threshold: float, log_type: str, pattern: str) -> DataSet:
    if logs.__len__() != 0:
        if threshold > 1 or threshold < 0:
            raise Exception("threshold must be a float number between 0 and 1.")
        match log_type:
            case "BGL":
                BGL_dataset = DataSet()
                length = logs.__len__()
                bound_idx = int(length * threshold)
                for idx in range(0, bound_idx):
                    log = logs[idx]
                    if re.search(pattern, log):
                        res = re.match(pattern, log).groups()
                        BGL_dataset.put_train_data(res[0], res[1])
                    else:
                        raise Exception("log pattern doesn't match.")
                for idx in range(bound_idx, length):
                    log = logs[idx]
                    if re.search(pattern, log):
                        res = re.match(pattern, log).groups()
                        BGL_dataset.put_test_data(res[0], res[1])
                    else:
                        raise Exception("log pattern doesn't match.")
                return BGL_dataset
            case _:
                raise Exception("can't cope with this log type ")


class Message:
    def __init__(self):
        self._message_list = []

    def add_user_message(self, message: str):
        self._message_list.append({"role": "user", "content": message})

    def add_system_message(self, message: str):
        self._message_list.append({"role": "system", "content": message})

    def add_assistant_message(self, message: str):
        self._message_list.append({"role": "assistant", "content": message})

    def get_message_json(self) -> str:
        return json.dumps(self._message_list)

    def get_message_obj_origin(self) -> list[dict]:
        return self._message_list

    def get_message_obj_copy(self) -> list[dict]:
        return self._message_list.copy()
