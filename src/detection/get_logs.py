"""get_logs.py

This module is used to load the new_log dataset.
@author: qi7876
"""

def get_logs(dataset_name: str):
    if dataset_name == "BGL":
        def load_and_preprocess_dataset(dataset_path, num_max_logs) -> list[str]:
            log_list = []
            log_count = 0
            with open(dataset_path, "r") as file:
                for line in file:
                    if log_count >= num_max_logs:
                        break

                    parts = line.split(" ")
                    new_line = " ".join(parts[6:])

                    log_list.append(new_line)
                    log_count += 1

            return log_list

    elif dataset_name == "Thunderbird":
        def load_and_preprocess_dataset(dataset_path, num_max_logs) -> list[str]:
            log_list = []
            log_count = 0
            with open(dataset_path, "r") as file:
                for line in file:
                    if log_count >= num_max_logs:
                        break

                    parts = line.split(" ")
                    new_line = " ".join(parts[6:])

                    log_list.append(new_line)
                    log_count += 1

            return log_list
    else:
        load_and_preprocess_dataset = None

    return load_and_preprocess_dataset