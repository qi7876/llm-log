"""dataset.py

This module is used to load the log dataset.
@author: qi7876
"""

def loadDataset(dataset_path, num_max_logs) -> list[str]:
    log_list = []
    log_count = 0
    with open(dataset_path, "r") as file:
        for line in file:
            if log_count >= num_max_logs:
                break
            if line.startswith("- "):
                line = line[2:]
            log_list.append(line)
            log_count += 1

    return log_list
