"""extract_anomal_logs.py

Extract anomal logs from a dataset.
@author: qi7876
"""

def extractAnomalLogs(dataset_path, extracted_logs_path) -> int:
    with open(dataset_path, "r") as dataset, open(extracted_logs_path, "w") as extracted_file:
        for line in dataset:
            if line.startswith("- "):
                continue
            else:
                extracted_file.write(line)

    return 0

if __name__ == "__main__":
    dataset_path = "../dataset/BGL/BGL_2k.log"
    extracted_logs_path = "extracted_logs.txt"
    extractAnomalLogs(dataset_path, extracted_logs_path)