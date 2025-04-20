import chromadb
import os
import shutil
from datetime import datetime
import uuid


def create_rag_dataset(origin_dataset_path, processed_dataset_path):
    with open(origin_dataset_path, "r") as o, open(processed_dataset_path, "w") as p:
        for line in o:
            if line.startswith("- "):
                p.write("This is a normal log: " + line[2:])
            else:
                first_space_index = line.find(' ')
                p.write(
                    "This is a log with an anomaly " + line[:first_space_index] + ": " + line[first_space_index + 1:])
    return 0


if __name__ == "__main__":
    line_num_file = "./line_num.txt"
    original_dataset = "../dataset/BGL/BGL_2k.log"
    new_dataset = "./BGL_extracted.log"
    processed = "rag_dataset.txt"

    with open(line_num_file, 'r') as line_num, open(original_dataset, 'r') as dataset:
        # Read all line numbers into a list and convert to integers
        line_numbers = [int(line.strip()) for line in line_num if line.strip()]

        # Read all lines from the original dataset
        all_lines = dataset.readlines()

        # Extract the lines corresponding to the line numbers
        # Note: Line numbers typically start from 1, while list indices start from 0
        extracted_lines = [all_lines[num - 1] for num in line_numbers if 0 < num <= len(all_lines)]

    # Write the extracted lines to a new dataset file
    with open(new_dataset, 'w') as output:
        new_lines = []
        for line in extracted_lines:
            parts = line.split(" ")
            new_line = parts[0] + " " + " ".join(parts[6:])
            if new_line not in new_lines:
                output.write(new_line)
                new_lines.append(new_line)

    print(f"Extraction complete. {len(extracted_lines)} lines extracted to {new_dataset}")

    create_rag_dataset(new_dataset, processed)
