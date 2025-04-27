line_num_file = "./line_num.txt"
original_dataset = "../../../dataset/liberty2/liberty2.sub.key_event"
original_dataset_name = "liberty2" # BGL, Thunderbird, liberty2
new_dataset = "./extracted.log"

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
        if original_dataset_name == "BGL":
            new_line = parts[0] + " " + " ".join(parts[6:])
        elif original_dataset_name == "Thunderbird":
            new_line = parts[0] + " " + " ".join(parts[7:])
        elif original_dataset_name == "liberty2":
            new_line = parts[0] + " " + " ".join(parts[8:])
        else:
            new_line = line

        # Deduplicate
        if new_line not in new_lines:
            output.write(new_line)
            new_lines.append(new_line)

print(f"Extraction complete. {len(new_lines)} lines extracted to {new_dataset}")
