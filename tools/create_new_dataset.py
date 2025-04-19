line_num_file = "./line_num.txt"
original_dataset = "../dataset/BGL/BGL_2k.new_log"
new_dataset = "../dataset/BGL/BGL_extracted.new_log"  # Path for the new dataset

with open(line_num_file, 'r') as line_num, open(original_dataset, 'r') as dataset:
    # Read all line numbers into a list and convert to integers
    line_numbers = [int(line.strip()) for line in line_num if line.strip()]
    
    # Read all lines from the original dataset
    all_lines = dataset.readlines()
    
    # Extract the lines corresponding to the line numbers
    # Note: Line numbers typically start from 1, while list indices start from 0
    extracted_lines = [all_lines[num-1] for num in line_numbers if 0 < num <= len(all_lines)]

# Write the extracted lines to a new dataset file
with open(new_dataset, 'w') as output:
    output.writelines(extracted_lines)

print(f"Extraction complete. {len(extracted_lines)} lines extracted to {new_dataset}")

