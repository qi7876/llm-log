def filter_lines(input_file, output_file):
    try:
        with (
            open(input_file, "r", encoding="utf-8") as infile,
            open(output_file, "w", encoding="utf-8") as outfile,
        ):
            for line in infile:
                if "-" in line or "+" in line:
                    outfile.write(line)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


input_file = "../prototype/output.txt"
output_file = "output.txt"
filter_lines(input_file, output_file)
