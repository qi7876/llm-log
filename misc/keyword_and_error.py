total_line = 0
keyword_counter = 0
keyword_normal = 0
keyword_anomal = 0

keyword = "exception"

with open("../dataset/BGL/BGL.log", 'r', encoding="utf-8") as file:
    for line in file:
        total_line += 1
        line = line.strip()
        if keyword in line:
            keyword_counter += 1
            if line.startswith("- "):
                keyword_normal += 1
            else:
                keyword_anomal += 1

print("Keyword:", keyword)
print("The number of total logs:", total_line)
print(f"The number of logs that contain \"{keyword}\":", keyword_counter)
print(f"Contains \"{keyword}\" and normal:", keyword_normal)
print(f"Contains \"{keyword}\" and anomal:", keyword_anomal)
print("=====================================================================")
print(f"The percentage of normal logs containing \"{keyword}\":", keyword_normal / keyword_counter)