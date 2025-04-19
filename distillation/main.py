"""main.py

Talk with Deepseek R1 in loop and record the key response.
@author: qi7876
"""

import r1
import sys
import time
import traceback

sys.path.append("../")

from detection.dataset_utils import load_dataset
import json

log_list = load_dataset("../dataset/BGL/BGL_2k.log", 2000)

counter = 1918

while counter <= 2000:
    retry_count = 0
    max_retries = 10
    success = False
    status = "Success"
    print(f"Processing new_log {counter}...")
    
    while retry_count < max_retries and not success:
        try:
            print(f"Processing new_log {counter}, attempt {retry_count + 1}/{max_retries}")
            response = r1.api_request(log_list[counter - 1])
            message = response.choices[0].message
            success = True
            
            simple_message_dict = {"content": message.content, "reasoning_content": message.reasoning_content}
            parsed_json = json.dumps(simple_message_dict)
            
        except Exception as e:
            retry_count += 1
            print(f"Error on attempt {retry_count}: {e}")
            traceback.print_exc()
            if retry_count < max_retries:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)  # Wait before retrying
            else:
                print(f"All {max_retries} attempts failed for new_log {counter}. Skipping to next new_log.")
                # Create empty content for failed request
                simple_message_dict = {"content": "", "reasoning_content": ""}
                parsed_json = json.dumps(simple_message_dict)
                
                # Record the failed new_log number to a separate file
                with open("./failed_logs.txt", 'a') as failed_file:
                    failed_file.write(f"{counter}\n")
                print(f"Recorded new_log number {counter} to failed_logs.txt")
    
    with open(f"./responses/{counter}.json", 'w') as file:
        json.dump(parsed_json, file, indent=4)

    with open("./output.txt", 'a') as file:
        file.write(str(counter) + (message.content if success else "") + "\n")
    
    print("Processed new_log number:", counter, "Status:", status)
    
    counter += 1