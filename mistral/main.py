import json

import llm
import configparser
import message
import  evaluate
config = configparser.ConfigParser()
config.read('proj.config', 'utf-8')
URL = config['OLLAMA']['URL']
BGL_LOG_PATH = config['DATASET']['BGL']
BGL_PATTERN = config['LOG_PATTERN']['BGL_PATTERN']
OUTPUT_PATTERN = config['LLM_OUTPUT_PATTERN']['MISTRAL_PATTERN']
PROMPT = config['PROMPT']['PROMPT_SIMPLE']
with open(config['FILE']['MISTRAL']) as f:
    OPTIONS = json.load(f)
# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # llm.interactive(URL, [{"role": "user", "content": "hello!"}])

    log_list = message.read_log_file(BGL_LOG_PATH)
    dataset = message.convert_log_list_to_dataset(log_list, 0, "BGL", BGL_PATTERN)
    evl = evaluate.Evaluate()
    for i in range(0, 100):
        data = dataset.get_random_test_data()
        print("label:"+data[0])
        print(f"log:{data[1]}")
        message_cls = message.Message()
        message_cls.add_system_message(PROMPT)
        message_cls.add_user_message(
            "1117869872 2005.06.04 R04-M1-N4-I:J18-U11 2005-06-04-00.24.32.432192 R04-M1-N4-I:J18-U11 " +
            "RAS APP FATAL ciod: failed to read message prefix on control stream (CioStream socket to 172.16.96.116:33569")
        message_cls.add_assistant_message("##yes## Because system read message prefix on control stream.")
        message_cls.add_user_message(data[1])
        output = llm.interactive(URL, message_cls.get_message_obj_origin(), OPTIONS)
        evl.extract_and_record_result(data[0], output, 'BGL', OUTPUT_PATTERN)
    evl.get_evaluate_data()
