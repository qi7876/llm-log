import logging
import time
import random
import threading
from queue import deque
from llama_cpp import Llama

def create_logger():
    logger = logging.getLogger("LlamaLogger")
    logger.setLevel(logging.DEBUG)
    
    handler = logging.FileHandler("logs.txt")
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

logger = create_logger()

# 创建缓冲区
small_buffer = deque(maxlen=20)  # 小缓冲区，最多存储 20 条日志记录
large_buffer = deque(maxlen=20)  # 大缓冲区，最多存储 1000 条日志记录

# 全局计数器和全局计时器
global_counter = 0  # 记录写入日志的数量
last_check_time = time.time()  # 记录上次检查的时间

# 初始化llama模型
llama_model = Llama(model_path="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf", max_context_tokens=32768)  # 指定 Llama 模型的路径
prompt = "The logs are below. You should analyze which log is anomaly and return it to me."  # llama 模型的提示词

log_levels = [logging.INFO, logging.WARNING, logging.ERROR]  # 定义三种日志级别

def log_message():
    global global_counter
    
    # 生成随机日志消息
    message = f"Test log message {random.randint(1, 100)}"
    level = random.choice(log_levels)
    
    logger.log(level, message)
    print(f"Logging message: {message} with level: {level}")
    
    # 写入小缓冲区并更新大缓冲区
    small_buffer.append((level, message))  # 将日志消息加入小缓冲区
    print(f"Small buffer size: {len(small_buffer)}")
    
    # 如果小缓冲区满了，将其内容移至大缓冲区
    if len(small_buffer) == small_buffer.maxlen:
        print("Small buffer is full. Moving logs to large buffer.")  # Debug: 小缓冲区已满
        while small_buffer:
            log = small_buffer.popleft()
            large_buffer.append(log)
        print(f"Large buffer size after moving: {len(large_buffer)}")  # Debug: 输出大缓冲区的大小
    
    # 更新全局计数器
    global_counter += 1
    print(f"Global counter updated to: {global_counter}")

# llama模型调用函数
def llama_process():
    global global_counter, last_check_time
    
    while True:
        current_time = time.time()
        
        # 检查条件：如果计数器达到小缓冲区大小，或每 3 秒检查一次且缓冲区内有日志
        if global_counter >= 20 or (current_time - last_check_time >= 3 and global_counter > 0):
            print("Condition met for llama processing.")  # Debug: 输出满足调用llama模型的条件
            
            # 将小缓冲区和大缓冲区内容整合为字符串
            logs_text = "\n".join([f"{level} - {msg}" for level, msg in list(small_buffer) + list(large_buffer)])
            input_text = f"{prompt}\n{logs_text}"
            
            # 调用llama模型并打印结果
            print("Calling llama model with current logs.")
            output = llama_model(input_text)
            print(f"Llama model output: {output}")  # Debug: 输出llama模型的结果
            
            # 将输出写入文本文件 "llama_output.txt"
            output_text = output['choices'][0]['text']
            with open("llama_output.txt", "a") as f:
                f.write("========================\n" + output_text + "\n")
            
            global_counter = 0
            last_check_time = current_time
            print("Global counter and last check time reset.")
        
        time.sleep(0.5)

llama_thread = threading.Thread(target=llama_process, daemon=True)
llama_thread.start()
print("Llama processing thread started.")

# 模拟日志写入
try:
    while True:
        log_message()
        time.sleep(random.uniform(0.1, 0.5))
except KeyboardInterrupt:
    print("Logging stopped.")