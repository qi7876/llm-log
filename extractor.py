
import os
import logging
import queue
import signal

def run_extract(output_q, res_q, event, wait_time:int):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while(True):
        try:
            output_dic = output_q.get(timeout=wait_time)
            output = output_dic['output']
            log = output_dic['log']
            res = default_extract(output)
            res_q.put({"log":log, 'res':res})
        except queue.Empty:
            logging.info(f"Process{os.getpid()} is empty.")
        else:
            continue
        if event.is_set():
            logging.info(f"Process{os.getpid()} exits.")
            break

def default_extract(str):
    idx1 = str.find('<|assistant|>')
    idx2 = str.find('<',idx1+13)
    idx3 = str.find('>', idx2+1)
    if (idx1!= -1 and idx2 != -1 and idx3 != -1):
        res = str[idx2+1:idx3].strip()
        return res
    else:
        raise Exception(f"Incorrect output pattern:{str}")