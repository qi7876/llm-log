
import os
import logging
import signal
from datasets import load_dataset


def sender(q, event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    dataset = load_dataset("dataset/BGL_2k", split = "test")
    for data in dataset:
        if event.is_set():
            logging.info(f"Process{os.getpid()} exits.")
            break
        q.put({'raw_log':data['log'], 'label':data['label']})