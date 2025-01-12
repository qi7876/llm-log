import signal
import sys
import os
import queue
import configparser
import extractor
import logging
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from transformers import TextIteratorStreamer
from datasets import load_dataset
from multiprocessing import Queue
from multiprocessing import Process
import configparser

def run_detector(log_q, output_q, config:configparser.ConfigParser, event, wait_time:int):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    PROMPT = config["LLM"]["PROMPT"]
    model, tokenizer = load_model()
    while(True):
        try:
            log = log_q.get(timeout=wait_time)
            output = generate(PROMPT, log['raw_log'], tokenizer, model)
            output_q.put({"log":log, "output": output})
        except queue.Empty:
            logging.info(f"Process{os.getpid()} is empty.")
        else:
            continue
        
        if event.is_set():
            logging.info(f"Process{os.getpid()} exits.")
            break


def load_model(max_seq_length=2048, dtype=None, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Phi-3.5-mini-instruct", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit
    )
    FastLanguageModel.for_inference(model) 
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = 'phi-3', # You must provide a template and EOS token
)
    return model, tokenizer

def generate(prompt:str, log:str, tokenizer, model)->str:
    message = [{'role':'system','content':prompt},{'role':'user', 'content':log}]
    inputs = tokenizer.apply_chat_template(
        message,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")
    outputs = model.generate(input_ids=inputs, max_new_tokens = 128, use_cache = True)
    resps = tokenizer.batch_decode(outputs)
    return resps[0]
