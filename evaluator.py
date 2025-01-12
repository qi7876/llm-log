
import os
import logging
from multiprocessing import Queue
import queue
import signal

class Eval:
    def __init__(self):
        self.TP,self.FP,self.TN,self.FN,self.all_count = 0,0,0,0,0
      
    def eval(self, predict, true_label):
        self.all_count += 1
        match [predict,true_label]:
            case ['1',1]:
                self.TP += 1
            case ['1',0]:
                self.FP += 1
            case ['0',0]:
                self.TN += 1
            case ['0',1]:
                self.FN += 1
            case _:
                raise Exception(f"Unkown Result Pair:predict={predict}, true={true_label}")
    def print(self):
        print(f"FN:{self.FN} FP:{self.FP} TN:{self.TN} TP:{self.TP}")
        if self.TP != 0:
            precision = self.TP/float(self.TP+self.FP)
            recall = self.TP/float(self.TP+self.FN)
            f1 = 2*precision*recall/float(precision+recall)
            print(f"Precision:{precision}, Recall:{recall}, F1:{f1}")


def run_evaluate(q: Queue, wait_time:int, event):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    eval = Eval()
    while(True):
        try:
            res_dic = q.get(timeout=wait_time)
            res = res_dic['res']
            log = res_dic['log']
            label = log['label']
            eval.eval(res, label)
            eval.print()
        except queue.Empty:
            logging.info(f"Process{os.getpid()} is empty.")
        else:
            continue
        if event.is_set():
            logging.info(f"Process{os.getpid()} exits.")
            break