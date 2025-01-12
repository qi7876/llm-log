import multiprocessing
import signal
import sys
import configparser
import traceback
import detector
import extractor
import evaluator
import sender
import logging
from multiprocessing import Manager
from multiprocessing import Process
config = configparser.ConfigParser()
config.read('proj.config', 'utf-8')

QUEUE_SIZE = config.getint('SYSTEM_CONFIG','QUEUE_SIZE')
PROCESS_NUM = config.getint('SYSTEM_CONFIG','PROCESS_NUM')
WAIT_TIME = config.getint('SYSTEM_CONFIG', 'WAIT_TIME')


logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='main.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
def create_work_processes(pro_num:int, run, args)->list[Process]:
    p_list = []
    for i in range(0,pro_num):
        p = Process(target=run,args=args)
        p.start()
        p_list.append(p)
        logging.info(f'work process ({run.__name__} {i}) start successfully.')
    return p_list

def stop_all_processes(event):
    event.set()

def wait_processes_exit(p_list:list[Process]):
    for p in p_list:
        p.join()

class SignalHelper:
    def __init__(self, stop_event, q_list:list):
        self.stop_event = stop_event
        self.p_list = []
        self.q_list = q_list
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def add_processes(self, p_list:list[Process]):
        self.p_list += p_list

    def signal_handler(self, signal, frame):
        stop_all_processes(self.stop_event)
        wait_processes_exit(self.p_list)
        logging.info("All process exits successfully(After CTRL-C).")
        sys.exit(0)


def main():
    multiprocessing.set_start_method("spawn")
    manager = Manager()
    log_queue = manager.Queue(QUEUE_SIZE)
    output_queue = manager.Queue(QUEUE_SIZE)
    result_queue = manager.Queue(QUEUE_SIZE)
    stop_event = manager.Event()
    signalHelper = SignalHelper(stop_event, [log_queue, output_queue, result_queue])
    p_list1 = create_work_processes(PROCESS_NUM, evaluator.run_evaluate, (result_queue, WAIT_TIME, stop_event))
    signalHelper.add_processes(p_list1)
    p_list2 = create_work_processes(PROCESS_NUM, extractor.run_extract, (output_queue, result_queue, stop_event, WAIT_TIME))
    signalHelper.add_processes(p_list2)
    p_list3 = create_work_processes(PROCESS_NUM, detector.run_detector, (log_queue, output_queue, config, stop_event, WAIT_TIME))
    signalHelper.add_processes(p_list3)
    sender_process = Process(target=sender.sender, args = (log_queue, stop_event))
    sender_process.start()
    signalHelper.add_processes([sender_process])
    sender_process.join()
    stop_all_processes(stop_event)
    wait_processes_exit(p_list1+p_list2+p_list3)
    logging.info("main process exits successfully.")

if __name__ == '__main__':
    main()