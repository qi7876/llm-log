import random
import sentencepiece as spm
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from torch import optim
import matplotlib.pyplot as plt
import torch.nn as nn
import logging
VOCAB_SIZE = 5000
EMBEDDING_DIM = 50
SEQ_MAX_LENGTH = 150
BATCH_SIZE = 32
NUM_WORKERS = 0
TRAIN_FILE_PATH = 'siamese_network_BGL.csv'
TEST_FILE_PATH = 'siamese_network_Thunderbird.csv'
EPOCH_NUM = 4
PAD_IDX = 0
TRAIN_MODEL_FILE = 'Siamese.pt'
THRESHOLD = 0.5
logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename='net.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
def train_tokenizer():
    file_list = []
    for system in ['Android', 'BGL', 'Apache', 'Hadoop', 'HDFS', 'HealthApp', 'HPC', 'Linux', 'Mac', 'OpenSSH', 'Windows', 'Spark', 'OpenSSH', 'Zookeeper', 'Proxifier']:
        file_list.append(f'loghub/{system}/{system}_2k.log')
    spm.SentencePieceTrainer.train(
        input=file_list,  # 输入文件路径
        model_prefix='m',  # 输出模型名称前缀
        vocab_size=VOCAB_SIZE,  # 词汇表大小
        model_type='bpe',  # 分词算法类型，可选值：unigram、bpe、char、word
        character_coverage=0.9995,  # 字符覆盖比例，对于中文推荐设置为1.0
        pad_id=PAD_IDX,  # 设置 padding ID 为 0
        unk_id=1,  # 设置未知符号 ID 为 1
        bos_id=2,  # 设置序列开始符号 ID 为 2
        eos_id=3   # 设置序列结束符号 ID 为 3
    )

class SiameseNet(torch.nn.Module):
    def __init__(self, hiddensize:int=64,num_layer:int=2,outputsize:int=5):
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx= PAD_IDX)
        self.lstm = torch.nn.LSTM(EMBEDDING_DIM, hidden_size=hiddensize,num_layers= num_layer, batch_first= True, bidirectional=True)
        self.fc = torch.nn.Linear(2*hiddensize, outputsize)
        
    
    def forward_one(self, seq):
        seq_embedding = self.embedding(seq)
        seq_lstm, _ = self.lstm(seq_embedding)
        output = self.fc(torch.mean(seq_lstm, dim=1))
        return output
    
    def forward(self, seq1, seq2):
        output1 = self.forward_one(seq1)
        output2 = self.forward_one(seq2)
        return output1, output2



# class SiameseNet(torch.nn.Module):
#     def __init__(self, outputsize:int):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx= PAD_IDX)
#         self.cnn = nn.Sequential(
#             nn.Conv2d(1, 4, kernel_size=5),
#             nn.BatchNorm2d(4),
#             nn.ReLU(),

#             nn.Conv2d(4, 8, kernel_size=5),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),

#             nn.Conv2d(8, 8, kernel_size=3),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),

#             nn.Flatten()
#         )

#         # self.fc = torch.nn.Linear(44800, outputszie)
#         self.fc = nn.Sequential(
#             nn.Linear(44800, 5)
#         )
        
    
#     def forward_one(self, seq):
#         seq_embedding = self.embedding(seq)
#         seq_embedding = seq_embedding.unsqueeze(1)
#         seq_cnn = self.cnn(seq_embedding)
#         output = self.fc(seq_cnn)
#         return output
    
#     def forward(self, seq1, seq2):
#         output1 = self.forward_one(seq1)
#         output2 = self.forward_one(seq2)
#         return output1, output2

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
        self.dis_layer = nn.PairwiseDistance(p=2)

    def forward(self, output1 , output2, label):#label 0 means output1 is similar to output2.
        euclidean_distance = self.dis_layer(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class SiameseNetDataSet(Dataset):
    def __init__(self, data_file:str):
        df = pd.read_csv(data_file)
        log_dic = {}
        for _, row in df.iterrows():
            label = row['label']
            log = row['raw_log']
            if label in log_dic:
                log_dic[label].append(log)
            else:
                log_dic[label] = [log]
        sp = spm.SentencePieceProcessor(model_file='m.model')
        self.length = 0
        ids_dic = {}
        for label, log_list in log_dic.items():
            logs = list(set(log_list))
            ids_list = []
            for log in logs:
                ids = sp.encode_as_ids(log)
                padded_ids = ids + [sp.pad_id()] * (SEQ_MAX_LENGTH - len(ids)) if len(ids) <= SEQ_MAX_LENGTH else ids[: SEQ_MAX_LENGTH]
                ids_list.append(padded_ids)
            if ids_list.__len__() > 1:
                ids_dic[label] = ids_list
                self.length += ids_list.__len__()
        self.log_dic = ids_dic
        self.log_keys = list(ids_dic.keys())
        if self.log_keys.__len__() <= 1:
            raise Exception('Dataset file only has one kind of log.')
    
    def __getitem__(self, index):
        choice_flag = random.randint(0,1)
        first_label = random.choice(self.log_keys)
        if choice_flag:
            second_label = random.choice(self.log_keys)
            while second_label == first_label:
                second_label = random.choice(self.log_keys)
            ids1 = random.choice(self.log_dic[first_label])
            ids2 = random.choice(self.log_dic[second_label])
            return torch.tensor(ids1), torch.tensor(ids2), torch.tensor(choice_flag)
        else:
            ids1 = random.choice(self.log_dic[first_label])
            ids2 = random.choice(self.log_dic[first_label])
            if ids1 == ids2:
                copy = self.log_dic[first_label].copy()
                copy.remove(ids2)
                ids2 = random.choice(copy)
            return torch.tensor(ids1), torch.tensor(ids2), torch.tensor(choice_flag)
    
    def __len__(self):
        return self.length
            


def train():
    net = SiameseNet(outputsize=5).cuda()
    criterion = ContrastiveLoss().cuda()
    # net = SiameseNet(128, 1, 5)
    # criterion = ContrastiveLoss()
    train_dataset = SiameseNetDataSet(TRAIN_FILE_PATH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,  num_workers=NUM_WORKERS)
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
    iteration_number = 0
    counter = []
    loss_history = []
    for i in range(0, EPOCH_NUM):
        net.train()
        j = 0
        for ids1, ids2, choice_flag in train_dataloader:
            ids1 = ids1.cuda()
            ids2 = ids2.cuda()
            choice_flag = choice_flag.cuda()
            optimizer.zero_grad()
            output1, output2 = net(ids1, ids2)
            loss = criterion(output1, output2, choice_flag)
            loss.backward()
            optimizer.step()
            if j %10 == 0 :
                logging.info("Epoch number {}\n Current loss {}\n".format(i,loss.item()))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss.item())
    plt.plot(counter,loss_history)
    plt.show()
    torch.save(net.state_dict(),TRAIN_MODEL_FILE)

def eval():
    net = SiameseNet().cuda()
    net.load_state_dict(torch.load(TRAIN_MODEL_FILE))
    test_dataset = SiameseNetDataSet(TEST_FILE_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    acc, fp, fn, tp, tn = 0, 0, 0, 0, 0
    net.eval()
    with torch.no_grad():
        for ids1, ids2, flag in test_dataloader:
            ids1 = ids1.cuda()
            ids2 = ids2.cuda()
            flag = flag.cuda()
            output1, output2 = net(ids1, ids2)
            dist = F.pairwise_distance(output1, output2)
            acc_tmp, fp_tmp, tp_tmp, fn_tmp, tn_tmp = calculate_acc(dist, flag)
            acc += acc_tmp
            fp += fp_tmp
            fn += fn_tmp
            tn += tn_tmp
            tp += tp_tmp
            # logging.info(f"dist:{dist}\n flag:choiceflag:{flag}")
    sum_num = test_dataset.__len__()
    logging.info(f"fp:{fp} tp:{tp} fn:{fn} tn:{tn} acc:{acc} sum:{sum_num}")
    acc_r = acc/sum_num
    logging.info(f" acc_rate:{acc_r}")

def calculate_acc(dist, label_tensor):
    dist = torch.where(dist > THRESHOLD, 1, 0)
    acc_tensor = torch.where(dist == label_tensor, 1, 0)
    fp_tensor = torch.where((dist == 1) & (label_tensor == 0), 1, 0)
    tp_tensor = torch.where((dist == 1) & (label_tensor == 1), 1, 0)
    fn_tensor = torch.where((dist == 0) & (label_tensor == 1), 1, 0)
    tn_tensor = torch.where((dist == 0) & (label_tensor == 0), 1, 0)
    return torch.sum(acc_tensor), torch.sum(fp_tensor), torch.sum(tp_tensor), torch.sum(fn_tensor), torch.sum(tn_tensor)



    


    
        

def test():

    sp = spm.SentencePieceProcessor(model_file='m.model')
    # 使用模型进行分词
    sentence = "2016-09-28 04:30:30, Info                  CBS    Loaded Servicing Stack v6.1.7601.23505 with Core: C:\Windows\winsxs\amd64_microsoft-windows-servicingstack_31bf3856ad364e35_6.1.7601.23505_none_681aa442f6fed7f0\cbscore.dll"
    tokens = sp.encode_as_pieces(sentence)
    ids = sp.encode_as_ids(sentence)
    print(tokens)
    print(len(ids))
    # 加载预训练模型的分词器
    # tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")

    # # 示例文本
    # text = "1118767015 2005.06.14 2005-06-14-09.36.55.370589 R24-M1-N3-C:J16-U01 RAS KERNEL FATAL data address: 0x00000002"

    # # 使用分词器进行分词
    # tokens = tokenizer.tokenize(text)
    # print(tokens)

# train_tokenizer()
# train()
eval()
