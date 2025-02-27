import random
import fasttext
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
import plotly.graph_objects as go
import numpy as np
import torch
import seaborn as sns
import re
import knn
VOCAB_SIZE = 5000
# EMBEDDING_DIM = 50
EMBEDDING_DIM = 300
SEQ_MAX_LENGTH = 50
BATCH_SIZE = 32
NUM_WORKERS = 0
TRAIN_FILE_PATH = 'siamese_network_BGL.csv'
TEST_FILE_PATH = 'siamese_network_Thunderbird.csv'
# TRAIN_FILE_PATH = 'siamese_network_Thunderbird.csv'
# TEST_FILE_PATH = 'siamese_network_BGL.csv'
FAST_TEXT_MODEL = 'cc.en.300.bin'
EPOCH_NUM = 3
PAD_IDX = 0
NUM_PATTERN = r'((?=[^a-zA-Z])(?:0[xX]?)?[0-9a-fA-F]+|\d+)(?=[^a-zA-Z0-9])'
TRAIN_MODEL_FILE = 'Siamese.pt'
THRESHOLD = 0.5
OUTPUT_SIZE = 3
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

# class SiameseNet(torch.nn.Module):
#     def __init__(self, hiddensize:int=64,num_layer:int=2,outputsize:int=OUTPUT_SIZE):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx= PAD_IDX)
#         self.lstm = torch.nn.LSTM(EMBEDDING_DIM, hidden_size=hiddensize,num_layers= num_layer, batch_first= True, bidirectional=True)
#         self.fc = nn.Sequential(
#             nn.Linear(hiddensize * 2, outputsize),
#         )
        
    
#     def forward_one(self, seq):
#         seq_embedding = self.embedding(seq)
#         seq_lstm, _ = self.lstm(seq_embedding)
#         output = self.fc(torch.mean(seq_lstm, dim=1))
#         return output
    
#     def forward(self, seq1, seq2):
#         output1 = self.forward_one(seq1)
#         output2 = self.forward_one(seq2)
#         return output1, output2



class SiameseNet(torch.nn.Module):
    def __init__(self, outputsize:int=OUTPUT_SIZE):
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, padding_idx= PAD_IDX)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(4, 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Flatten()
        )

        # self.fc = torch.nn.Linear(44800, outputszie)
        self.fc = nn.Sequential(
            nn.Linear(44800, outputsize)
        )
        
    
    def forward_one(self, seq):
        seq_embedding = self.embedding(seq)
        seq_embedding = seq_embedding.unsqueeze(1)
        seq_cnn = self.cnn(seq_embedding)
        output = self.fc(seq_cnn)
        return output
    
    def forward(self, seq1, seq2):
        output1 = self.forward_one(seq1)
        output2 = self.forward_one(seq2)
        return output1, output2
    
# class SiameseNetWithPretrainedEmbedding(torch.nn.Module):
#     def __init__(self, outputsize:int=OUTPUT_SIZE):
#         super().__init__()
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
#             nn.Linear(44800, outputsize)
#         )
        
    
#     def forward_one(self, seq_embedding):
#         seq_embedding = seq_embedding.unsqueeze(1)
#         seq_cnn = self.cnn(seq_embedding)
#         output = self.fc(seq_cnn)
#         return output
    
#     def forward(self, seq1, seq2):
#         output1 = self.forward_one(seq1)
#         output2 = self.forward_one(seq2)
#         return output1, output2

class SiameseNetWithPretrainedEmbedding(torch.nn.Module):
    def __init__(self, outputsize:int=OUTPUT_SIZE):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=6, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(4, 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Flatten()
        )

        # self.fc = torch.nn.Linear(44800, outputszie)
        self.fc = nn.Sequential(
            nn.Linear(19312, outputsize)
        )
        
    
    def forward_one(self, seq_embedding):
        seq_embedding = seq_embedding.unsqueeze(1)
        seq_cnn = self.cnn(seq_embedding)
        output = self.fc(seq_cnn)
        return output
    
    def forward(self, seq1, seq2):
        output1 = self.forward_one(seq1)
        output2 = self.forward_one(seq2)
        return output1, output2

class TripletNet(torch.nn.Module):
    def __init__(self, outputsize:int=OUTPUT_SIZE):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=6, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(),

            nn.Conv2d(4, 8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Flatten()
        )

        # self.fc = torch.nn.Linear(44800, outputszie)
        self.fc = nn.Sequential(
            nn.Linear(19312, outputsize)
        )
        
    
    def forward_one(self, seq_embedding):
        seq_embedding = seq_embedding.unsqueeze(1)
        seq_cnn = self.cnn(seq_embedding)
        output = self.fc(seq_cnn)
        return output
    
    def forward(self, seq1, seq2, seq3):
        output1 = self.forward_one(seq1)
        output2 = self.forward_one(seq2)
        output3 = self.forward_one(seq3)
        return output1, output2, output3

class FastSiameseNet(torch.nn.Module):
    def __init__(self, outputsize:int=OUTPUT_SIZE):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, outputsize)
        )
    
    def forward_one(self, seq):
        output = self.fc(torch.mean(seq, dim=1))
        return output
    
    def forward(self, seq1, seq2):
        output1 = self.forward_one(seq1)
        output2 = self.forward_one(seq2)
        return output1, output2
    
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

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.dis_layer = nn.PairwiseDistance(p=2)

    def forward(self, anchor , pos, neg):#label 0 means output1 is similar to output2.
        pos_distance = self.dis_layer(anchor, pos)
        neg_distance = self.dis_layer(anchor, neg)
        loss_triplet = torch.mean(torch.clamp(torch.pow(pos_distance, 2) - torch.pow(neg_distance, 2) + self.margin, min=0.0))
        return loss_triplet
    

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

class TripletNetDataSet(Dataset):
    def __init__(self, data_file:str, is_test:bool):
        df = pd.read_csv(data_file)
        log_dic = {}
        for _, row in df.iterrows():
            label = row['label']
            log = row['raw_log']
            if label in log_dic:
                log_dic[label].append(log)
            else:
                log_dic[label] = [log]
        self.length = 0
        ids_dic = {}
        model = fasttext.load_model(FAST_TEXT_MODEL)
        for label, log_list in log_dic.items():
            logs = list(set(log_list))
            ids_list = []
            for log in logs:
                if is_test:
                    log = re.sub(NUM_PATTERN, '', log)
                tokens = log.split()
                if len(tokens) > SEQ_MAX_LENGTH:
                    tokens = tokens[: SEQ_MAX_LENGTH]
                if len(tokens) < SEQ_MAX_LENGTH:
                    tokens += [''] * (SEQ_MAX_LENGTH - len(tokens))
                vectors = np.array([model.get_word_vector(token) if token else np.zeros(EMBEDDING_DIM) for token in tokens])
                ids_list.append(vectors)
            if ids_list.__len__() > 1:
                ids_dic[label] = ids_list
                self.length += ids_list.__len__()
        self.log_dic = ids_dic
        self.log_keys = list(ids_dic.keys())
        if self.log_keys.__len__() <= 1:
            raise Exception('Dataset file only has one kind of log.')
    
    def __getitem__(self, index):
        first_label, second_label = random.sample(self.log_keys, 2)
        anchor, pos = random.sample(self.log_dic[first_label], 2)
        neg = random.choice(self.log_dic[second_label])
        return torch.from_numpy(anchor).float(), torch.from_numpy(pos).float(), torch.from_numpy(neg).float()
    
    def __len__(self):
        return self.length

class SiameseNetDataSetWithPretrainedEmbdedding(SiameseNetDataSet):
    def __init__(self, data_file:str, is_test:bool):
        df = pd.read_csv(data_file)
        log_dic = {}
        for _, row in df.iterrows():
            label = row['label']
            log = row['raw_log']
            if label in log_dic:
                log_dic[label].append(log)
            else:
                log_dic[label] = [log]
        self.length = 0
        ids_dic = {}
        model = fasttext.load_model(FAST_TEXT_MODEL)
        for label, log_list in log_dic.items():
            logs = list(set(log_list))
            ids_list = []
            for log in logs:
                if is_test:
                    log = re.sub(NUM_PATTERN, '', log)
                tokens = re.split('=|:|,|')
                if len(tokens) > SEQ_MAX_LENGTH:
                    tokens = tokens[: SEQ_MAX_LENGTH]
                if len(tokens) < SEQ_MAX_LENGTH:
                    tokens += [''] * (SEQ_MAX_LENGTH - len(tokens))
                vectors = np.array([model.get_word_vector(token) if token else np.zeros(EMBEDDING_DIM) for token in tokens])
                ids_list.append(vectors)
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
            return torch.from_numpy(ids1).float(), torch.from_numpy(ids2).float(), torch.tensor(choice_flag)
        else:
            ids1 = random.choice(self.log_dic[first_label])
            ids2_candidate = random.sample(self.log_dic[first_label], 2)
            if np.allclose(ids1, ids2_candidate[0]):
                ids2 = ids2_candidate[1]
            else:
                ids2 = ids2_candidate[0]
            return torch.from_numpy(ids1).float(), torch.from_numpy(ids2).float(), torch.tensor(choice_flag)
    
    def __len__(self):
        return self.length

class NormalDataSet(Dataset):
    def __init__(self, path:str):
        self.df = pd.read_csv(path)
        self.model = fasttext.load_model(FAST_TEXT_MODEL)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = row['label']
        log = row['raw_log']
        log = re.sub(NUM_PATTERN, '', log)
        # sp = spm.SentencePieceProcessor(model_file='m.model')
        # ids = sp.encode_as_ids(log)
        # padded_ids = ids + [sp.pad_id()] * (SEQ_MAX_LENGTH - len(ids)) if len(ids) <= SEQ_MAX_LENGTH else ids[: SEQ_MAX_LENGTH]
        model = self.model
        tokens = log.split()
        if len(tokens) > SEQ_MAX_LENGTH:
            tokens = tokens[: SEQ_MAX_LENGTH]
        
        if len(tokens) < SEQ_MAX_LENGTH:
            tokens += [''] * (SEQ_MAX_LENGTH - len(tokens))
        vectors = np.array([model.get_word_vector(token) if token else np.zeros(EMBEDDING_DIM) for token in tokens])

        return torch.from_numpy(vectors).float(), label[1:]
    
    def __len__(self):
        return self.df.__len__()



# def train():
#     # net = SiameseNet(outputsize=OUTPUT_SIZE).cuda()
#     net = SiameseNetWithPretrainedEmbedding(outputsize=OUTPUT_SIZE).cuda()
#     # net = TripletNet(outputsize=OUTPUT_SIZE).cuda()
#     # net = FastSiameseNet(outputsize=OUTPUT_SIZE).cuda()
#     criterion = ContrastiveLoss().cuda()
#     # criterion = TripletLoss().cuda()
#     # train_dataset = SiameseNetDataSet(TRAIN_FILE_PATH)
#     train_dataset = SiameseNetDataSetWithPretrainedEmbdedding(TRAIN_FILE_PATH)
#     # train_dataset = TripletNetDataSet(TRAIN_FILE_PATH)
#     train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,  num_workers=NUM_WORKERS)
#     optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
#     iteration_number = 0
#     counter = []
#     loss_history = []
#     for i in range(0, EPOCH_NUM):
#         net.train()
#         j = 0
#         for ids1, ids2, choice_flag in train_dataloader:
#             ids1 = ids1.cuda()
#             ids2 = ids2.cuda()
#             choice_flag = choice_flag.cuda()
#             optimizer.zero_grad()
#             output1, output2 = net(ids1, ids2)
#             loss = criterion(output1, output2, choice_flag)
#             loss.backward()
#             optimizer.step()
#             if j %10 == 0 :
#                 logging.info("Epoch number {}\n Current loss {}\n".format(i,loss.item()))
#                 iteration_number +=10
#                 counter.append(iteration_number)
#                 loss_history.append(loss.item())
#     plt.plot(counter,loss_history)
#     plt.show()
#     torch.save(net.state_dict(),TRAIN_MODEL_FILE)


def train():
    # net = SiameseNet(outputsize=OUTPUT_SIZE).cuda()
    # net = SiameseNetWithPretrainedEmbedding(outputsize=OUTPUT_SIZE).cuda()
    net = TripletNet(outputsize=OUTPUT_SIZE).cuda()
    # criterion = ContrastiveLoss().cuda()
    criterion = TripletLoss().cuda()
    # train_dataset = SiameseNetDataSet(TRAIN_FILE_PATH)
    # train_dataset = SiameseNetDataSetWithPretrainedEmbdedding(TRAIN_FILE_PATH)
    train_dataset = TripletNetDataSet(TRAIN_FILE_PATH, is_test=False)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,  num_workers=NUM_WORKERS)
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
    iteration_number = 0
    counter = []
    loss_history = []
    for i in range(0, EPOCH_NUM):
        net.train()
        j = 0
        for anchor, pos, neg in train_dataloader:
            anchor = anchor.cuda()
            pos = pos.cuda()
            neg = neg.cuda()
            optimizer.zero_grad()
            output1, output2, output3 = net(anchor, pos, neg)
            loss = criterion(output1, output2, output3)
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
    # net = SiameseNet().cuda()
    net = SiameseNetWithPretrainedEmbedding().cuda()
    # net = TripletNet().cuda()
    # net = FastSiameseNet().cuda()
    net.load_state_dict(torch.load(TRAIN_MODEL_FILE))
    # test_dataset = SiameseNetDataSet(TEST_FILE_PATH)
    test_dataset = SiameseNetDataSetWithPretrainedEmbdedding(TEST_FILE_PATH, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    acc, fp, fn, tp, tn = 0, 0, 0, 0, 0
    net.eval()
    with torch.no_grad():
        for ids1, ids2, flag in test_dataloader:
            ids1 = ids1.cuda()
            ids2 = ids2.cuda()
            flag = flag.cuda()
            output1 = net.forward_one(ids1)
            output2 = net.forward_one(ids2)
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


def eval_pretrained():
    test_dataset = SiameseNetDataSetWithPretrainedEmbdedding(TEST_FILE_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    acc, fp, fn, tp, tn = 0, 0, 0, 0, 0
    for ids1, ids2, flag in test_dataloader:
        output1 = torch.mean(ids1, dim=1)
        output2 = torch.mean(ids2, dim=1)
        dist = F.pairwise_distance(output1, output2)
        acc_tmp, fp_tmp, tp_tmp, fn_tmp, tn_tmp = calculate_acc(dist, flag)
        acc += acc_tmp
        fp += fp_tmp
        fn += fn_tmp
        tn += tn_tmp
        tp += tp_tmp
    sum_num = test_dataset.__len__()
    logging.info(f"fp:{fp} tp:{tp} fn:{fn} tn:{tn} acc:{acc} sum:{sum_num}")
    acc_r = acc/sum_num
    logging.info(f" acc_rate:{acc_r}")


# def eval():
#     # net = SiameseNet().cuda()
#     # net = SiameseNetWithPretrainedEmbedding().cuda()
#     net = TripletNet().cuda()
#     net.load_state_dict(torch.load(TRAIN_MODEL_FILE))
#     # test_dataset = SiameseNetDataSet(TEST_FILE_PATH)
#     # test_dataset = SiameseNetDataSetWithPretrainedEmbdedding(TEST_FILE_PATH)
#     test_dataset = TripletNetDataSet(TEST_FILE_PATH)
#     test_dataloader = DataLoader(test_dataset, batch_size=8)
#     acc, fp, fn, tp, tn = 0, 0, 0, 0, 0
#     net.eval()
#     with torch.no_grad():
#         for ids1, ids2, ids3 in test_dataloader:
#             ids1 = ids1.cuda()
#             ids2 = ids2.cuda()
#             ids3 = ids3.cuda()
#             output1, output2 = net(ids1, ids2)
#             dist = F.pairwise_distance(output1, output2)
#             acc_tmp, fp_tmp, tp_tmp, fn_tmp, tn_tmp = calculate_acc(dist, flag)
#             acc += acc_tmp
#             fp += fp_tmp
#             fn += fn_tmp
#             tn += tn_tmp
#             tp += tp_tmp
#             # logging.info(f"dist:{dist}\n flag:choiceflag:{flag}")
#     sum_num = test_dataset.__len__()
#     logging.info(f"fp:{fp} tp:{tp} fn:{fn} tn:{tn} acc:{acc} sum:{sum_num}")
#     acc_r = acc/sum_num
#     logging.info(f" acc_rate:{acc_r}")

def statistic():
    # net = SiameseNet().cuda()
    net = SiameseNetWithPretrainedEmbedding().cuda()
    # net = TripletNet().cuda()
    # net = FastSiameseNet().cuda()
    # net = SiameseNetWithPretrainedEmbedding().cuda()
    net.load_state_dict(torch.load(TRAIN_MODEL_FILE))
    test_dataset = NormalDataSet(TEST_FILE_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    net.eval()
    output_list = []
    labels = []
    with torch.no_grad():
        for ids, label in test_dataloader:
            ids = ids.cuda()
            output = net.forward_one(ids)
            labels += list(label)
            output_list += list(torch.unbind(output, dim=0))
    x_vals = [tensor[0].item() for tensor in output_list]
    y_vals = [tensor[1].item() for tensor in output_list]
    z_vals = [tensor[2].item() for tensor in output_list]
    clrs = [int(label) for label in labels]
    fig = go.Figure(data=[go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='markers', text=labels, marker=dict(size=3, color=clrs, colorscale='Viridis', opacity=0.8, showscale=True))])
    fig.update_traces(hoverinfo='text')
    fig.show()


def statistic_fasttext():
    # net = SiameseNet().cuda()
    # net = SiameseNetWithPretrainedEmbedding().cuda()
    # net = TripletNet().cuda()
    # net.load_state_dict(torch.load(TRAIN_MODEL_FILE))
    test_dataset = NormalDataSet(TEST_FILE_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    # net.eval()
    output_list = []
    labels = []
    with torch.no_grad():
        for ids, label in test_dataloader:
            ids = ids.cuda()
            output = torch.mean(ids, dim=1)
            labels += list(label)
            output_list += list(torch.unbind(output, dim=0))
    x_vals = [tensor[0].item() for tensor in output_list]
    y_vals = [tensor[1].item() for tensor in output_list]
    z_vals = [tensor[2].item() for tensor in output_list]
    clrs = [int(label) for label in labels]
    fig = go.Figure(data=[go.Scatter3d(x=x_vals, y=y_vals, z=z_vals, mode='markers', text=labels, marker=dict(size=3, color=clrs, colorscale='Viridis', opacity=0.8, showscale=True))])
    fig.update_traces(hoverinfo='text')
    fig.show()
    

def calculate_acc(dist, label_tensor):
    dist = torch.where(dist > THRESHOLD, 1, 0)
    acc_tensor = torch.where(dist == label_tensor, 1, 0)
    fp_tensor = torch.where((dist == 1) & (label_tensor == 0), 1, 0)#意味着两个应该相似，实际预测不相似
    tp_tensor = torch.where((dist == 1) & (label_tensor == 1), 1, 0)#不相似
    fn_tensor = torch.where((dist == 0) & (label_tensor == 1), 1, 0)#应该不相似，实际预测相似
    tn_tensor = torch.where((dist == 0) & (label_tensor == 0), 1, 0)
    return torch.sum(acc_tensor), torch.sum(fp_tensor), torch.sum(tp_tensor), torch.sum(fn_tensor), torch.sum(tn_tensor)



def test_knn():
    hnsw = knn.HNSW(threshold=0.25, hnsw_path=None, knn_store_path=None)
    net = SiameseNetWithPretrainedEmbedding().cuda()
    net.load_state_dict(torch.load(TRAIN_MODEL_FILE))
    test_dataset = NormalDataSet(TEST_FILE_PATH)
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    correct_num, ambiguous_num, data_num = 0, 0, len(test_dataset)
    k = 5
    net.eval()
    with torch.no_grad():
        for ids, label in test_dataloader:
            ids = ids.cuda()
            output = net.forward_one(ids)
            output = output.cpu()
            output_list = list(torch.unbind(output, dim=0))
            correct_tmp, ambiguous_tmp = hnsw.query_and_adjust(output_list, k, label)
            correct_num += correct_tmp
            ambiguous_num += ambiguous_tmp
    print(f"correct_num:{correct_num} ambiguous_num:{ambiguous_num} data_num:{data_num} ")
    if data_num != 0 and data_num != ambiguous_num:
        print(f"correct_rate:{correct_num/(data_num - ambiguous_num)} ambiguous_rate:{ambiguous_num/data_num}")
    
        

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
# eval_pretrained()
# statistic_fasttext()
# eval()
# statistic()
test_knn()
# print(re.sub(NUM_PATTERN, '', '2016-09-28 04:30:30, Info                  CBS    Loaded Servicing Stack v6.1.7601.23505 with Core: C:\Windows\winsxs\amd64_microsoft-windows-servicingstack_31bf3856ad364e35_6.1.7601.23505_none_681aa442f6fed7f0\cbscore.dll'))
