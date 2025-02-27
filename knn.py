import hnswlib
import numpy as np
import torch
import configparser
import pickle

KNN_STORE_PATH = 'knn_store.pkl'
HNSW_STORE_PATH = 'hnsw.model'
# config = configparser.ConfigParser()
# if config.has_option('SYSTEM_CONFIG', 'HNSW_IDX_PROPERTY'):
#     HNSW_IDX = config.getint('SYSTEM_CONFIG','HNSW_IDX_PROPERTY')
# else:
#     HNSW_IDX = 0
DIM = 3
MAX_ELEMENTS = 100000
class HNSW:
    def __init__(self, threshold:int, hnsw_path:str, knn_store_path:str, M=16, efC=100, efS=100, allow_replace_deleted = False):
        if hnsw_path is not None and knn_store_path is not None:
            self.p = hnswlib.Index(space='l2', dim=DIM)
            self.p.load_index(hnsw_path)
            with open(knn_store_path, 'rb') as f:
                knn_store_data = pickle.load(f)
                self.node_idx_table = knn_store_data['label_list']
                self.p_idx = knn_store_data['p_idx']
                self.threshold = knn_store_data['threshold']
        else:
            self.p = hnswlib.Index(space='l2', dim=DIM)
            self.p.init_index(max_elements=MAX_ELEMENTS, ef_construction=efC, M=M, allow_replace_deleted=allow_replace_deleted)
            self.p.set_ef(efS)
            self.node_idx_table = ['']*MAX_ELEMENTS
            self.threshold = threshold
            self.p_idx = 0

    def add(self, data_vec:list[torch.Tensor], label:list[str]):
        increment = len(data_vec)
        if increment != 0:
            stacked_data = torch.stack(data_vec)
            data_numpy = stacked_data.numpy()
            idx_numpy = np.arange(self.p_idx, self.p_idx+increment)
            self.p.add_items(data_numpy, idx_numpy)
            for i in range(increment):
                self.node_idx_table[self.p_idx] = {'label':label[i], 'weight':1}
                self.p_idx += 1

    def query(self, data:list[torch.Tensor], k)->tuple[list[str],list[list]]:#返回k个最近邻的标签，以及对应满足条件的最近邻,分别对应最近标签组成的list和每个数据相关的多个有效最近邻
        data_numpy = np.array(data)
        try:
            ids, distances = self.p.knn_query(data_numpy, k=k, num_threads=1, filter = None)
        except RuntimeError:
            ids, distances = np.ones((len(data_numpy), k), dtype=int), np.zeros((len(data_numpy), k), dtype=int)
        filtered_ids = []
        all_data_predicted_labels = []
        for j in range(len(data_numpy)):#针对每个数据
            filtered_ids_j = []
            for i in range(k):#数据的knn
                dis = distances[j, i]
                if dis <= self.threshold and dis > 0:
                    filtered_ids_j.append(ids[j, i])
            data_attr_dict = {}#统计knn的标签
            for filtered_label in filtered_ids_j:
                if data_attr_dict.get(self.node_idx_table[filtered_label]['label']) is None:
                    data_attr_dict[self.node_idx_table[filtered_label]['label']] = self.node_idx_table[filtered_label]['weight']
                else:
                    data_attr_dict[self.node_idx_table[filtered_label]['label']] += self.node_idx_table[filtered_label]['weight']
            if len(data_attr_dict) == 0:#没有最近邻
                all_data_predicted_labels.append(None)
                filtered_ids.append([])
                continue
            #找到最多的标签
            max_value = max(data_attr_dict.values())
            majority_labels = [key for key, value in data_attr_dict.items() if value == max_value]
    
            if len(majority_labels) == 1 and max_value > 1:#如果标签具有显著优势，选择此标签
                all_data_predicted_labels.append(majority_labels[0])
                related_ids = [id for id in filtered_ids_j if self.node_idx_table[id]['label'] == majority_labels[0]]
                filtered_ids.append(related_ids)

            else:#否则，无法预测
                all_data_predicted_labels.append(None)
                filtered_ids.append([])
        return all_data_predicted_labels, filtered_ids

    def add_weight(self, idx:str, weight:int):
        self.node_idx_table[idx]['weight'] += weight
    
    def query_and_adjust(self, data:list[torch.Tensor], k, true_label_list:list[str])-> tuple[int, int]:#按顺序返回准确预测数，不能预测数与总数
        all_data_predicted_labels, related_ids = self.query(data, k)
        new_node = []
        new_label = []
        correct_cnt, ambiguous_cnt = 0, 0
        for i in range(len(data)):
            predicted_label = all_data_predicted_labels[i]
            if predicted_label is None:
                new_node.append(data[i])
                new_label.append(true_label_list[i])
                ambiguous_cnt += 1
            else:
                related_ids_i = related_ids[i]
                for related_id in related_ids_i:
                    self.add_weight(related_id, 1)
                if predicted_label == true_label_list[i]:
                    correct_cnt += 1
        self.add(new_node, new_label)
        return correct_cnt, ambiguous_cnt
                

    def save(self, hnsw_path, knn_store_path):
        self.p.save_index(hnsw_path)
        knn_store_data = {'label_list':self.node_idx_table, 'p_idx':self.p_idx, 'threshold':self.threshold}
        with open(knn_store_path, 'wb') as f:
            pickle.dump(knn_store_data, f)
        
