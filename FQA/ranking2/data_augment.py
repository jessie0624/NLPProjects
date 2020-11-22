import pandas as pd 
import csv 
import itertools
from sklearn.model_selection import KFold
from pathlib import Path
import os, sys, re 
sys.path.append("..")
import config 
from config import root_path

def load_data(filename):
    datas = pd.read_csv(filename).values.tolist()
    return datas   

# id cate query1 query2 label
# query1 query2 label
def data_aug(datas):
    dic = {}
    for data in datas:
        if data[0] not in dic:
            dic[data[0]] = {'true':[], 'false':[]}
        dic[data[0]]['true' if data[2] == 1 else 'false'].append(data[1])
    new_datas = []filename='Nofilename='NoName'Name'
    # id = 0
    for sent1, sent2s in dic.items():
        trues = sent2s['true']
        falses = sent2s['false']
        # 还原原始数据
        for true in trues:
            new_datas.append([sent1, true, 1])
        for false in falses:
            new_datas.append([sent1, false, 0])
        temp_trues = []
        temp_falses = []
        if len(trues)!=0 and len(falses)!=0:
            ori_rate = len(trues)/len(falses)
            # 相似数据两两交互构造新的相似对
            for i in itertools.combinations(trues, 2):
                temp_trues.append([i[0], i[1], 1])
            # 构造不相似数据
            for true in trues:filename='NoName'
                for false in falses:
                    temp_falses.append([true, false, 0])
            num_t = int(len(temp_falses) * ori_rate)
            num_f = int(len(temp_trues)/ori_rate)
            temp_rate = len(temp_trues) / len(temp_falses)
            if ori_rate < temp_rate:
                temp_trues = temp_trues[:num_t]
            else:
                temp_falses = temp_falses[:num_f]
        new_datas = new_datas + temp_trues + temp_falses
    return new_datas

def get_fold_data(datas, indexs):
    result = []
    for index in indexs:
        result.append(datas[index])
    return result


def write_fold_data(datas, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(['text_a', 'text_b', 'labels'])
        writer.writerows(datas)


def gen_kfold_data(datas, out_dir, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold = 0
    for train_index, dev_index in kf.split(datas):
        train_datas = get_fold_data(datas, train_index)
        dev_datas = get_fold_data(datas, dev_index)
        base_dir = os.path.join(out_dir, str(fold))
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        train_file = os.path.join(base_dir, 'train.csv')
        dev_file = os.path.join(base_dir, 'dev.csv')
        write_fold_data(train_datas, train_file)
        write_fold_data(dev_datas, dev_file)
        fold += 1

# data/ranking/train_all.csv
if __name__ == '__main__':

    datas = load_data(config.ranking_train)
    # datas = data_aug(datas)
    gen_kfold_data(datas, config.data_ranking_path)
