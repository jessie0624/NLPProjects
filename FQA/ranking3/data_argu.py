import pandas as pd 
import csv 
import itertools
from sklearn.model_selection import KFold
from pathlib import Path
import os, sys, re 
sys.path.append("..")
import config 
from config import root_path, data_path



def read_file(folder, file_list, col=None):
    df = pd.DataFrame(columns=col)
    for filename in file_list:
        file_path = Path(data_path/folder/filename)
        df = df.append(pd.read_csv(file_path, sep='\t', header=None, names=col))
    return df                  


  

# def data_clean(): # 同义词替换  
#     """
#     对于
#     """

def get_fold_data(datas, indexs):
    result = []
    for index in indexs:
        result.append(datas[index])
    return result


def write_fold_data(datas, filename):
    with open(filename, 'w', nfilename='NoName', newline='', encoding='utf-8') as f:
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





