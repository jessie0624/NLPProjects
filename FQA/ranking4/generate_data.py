import pandas as pd 
import sys, os
from pathlib import Path
from sklearn.model_selection import train_test_split
sys.path.append('..')
import config
from config import *

datas = pd.read_csv(os.fspath(Path(data_path / 'ranking_raw_data/atec_all_org.csv')))

train_set, dev_set = train_test_split(datas, test_size=0.2, random_state=42, stratify=datas[['labels']])

# # test_set = test_set.reset_index()
# # test_set2, dev_set = train_test_split(test_set, test_size=0.3, random_state=42, stratify=datas[['labels']])

# Path(data_path / "ranking4").mkdir(parents=True, exist_ok=True)

# data_intention_path.mkdir(parents=True, exist_ok=True)

train_set.to_csv(os.fspath(Path(data_path / "ranking/train.csv")), index=False)
dev_set.to_csv(os.fspath(Path(data_path / "ranking/dev.csv")), index=False)

# dev_data = pd.read_csv(os.fspath(Path(data_path / 'ranking4/dev.csv')))
# # dev_data = dev_data.reset_index(inplace=True)
# dev_set, test_set = train_test_split(dev_data, test_size=0.3, random_state=42, stratify=dev_data[['labels']])

# dev_set.to_csv(os.fspath(Path(data_path / "ranking4/dev.csv")), index=False)
# test_set.to_csv(os.fspath(Path(data_path / "ranking4/test.csv")), index=False)

# print(datas.shape)
# print(train_set.shape)
# print(test_set.shape)

# print(len(train_set.values.tolist())) # 81104 + 34759 = 115863
# # print(train_set[:10])
# # print(len(set(train_set.values.tolist())))

# print(len(test_set.values.tolist()))
# # print(len(set(test_set.values.tolist())))