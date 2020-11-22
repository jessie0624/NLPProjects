import sys
sys.path.append('../')
import config
import pandas as pd 
import numpy as np 
from utils.jiebaSegment import *
import re 

# data = pd.read_csv(config.ranking_train)
# print(data[data.labels==1].shape) # 25k   68k
# print(data[data.labels==0].shape) # 94k   133k  2:1

# data = data.sample(frac=1).reset_index(drop=True)
# data[:int(data.shape[0]*0.8)].to_csv(os.fspath(config.ranking_bert_train), index=False)
# data[int(data.shape[0]*0.8):].to_csv(os.fspath(config.ranking_bert_dev), index=False)

# data = pd.read_csv(os.fspath(config.rank_path/"bm25_test.csv"))
# p_data = data[data.labels==1]['score']
# print(p_data.describe()) # 210258, mean: 11, std: 6.49, min:0, max: 172, 75%:14.6, 50%: 10.9, 25: 6.7
# n_data = data[data.labels==0]['score']
# print(n_data.describe()) # 177637, mean: 7.9, std: 6.9, min:0, max: 169, 75%: 11, 50% 6.9, 25%: 2.46

# score = data['score'].values
# labels = data['labels'].values

# pccs = np.corrcoef(score, labels)
# print('pccs: ', pccs) # 0.2243

# print(data[data.labels==1][data.score<3])
# ranking_raw_path
# data3 = []
# with open(config.ranking_raw_path/"task3_train.txt", 'r') as rf:
#     for line in rf.readlines():
#         data3.append(line.strip().split('\t'))

# data2 = pd.read_csv(os.fspath(config.ranking_raw_path/"atec_nlp_sim_train.csv"),sep='\t',header=None, names=['text_a', 'text_b', 'labels'])
# data1 = pd.read_csv(os.fspath(config.ranking_raw_path/"atec_nlp_sim_train_add.csv"), sep='\t',header=None, names=['text_a', 'text_b', 'labels'])
# df3 = pd.DataFrame(data3, columns=['text_a', 'text_b', 'labels'])
# data = pd.concat([data1, data2, df3],ignore_index=True)
# print(data.shape)
# data.drop_duplicates(subset=['text_a','text_b', 'labels'], keep='first', inplace=True)
# print(data.shape)
# # df = pd.read_csv(os.fspath(config.ranking_train))
# # print(df.shape)
# # df.drop_duplicates(subset=['text_a','text_b', 'labels'], keep='first', inplace=True)
# # print(df.shape) # 119k 数据集
# data.to_csv(os.fspath(config.ranking_train), index=False)


# dev = pd.read_csv(config.ranking_bert_dev)
# train = pd.read_csv(config.ranking_bert_train)

# print(dev[dev.labels==1].shape, dev[dev.labels==0].shape) 13821, 26670
# print(train[train.labels==1].shape, train[train.labels==0].shape)54840 107122  ## neg 2, pos 1

# gbm_path = config.result_path/ "gbm_data.csv"
# data = pd.read_csv(gbm_path)
# print(data.shape)
# print(data[:2])
# data2 = data.groupby()

# train['a_len'] = train['text_a'].apply(lambda x: len(x))
# train['b_len'] = train['text_b'].apply(lambda x: len(x))
# train['total_len'] = train['a_len'] + train['b_len']
# # print(train['a_len'].describe())
# # print(train['b_len'].describe())
# # print(train['total_len'].describe())

# print(train[train.a_len>30]['a_len'].describe()) # 40 

# gbm_path = config.result_path/ "gbm_data.csv"
# data = pd.read_csv(gbm_path)

# data.groupby()

data = pd.read_csv(config.ranking_train)

def clean(sentence):
    sentence = re.sub(r'[0-9\.]+%?', '[数字x]', sentence)
    sentence = re.sub(r' ', '', sentence)
    sentence = re.sub(r'', '', sentence)
    sentence = re.sub(r'÷', '[符号x]', sentence)
    sentence = re.sub(r'yib', '', sentence)
    sentence = re.sub(r'＂','', sentence)
    
    return sentence
data['text_a'] = data.text_a.apply(lambda x: clean(x))
data['text_b'] = data.text_b.apply(lambda x: clean(x))
data.to_csv(config.ranking_train_clean, index=False)

    

