"""
DESC: 细排算法
模型: lightGBM
特征: 
- 各种计算字符串间距离得分(lcs, jaccard, bm25, edit_dist), 
- 各种embedding + 相似度评估的算法计算出的 相似度.         
    具体embedding: embedding: w2v, fasttext, tfidf,
    相似度评估算法:　 cos, eurl, pearson, wmd
- BERT相似度比较. q1 q2 用classification模型计算相似度得分.

将这些特征组合起来, 作为lightGBM 的训练特征, 进行训练.             

特征的特点: 
"""
import os, sys
import csv,re
import logging

import lightgbm as lgb
import pandas as pd  
import joblib 
# from tqdm import tqdm 
from pathlib import Path

sys.path.append("..")
from config import root_path, rank_path,result_path, ranking_bert_train, ranking_bert_dev
from ranking.matchnn import MatchingNN
from ranking.similarity import TextSimilarity 
# from retrieval.hnsw_faiss import wam 

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import f1_score


import numpy as np     
import time        
from utils.tools import create_logger
logger = create_logger(os.fspath(root_path/'log/train_gbm'))

# tqdm.pandas()/

# parameters for lightgbm
params = {
    "boosting_type":  "gbdt", 
    "max_depth":       -1,
    "objective":      "binary",
    # "nthread":         10,
    "num_leaves":      256,
    # "learning_rate":   0.01,
    # "max_bin":         10,
    # # "subsample_for_bin": 200,
    # "subsample":        0.8,
    # # "subsample_freq":   5,
    # "colsample_bytree": 0.8,
    # # "reg_alpha":        5,
    # # "reg_lambda":       10,
    # "min_split_gain":   0.5,
    # # "min_child_weight":  1,
    # "min_child_samples": 5,
    # "scale_pos_weight":  1,
    "metric":            "auc",
    "n_iter":            100
}  # "auc"
def clean(sentence):
    sentence = re.sub(r'[0-9\.]+%?', '[数字x]', sentence)
    sentence = re.sub(r' ', '', sentence)
    sentence = re.sub(r'', '', sentence)
    sentence = re.sub(r'÷', '[符号x]', sentence)
    sentence = re.sub(r'yib', '', sentence)
    sentence = re.sub(r'＂','', sentence)
    
    return sentence
class Data(object):
    def __init__(self, do_train=True, bert=False, train_path=None):
        self.ts = TextSimilarity()
        if bert:
            self.matchingNN = MatchingNN()
        if do_train:
            logger.info("Training mode")
            self.train = pd.read_csv(os.fspath(ranking_bert_train))
            self.train['text_a'] = self.train['text_a'].apply(lambda x: clean(x))
            self.train['text_b'] = self.train['text_b'].apply(lambda x: clean(x))
            self.data = self.generate_feature(self.train, do_train=do_train, bert=bert)
        else:
            logger.info("Predicting mode")
            self.test = pd.read_csv(os.fspath(ranking_bert_dev))
            self.test['text_a'] = self.test['text_a'].apply(lambda x: clean(x))
            self.test['text_b'] = self.test['text_b'].apply(lambda x: clean(x))
            self.testdata = self.generate_feature(self.test,do_train=do_train, bert=bert)

    def generate_feature(self, data, do_train=True, bert=False):
        logger.info("Generating manual features")
        data = pd.concat([data, pd.DataFrame.from_records(
            data.apply(lambda row: self.ts.generate_all(
                row['text_a'],
                row['text_b']),
                axis=1))], axis=1)
        logger.info("Generating deep matching features..")
        if bert:
            data['matching_score'] = data.apply(lambda row: self.matchingNN.predict(
                                            row['text_a'],
                                            row['text_b'])[1], axis=1)
        if do_train:
            data.to_csv(os.fspath(result_path / "gbm_train_data2.csv"), index=False)
        else:
            data.to_csv(os.fspath(result_path / "gbm_test_data2.csv"), index=False)
        return data 

class RANK(object):
    def __init__(self, train_file, dev_file, model_path):
        self.train_data = pd.read_csv(train_file)
        self.dev_data = pd.read_csv(dev_file)
        print(self.train_data.shape) # 161k
        self.model_path = model_path

        self.columns = [i for i in self.train_data.columns if 'text' not in i and 'w2v_wmd' not in i] 
        self.train_col = [i for i in self.columns if 'labels' not in i ]
        print(self.train_col)
        # 下面得到ndarray的数据集
        self.X_train, self.Y_train = self.train_data[self.train_col].values, self.train_data['labels'].values
        self.X_dev,self.Y_dev = self.dev_data[self.train_col].values, self.dev_data['labels'].values
        
    def train(self):
        self.clf = lgb.LGBMClassifier(**params)
        x_train, x_val, y_train, y_val = train_test_split(self.X_train, self.Y_train, test_size=0.2)
        logger.info('start training ...')
        st = time.time()
        # print(set(y_train.values))
        print(y_train.shape) # 129569
        self.clf.fit(x_train, y_train)
        y_pred = self.clf.predict(x_val)
        train_scores = self.clf.score(x_train, y_train)
        y_dev_pred = self.clf.predict(self.X_dev)
        et = time.time()
        logger.info('lgbm {} train scores:{} val f1score:{} dev f1score:{}'.format(et-st,\
             train_scores, f1_score(y_pred, y_val), f1_score(y_dev_pred, self.Y_dev)))
        joblib.dump(self.clf, self.model_path)

    def predict(self, data):
        self.clf = joblib.load(self.model_path)
        data = data[self.train_col]
        result = self.clf.predict_proba(data)
        return result

if __name__ == '__main__':
    train_file = result_path / "gbm_train_data2.csv"
    dev_file = result_path / "gbm_test_data2.csv"
    model_path = os.fspath(rank_path/'lgb')

    if not train_file.exists():
        data = Data(do_train=True, bert=False, train_path=train_file)
    if not dev_file.exists():
        data = Data(do_train=False, bert=False, train_path=dev_file)
    rank = RANK(train_file, dev_file, model_path)
    rank.train()