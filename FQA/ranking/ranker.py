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
import csv
import logging

import lightgbm as lgb
import pandas as pd  
import joblib 
from tqdm import tqdm 
from pathlib import Path

sys.path.append("..")
from config import root_path,result_path, ranking_bert_train, ranking_bert_dev
from ranking.matchnn import MatchingNN
from ranking.similarity import TextSimilarity 
# from retrieval.hnsw_faiss import wam 

from sklearn.model_selection import train_test_split
import numpy as np                  

logger = logging.getLogger(__name__)

tqdm.pandas()

# parameters for lightgbm
params = {
    "boosting_type":  "gbdt", 
    "max_depth":       5,
    "objective":      "lambdarank",
    "nthread":         3,
    "num_leaves":      64,
    "learning_rate":   0.05,
    "max_bin":         512,
    "subsample_for_bin": 200,
    "subsample":        0.5,
    "subsample_freq":   5,
    "colsample_bytree": 0.8,
    "reg_alpha":        5,
    "reg_lambda":       10,
    "min_split_gain":   0.5,
    "min_child_weight":  1,
    "min_child_samples": 5,
    "scale_pos_weight":  1,
    "max_position":      20,
    "group":             "name:groupId",
    "metric":            "auc"
}

class RANK(object):
    def __init__(self, do_train=True,bert=False, model_path=os.fspath(root_path / "model/ranking/lightgbm_wo_tfidf")):
        self.ts = TextSimilarity()
        if bert:
            self.matchingNN = MatchingNN()
        if do_train:
            logger.info("Training mode")
            self.train = pd.read_csv(os.fspath(ranking_bert_train))
                                    # sep='\t',
                                    # header=None,
                                    # nrows=10000,
                                    # names=['question1', 'question2', 'target'])
            self.data = self.generate_feature(self.train, do_train=do_train, bert=bert)
            self.columns = [i for i in self.train.columns if 'text' not in i]
            self.trainer()
            self.save(model_path)
        else:
            logger.info("Predicting mode")
            self.test = pd.read_csv(os.fspath(ranking_bert_dev))
                                    # sep='\t',
                                    # header=None,
                                    # names=['question1', 'question2', 'target'])
            self.testdata = self.generate_feature(self.test,do_train=do_train, bert=bert)
            self.gbm = joblib.load(model_path)
            self.predict(self.testdata)

    
    def generate_feature(self, data, do_train=True, bert=False):
        """
        @desc: 生成模型训练所需要的特征 
               包括:similarity 里面提到的所有embedding 与 评估算法组合得到的 相似度得分, 以及采用BERT进行相似度预测的得分.
        @param:
            - data: dataframe(原始数据集 包含 question1, question2, label)
        @return: 
            - ret: dataframe(增加相似度得分的dataframe)
        """
        logger.info("Generating manual features")
        if do_train:
            data_path = Path(result_path/"gbm_data.csv")
            if data_path.exists():
                data = pd.read_csv(os.fspath(data_path))
                columns = [i for i in data.columns if 'tfidf' not in i]
                print(columns)
                return data[columns]


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
            data.to_csv(os.fspath(result_path / "gbm_data.csv"), index=False)
        else:
            data.to_csv(os.fspath(result_path / "ranker_test_data.csv"), index=False)
        return data 
    
    def trainer(self):
        logger.info("Training lightgbm model.")
        self.gbm = lgb.LGBMRanker(metric='auc')
        columns = [i for i in self.data.columns if i not in ['text_a', 'text_b', 'labels']]
        X_train, X_test, y_train, y_test = train_test_split(
                                                    self.data[columns], 
                                                    self.data['labels'],
                                                    test_size=0.3,
                                                    random_state=42)
        query_train = [X_train.shape[0]]
        query_val = [X_test.shape[0]]
        print(query_train)
        print(query_val)
        self.gbm.fit(X_train, y_train, 
                     group=query_train, 
                     eval_set=[(X_test, y_test)],
                     eval_group=[query_val],
                     eval_at=[5, 10, 20],
                     early_stopping_rounds=50)
        
    def save(self, model_path):
        logger.info("Saving lightgbm model.")
        joblib.dump(self.gbm, model_path)
    
    def predict(self, data):
        """
        @desc: 预测
        @param:
            - data: dataframe generate_feature 的输出
        @return:
            - result[list]: 所有query-candidate对儿的得分
        """
        columns = [i for i in data.columns if i not in ['text_a', 'text_b', 'labels']]
        result = self.gbm.predict(data[columns])
        print(data.shape, result.shape)
        return result 

if __name__ == '__main__':
    rank = RANK(do_train=True,bert=False)