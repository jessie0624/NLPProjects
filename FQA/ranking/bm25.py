"""
DESC: 训练一个bm25模型 用于排序, 模型保存路径model/ranking/
"""

import math 
import os, sys
from collections import Counter  
import csv  
import jieba 
import jieba.posseg as pseg  
import numpy as np               
import pandas as pd
import joblib
from pathlib import Path
import logging
from six import iteritems 
sys.path.append("..")
import config 
from config import root_path, rank_path,  ranking_train, stop_words_path
from utils.tools import create_logger
# logger = logging.getLogger(__name__)
logger = create_logger(os.fspath(root_path/'log/ranking_bm25'))
class Corpus(object):
    """
    返回corpus list, type : List[List]
    """
    def __init__(self, train_path=ranking_train):
        self.data = pd.read_csv(os.fspath(train_path))#, sep='\t', header=None,
                # names=['question1', 'question2', 'target'])
        self.corpus = list(self.data['question2'].apply(lambda x: jieba.lcut(str(x))))

class BM25(object):
    """
    score(q, d) = \sum(i=1-n) Wi*R(qi, d)
    Wi = log((N-nqi+0.5)/(nqi + 0.5))
    R(qi,d) = (fi*(k1+1 / fi+K))(qfi*(k2+1 / qfi+k2))
    K = k1(1-d+d*(dl/avgdl))
    """
    def __init__(self, do_train=True, save_path=rank_path):
        """
        @param:
            - corpus: List[List[]]
            - do_train: True  runing trainning
        """
        self.stopwords = self.load_stop_word()
        
        if do_train:
            self.corpus_obj = Corpus()
            self.corpus = self.corpus_obj.corpus
            self.initialize()
            self.saver(save_path)
        else:
            self.load(save_path)

    def initialize(self):
        self.corpus_size = len(self.corpus)
        self.avgdl = sum(map(lambda x: float(len(x)), self.corpus))/self.corpus_size
        # 单词在每个文档中出现的频率  fi = self.tf_list[indelogger = create_logger(os.fspath(root_path/'log/train_matchnn'))x][qi]
        self.tf_list = list(map(lambda x: Counter([word for word in x]), self.corpus))
        # 单词出现的文档个数
        self.nq = Counter([word for doc in self.tf_list for word in doc])
        # 单词的逆文档频率
        self.idf = {word: math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5) for word, freq in self.nq.items()}
    
    def load_stop_word(self):
            return stop_words_path.open(mode='r',encoding='utf-8').read(\
            ).strip().split('\n')

    def saver(self, save_path):
        print('save model')
        joblib.dump(self.idf, os.fspath(save_path / 'bm25_idf.bin'))
        joblib.dump(self.avgdl, os.fspath(save_path / 'bm25_avgdl.bin'))
    
    def load(self, save_path):
        self.idf = joblib.load(os.fspath(save_path / 'bm25_idf.bin'))
        self.avgdl = joblib.load(os.fspath(save_path / 'bm25_avgdl.bin'))

    def get_score(self, query, doc, k1=1.2, k2=200, b=0.75):
        """
        q: query
        d: document
        """ 
        stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
        words = pseg.cut(query)
        fi = {}
        qfi = {}
        for word, flag in words:
            if flag not in stop_flag and word not in self.stopwords:
                fi[word] = doc.count(word)
                qfi[word] = query.count(word)
        K = k1 * (1 - b + b * (len(doc)/self.avgdl))
        ri = {}
        for key in fi:
            ri[key] = fi[key] * (k1+1) * qfi[key] * (k2+1) / ((fi[key] + K) * (qfi[key] + k2))
        score = 0
        for key in ri:
            score += self.idf.get(key, 20.0) * ri[key]
        return score


if __name__ == "__main__":
    bm25 = BM25(do_train=True)