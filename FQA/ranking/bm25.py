import jieba.posseg as pseg 
import codecs
import math 
from collections import Counter 
from gensim import corpora
from gensim.summarization import bm25
import pandas as pd 
import numpy as np 
import joblib
from pathlib import Path
import os, sys, re 
sys.path.append("..")
import config 
from config import root_path, rank_path,  ranking_train, stop_words_path
from utils.tools import create_logger
from utils.jiebaSegment import *

logger = create_logger(os.fspath(root_path/'log/ranking_bm25'))
class Corpus(object):
    """
    返回corpus list, type : List[List]
    """
    def __init__(self, train_path=ranking_train):
        self.seg_obj = Seg()
        # self.seg_obj.load_userdict(os.fspath(config.user_dict))
        # 结巴分词后的停用词词性[标点符号，连词，助词，副词，介词，时语素，的，数词，方位词，代词]
        self.stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']
        if not Path(rank_path/"bm25_corpus.txt").exists():
            self.data = pd.read_csv(os.fspath(train_path))
            # self.data['len_b'] = self.data['text_b'].apply(lambda x: self.tokenization(str(x)))
            # self.data[self.data['len_b']>0].to_csv(os.fspath(train_path), index=False)
            self.corpus = list(self.data['text_b'].apply(lambda x: self.tokenization(str(x))))
            self.save_corpus()
        else:
            self.corpus = self.load_corpus()
            print('loading done')

    def tokenization(self, text):
        result = []
        words = pseg.cut(text)
        for word, flag in words:
            if flag not in self.stop_flag and word not in self.seg_obj.stopwords:
                result.append(word)
        return result 
    
    def save_corpus(self):
        with open(Path(rank_path/"bm25_corpus.txt"), 'w') as wf:
            for item in self.corpus:
                for word in item:
                    wf.write(word + ' ')
                wf.write('\n')                

    def load_corpus(self):
        ret = []
        with open(Path(rank_path/"bm25_corpus.txt"), 'r') as rf:
            for line in rf.readlines():
                cur = line.strip().split(' ')
                ret.append(cur)
        return ret 

class BM25(object):
    def __init__(self, do_train=True, save_path=rank_path):
        self.corpus_obj = Corpus()
        self.corpus = self.corpus_obj.corpus
        if do_train:
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
        # words = pseg.cut(query)
        words = self.corpus_obj.tokenization(query)
        fi = {}
        qfi = {}
        for word in words:
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
    
    def test(self):
        data = pd.read_csv(os.fspath(ranking_train))
        data['score'] = data.apply(lambda x: self.get_score(str(x['text_a']), str(x['text_b'])), axis=1)
        data.to_csv(os.fspath(rank_path/"bm25_test.csv"), index=False)
        score = data['score'].values
        labels = data['labels'].values
    
        pccs = np.corrcoef(score, labels)
        print('pccs: ', pccs)

if __name__ == "__main__":
    bm25 = BM25(do_train=True)
    bm25.test()
