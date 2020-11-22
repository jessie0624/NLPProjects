import jieba.posseg as pseg 
import codecs
import math 
from collections import Counter 
from gensim import corpora
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD
# from gensim.summarization import bm25
from gensim import corpora, models, similarities
import pandas as pd 
import numpy as np 
import joblib
from pathlib import Path
from typing import List
import os, sys, re 
sys.path.append("..")
import config 
from ranking.bm25 import BM25 
from config import root_path, rank_path,  ranking_train, stop_words_path
from utils.tools import create_logger
from utils.jiebaSegment import *
logger = create_logger(os.fspath(root_path/'log/ranking_sim'))
class SentEmbedding():
    def __init__(self):
        # super(SentEmbedding, self).__init__()
        self.dim = 200
        logger.info("loading dictionary...")
        self.dictionary = corpora.Dictionary.load(os.fspath(rank_path / "ranking.dict"))
        
        logger.info(" load corpus")
        self.corpus = corpora.MmCorpus(os.fspath(rank_path / "ranking.mm"))

        logger.info(" load tfidf/lsi/lda")
        self.tfidf = models.TfidfModel.load(os.fspath(rank_path / "tfidf"))
        self.lsi = models.LsiModel.load(os.fspath(rank_path / "lsi"))
        self.lda = models.LsiModel.load(os.fspath(rank_path / "lda"))

        logger.info(" load word2vec")
        self.w2v_model = models.KeyedVectors.load(os.fspath(rank_path / "w2v"))

        logger.info(" load fasttext")
        self.fasttext = models.FastText.load(os.fspath(rank_path / "fast"))

        logger.info(" load sif weight")
        self.sif_weight = self.load_sif_weight()

    def tokenize(self, str1):
        seg_obj = Seg()
        seg_obj.load_userdict(os.fspath(config.stop_words_path))
        words = seg_obj.cut(str1)
        # return [" ".join(words), set(words)]
        return words
    
    def remove_pc(self, X, npc, emb_type):
        file_path = os.fspath(rank_path) + "/svd_"+emb_type + ".txt"
        pc = np.loadtxt(file_path, delimiter=",")
        # print(pc.shape)
        pc = pc.reshape(1, 200)
        # print(pc)

        if npc == 1:
            XX = X - X.dot(pc.transpose()) * pc 
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
        return XX
    
    def load_sif_weight(self):
        sif_weight = dict()
        with open(config.rank_path/'sif-weight.txt', 'r') as f:
            for line in f.readlines():
                ret =line.strip().split()
                if len(ret) <2: continue
                sif_weight[ret[0]] = float(ret[1])
        return sif_weight

    def get_weight(self, sent, weight_type=None):
        ## 计算权重
        if not isinstance(sent, List):
            sent = self.tokenize(sent)

        if weight_type == None:
            weight = np.array([1.0 for _ in range(len(sent))])  
        elif weight_type == 'tfidf':
            tfidf_vec = self.tfidf[sent]
            vec_dict = dict()
            for (k, v) in tfidf_vec:
                vec_dict[k] = v
            vec = [self.dictionary.token2id[token] if token in self.dictionary.token2id else 'UNK' for token in sent ]
            weight = np.array([vec_dict[t] if t in vec_dict else 0.0 for t in vec]).reshape(1, -1)
        elif weight_type == 'sif':
            weight = np.array([self.sif_weight[word] if word in self.sif_weight else 0.0 for word in sent]).reshape(1, -1)
        return weight

    def get_embed(self, sent, emb_type='w2v', weight_type=None):
        sent = self.tokenize(sent)
        # 获取权重 
        weight = self.get_weight(sent, weight_type).reshape(1,-1)

        # 获取句子的词嵌入embed
        if emb_type in ['w2v', 'fasttext']:
            model = self.w2v_model if emb_type == 'w2v' else self.fasttext
            # 得到sent_len * dim 矩阵 根据该矩阵可以获取句向量
            sent_emb = np.array([model.wv.get_vector(word) if word in model.wv.vocab.keys() \
                else np.random.randn(self.dim) for word in sent]).reshape(-1, self.dim)
        # elif emb_type in ['tfidf', 'lsi', 'lda']:
        #     tfidf_vec = self.tfidf[sent]
        if len(sent) > 0:
            sent_vec = np.dot(weight, sent_emb)/(len(sent)) # 1*300
        else:
            sent_vec = np.random.randn(1, self.dim)
        
        if weight_type == 'sif':
            npc = 1
            sent_vec = self.remove_pc(sent_vec, npc, emb_type)
        
        return sent_vec
    
class TextSimilarity(SentEmbedding):
    def __init__(self):
        super(TextSimilarity, self).__init__()
        logger.info(" load bm25")
        self.bm25 = BM25(do_train=False)#.load(rank_path) 
    
    def lcs(self, str1, str2):
        """
        @decs: 最长公共子串 longest common substring
        @param:
            - str1: 字符串a
            - str2: 字符串b
        @return:
            - ratio: 最长公共子串 占 输入a和b中较短字符串的 比例
        """
        if not str1 or not str2: return 0

        m, n = len(str1), len(str2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1] / min(m, n)

    def editDistance(self, str1, str2):
        """
        @decs: 由str1到str2的编辑距离
        @param:
            - str1
            - str2
        @return:
            - 最小编辑距离 占 两个字符串总长度的 比例
        """
        if not str1 or not str2:
            return 1
        m, n = len(str1), len(str2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            dp[i][0] = i 
        for j in range(n+1):
            dp[0][j] = j 
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                        # dp[i-1][j-1]: 修改str2[j] 使其等于str1[i]
                        # dp[i][j-1]: 删除str2[j] 那么删除以后就等于 dp[i][j-1] + 1
                        # dp[i-1][j]: 在str2[j]插入一个str1[i],那么此处就相等了 j就自动往后移动位置,i 相对来说就是前一个位置.
        return (m + n - dp[-1][-1]) / (m + n)

    def JaccardSim(self, str1, str2):
        """
        @desc: 计算Jaccard 相似系数
        @param:
            - str1
            - str2
        @return:
            - jaccard相似度 len(s1 & s2)/len(s1|s2)
        """
        set1, set2 = set(self.tokenize(str1)), set(self.tokenize(str2))
        if len(set1 | set2)==0:
            return 0.0
        return 1.0 * len(set1 & set2) / len(set1 | set2)
    
    @staticmethod
    def cos_sim(a, b):
        a, b = np.array(a), np.array(b)
        return np.sum(a * b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))
    
    @staticmethod
    def eucl_sim(a, b):
        a, b = np.array(a), np.array(b)
        return 1 / (1 + np.sqrt((np.sum(a - b)**2)))

    @staticmethod
    def pearson_sim(a, b):
        a, b = np.array(a), np.array(b)
        a -= np.average(a)
        b -= np.average(b)    
        return np.sum(a * b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))
    
  
    def tokenSimilarity(self, str1, str2, method='w2v', weight=None, sim='cos'):
        
        str1_emb = self.get_embed(str1, method, weight)
        str2_emb = self.get_embed(str2, method, weight)
        
        if sim == 'cos':
            result = TextSimilarity.cos_sim(str1_emb, str2_emb)
        elif sim == 'eucl':
            result = TextSimilarity.eucl_sim(str1_emb, str2_emb)
        elif sim == 'pearson':
            result = TextSimilarity.pearson_sim(str1_emb, str2_emb)
        elif sim == 'wmd' and method in ['w2v', 'fasttext']:
            model = self.w2v_model if method == 'w2v' else self.fasttext
            result = model.wmdistance(" ".join(self.tokenize(str1)), " ".join(self.tokenize(str2)))
        return result
    
    def generate_all(self, str1, str2):
        return {
            "lcs":        self.lcs(str1, str2),
            "edit_dist":  self.editDistance(str1, str2),
            "jaccard":    self.JaccardSim(str1, str2),
            "bm25":       self.bm25.get_score(str1, str2),
            # "w2v_cos":    self.tokenSimilarity(str1, str2, method='w2v', weight=None, sim='cos'),
            # "w2v_eucl":   self.tokenSimilarity(str1, str2, method='w2v', weight=None, sim='eucl'),
            # "w2v_pearson":   self.tokenSimilarity(str1, str2, method='w2v', weight=None, sim='pearson'),
            "w2v_wmd":       self.tokenSimilarity(str1, str2, method='w2v',weight=None,  sim='wmd'),
            # "fast_cos":      self.tokenSimilarity(str1, str2, method='fasttext',weight=None,  sim='cos'),
            # "fast_eucl":     self.tokenSimilarity(str1, str2, method='fasttext',weight=None,  sim='eucl'),
            # "fast_pearson":  self.tokenSimilarity(str1, str2, method='fasttext',weight=None,  sim='pearson'),
            "fast_wmd":      self.tokenSimilarity(str1, str2, method='fasttext', weight=None, sim='wmd'),
            "w2v_cos_sif":    self.tokenSimilarity(str1, str2, method='w2v', weight='sif', sim='cos'),
            "w2v_eucl_sif":   self.tokenSimilarity(str1, str2, method='w2v', weight='sif', sim='eucl'),
            "w2v_pearson_sif":   self.tokenSimilarity(str1, str2, method='w2v', weight='sif', sim='pearson'),
            # "w2v_wmd":       self.tokenSimilarity(str1, str2, method='w2v',weight=None,  sim='wmd'),
            "fast_cos_sif":      self.tokenSimilarity(str1, str2, method='fasttext',weight='sif',  sim='cos'),
            "fast_eucl_sif":     self.tokenSimilarity(str1, str2, method='fasttext',weight='sif',  sim='eucl'),
            "fast_pearson_sif":  self.tokenSimilarity(str1, str2, method='fasttext',weight='sif',  sim='pearson'),
            # "fast_wmd":      self.tokenSimilarity(str1, str2, method='fasttext', weight=None, sim='wmd'),

        }
