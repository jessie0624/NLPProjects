"""
Decs: 实现text similarity
"""
import logging 
import os, sys 
import jieba
import jieba.posseg as pseg 
import numpy as np                  
from gensim import corpora, models, similarities

sys.path.append("..")
import config 
from config import rank_path
# from retrieval.hnsw_faiss import wam 
from ranking.bm25 import BM25 
from collections import Counter

logger = logging.getLogger(__name__)

class TextSimilarity(object):
    def __init__(self):
        logger.info(" load dictionary...")
        self.dictionary = corpora.Dictionary.load(os.fspath(rank_path / "ranking.dict"))

        logger.info(" load corpus")
        self.corpus = corpora.MmCorpus(os.fspath(rank_path / "ranking.mm"))

        logger.info(" load tfidf")
        self.tfidf = models.TfidfModel.load(os.fspath(rank_path / "tfidf"))
  
        logger.info(" load bm25")
        self.bm25 = BM25(do_train=False)#.load(rank_path) 

        logger.info(" load word2vec")
        self.w2v_model = models.KeyedVectors.load(os.fspath(rank_path / "w2v"))

        logger.info(" load fasttext")
        self.fasttext = models.FastText.load(os.fspath(rank_path / "fast"))
    
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

    @classmethod
    def tokenize(cls, str1):
        """
        @desc: 返回字符串的分词后的结果 用空格隔开的字符串和集合
        @param:
            - str1: 需要分词的子串
        @return:
            - List[str, set]
        """
        words = [word for word in jieba.cut(str1)]
        return [" ".join(words), set(words)]
    
    def JaccardSim(self, str1, str2):
        """
        @desc: 计算Jaccard 相似系数
        @param:
            - str1
            - str2
        @return:
            - jaccard相似度 len(s1 & s2)/len(s1|s2)
        """
        set1, set2 = self.tokenize(str1)[1], self.tokenize(str2)[1]
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
    

    def tokenSimilarity(self, str1, str2, method='w2v', sim='cos', most_common=False):
        """
        @desc: 基于分词求相似度, 默认使用cos_sim余弦相似度, 默认使用前20个最频繁的词项进行计算(wam)
        @param:
            - str1
            - str2
            - method: 词向量选择 支持w2v, tfidf, fasttext
            - sim: 相似度方法选择 支持 cos, pearson, eucl
        @return:
            - 相似度值
        """
        str1, str2 = self.tokenize(str1)[0], self.tokenize(str2)[0]
        str1_emb, str2_emb, model = None, None, None
        result = None 
        # 下面得到的就是针对分词后的str1 和 str2得到他们的embedding. 然后计算wam 沿着句子长度方向进行叠加.
        if most_common:
            str1 = " ".join([word[0] for word in Counter(str1.split()).most_common(20)])
            str2 = " ".join([word[0] for word in Counter(str2.split()).most_common(20)])
        # 获取emb 和 model
        if method in ['w2v', 'fast']: # wam
            model = self.w2v_model if method == 'w2v' else self.fasttext
            str1_emb = np.array([model.wv.get_vector(word) if word in model.wv.vocab.keys() \
                    else np.random.randn(1, 300) for word in str1.split()]).mean(axis=0).reshape(1, -1)
            str2_emb = np.array([model.wv.get_vector(word) if word in model.wv.vocab.keys() \
                    else np.random.randn(1, 300) for word in str2.split()]).mean(axis=0).reshape(1, -1)
        else:
            NotImplementedError

        if str1_emb is not None and str2_emb is not None:
            if sim == 'cos':
                result = TextSimilarity.cos_sim(str1_emb, str2_emb)
            elif sim == 'eucl':
                result = TextSimilarity.eucl_sim(str1_emb, str2_emb)
            elif sim == 'pearson':
                result = TextSimilarity.pearson_sim(str1_emb, str2_emb)
            elif sim == 'wmd' and model:
                result = model.wmdistance(str1, str2)
        return result 

    def generate_all(self, str1, str2):
        return {
            "lcs":        self.lcs(str1, str2),
            "edit_dist":  self.editDistance(str1, str2),
            "jaccard":    self.JaccardSim(str1, str2),
            "bm25":       self.bm25.get_score(str1, str2),
            "w2v_cos":    self.tokenSimilarity(str1, str2, method='w2v', sim='cos'),
            "w2v_eucl":   self.tokenSimilarity(str1, str2, method='w2v', sim='eucl'),
            "w2v_pearson":   self.tokenSimilarity(str1, str2, method='w2v', sim='pearson'),
            "w2v_wmd":       self.tokenSimilarity(str1, str2, method='w2v', sim='wmd'),
            "fast_cos":      self.tokenSimilarity(str1, str2, method='fast', sim='cos'),
            "fast_eucl":     self.tokenSimilarity(str1, str2, method='fast', sim='eucl'),
            "fast_pearson":  self.tokenSimilarity(str1, str2, method='fast', sim='pearson'),
            "fast_wmd":      self.tokenSimilarity(str1, str2, method='fast', sim='wmd'),
        }




        