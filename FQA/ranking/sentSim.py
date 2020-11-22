"""
1. 实现embedding 表示
2. 实现similarity 计算
"""
import jieba.posseg as pseg 
import codecs
import math 
from collections import Counter 
from gensim import corpora
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
from config import root_path, rank_path,  ranking_train, stop_words_path, ranking_train_clean
from utils.tools import create_logger
from utils.jiebaSegment import *
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD


logger = create_logger(os.fspath(root_path/'log/ranking_sentSim'))
def clean(sentence):
    sentence = re.sub(r'[0-9\.]+%?', '[数字x]', sentence)
    sentence = re.sub(r' ', '', sentence)
    sentence = re.sub(r'', '', sentence)
    sentence = re.sub(r'÷', '[符号x]', sentence)
    sentence = re.sub(r'yib', '', sentence)
    sentence = re.sub(r'＂','', sentence)
    
    return sentence

class Corpus(object):
    """
    返回corpus list, type : List[List]
    """
    def __init__(self, min_freq=1, train_path=ranking_train):
        self.seg_obj = Seg()
        self.seg_obj.load_userdict(os.fspath(config.stop_words_path))
        self.data = self.data_reader(train_path)
        self.min_freq = min_freq
        self.preprocessor()

    def data_reader(self, file_path):
        samples = []
        df = pd.read_csv(os.fspath(file_path))
        df['text_a'] = df['text_a'].apply(lambda x: clean(x))
        df['text_b'] = df['text_b'].apply(lambda x: clean(x))
        for index in range(df.shape[0]):
            for item in list(df.iloc[index].values[:-1]):
                samples.append(item)
        return list(set(samples))

    def preprocessor(self):
        logger.info("loading data..")
        self.data = [self.seg_obj.cut(sent.strip()) for sent in self.data]
        self.freq = Counter([word for sent in self.data for word in sent])
        self.data = [[word for word in sent if self.freq[word]>self.min_freq] for sent in self.data]
        logger.info("building dictionary...")
        self.dictionary = corpora.Dictionary(self.data)
        self.dictionary.save(os.fspath(rank_path/"ranking.dict"))
        self.corpus = [self.dictionary.doc2bow(text) for text in self.data]
        corpora.MmCorpus.serialize(os.fspath(rank_path/'ranking.mm'), self.corpus)

class TrainLM(Corpus):
    def __init__(self):
        super(TrainLM, self).__init__()
        self.dim = 200

    def w2v(self):
        logger.info("train word2vec model...")
        if os.path.exists(os.fspath(rank_path / 'w2v')):
            self.w2v_model = KeyedVectors.load(os.fspath(rank_path / 'w2v'))
        else:
            self.w2v_model = models.Word2Vec(min_count=2, window=2, size=self.dim, sample=6e-5,
                            alpha=0.03, min_alpha=0.0007, negative=15, workers=4, iter=7)
            self.w2v_model.build_vocab(self.data)
            self.w2v_model.train(self.data, total_examples=self.w2v_model.corpus_count,
                            epochs=15, report_delay=1)
            self.w2v_model.save(os.fspath(rank_path/'w2v'))
    
    def fasttext(self):
        logger.info("train fast text model...")
        if os.path.exists(os.fspath(rank_path / 'fast')):
            self.fasttext = KeyedVectors.load(os.fspath(rank_path / 'fast'))
        else:
            self.fasttext = models.FastText(self.data, size=self.dim, window=3, min_count=1,
                        iter=10, min_n=3, max_n=6, word_ngrams=2)
            self.fasttext.save(os.fspath(rank_path/'fast'))
    def tfidf(self):
        self.tfidf = models.TfidfModel(self.corpus, normalize=True)
        # self.tfidf_corp = self.tfidf[self.corpus]
        self.tfidf.save(os.fspath(rank_path/'tfidf'))

    def lsi(self):
        self.lsi = models.LsiModel(self.corpus)
        # self.lsi_corp = self.lsi[self.corpus]
        self.lsi.save(os.fspath(rank_path/'lsi'))
    
    def lda(self):
        self.lda = models.LdaModel(self.corpus)
        # self.lda_corp = self.lda[self.corpus]
        self.lda.save(os.fspath(rank_path/'lda'))

    def get_sif_weight(self, a=3e-4):
        """
        根据每个词频计算sif权重
        """
        if os.path.exists(config.rank_path/'sif-weight.txt'):
            self.sif_weight = self.load_sif_weight()
            return self.sif_weight

        N = sum(self.freq.values())
        self.sif_weight = dict()
        with open(config.rank_path/'sif-weight.txt', 'wb') as f:
            for key, value in self.freq.items():
                self.sif_weight[key] = a/(a + value/N)
                f.write('{0} {1}'.format(key, str(self.sif_weight.get(key))).encode('utf-8'))
                f.write('\n'.encode('utf-8'))
        logger.info('SIF 权重更新完成')
        return self.sif_weight

    def load_sif_weight(self):
        sif_weight = dict()
        with open(config.rank_path/'sif-weight.txt', 'r') as f:
            for line in f.readlines():
                ret =line.strip().split(' ')
                # print(ret)
                sif_weight[ret[0]] = float(ret[1])
        return sif_weight
    
    def compute_pc(self, X, npc, emb_type):
        svd = TruncatedSVD(n_components=npc, n_iter=5, random_state=0)
        svd.fit(X)
        return svd.components_

    def sif_no_rem(self, emb_type='w2v'):
        sent_vec = []
        if emb_type == 'w2v': model = self.w2v_model
        elif emb_type =='fasttext':model = self.fasttext
    
        for index, sent in enumerate(self.data): # 计算每个句子的句向量得到N * 300 的 句向量库。
            # assert len(sent) != 0, print(sent)
            emb = np.array([model.wv.get_vector(word) if word in model.wv.vocab.keys() \
                    else np.random.randn(self.dim) for word in sent]).reshape(-1, self.dim) # Seq * 300 
            weight = np.array([self.sif_weight[word] for word in sent])
            # print(np.dot(weight, emb).shape) 句子为在去除停用词或者结巴分词后为空
            if len(sent) > 0:
                sent_vec.append(np.dot(weight, emb)/(len(sent)))
            else:
                sent_vec.append(np.dot(weight, emb)/(len(sent)+1))
        return np.stack(sent_vec).reshape(-1, self.dim)

    def train(self):
        self.w2v()
        self.fasttext()
        self.tfidf()
        self.lsi()
        self.lda()
        self.get_sif_weight()
        npc = 1
        emb_type = 'w2v'
        svd_com_w2v = self.compute_pc(self.sif_no_rem('w2v'), npc, 'w2v')
        np.savetxt(os.fspath(rank_path) +"/svd_"+emb_type + ".txt", np.array(svd_com_w2v),  delimiter=",")
        emb_type = 'fasttext'
        svd_com_ft = self.compute_pc(self.sif_no_rem('fasttext'), npc, 'fasttext')
        np.savetxt(os.fspath(rank_path) +"/svd_"+emb_type + ".txt", np.array(svd_com_ft),  delimiter=",")
        print(svd_com_ft[:10])
        print('svd w2v:')
        print(svd_com_w2v[:10])
        logger.info("train LM done...")


if __name__ == "__main__":
    tlm = TrainLM()
    tlm.train()
