"""
Desc: 训练TFIDF, FastText, Word2vec等模型. 用于细排的特征工程.
"""
import logging
import os, sys
from collections import Counter
import jieba 
from gensim import corpora, models
import pandas as pd 
from pathlib import Path
sys.path.append("..")
import config 
from config import root_path, ranking_train, ranking_dev, \
    ranking_test, stop_words_path, rank_path

logger = logging.getLogger(__name__)


class Corpus(object):
    def __init__(self):
        self.data = self.data_reader(ranking_train) + \
                    self.data_reader(ranking_test) + \
                    self.data_reader(ranking_dev)
        print(self.data[:10])
        # self.data.to_csv(os.fspath(root_path / "result/trainlm_data.csv"), index=False)
        self.stopwords = stop_words_path.open(mode='r',encoding='utf-8').read(\
            ).strip().split('\n')
        self.preprocessor()

    def data_reader(self, path):
        samples = []
        #with open() as rf:
        df = pd.read_csv(os.fspath(path))

        samples = []
        for index in range(df.shape[0]):
            for item in list(df.iloc[index].values)[:-1]:
                samples.append(item)
        return samples

    def preprocessor(self):
        """
        @desc: 分词, 并生成tfidf 所需要的数据
        """
        logger.info("loading data ...")
        
        self.data = [[word for word in jieba.cut(sent) if word not in self.stopwords
        ] for sent in self.data]
        self.freq = Counter([word for sent in self.data for word in sent])
        self.data = [[word for word in sent if self.freq[word] > 1] for sent in self.data]
        
        logger.info(" building dictionary ...")
        self.dictionary = corpora.Dictionary(self.data)
        self.dictionary.save(os.fspath(rank_path / 'ranking.dict'))
        self.corpus = [self.dictionary.doc2bow(text) for text in self.data]
        corpora.MmCorpus.serialize(os.fspath(rank_path / 'ranking.mm'), self.corpus)


class Trainer(object):
    def __init__(self):
        self.corp = Corpus()
        self.train()
        self.saver()

    def train(self):
        logger.info(" train tfidf model ...")
        self.tfidf = models.TfidfModel(self.corp.corpus, normalize=True)
        
        logger.info(" trian word2vec model ...")
        self.w2v = models.Word2Vec(min_count=2, window=2, size=300, sample=6e-5,
                            alpha=0.03, min_alpha=0.0007, negative=15, workers=4, iter=7)
        self.w2v.build_vocab(self.corp.data)
        self.w2v.train(self.corp.data, total_examples=self.w2v.corpus_count,
                        epochs=15, report_delay=1)
        
        logger.info(" train fasttext model ...")
        self.fast = models.FastText(self.corp.data, size=300, window=3, min_count=1,
                        iter=10, min_n=3, max_n=6, word_ngrams=2)

    def saver(self):
        logger.info(' save tfidf model ...')
        self.tfidf.save(os.fspath(rank_path / 'tfidf'))

        logger.info(' save word2vec model ...')
        self.w2v.save(os.fspath(rank_path / 'w2v'))
        
        logger.info(' save fasttext model ...')
        self.fast.save(os.fspath(rank_path / 'fast'))

if __name__ == "__main__":
    Trainer()
