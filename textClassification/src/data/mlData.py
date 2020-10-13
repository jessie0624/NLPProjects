import numpy as np
import pandas as pd 
import json
import os
from __init__ import * 
from src.utils import config
from src.utils.tools import create_logger, wam, query_cut
from src.word2vec.embedding import Embedding

logger = create_logger(config.log_dir + 'data.log')

class MLData(object):

    def __init__(self, debug_mode=False):
        """
        init ML dataset,
        @param: debug_mode: if debug_mode, only deal 10000 data
        """
        self.debug_mode = debug_mode
        self.em = Embedding()
        self.em.load()
        self.preprocessor()
    
    def preprocessor(self):
        """
        process data, segment, transform label to id
        """
        logger.info("load data")
        self.train = pd.read_csv(config.root_path + '/data/train_clean.csv', sep='\t').dropna()
        self.dev = pd.read_csv(config.root_path + '/data/dev_clean.csv', sep='\t').dropna()
        
        if self.debug_mode:
            self.train = self.train.sample(n=1000).reset_index(drop=True)
            self.dev = self.dev.sample(n=100).reset_index(drop=True)
        
        # 1. 分词 (由于数据已经分好词,这里就跳过分词, 如果原数据没有分词,可以通过 jieba.cut() 来分词)
        # 2. 去除停用词
        # self.stopWords = open(config.root_path + '/data/stopwords.txt', encoding="utf-8").read().splitlines()
        self.train["queryCutRMStopWord"] = self.train.text.apply(
            lambda x: [word for word in x.split(" ") if word not in self.em.stopWords])
        self.dev["queryCutRMStopWord"] = self.dev.text.apply(
            lambda x: [word for word in x.split(" ") if word not in self.em.stopWords])
        # 3. 将label转成id
        if os.path.exists(config.root_path + '/data/label2id.json'):
            labelNameToIndex = json.load(
                open(config.root_path + '/data/label2id.json', encoding="utf-8"))
        else:
            labelName = self.train["label"].unique() # 全部的label
            labelIndex = list(range(len(labelName))) # 全部的label标签
            labelNameToIndex = dict(zip(labelName, labelIndex)) # label 标签对应 index
            with open(config.root_path + '/data/label2id.json', 'w', encoding='utf-8') as f:
                json.dump({k: v for k, v in labelNameToIndex.items()}, f)
        self.train["labelIndex"] = self.train["label"].map(labelNameToIndex)
        self.dev["labelIndex"] = self.dev["label"].map(labelNameToIndex)

    def process_data(self, method="word2vec"):
        """
        generate date use for sklearn
        method: word2vec, tfidf, fasttext
        return X_train, X_test, y_train, y_test
        """
        X_train = self.get_feature(self.train, method)
        X_test = self.get_feature(self.dev, method)
        y_train = self.train["labelIndex"]
        y_test = self.dev["labelIndex"]
        return X_train, X_test, y_train, y_test

    def get_feature(self, data, method="word2vec"):
        """
        generate feature
        data: input dataset
        method: word2vec, tfidf, fasttext
        """
        if method == "tfidf":
            data = [" ".join(query) for query in data["queryCutRMStopWord"]]
            return self.em.tfidf.transform(data)
        elif method == "word2vec":
            return np.vstack(data["queryCutRMStopWord"].apply(lambda x: wam(x, self.em.w2v)))
        elif method == "fasttext":
            return np.vstack(data["queryCutRMStopWord"].apply(lambda x: wam(x, self.em.fast)))
        else:
            NotImplementedError