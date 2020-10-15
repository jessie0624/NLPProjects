
import logging
import re
import time
from datetime import timedelta
from logging import handlers

from sklearn import metrics
import numpy as np
import jieba
import torch 
from tqdm import tqdm 
tqdm.pandas()


def query_cut(query):
    '''
    @description: word segment 分词
    @param {type} query: input data
    @return:
    list of cut word
    '''
    return list(jieba.cut(query))

def create_logger(log_path):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger

def get_score(Train_label, Test_label, Train_predict_label, Test_predict_label):
    """
    get model score
    return train_acc, test_acc, recall, f1
    """
    return metrics.accuracy_score(Train_label, Train_predict_label), \
            metrics.accuracy_score(Test_label, Test_predict_label), \
            metrics.recall_score(Test_label, Test_predict_label, average='micro'), \
            metrics.f1_score(Test_label, Test_predict_label, average='weighted')

def wam(sentence, w2v_model, method="mean", aggregate=True):
    """
    通过word average model 生成词向量
    sentence: 以空格分割的句子
    w2v_model: word2vec模型
    method:聚合方法
    aggregate: 是否进行聚合
    """
    arr = np.array([
        w2v_model.wv.get_vector(s) for s in sentence
        if s in w2v_model.wv.vocab.keys()
    ])
    if not aggregate:
        return arr
    if len(arr) > 0:
        if method == "mean":
            return np.mean(np.array(arr), axis=0) # 沿着句子长度的方向算均值或max 去除句子长度导致的矩阵的问题
        elif method == "max":
            return np.max(np.array(arr), axis=0)
        else:
            raise NotImplementedError
    else:
        return np.zeros(300)

def padding(indice, max_length, pad_idx=0):
    pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
    return torch.tensor(pad_indice)

def get_time_dif(start_time):
    """获取已使用的时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

