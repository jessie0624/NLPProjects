"""
@DESC: 该模块是实现基于句向量的HNSW的召回

句向量的训练方法有： 
- TFIDF + W2V 的句向量
- WAM 的W2V的句向量
"""
import os, sys 
import pandas as pd
import matplotlib as mpl
import numpy as np
from nltk.probability import FreqDist
import time
sys.path.append('..')
import config
from utils.jiebaSegment import *
from sentenceSimilarity import SentenceSimilarity, HNSW
import argparse
from typing import List 
import operator
from functools import reduce

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese

def read_corpus():
    qList = []
    qList_kw = [] #问题的关键词列表
    aList = []
    data = pd.read_csv(config.retrieval_data)[:10000] #取前100000做demo
    data_ls = np.array(data).tolist()
    for t in data_ls:
        qList.append(str(t[1]))
        qList_kw.append(seg.cut(str(t[1]))) # 去除停用词以后的
        aList.append(str(t[2]))
    return data, qList_kw, qList, aList

# def train_w2v():
#     seg = Seg()
#     seg.load_userdict(os.fspath(config.user_dict))
#     # 读取数据
#     data, List_kw, questionList, answerList = read_corpus()
#     ss = SentenceSimilarity(seg)
#     ss.set_sentences(questionList)


if __name__ == '__main__':
    # 设置外部词
    parser = argparse.ArgumentParser(description="tmodel1 caculate retrieval")                           
    parser.add_argument('--eval', '-e', default=True) 
    parser.add_argument('--model', '-m', default='w2v', help='only support w2v')
    args = parser.parse_args() 
    eval_flag, model_type = args.eval, args.model
    seg = Seg()
    seg.load_userdict(os.fspath(config.user_dict))
    # 读取数据
    data, List_kw, questionList, answerList = read_corpus()
    # 初始化模型
    # ss = SentenceSimilarity(seg)
    ss = HNSW(seg)
    ss.set_sentences(questionList)
    ss.word2vec()
    # hnsw = HNSW(seg)
    ss.build_hnsw() #得到index
    if eval_flag:
        wrong_ret = ss.evaluate()
        print("wrong recall")
        for index in wrong_ret:
            print(questionList[index])
            print([questionList[val] for val in wrong_ret[index]])

    while True:
        question = input("请输入问题(q退出): ")
        if question == 'q':
            break
        time1 = time.time()
        I, D = ss.search(question)
        print("亲，我们给您找到的答案是： {}".format(answerList[I[0][0]]))
            
        time2 = time.time()
        cost = time2 - time1
        print('Time cost: {} s'.format(cost))
        for i in range(len(I[0])):
            print("same questions： {},                distance: {}".format(questionList[I[0][i]],  D[0][i]))
            


        #for i in len(I):
        # print(questionList[I[0]],answerList[I[0]], D[0])
            # question_k = ss.similarity_k(question, 5)
            # print("亲，我们给您找到的答案是： {}".format(answerList[question_k[0][0]]))
            # for idx, score in zip(*question_k):
            #     print("same questions： {},                score： {}".format(questionList[idx], score))
            # time2 = time.time()
            # cost = time2 - time1
            # print('Time cost: {} s'.format(cost))
       