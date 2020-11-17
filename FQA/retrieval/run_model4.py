"""
@DESC: 该模块是实现基于句向量的HNSW的召回

句向量的训练方法有： 
- SIF + W2V w/o 去除主成分
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
from sentenceSimilarity import SentenceSimilarity, HNSW, SIF 
import argparse
from typing import List 
import operator
from functools import reduce

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese

def read_corpus():
    qList = []
    qList_kw = [] #问题的关键词列表
    aList = []
    data = pd.read_csv(config.clean_data)[:100000] #取前100000做demo
    data_ls = np.array(data).tolist()
    for t in data_ls:
        qList.append(str(t[0]))
        qList_kw.append(seg.cut(str(t[0]))) # 去除停用词以后的
        aList.append(str(t[1]))
    return data, qList_kw, qList, aList


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
    ss = SIF(seg)
    # ss.set_sentences(questionList, False)
    ss.set_sentences(questionList, True)
    ss.word2vec()
    ss.build_hnsw() #得到index
    if eval_flag:
        wrong_ret = ss.evaluate()
        print("wrong recall")
        for index in wrong_ret:
            print(questionList[index])
            print([questionList[val] for val in wrong_ret[index]])
    # while True:
    #     sent1 = input("input sent1:")
    #     sent2 = input("input sent2:")
    #     ss.cossim(sent1, sent2)

    # while True:
    #     question = input("请输入问题(q退出): ")
    #     if question == 'q':
    #         break
    #     time1 = time.time()
    #     I, D = ss.search(question)
    #     print("亲，我们给您找到的答案是： {}".format(answerList[I[0][0]]))
            
    #     time2 = time.time()
    #     cost = time2 - time1
    #     print('Time cost: {} s'.format(cost))
    #     for i in range(len(I[0])):
    #         print("same questions： {},                distance: {}".format(questionList[I[0][i]],  D[0][i]))