"""
@DESC：基于tfidf, lsi, lda 模型，最简单的检索实现。
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
from sentenceSimilarity import SentenceSimilarity
import argparse

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese


def read_corpus():
    qList = []
    qList_kw = [] #问题的关键词列表
    aList = []
    data = pd.read_csv(config.clean_data3)[:100000] #取前1000做demo
    data_ls = np.array(data).tolist()
    for t in data_ls:
        qList.append(str(t[0]))
        qList_kw.append(seg.cut(str(t[0]))) # 去除停用词以后的
        aList.append(str(t[1]))
    return data, qList_kw, qList, aList


def plot_words(wordList):
    fDist = FreqDist(wordList)
    #print(fDist.most_common())
    print("单词总数: ",fDist.N())
    print("不同单词数: ",fDist.B())
    fDist.plot(10)


if __name__ == '__main__':
    # 设置外部词
    parser = argparse.ArgumentParser(description="tmodel1 caculate retrieval")                           
    parser.add_argument('--eval', '-e', default=True) 
    parser.add_argument('--model', '-m', default='tfidf', help='only support tfidf, lsi, lda')
    args = parser.parse_args() 
    eval_flag, model_type = args.eval, args.model
    seg = Seg()
    seg.load_userdict(os.fspath(config.user_dict))
    # 读取数据
    data, List_kw, questionList, answerList = read_corpus()
    # 初始化模型
    ss = SentenceSimilarity(seg)
    ss.set_sentences(questionList)
    if model_type == 'tfidf':
        ss.TfidfModel()         # tfidf模型
    elif model_type == 'lsi':
        ss.LsiModel()         # lsi模型
    elif model_type == 'lda':
        ss.LdaModel()         # lda模型
    ss.create_sim_index()
    if eval_flag:
        ## evaluate: 
        eval_questions = list(data.iloc[1000:2000]['title'])
        y_pred, scores,top_3 = ss.get_recall_result(eval_questions, 10)
        recall, wrong_index = ss.recall_topk(y_pred, list(range(1000,2000)), 10)
        for que in wrong_index:
            print(questionList[que], [questionList[i] for i in wrong_index[que]])
        print("recall result:", recall)
        print("recall scores:", np.average(scores))
        print("recall top3:", np.average(top_3))
    # else:
    while True:
        # while True:
        question = input("请输入问题(q退出): ")
        if question == 'q':
            break
        time1 = time.time()
        question_k = ss.similarity_k(question, 5)
        print("亲，我们给您找到的答案是： {}".format(answerList[question_k[0][0]]))
        for idx, score in zip(*question_k):
            print("same questions： {},                score： {}".format(questionList[idx], score))
        time2 = time.time()
        cost = time2 - time1
        print('Time cost: {} s'.format(cost))
