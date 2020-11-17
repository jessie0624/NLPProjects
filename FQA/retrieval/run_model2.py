"""
@DESC：基于model1 的3个模型tfidf, lsi, lda，增加了检索时 倒排索引的优化
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
from typing import List 
import operator
from functools import reduce

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # enable chinese

def read_corpus():
    qList = []
    qList_kw = [] #问题的关键词列表
    aList = []
    data = pd.read_csv(config.retrieval_data)[:50000] #取前100000做demo
    data_ls = np.array(data).tolist()
    for t in data_ls:
        qList.append(str(t[1]))
        qList_kw.append(seg.cut(str(t[1]))) # 去除停用词以后的
        aList.append(str(t[2]))
    return data, qList_kw, qList, aList


def plot_words(wordList):
    fDist = FreqDist(wordList)
    #print(fDist.most_common())
    print("单词总数: ",fDist.N())
    print("不同单词数: ",fDist.B())
    fDist.plot(10)


def invert_idxTable(qList_kw):  # 定一个简单的倒排表
    invertTable = {}
    for idx, tmpLst in enumerate(qList_kw):
        for kw in tmpLst:
            if kw in invertTable.keys():
                invertTable[kw].append(idx)
            else:
                invertTable[kw] = [idx]
    return invertTable


def filter_questionByInvertTab(inputQuestionKW, questionList, answerList, invertTable):
    idxLst = []
    questions = []
    answers = []
    for kw in inputQuestionKW:
        if kw in invertTable.keys():
            idxLst.extend(invertTable[kw])
    idxSet = set(idxLst)
    for idx in idxSet:
        questions.append(questionList[idx])
        answers.append(answerList[idx])
    return questions, answers

def search(seg, model, question, questionList, answerList, invertTable, num):
    if not isinstance(question, List):
        question = [question]
    # 利用关键词匹配得到与原来相似的问题集合
    y_pred = []
    for i in range(len(question)):
        queList, ansList = filter_questionByInvertTab(seg.cut(question[i]), questionList,\
                answerList, invertTable)
        ss = SentenceSimilarity(seg)
        ss.set_sentences(queList)
        if model == 'tfidf':
            ss.TfidfModel()  # tfidf模型
        elif model == 'lsi':
            ss.LsiModel()    # lsi模型
        elif model == 'lda':
            ss.LdaModel()   # lda模型
        i_pred = ss.get_recall_result(queList, num) #recall num
        y_pred.append(reduce(operator.add, i_pred))
    
    recall, wrong_index = ss.recall_topk(y_pred, list(range(num)), k)

    return recall, wrong_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="tmodel1 caculate retrieval")                           
    parser.add_argument('--eval', '-e', default=False) 
    parser.add_argument('--e_num', default=1000)
    parser.add_argument('--k', default=10)
    parser.add_argument('--model', '-m', default='tfidf', help='only support tfidf, lsi, lda')
    args = parser.parse_args() 
   
    eval_flag, model_type = args.eval, args.model
    num, k = args.e_num, args.k
    
    seg = Seg()
    #　加载jieba userdict 
    seg.load_userdict(os.fspath(config.user_dict))
    # 读取数据
    data, qList_kw, questionList, answerList = read_corpus()
    
    """简单的倒排索引"""
    # 计算倒排表
    invertTable = invert_idxTable(qList_kw)

    if eval_flag:
        ## evaluate: 
        ss = SentenceSimilarity(seg)
        # 
        eval_questions = list(data.iloc[:num]['custom']) 

        recall, wrong_index = search(seg, args.model, eval_questions, questionList, answerList, invertTable, k)
    
        for que in wrong_index:
            print(que, [questionList[i] for i in wrong_index[que]])
        print("recall result:", recall)
    else:

        while True:
            question = input("请输入问题(q退出): ")
            time1 = time.time()
            if question == 'q':
                break
            inputQuestionKW = seg.cut(question)
           # 利用关键词匹配得到与原来相似的问题集合
            questionList_s, answerList_s = filter_questionByInvertTab(inputQuestionKW, questionList, answerList,
                                                                    invertTable)
            print(questionList_s)
            # if len(questionList_s) > 1:
            #     questionList = questionList_s
            #     answerList = answerList_s
            # questionList, answerList  = search(seg, question, questionList, answerList, invertTable)

            # 初始化模型
            ss = SentenceSimilarity(seg)
            ss.set_sentences(questionList_s)
            ss.TfidfModel()  # tfidf模型
            # ss.LsiModel()         # lsi模型
            # ss.LdaModel()         # lda模型
            ss.create_sim_index()
            question_k = ss.similarity_k(question, 5)
            # print(question_k)
            # print(answerList_s)

            print("亲，我们给您找到的答案是： {}".format(answerList_s[question_k[0][0]]))
            # print(question_k)
            for idx, score in zip(*question_k):
                print("same questions： {},                score： {}".format(questionList_s[idx], score))
            time2 = time.time()
            cost = time2 - time1
            print('Time cost: {} s'.format(cost))