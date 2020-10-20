"""
Desc: 建立fasttext模型,判断用户输入是否属于业务咨询
这是一个简单的二分类,实际中会涉及到多分类,
    比如业务咨询包含哪些业务咨询,或者闲聊也分为感情,天气等闲聊

目前状态 vs 优化空间
    1. 根据对话数据集,进行分类标注,标注的依据是 对话中是否包含关键. 
        1.1 关键词提取. 
            目前采用的是:基于jieba.pesg分词, 提取句子中的名词,动名词等.这是一个非常简单暴力的方式.
            优化空间: 可以根据 句法分析 提取关键词. (TODO)
        1.2 样本标注.               
            目前采用的是: 是否包含关键词 来标注句子 是咨询业务的还是闲聊的.
            优化空间: 可以加上人为标注.  (目前没有时间和人力 做不到)
   
"""
import logging 
import sys, os 
sys.path.append('..')
import fasttext 
import jieba.posseg as pseg 
import pandas as pd 
from tqdm import tqdm 
import config 
from config import root_path 
from preprocessor import clean, filter_content
from pathlib import Path 
from typing import List 

tqdm.pandas()
logger = logging.getLogger(__name__)

class Intention(object):
    def __init__(self,
                 data_path=config.train_path, # original data path
                 sku_path=config.ware_path, # sku file path 
                 model_path=None, # saved model path 
                 kw_path=None, # key word file path 
                 model_train_file=config.business_train, # path to save training data for intention.
                 model_test_file=config.business_test): # path to save test data for intention

        self.model_path = model_path
        self.data = pd.read_csv(data_path)
        if model_path and Path(model_path).exists():
            self.fast = fasttext.load_model(model_path)
        else:
            self.kw = self.build_keyword(sku_path, to_file=kw_path)
            self.data_process(model_train_file)
            self.fast = self.train(model_train_file, model_test_file)

    def build_keyword(self, sku_path, to_file):
        """
        @desc: 构建业务咨询相关关键词,并保存(读取语料中的名词和提供的sku)
        @param:
            - sku_path: sku文件路径
            - to_file: 关键词保存路径
        @return: 关键词list
        """
        logger.info("Building keywords")
        if to_file.exists():
            return set(to_file.open(mode='r').read().strip().split('\n'))
        
        tokens = []
        tokens = self.data["custom"].dropna().apply(lambda x: [
            token for token, pos in pseg.cut(x) if pos in ['n', 'vn', 'nz']
        ])
        key_words = set([tk for idx, sample in tokens.iteritems() for tk in sample if len(tk) > 1])
        logger.info("key words build.")
        lines = sku_path.open(mode='r').read().strip().split('\n')
        key_words |= set([item.strip().split('\t')[1] for item in lines[1:]])
        logger.info("Sku words merged.")
        if to_file is not None:
            with open(to_file, "w") as wf:
                for item in key_words:
                    wf.write(item + '\n')
        return key_words
    
    
    def data_process(self, model_data_file):
        """
        @desc: 判断咨询中是否包含业务关键词,如果包含label为1, 否则为0
               并处理成fasttext 需要的数据格式
        @param:
            - model_data_file: 模型训练数据保存路径
        @return
        """
        logger.info("Processing data.")
        
        if model_data_file.exists():
            return 
        
        self.data['is_bussiness'] = self.data['custom'].progress_apply(
            lambda x: 1 if any(kw in x for kw in self.kw) else 0)
        
        with open(model_data_file, "w") as wf:
            for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
                outline = clean(row['custom']) + "\t__label__" + str(int(row["is_bussiness"])) + "\n"
                wf.write(outline)
    
    def train(self, model_data_file, model_test_file):
        """
        @desc: 读取模型训练数据,并保存
        @param:
            - model_data_file: 模型训练数据位置
            - model_test_file: 模型验证文件位置
        @return:
            - fasttext model
        """
        logger.info("Trainning classifier..")
        # fasttext.train_supervised(input='cooking.train'
        classifier = fasttext.train_supervised(input=model_data_file)#, dim=100, epoch=10, \
            # lr=0.1, wordNgrams=2, loss='softmax', thread=5, verbose=True)
        self.test(classifier, model_test_file)
        classifier.save_model(self.model_path)
        logger.info("Model saved")
        return classifier 
    
    def test(self, classifier, model_test_file):
        """
        @desc: 验证模型
        @param:
            - classifier: model
            - model_test_file: 测试数据路径
        @return
        """
        logger.info("Testing trained model.")
        test = pd.read_csv(config.test_path).fillna("")
        test["is_business"] = test['custom'].apply(lambda x: 1 if any(kw in x for kw in self.kw) else 0)
        for index, row in tqdm(test.iterrows(), total=test.shape[0]):
            outline = clean(row["custom"]) + "\t__label__" + str(
                int(row['is_business']) + '\n')
            model_test_file.write_text(outline)
         
        result = classifier.test(model_test_file) 
        # fasttext.test的 output:[num_samples, top1-precision, top1-recall], 
        # 可以通过设置参数k=5, 获得top5 precision, top5 recall. 
        # The precision is the number of correct labels among the labels predicted by fastText. 
        # The recall is the number of labels that successfully were predicted, among all the real labels. 
        # F1 score
        
        print("prec {} recall {} f1 {}".format(result[1], result[2], result[1] * result[2] * 2 / (result[2] + result[1])))


    def predict(self, text):
        """
        @desc: 预测
        @param:
            - text:文本
        @return:
            - label 
            - score
        """
        logger.info("Predicting")
        label, score = self.fast.predict(clean(filter_content(text)))
        return label, score

if __name__ == "__main__":
    it = Intention(config.train_path, 
                    config.ware_path,
                    model_path=config.ft_path,
                    kw_path=config.keyword_path)
    print(it.predict("怎么申请报价呢?"))
    print(it.predict("你好呀"))
