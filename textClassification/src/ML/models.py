"""
This ML models for text classification
- if using feature_engineer, we just train lgb model
- if not using feature_engineer, then we can train RF, LR, NB, SVM, LGB models  to compare the result.
"""
import os 
import lightgbm as lgb
import torchvision
import json
import pandas as pd 
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from transformers import BertModel,BertTokenizer

from src.data.mlData import MLData
from src.utils import config
from src.utils.config import root_path
from src.utils.tools import get_score, create_logger
# from src.utils.tools import (Grid_Train_model, bayes_parameter_opt_lgb,
#                             create_logger, formate_data, get_score)
# from src.utils.feature import (get_embedding_feature, get_img_embedding, 
#                                 get_lda_features, get_pretrain_embedding,
#                                 get_autoencoder_feature, get_basic_feature)

logger = create_logger(config.log_dir + 'model.log')


class Models(object):
    def __init__(self, feature_engineer=False):
        
        self.ml_data = MLData(debug_mode=True)
        if feature_engineer:
            self.model = lgb.LGBMClassifier(objective="nulticlass",
                                            n_jobs=10,
                                            num_class=33,
                                            num_leaves=30,
                                            reg_alpha=10,
                                            reg_lambda=200,
                                            max_depth=3,
                                            learning_rate=0.05,
                                            n_estimators=2000,
                                            bggging_freq=1,
                                            bagging_fraction=0.9,
                                            feature_fraction=0.8,
                                            seed=1440)
        else:
            self.models = [
                RandomForestClassifier(n_estimators=500,
                                       max_depth=5,
                                       random_state=0),
                LogisticRegression(solver='liblinear', random_state=0),
                MultinomialNB(),
                SVC(),
                lgb.LGBMClassifier(objective='multiclass',
                                   n_jobs=10,
                                   num_class=33,
                                   num_leaves=30,
                                   reg_alpha=10,
                                   reg_lambda=200,
                                   max_depth=3,
                                   learning_rate=0.05,
                                   n_estimators=2000,
                                   bagging_freq=1,
                                   bagging_fraction=0.8,
                                   feature_fraction=0.8),
            ]
        
    def model_select(self, X_train, X_test, y_train, y_test, feature_method="tfidf"):
        """
        using different embedding features to train common ML models
        feature_method: tfidf, word2vec, fasttext
        此处需要注意的是, MultinomialNB() 适合使用TFIDF,CountVectorizer 词向量,尤其是CountVectorizer, 符合MultinomialNB的定义. 不能用word2vec作为词向量特征因为 训练样本不能为负数.
        # MultinomialNB() 训练样本 不能是负数,
        # 多项式分布擅长的是分类型变量，在其原理假设中，P(xi|Y)的概率是离散的，并且不同xi下的P(xi|Y)相互独立，互不不影响。虽然sklearn中的多项式分布也可以处理连续型变量，但现实中，
        # 如果我们真的想要处理连续型变量，应当使用高斯朴素贝叶斯。
        # 多项式实验中的实验结果都很具体，它所涉及的特征往往是次数，频率，计数，出现与否这样的概念，这些概念都是离散的正整数，因此，sklearn中的多项式朴素贝叶斯不接受负值的输入。
        # 由于这样的特性，多项式朴素贝叶斯的特征矩阵经常是稀疏矩阵（不一定总是稀疏矩阵），并且它经常被用于文本分类。我们可以使用著名的TF-IDF向量技术，也可以使用常见并且简单的单词计数向量手段与贝叶斯配合使用。这两种手段都属于常见的文本特征提取的方法，可以很简单地通过sklearn来实现。
        # 从数学的角度来看，在一种标签类别Y=c下，有一组分别对应特征的参数向量θ(1,n)，其中n表示特征的总数。一个θci表示这个标签类别下的第i个特征所对应的参数。这个参数被我们定义为：
        # $$θ_{c_i} = \frac {特征Xi在Y=特征在c这个分类下的所有样本的取值总和} {所有特征在Y=特征在c这个分类下的所有样本的取值总和}$$
        """
        for model in self.models:
            model_name = model.__class__.__name__
            print(model_name)
            if model_name == 'MultinomialNB' and feature_method != 'tfidf':
                print("skip")
                continue
            
            # print(X_train.shape, y_train.shape)
            clf = model.fit(X_train, y_train)
            Test_predict_label = clf.predict(X_test)
            Train_predict_label = clf.predict(X_train)
            train_acc, test_acc, recall, f1 = get_score(y_train, y_test, 
                                            Train_predict_label,
                                            Test_predict_label)
                        
            logger.info(model_name + '_' + 'Train accuracy %s' % train_acc)
            logger.info(model_name + '_' + 'Test accuracy  %s' % test_acc)
            logger.info(model_name + '_' + 'Test recall    %s' % recall)
            logger.info(model_name + '_' + 'Test f1        %s' % f1)
    

    def feature_engineer(self):
        """
        building all kinds of features
        return:  X_train, X_test, y_train, y_test
        """
        logger.info("generate embedding feature")
        # 获取tfifd, word2ve feature
        train_tfidf, train = get_embedding_feature(self.ml_data.train, 
                                                    self.ml_data.em.tfidf,
                                                    self.ml_data.em.w2v)

    