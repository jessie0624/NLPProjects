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
from src.utils.tools import get_score, create_logger, query_cut
from src.utils.feature import (get_embedding_feature, get_pretrain_embedding,
                                get_basic_feature, formate_data, get_lda_features)
# from src.utils.tools import (Grid_Train_model, bayes_parameter_opt_lgb,
#                             create_logger, formate_data, get_score)
# from src.utils.feature import (get_embedding_feature, get_img_embedding, 
#                                 get_lda_features, get_pretrain_embedding,
#                                 get_autoencoder_feature, get_basic_feature)

logger = create_logger(config.log_dir + 'model.log')


class Models(object):
    def __init__(self, feature_engineer=False):
        model_path = config.root_path + "/model/bert"
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path)
        self.bert = self.bert.to(config.device)
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
        test_tfidf, test = get_embedding_feature(self.ml_data.dev, 
                                                    self.ml_data.em.tfidf,
                                                    self.ml_data.em.w2v)
        
        logger.info("generate basic feature")
        # 获取nlp 基本特征
        train, test = get_basic_feature(train), get_basic_feature(test)

        logger.info("generate model feature")
        # # 加载图书封面的文件 not enabled here
        # cover = os.listdir(config.root_path + '/data/book_cover/book_cover/')
        # # 根据title匹配图书封面

        logger.info("generate bert feature")
        train["bert_embedding"] = train["text"].progress_apply(
            lambda x: get_pretrain_embedding(x, self.bert_tokenizer, self.bert))
        test["bert_embedding"] = test["text"].progress_apply(
            lambda x: get_pretrain_embedding(x, self.bert_tokenizer, self.bert))
        
        logger.info("generate lda feature")
        # 生成 bag of word 格式数据
        train["bow"] = train["queryCutRMStopWord"].apply(
            lambda x: self.ml_data.em.lda.id2word.doc2bow(x))
        test["bow"] = test["queryCutRMStopWord"].apply(
            lambda x: self.ml_data.em.lda.id2word.doc2bow(x))
        
        # 在bag of word基础上 得到lda embedding
        train["lda_embedding"]= map(lambda x: get_lda_features(x, self.ml_data.em.lda), train["bow"])
        test["lda_embedding"] = map(lambda x: get_lda_features(x, self.ml_data.em.lda), test["bow"])

        logger.info("formate dat")
        # 将所有特征拼接到一起
        train = formate_data(train, train_tfidf)
        test = formate_data(test, test_tfidf)
        # 生成训练,测试数据
        cols = [x for x in train.columns if str(x) not in ["labelIndex"]]
        X_train = train[cols]
        X_test = test[cols]
        train["labelIndex"] = train["labelIndex"].astype(int)
        test["labelIndex"] = test["labelIndex"].astype(int)
        y_train, y_test = train["labelIndex"], test["labelIndex"]
        return X_train, X_test, y_train, y_test

    def param_search(self, search_method='grid'):
        """
        use param search tech to find best param
        @param:
            - search_method: grid or bayesian optimization
        """
        # 使用网格搜索或者贝叶斯优化 寻找最优参数
        if search_method == "grid":
            logger.info("use grid search")
            self.model = Grid_Train_model(self.model, self.X_train,
                                          self.X_test, self.y_train,
                                          self.y_test)
        # elif search_method == 'bayesian':
        #     logger.info("use bayesian optimization")
        #     trn_data = lgb.Dataset(data=self.X_train,
        #                            label=self.y_train,
        #                            free_raw_data=False)
        #     param = bayes_parameter_opt_lgb(trn_data)
        #     logger.info("best param", param)
        #     return param


    def unbalance_helper(self, imbalance_method="under_sampling", search_method="grid"):
        """
        handle unbalance data, then search best param
        @param:
            - imbalance_method: three option, under_sampling for ClusterCentroids, \
                                SMOTE for over_sampling, ensemble for BalancedBaggingClassifier
            - search_method: two options. grid or bayesian optimization
        @return: None
        """
        logger.info("get all feature")
        self.X_train, self.X_test, self.y_train, self.y_test = self.feature_engineer()
        if imbalance_method == "over_sampling":
            logger.info("Use SMOTE deal with unbalance data ")
            self.X_train, self.y_train = SMOTE().fit_resample(self.X_train, self.y_train)
            self.X_test, self.y_test = SMOTE().fit_resample(self.X_test, self.y_test)
            model_name = "lgb_over_sampling"
        elif imbalance_method == "under_sampling":
            logger.info("Use ClusterCentroids deal with unbalance data ")
            self.X_train, self.y_train = ClusterCentroids(random_state=0).fit_resample(self.X_train, self.y_train)
            self.X_test, self.y_test = ClusterCentroids(random_state=0).fit_resample(self.X_test, self.y_test)
            model_name = "lgb_under_sampling"
        elif imbalance_method == "ensemble":
            self.model = BalanceBaggingClassifier(
                base_estimator=DecisionTreeClassifier(),
                sampling_strategy="auto",
                replacement=False,
                random_state=0)
            model_name = 'ensemble'
        
        logger.info("search best param")
        # 使用set_param 将搜索到的最优参数设置为模型的参数
        if imbalance_method != "ensemble":
            # 使用参数搜索技术
            param = {}
            param["params"] = {}
            param["params"]["num_leaves"] = 3
            param["params"]["max_depth"] = 5
            self.model = self.model.set_params(**param["params"])
        logger.info("fit model")
        self.model.fit(self.X_train, self.y_train)
        # 1.预测测试的label
        # 2.预测训练的label
        # 3.计算precision, accuracy, recall, f1_score
        Test_predict_label = self.model.predict(self.X_test)
        Train_predict_label = self.model.predict(self.X_train)
        train_acc, test_acc, recall, f1 = get_score(self.y_train, self.y_test, Train_predict_label, Test_predict_label)
        # 输出训练集的准确率
        logger.info('Train accuracy %s' % train_acc)
        # 输出测试集的准确率
        logger.info('test accuracy %s' % test_acc)
        # 输出recall
        logger.info('test recall %s' % recall)
        # 输出F1-score
        logger.info('test F1_score %s' % f1)
        self.save(model_name)        
    
    def process(self, title, desc):
        # 处理数据, 生成预测时所需要的特征
        df = pd.DataFrame([[title, desc]], columns=['title', 'desc'])
        df["text"] = df["title"] + df["desc"]
        df["queryCut"] = df["text"].apply(query_cut)
        df["queryCutRMStopWord"] = df["queryCut"].apply(
            lambda x: [word for word in x if word not in self.ml_data.em.stopWords])
        df_tfidf, df = get_embedding_feature(df, self.ml_data.em.tfidf, self.ml_data.em.w2v)
        print("generate basic feature")
        df = get_basic_feature(df)
        print("generate bert feature")
        df["bert_embedding"] = df.text.progress_apply(
            lambda x: get_pretrain_embedding(x, self.bert_tokenizer, self.bert))
        print("generate lda feature")
        df["bow"] = df["queryCutStopWord"].apply(
            lambda x: self.ml_data.em.lda.id2word.doc2bow(x))
        df["lda"] = list(map(lambda doc: get_lda_features(doc, self.ml_data.em.lda)))
        print("formate data")
        df["labelIndex"] = 1
        df = formate_data(df, df_tfidf)
        cols = [x for x in df.columns if str(x) not in ["labelIndex"]]
        X_trian = df[cols]
        return X_train

    def predict(self, title, desc):
        """
        根据输入的 title, desc 预测图书类别
        """
        inputs = self.process(title, desc)
        label = self.ix2label[self.model.predict(inputs)[0]]
        proba = np.max(self.model.predict_proba(inputs))
        return laebl, proba 
        
    def save(self, model_name):
        folder = config.root_path + "/model/ml_model/"
        if os.path.exists(folder):
            os.mkdirs(folder)
        joblib.dump(self.model_name, folder + model_name)
    
    def load(self, path):
        self.model = joblib.load(path)
