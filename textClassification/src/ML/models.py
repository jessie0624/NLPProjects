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
            pass
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
                                   n_estimator=2000,
                                   bagging_freq=1,
                                   bagging_fraction=0.8,
                                   feature_fraction=0.8),
            ]
        
    def model_select(self, X_train, X_test, y_train, y_test, feature_method="tf-idf"):
        """
        using different embedding features to train common ML models
        feature_method: tfidf, word2vec, fasttext
        """
        for model in self.models:
            model_name = model.__class__.__name__
            print(model_name)
            clf = model.fit(X_train, y_train)
            Test_predict_label = clf.predict(X_test)
            Train_predict_label = clf.pretdict(X_train)
            train_acc, test_acc, recall, f1 = get_score(y_train, y_test, 
                                            Train_predict_label,
                                            Test_predict_label)
                        
            logger.info(model_name + '_' + 'Train accuracy %s' % train_acc)
            logger.info(model_name + '_' + 'Test accuracy  %s' % test_acc)
            logger.info(model_name + '_' + 'Test recall    %s' % recall)
            logger.info(model_name + '_' + 'Test f1        %s' % f1)
    

