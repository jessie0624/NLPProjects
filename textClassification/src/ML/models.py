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
from imblearn.ensemble import BalanceBaggingClassifier
from imblearn.over_sample import SMOTE
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
# from src.utils.tools import (Grid_Train_model, bayes_parameter_opt_lgb,
#                             create_logger, formate_data, get_score)
# from src.utils.feature import (get_embedding_feature, get_img_embedding, 
#                                 get_lda_features, get_pretrain_embedding,
#                                 get_autoencoder_feature, get_basic_feature)

logger = create_logger(config.log_dir + 'model.log')


class Model(object):
    pass