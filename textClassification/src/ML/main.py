import argparse

from __init__ import *
from src.utils import config
from src.utils.tools import create_logger
from src.ML.models import Models


parser = argparse.ArgumentParser(description="chinese text classification")
parser.add_argument("--feature_engineering", 
                    default = False,
                    type = bool,
                    required = False,
                    help = 'where us feature engineering')
parser.add_argument('--search_method',
                    default = 'grid',
                    type = str,
                    required = False,
                    help = 'grid / bayesian optimzation')

parser.add_argument('--unbalance',
                     default = True,
                     type = bool,
                     required = False,
                     help = 'wether use imbalance tech')

parser.add_argument('--imbalance_method',
                     default = 'other',
                     type = str,
                     required = False,
                     help = 'under_sampling, over_sampling, ensemble, other')

parser.add_argument('--model_name',
                     default = 'lgb_under_sampling',
                     type = str,
                     required = False,
                     help = 'model name')

args = parser.parse_args()

logger = create_logger(config.root_path + '/logs/main.log')

if __name__ == "__main__":
    # feature_engineering = args.feature_engineering
    feature_engineering = True
    m = Models()
    if feature_engineering:
        m.unbalance_helper(imbalance_method=args.imbalance_method, search_method=args.search_method)
    else:
        X_train, X_test, y_train, y_test = m.ml_data.process_data(method='tfidf') # n*5773(idf_size)
        logger.info('model select with tfidf')
        m.model_select(X_train, X_test, y_train, y_test, feature_method='tfidf')

        X_train, X_test, y_train, y_test = m.ml_data.process_data(method='word2vec') # n*300 学习的word2vec embedding
        logger.info('model select with word2vec')
        m.model_select(X_train, X_test, y_train, y_test, feature_method='word2vec')

        X_train, X_test, y_train, y_test = m.ml_data.process_data(method='fasttext') # n*300 word2vec embedding
        logger.info('model select with fasttext')
        m.model_select(X_train, X_test, y_train, y_test, feature_method='fasttext')
