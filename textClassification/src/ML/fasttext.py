import pandas as pd 
from tqdm import tqdm 
import fasttext
from fasttext import train_supervised
import os
from __init__ import *
from src.utils.config import root_path
from src.utils.tools import create_logger

logger = create_logger(root_path + '/logs/Fasttext.log')

class Fasttext(object):
    def __init__(self,
                 train_raw_path=root_path + "/data/train_clean.csv",
                 test_raw_path=root_path + "/data/test_clean.csv",
                 model_train_file=root_path + '/data/fast_train.csv',
                 model_test_file=root_path + '/data/fast_test.csv',
                 model_path=None):
        """
        Init the class,
        if model_path is None, then train a fasttext model, 
        else load model then predict

        @param:
            - model_path: fasttext model path
        """
        if model_path is None:
            self.Train_raw_data = pd.read_csv(train_raw_path, sep='\t')
            self.Test_raw_data = pd.read_csv(test_raw_path, sep='\t')
            self.data_process(self.Train_raw_data, model_train_file)
            self.data_process(self.Test_raw_data, model_test_file)
            self.train(model_train_file, model_test_file)
        else:
            self.fast = fasttext.load_model(model_path)

    def data_process(self, raw_data, model_data_file):
        """
        process raw data and save to model_data_file
        """
        if os.path.exists(model_data_file):
            return 
        with open(model_data_file, 'w') as f:
            for index, row in tqdm(raw_data.iterrows(), total=raw_data.shape[0]):
                outline = row['text'] + "\t__label__" + str(int(row["category_id"])) + '\n'
                f.write(outline)

    def train(self, model_train_file, model_test_file):
        self.classifier = fasttext.train_supervised(model_train_file, label="__label__", dim=50, epoch=5, lr=0.1, wordNgrams=2, loss='softmax', thread=50, verbose=True)
        self.test(model_train_file, model_test_file)
        self.classifier.save_model(root_path + '/model/fasttext.model')
    
    def test(self, model_train_file, model_test_file):
        test_result = self.classifier.test(model_test_file)
        train_result = self.classifier.test(model_train_file)
        print(test_result[1], test_result[2])
        print(train_result[1], train_result[2])

if __name__ == '__main__':
    content = Fasttext()