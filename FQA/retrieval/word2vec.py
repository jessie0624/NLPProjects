"""
Desc: 训练一个word2vec 词嵌入模型
目前用的是输入数据为空格切分的单个字, 通过word2vec找到Phrases 得到对应的词语,然后进行Word2vec模型训练

可以改进的方式: 可以做个对比  用结巴分词得到的词向量与这里的结果对比. 注意 这里 对语料的处理 [数字x] [SEP]等.
"""
import logging 
import multiprocessing 
import os, sys 
from time import time 
import pandas as pd 
from gensim.models import Word2Vec 
from gensim.models.phrases import Phraser, Phrases
sys.path.append('..')
import config
from config import root_path, train_raw, w2v_path
from utils.preprocessor import clean, read_file
from utils.tools import create_logger
# logger = logging.getLogger(__name__)
logger = create_logger(os.fspath(root_path/'log/retrieval_w2v'))
def read_data(file_path):
    """
    @desc: 读取数据,清洗
    @param:
        - file_path: 文件路径
    @return:
        - Training samples.
    """ 
    train = pd.DataFrame(read_file(file_path, True),
                        columns=['session_id', 'role', 'content']) # session_id, role, content
    train['clean_content'] = train['content'].apply(clean)
    return train

def train_w2v(train, to_file, epoch=5):
    """
    @desc: 训练word2vec model 保存模型
    @param:
        - train: 数据集dataframe
        - to_file: 模型保存路径
    @return: None
    """
    sent = [row.split() for row in train['clean_content']]
    phrases = Phrases(sent, min_count=5, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]
    print(sentences)

    # cores = multiprocessing.cpu_count()

    # # if os.path.exists(to_file):
    # #     w2v_model = gensim.models.Word2Vec.load(to_file)
    # # else:
    # w2v_model = Word2Vec(min_count=2, window=2, size=300, sample=6e-5, \
    #         alpha=0.03, min_alpha=0.0007, negative=15, workers=cores-1, iter=7, compute_loss=True)
    # t = time()
    # w2v_model.build_vocab(sentences)
    # print("Time to build vocab: {} mins".format(round((time() - t)/60, 2)))

    # t = time()
    # w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=epoch, report_delay=1)
    
    # print("Time to train vocab: {} mins".format(round((time() - t)/60, 2)))
    # w2v_model.save(os.fspath(to_file))
    # return w2v_model.get_latest_training_loss()

if __name__ == "__main__":
    train = read_data(train_raw)
    train_w2v(train, os.fspath(w2v_path) + str(111), 1)
    # for epoch in [1, 5, 10, 15]:
    #     logger.info('epoch : ' + str(epoch))
    #     logger.info(train_w2v(train, os.fspath(w2v_path) + str(epoch), epoch))

