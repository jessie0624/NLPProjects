"""
Desc: 采用faiss的HNSW包,训练一个NHSW召回模型
"""
import sys, os      
import time 
import numpy as np          
import pandas as pd 
from gensim.models import KeyedVectors
import faiss 

sys.path.append('..')
import config 
from preprocessor import clean     
import logging                 

logger = logging.getLogger(__name__)

def wam(sentence, w2v_model):
    """
    @desc: 通过word average model 生成句向量
    @param:
        - sentence: 以空格分割的句子
        - w2v_model: word2vec模型
    @return:
        - the sentence vector
    """
    arr = []
    miss_count, valid_count = 0, 0
    for word in clean(sentence).split():
        if word in w2v_model.wv.vocab.keys():
            arr.append(w2v_model.wv.get_vector(word))
            valid_count += 1
        else:
            # logger.info("word not in w2v_model {}".format(word))
            arr.append(np.random.randn(1, 300))
            miss_count += 1
    # logger.info("w2v_model missing word {}, valid word {}".format(miss_count, valid_count))
    return np.mean(np.array(arr), axis=0).reshape(1, -1)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set)) #也是命中个数 除以 ground-truth 的个数 求和后再除以用户个数。
            true_users += 1
    return sum_recall / true_users

class HNSW(object):
    def __init__(self, 
                w2v_path, 
                ef=config.ef_construction,
                M=config.M,
                model_path=None,
                data_path=None):
        self.w2v_model = KeyedVectors.load(str(w2v_path))
        self.data = self.load_data(str(data_path))
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.index = self.load_hnsw(str(model_path))
        elif data_path:
            self.index = self.build_hnsw(str(model_path), ef=ef, m=M)
        else:
            logger.error("No existing model and no building data provided")
    
    def load_data(self, data_path):
        """
        @desc: 读取数据,并生成句向量
        @param: 问答pair数据所在路径
        @return: 包含句向量的dataframe
        """
        data = pd.read_csv(data_path)
        data['custom_vec'] = data['custom'].apply(lambda x: wam(x, self.w2v_model))
        data['custom_vec'] = data['custom_vec'].apply(lambda x: x[0][0] if x.shape[1] != 300 else x)
        data = data.dropna()
        return data 
    
    def evaluate(self, vecs, topk=1):
        """
        @desc: 评估模型
        @param: text the query
        @return: None
        """
        logger.info("Evaluating...")
        index = self.load_hnsw(str(self.model_path))
        nq, d = vecs.shape 
        t0 = time.time()
        # if index is None:
        D, I = index.search(vecs, topk)
        # else:
        #     D, I = index.search(vecs, 1)
        t1 = time.time()
        
        missing_rate = (I == -1).sum() / float(nq)
        # recall_at_1 = (I == np.arange(nq)).sum() / float(nq)
        print(I)
        # print(D)
        for i in range(nq):
            print(self.data.iloc[i]['custom'], self.data.iloc[I[i][0]]['custom'], D[i][0])
        recall_at_topk = recall_at_k(np.arange(nq).reshape(-1, 1), I, topk)
        # logger.info("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
        #     (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))
        logger.info("recall_at_{}: {}".format(topk, recall_at_topk))
    
    def build_hnsw(self, to_file, ef=2000, m=64):
        """
        @desc: 训练hnsw模型
        @param: to_file: 模型保存目录
        @return: None
        """
        logger.info("Building hnsw index.")
        vecs = np.stack(self.data['custom_vec'].values).reshape(-1, 300)
        vecs = vecs.astype("float32")
        dim = self.w2v_model.vector_size

        index = faiss.IndexHNSWFlat(dim, m) # build index
        res = faiss.StandardGpuResources() # using a single GPU
        faiss.index_cpu_to_gpu(res, 0, index) # make it a GPU index
        index.hnsw.efConstruction = ef              
        logger.info("add")
        index.verbose = True
        print("xb: ", vecs.shape)

        print('dtype: ', vecs.dtype)
        index.add(vecs)  # add vectors to the index
        print("total: ", index.ntotal)
        faiss.write_index(index, str(to_file))
        self.evaluate(vecs[:10000], index)
        return index

    
    def load_hnsw(self, model_path):
        """
        @desc: 加载训练好的hnsw模型
        @param: model_path
        @return: hnsw 模型
        """
        logger.info("Loading hnsw index from {model_path}")
        hnsw = faiss.read_index(str(model_path))
        return hnsw 
    
    def search(self, text, k=5):
        """
        @desc: 通过hnsw检索
        @param: 
            - text: 检索句子
            - k=5: 检索返回的数量
        @return:
            - DataFrame contianing the customer inpute, assistance response and the distance to the query. 
        """
        logger.info("Searching for {text}")
        test_vec = wam(clean(text), self.w2v_model)
        k = 4
        D, I = self.index.search(test_vec, k)
        print(I)
        return pd.concat((self.data.iloc[I[0]]['custom'].reset_index(),
                          self.data.iloc[I[0]]['assistance'].reset_index(drop=True),
                          pd.DataFrame(D.reshape(-1, 1), columns=['q_distance'])), axis=1)


if __name__ == "__main__":
    hnsw = HNSW(config.w2v_path, config.ef_construction, config.M, config.hnsw_path, config.train_path)
    test = '我要转人工'
    print(hnsw.search(test, k=10))
    eval_vecs = np.stack(hnsw.data['custom_vec'].values).reshape(-1, 300).astype('float32')
    
    hnsw.evaluate(eval_vecs[:1000], 10)
    