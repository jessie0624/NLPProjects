"""
@DESC: 句子相似度模型
- tfidf
- lsi
- lda

"""
import os, sys
import gc
import tqdm
import numpy as np
from gensim import corpora, models, similarities
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from sentence import Sentence
from collections import defaultdict
sys.path.append('../')
import config
from typing import List
import time 
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD
import faiss     
import logging                 

logger = logging.getLogger(__name__)

class SentenceSimilarity():

    def __init__(self, seg):
        self.seg = seg

    def set_sentences(self, sentences,  stopword=True): #外部调用　获取self.sentences 分词后的句子
        self.sentences = []
        for i in range(0, len(sentences)): # sentences [[],[]]
            if not isinstance(sentences[i], List):
                self.sentences.append(Sentence(sentences[i], self.seg, i, stopword))
            else:
                for j in range(len(sentences[i])):
                    self.sentences.append(Sentence(sentences[i][j], self.seg, j, stopword))

    # 获取切过词的句子
    def get_cuted_sentences(self):
        cuted_sentences = []
        for sentence in self.sentences:
            cuted_sentences.append(sentence.get_cuted_sentence())
        return cuted_sentences

    # 构建其他模型前需要的 获取语 doc2bow 语料信息
    def simple_model(self, min_frequency = 1):
        self.texts = self.get_cuted_sentences() # jieba分词后的list[[],[],]
        # 删除低频词
        self.frequency = defaultdict(int)
        for text in self.texts:
            for token in text:
                self.frequency[token] += 1
        self.texts = [[token for token in text if self.frequency[token] > min_frequency]\
            for text in self.texts]
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus_simple = [self.dictionary.doc2bow(text) for text in self.texts]

    def word2vec(self):
        self.dim = 200
        self.simple_model() 
        if Path(config.retrieval_path / 'w2v').exists():
            self.w2v_model = KeyedVectors.load(os.fspath(config.retrieval_path / 'w2v'))
        else:
            logger.info('start train word2vec model')
            self.w2v_model = models.Word2Vec(min_count=2, window=2, size=self.dim, sample=6e-5,
                            alpha=0.03, min_alpha=0.0007, negative=15, workers=4, iter=7)
            self.w2v_model.build_vocab(self.texts)
            self.w2v_model.train(self.texts, total_examples=self.w2v_model.corpus_count,
                            epochs=15, report_delay=1)
            self.w2v_model.save(os.fspath(config.retrieval_path / 'w2v'))
            
            # self.vectors = self.w2v_model.getVectors()
            # print(self.vectors.show()) #w2v训练出的所有词向量

    # tfidf模型
    def TfidfModel(self):
        self.simple_model()
        self.model = models.TfidfModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple] # 这个是 [doc list [(wordid, freq), (wordid, freq)]]

    # lsi模型
    def LsiModel(self):
        self.simple_model()
         # 转换模型
        self.model = models.LsiModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple]

    # lda模型
    def LdaModel(self):
        self.simple_model()
        # 转换模型
        self.model = models.LdaModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple]

    def create_sim_index(self):
        self.index = similarities.MatrixSimilarity(self.corpus)
        # self.index.save(os.fspath(config.retrieval_lda))
    # 对新输入的句子（比较的句子）进行预处理 #这里得到的tfidf/lsi/lda的权重
    def sentence2vec(self, sentence):
        sentence = Sentence(sentence, self.seg)
        vec_bow = self.dictionary.doc2bow(sentence.get_cuted_sentence())
        return self.model[vec_bow] 

    def bow2vec(self):
        vec = []
        length = max(self.dictionary) + 1
        for content in self.corpus:
            sentence_vectors = np.zeros(length)
            for co in content:
                sentence_vectors[co[0]] = co[1]  # 将句子出现的单词的tf-idf表示放入矩阵中
            vec.append(sentence_vectors)
        return vec

    def similarity(self, sentence):
        sentence_vec = self.sentence2vec(sentence)

        sims = self.index[sentence_vec]
        sim = max(enumerate(sims), key=lambda item: item[1])

        index = sim[0]
        score = sim[1]
        sentence = self.sentences[index]

        sentence.set_score(score)
        return sentence  # 返回一个类

        # 求最相似的句子
    
    def similarity_k(self, sentence, k):
        sentence_vec = self.sentence2vec(sentence)

        sims = self.index[sentence_vec]
        sim_k = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)[:k]

        indexs = [i[0] for i in sim_k]
        scores = [i[1] for i in sim_k]
        return indexs, scores
    
    def get_recall_result(self, questions, k):
        ### 返回这些questions 找到相似的question对应的index  与正确的index计算 召回率
        ret = []
        score = []
        top_3 = []
        for sent in questions:
            sentence_vec = self.sentence2vec(str(sent))
            sims = self.index[sentence_vec]
            sim_k = sorted(enumerate(sims),key=lambda item: item[1], reverse=True)[:k]
            indexs = [i[0] for i in sim_k]
            score = [i[1] for i in sim_k]
            ret.append(indexs)
            score.append(np.average(score))
            top_3.append(np.average(score[:3]))
        return ret, score, top_3

     # 计算top k recall
    def recall_topk(self, y_pred, y_true, num = 5):
        """
        y_true: [0, 1, 2, 3]
        y_pred: [[0,1, 11], [11,1,2]]
        """
        # print(y_pred)
        if len(y_pred[0]) < num:
            print('error not enough pred')
            return -1
        if len(y_pred[0]) > num:
            y_pred = [i[:num] for i in y_pred]
        total_item = len(y_true)
        ret = [1.0  if y_true[i] in y_pred[i] else 0.0 for i in range(total_item) ]
        
        wrong_ret = dict()
        for index in range(len(ret)):
            if ret[index] == 0.0:
                wrong_ret[index] = y_pred[index]

        print('recall :',sum(ret)/total_item)
        return sum(ret)/total_item, wrong_ret

class HNSW(SentenceSimilarity):
    def __init__(self, seg, tfidf_weight=True):
        super(HNSW, self).__init__(seg)
        self.hnsw_path = config.hnsw_path
        self.tfidf_weight = tfidf_weight

    def build_hnsw(self, ef=2000, m=64): # 获取w2v HNSW 的词向量
        """
        @desc: 训练hnsw模型
        @param: to_file: 模型保存目录
        @return: None
        """
        self.creat_sent_embed()
        if self.hnsw_path.exists():
            self.index = self.load_hnsw(self.hnsw_path)
        else:
            print("Building hnsw index.")
            vecs = self.sent_vec.reshape(-1, self.dim)
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
            faiss.write_index(index, str(self.hnsw_path))
            self.index = index
            # self.evaluate(vecs[:10000],10)
            self.evaluate()
            # return index

    def load_hnsw(self, model_path):
        """
        @desc: 加载训练好的hnsw模型
        @param: model_path
        @return: hnsw 模型
        """
        logger.info("Loading hnsw index from {model_path}")
        hnsw = faiss.read_index(str(model_path))
        return hnsw 
    
    def evaluate(self, topk=10):
        """
        @desc: 评估模型
        @param: text the query
        @return: None
        """
        logger.info("Evaluating...")
        vecs = self.sent_vec.reshape(-1, self.dim).astype("float32")[1000:2000]
        index = self.load_hnsw(str(self.hnsw_path))
        nq, d = vecs.shape 
        t0 = time.time()
        D, I = index.search(vecs, topk)
        t1 = time.time()
        
        missing_rate = (I == -1).sum() / float(nq)
        
        recall_at_topk,wrong_ret = self.recall_topk(I, np.arange(1000,1000+nq), topk)
        
        logger.info("recall_at_{}: {}".format(topk, recall_at_topk))
        distance = np.array(D)
        top3 = np.average(distance[:][:3])
        total = np.average(distance)
        logger.info("recall top3:{}".format(top3))
        logger.info("recall total:{}".format(total))
        return wrong_ret

    def search(self, text, k=10):
        """
        @desc: 通过hnsw检索
        @param: 
            - text: 检索句子
            - k=5: 检索返回的数量
        @return:
            - DataFrame contianing the customer inpute, assistance response and the distance to the query. 
        """
        logger.info("Searching for "+text)
        # test_vec = wam(clean(text), self.w2v_model).astype('float32')
        test_vec = self.get_w2v_emb(text).astype('float32')
        D, I = self.index.search(test_vec, k)
        # print(I)
        return I, D 
    
    def creat_sent_embed(self):# 这个是词典库
        sent_vec = []
        
        if not self.tfidf_weight:
            for index, sent in enumerate(self.texts): # 计算每个句子的句向量得到N * 300 的 句向量库。
                emb = np.array([self.w2v_model.wv.get_vector(word) if word in self.w2v_model.wv.vocab.keys() \
                        else np.random.randn(self.dim) for word in sent]).reshape(-1, self.dim) # Seq * 300 
                sent_vec.append(np.average(emb, axis=0))
        else:
            self.model = models.TfidfModel(self.corpus_simple)
            self.corpus = self.model[self.corpus_simple]
            for index, sent in enumerate(self.texts): # 计算每个句子的句向量得到N * 300 的 句向量库。
                emb = np.array([self.w2v_model.wv.get_vector(word) if word in self.w2v_model.wv.vocab.keys() \
                        else np.random.randn(self.dim) for word in sent]).reshape(-1, self.dim) # Seq * 300 

                assert emb.shape[1] == self.dim, print(index, sent, emb.shape)
                tfidf_doc = self.corpus[index]
                doc_dict = dict()
                for (k, v) in tfidf_doc:
                    doc_dict[k] = v
                vec =[self.dictionary.token2id[token] if token in self.dictionary.token2id else 'UNK' for token in sent ]
                weight = np.array([doc_dict[t]  if t in doc_dict else 0.01 for t in vec]).reshape(1, -1)
                ret = np.dot(weight, emb).reshape(1, self.dim)
                sent_vec.append(np.dot(weight, emb)/len(weight))
        self.sent_vec = np.stack(sent_vec).reshape(-1, self.dim) #这个是得到句子库
        
    def get_w2v_emb(self, sent):
        sentence = Sentence(sent, self.seg)
        cut_sent = sentence.get_cuted_sentence()
        print(cut_sent)
        if not self.tfidf_weight:
            emb = np.array([self.w2v_model.wv.get_vector(word) if word in self.w2v_model.wv.vocab.keys() \
                        else np.random.randn(self.dim) for word in cut_sent]).reshape(-1, self.dim)
            sent_vec = np.average(emb, axis=0)
        else:
            self.model = models.TfidfModel(self.corpus_simple)
            self.corpus = self.model[self.corpus_simple]
            
            vec_bow = self.dictionary.doc2bow(cut_sent)
            tfidf_weigth = self.model[vec_bow] 
            print(vec_bow, tfidf_weigth)
            print(cut_sent)
            # print([(self.dictionary.id2token[id],we) if id in self.dictionary.id2token else 0.01  for (id,we) in tfidf_weigth ])
            emb = np.array([self.w2v_model.wv.get_vector(word) if word in self.w2v_model.wv.vocab.keys() \
                        else np.random.randn(self.dim) for word in cut_sent]).reshape(-1, self.dim)
            doc_dict = dict()
            for (k, v) in tfidf_weigth:
                doc_dict[k] = v
            vec =[self.dictionary.token2id[token] if token in self.dictionary.token2id else 'UNK' for token in cut_sent ]
            weight = np.array([doc_dict[t]  if t in doc_dict else 0.0 for t in vec]).reshape(1, -1)
            print(vec,weight)
            sent_vec = np.dot(weight, emb)/len(weight)
        return sent_vec

    def cossim(self, sent1, sent2):
        sim=cosine_similarity(self.get_w2v_emb(sent1), self.get_w2v_emb(sent2))
        print("sim for sent1: " + sent1 + ", and sent2："+sent2 + "is: ", sim)
    

class SIF(SentenceSimilarity): # 基于SIF的
    
    # 因为这些可以通过去除主成分去掉
    def __init__(self, seg):
        super(SIF, self).__init__(seg)
        # self.w2v_model
        self.hnsw_path = config.hnsw_path_sif
        
    def compute_pc(self, X, npc):
        svd = TruncatedSVD(n_components=npc, n_iter=5, random_state=0)
        svd.fit(X)
        return svd.components_

    # 去除主成分
    def remove_pc(self, X, npc):
        pc = self.compute_pc(X, npc)
        if npc == 1:
            XX = X - X.dot(pc.transpose()) * pc
        else:
            XX = X - X.dot(pc.transpose()).dot(pc)
        return XX
    
    def get_sif_weight(self, a=3e-4):
        """
        根据每个词频等信息计算sif权重
        """
        N = sum(self.frequency.values())
        self.sif_weight = dict()
        with open(config.retrieval_path / 'sif-weight.txt', 'wb') as f:
            for key, value in self.frequency.items():
                self.sif_weight[key] = a / (a + value/N)
                f.write("{0} {1}".format(key, str(self.sif_weight.get(key))).encode("utf-8"))
                f.write('\n'.encode("utf-8"))
        print("权重更新完成")
        return self.sif_weight

    # 句子词向量平均
    def ave_no_rem(self): 
        sent_vec = []
        for index, sent in enumerate(self.texts): # 计算每个句子的句向量得到N * 300 的 句向量库。
            emb = np.array([self.w2v_model.wv.get_vector(word) if word in self.w2v_model.wv.vocab.keys() \
                    else np.random.randn(300) for word in sent]).reshape(-1, 300) # Seq * 300 
            sent_vec.append(np.average(emb, axis=0))
        self.ave_sent = np.stack(sent_vec).reshape(-1, 300)
    # 句子词向量平均去除主成分
    def ave_with_rem(self):
        self.ave_no_rem()
        npc = 1
        self.ave_sent_rem = self.remove_pc(self.ave_sent, npc)
    # 带sif加权的词向量平均
    def sif_no_rem(self):
        sent_vec = []
        for index, sent in enumerate(self.texts): # 计算每个句子的句向量得到N * 300 的 句向量库。
            # assert len(sent) != 0, print(sent)
            emb = np.array([self.w2v_model.wv.get_vector(word) if word in self.w2v_model.wv.vocab.keys() \
                    else np.random.randn(self.dim) for word in sent]).reshape(-1, self.dim) # Seq * 300 
            weight = np.array([self.sif_weight[word] for word in sent])
            # print(np.dot(weight, emb).shape) 句子为在去除停用词或者结巴分词后为空
            sent_vec.append(np.dot(weight, emb)/(len(sent)+1))
        return np.stack(sent_vec).reshape(-1, self.dim)
    # 带sif加权的词向量平均去除主成分
    def sif_with_rem(self):
        # self.sif_no_rem()
        npc = 1
        return self.remove_pc(self.sif_no_rem(), npc)

    def build_hnsw(self, ef=2000, m=64): # 获取w2v HNSW 的词向量
        """
        @desc: 训练hnsw模型
        @param: to_file: 模型保存目录
        @return: None
        """
        self.get_sif_weight()
        self.sent_vec = self.sif_with_rem()
        if self.hnsw_path.exists():
            self.index = self.load_hnsw(self.hnsw_path)
            self.evaluate()
        else:
            print("Building hnsw index.")
            vecs = self.sent_vec.reshape(-1, self.dim)
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
            faiss.write_index(index, str(self.hnsw_path))
            self.index = index
            # self.evaluate(vecs[:10000],10)
            self.evaluate()

    def load_hnsw(self, model_path):
        """
        @desc: 加载训练好的hnsw模型
        @param: model_path
        @return: hnsw 模型
        """
        logger.info("Loading hnsw index from {model_path}")
        hnsw = faiss.read_index(str(model_path))
        return hnsw 

    def evaluate(self, topk=10):
        """
        @desc: 评估模型
        @param: text the query
        @return: None
        """
        logger.info("Evaluating...")
        vecs = self.sent_vec.reshape(-1, self.dim).astype("float32")[1000:2000]
        index = self.load_hnsw(str(self.hnsw_path))
        nq, d = vecs.shape 
        t0 = time.time()
        D, I = index.search(vecs, topk)
        t1 = time.time()
        
        missing_rate = (I == -1).sum() / float(nq)
        
        recall_at_topk,wrong_ret = self.recall_topk(I, np.arange(1000,1000+nq), topk)
        logger.info("recall_at_{}: {}".format(topk, recall_at_topk))
        distance = np.array(D)
        top3 = np.average(distance[:][:3])
        total = np.average(distance)
        logger.info("recall top3:{}".format(top3))
        logger.info("recall total:{}".format(total))
        return wrong_ret

    def search(self, text, k=30):
        """
        @desc: 通过hnsw检索
        @param: 
            - text: 检索句子
            - k=5: 检索返回的数量
        @return:
            - DataFrame contianing the customer inpute, assistance response and the distance to the query. 
        """
        logger.info("Searching for "+text)
        # test_vec = wam(clean(text), self.w2v_model).astype('float32')
        # test_vec = self.get_w2v_emb(text).astype('float32')
        D, I = self.index.search(test_vec, k)
        # print(I)
        return I, D 