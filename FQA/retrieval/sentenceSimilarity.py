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
from pathlib import Path
from sentence import Sentence
from collections import defaultdict
sys.path.append('../')
import config
from typing import List
import time 
from gensim.models import KeyedVectors
import faiss     
import logging                 

logger = logging.getLogger(__name__)

class SentenceSimilarity():

    def __init__(self, seg):
        self.seg = seg

    def set_sentences(self, sentences): #外部调用　获取self.sentences 分词后的句子
        self.sentences = []
        for i in range(0, len(sentences)): # sentences [[],[]]
            if not isinstance(sentences[i], List):
                self.sentences.append(Sentence(sentences[i], self.seg, i))
            else:
                for j in range(len(sentences[i])):
                    self.sentences.append(Sentence(sentences[i][j], self.seg, j))

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
        frequency = defaultdict(int)
        for text in self.texts:
            for token in text:
                frequency[token] += 1
        self.texts = [[token for token in text if frequency[token] > min_frequency]\
            for text in self.texts]
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus_simple = [self.dictionary.doc2bow(text) for text in self.texts]

    def word2vec(self):
        self.simple_model() 
        if Path(config.retrieval_path / 'w2v').exists():
            self.w2v_model = KeyedVectors.load(os.fspath(config.retrieval_path / 'w2v'))
        else:
            logger.info('start train word2vec model')
            self.w2v_model = models.Word2Vec(min_count=2, window=2, size=300, sample=6e-5,
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
        self.index.save(os.fspath(config.retrieval_lda))
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

    # 求最相似的句子
    # input: test sentence
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
        for sent in questions:
            sentence_vec = self.sentence2vec(str(sent))
            sims = self.index[sentence_vec]
            sim_k = sorted(enumerate(sims),key=lambda item: item[1], reverse=True)[:k]
            indexs = [i[0] for i in sim_k]
            ret.append(indexs)
        return ret 

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
        ret = [1.0   if y_true[i] in y_pred[i] else 0.0 for i in range(total_item) ]
        
        print("wrong recall")
        wrong_ret = dict()
        # for index, que in enumerate(questions):
        #     if ret[index] == 0.0:
        #         wrong_ret[que] = y_pred[index]
        for index in range(len(ret)):
            if ret[index] == 0.0:
                wrong_ret[index] = y_pred[index]

        print('recall :',sum(ret)/total_item)
        return sum(ret)/total_item, wrong_ret

class HNSW(SentenceSimilarity):
    def __init__(self, seg):
        super(HNSW, self).__init__(seg)
        # SentenceSimilarity.__init__(seg)
        self.hnsw_path = config.hnsw_path
    
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
            vecs = self.sent_vec.reshape(-1, 300)
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
        vecs = self.sent_vec.reshape(-1, 300).astype("float32")[:1000]
        index = self.load_hnsw(str(self.hnsw_path))
        nq, d = vecs.shape 
        t0 = time.time()
        D, I = index.search(vecs, topk)
        t1 = time.time()
        
        missing_rate = (I == -1).sum() / float(nq)
        
        # for i in range(nq):
        #     print(self.data.iloc[i]['custom'], self.data.iloc[I[i][0]]['custom'], D[i][0])
        recall_at_topk,wrong_ret = self.recall_topk(I, np.arange(nq), topk)
        logger.info("recall_at_{}: {}".format(topk, recall_at_topk))
        return wrong_ret

    def search(self, text, k=5):
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
        self.model = models.TfidfModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple]
        for index, sent in enumerate(self.texts): # 计算每个句子的句向量得到N * 300 的 句向量库。
            emb = np.array([self.w2v_model.wv.get_vector(word) if word in self.w2v_model.wv.vocab.keys() \
                    else np.random.randn(300) for word in sent]).reshape(-1, 300) # Seq * 300 

            assert emb.shape[1] == 300, print(index, sent, emb.shape)
            tfidf_doc = self.corpus[index]
            doc_dict = dict()
            for (k, v) in tfidf_doc:
                doc_dict[k] = v
            vec =[self.dictionary.token2id[token] if token in self.dictionary.token2id else 'UNK' for token in sent ]
            weight = np.array([doc_dict[t]  if t in doc_dict else 0.0 for t in vec]).reshape(1, -1)
            # print(weight.shape, emb.shape, sent, index)
            ret = np.dot(weight, emb).reshape(1, 300)
            # print(ret.shape, sent, index)
            sent_vec.append(np.dot(weight, emb))
            # print(np.array(sent_vec).shape)
        self.sent_vec = np.stack(sent_vec).reshape(-1, 300) #这个是得到词典库
        
    def get_w2v_emb(self, sent):

        self.model = models.TfidfModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple]
        
        sentence = Sentence(sent, self.seg)
        cut_sent = sentence.get_cuted_sentence()
        vec_bow = self.dictionary.doc2bow(cut_sent)
        tfidf_weigth = self.model[vec_bow] 
        print(vec_bow, tfidf_weigth)
        print(cut_sent)
        emb = np.array([self.w2v_model.wv.get_vector(word) if word in self.w2v_model.wv.vocab.keys() \
                    else np.random.randn(300) for word in cut_sent]).reshape(-1, 300)
        doc_dict = dict()
        for (k, v) in tfidf_weigth:
            doc_dict[k] = v
        vec =[self.dictionary.token2id[token] if token in self.dictionary.token2id else 'UNK' for token in cut_sent ]
        weight = np.array([doc_dict[t]  if t in doc_dict else 0.0 for t in vec]).reshape(1, -1)
        sent_vec = np.dot(weight, emb)
        return sent_vec