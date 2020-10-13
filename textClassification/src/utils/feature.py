import numpy as np  
import copy 

from __init__ import * 

# from src.utils.tools import wam, format_data


def get_embedding_feature(data, tfidf, embedding_model):
    """
    get_embedding_feature:
    data: train/dev data, tfidf:tfidf embedding, embedding_model: w2v?

    tfidf, word2vec -> max/mean,
    word2vec n-gram(2, 3, 4) -> max/mean, label embedding-> max/mean
    @param:
        mldata, input data set, mldata class instance
    @return:
        train_tfidf, tfidf of train data set
        test_tfidf, tfidf of test data set
        train, train data set
        test, test data set
    """
    # 根据过滤停用词后数据 获取tfidf特征
    data["queryCutRMStopWords"] = data["querySRMStopWord"].apply(lambda x: " ".join(x))
    tfidf_data = pd.DataFrame(tfidf.transform(data["queryCutRMStopWords"].tolist()).toarray())
    tfidf_data.columns = ["tfidf" + str(i) for i in range(tfidf_data.shape[1])]

    # 获取word2vec模型
    print("transfrom w2v") #here 300 embedidng comes from embedding.py file word2vec and fasttext model size = 300
    data["w2v"] = data["queryCutRMStopWords"].apply(lambda x: wam(x, embedding_model, aggregate=False)) # (seq_len, 300)

    # 深度copy 到 train 里面
    train = copy.deepcopy(data)
    # 加载所有类别, 获取类别的embedding, 并保存文件
    labelNameToIndex = json.load(open(config.root_path + '/data/label2id.json', encoding='utf-8'))
    labelIndexToName = {v: k for k, v in labelNameToIndex.items()}
    w2v_label_embedding = np.array([
        embedding_model.wv.get_vector(labelIndexToName[key]) for key in labelIndexToName
            if labelIndexToName[key] in embedding_model.wv.vocab.keys()
    ])

    joblib.dump(w2v_label_embedding, config.root_path + '/dev/w2v_label_embedding.pkl')
    
    # 根据未聚合的embedding 数据, 获取各类embedding 特征
    train = generate_feature(train, w2v_label_embedding, model_name='w2v')
    return tfidf_data, train 

def generate_feature(data, label_embedding, model_name='w2v'):
    """
    data: inout data  DataFrame
    label_embedding: all label embedding
    model_name: w2v means word2vec
    return data, DataFrame
    """
    print('generate w2v & fast label max/mean')
    # 首先在预训练的词向量中获取标签的词向量句子,每一行表示一个标签的embedding
    data[model_name + "_label_mean"] = data[model_name].process_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method="mean"))
    data[model_name + "_label_max"] = data[model_name].process_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method="max"))
    
    
    

def Find_Label_embedding(example_matrix, label_embedidng, method='mean'):
    """
    根据论文 <<Join embedding of words and labels>> 获取标签空间的词嵌入
    example_matrix(np.array 2D): input words embeddings
    label_embedding(np.array 2D): all label embedding
    return (np.array, 1D) the embedding by joint label and word
    """
    # 根据矩阵乘法来计算label和words 之间的相似度 
    similarity_matrix = np.dot(example_matrix, label_embedding.T) / (
        np.linalg.norm(example_matrix) * (np.linalg.norm(label_embedding)))
    # 然后对相似矩阵进行均值池化, 则得到了"类别-词语"的注意力机制
    # 这里可以使用max-pooling 和 mean-pooling
    attention = similarity_matrix.max()
    attention = softmax(attention)
    attention_embedding = example_matrix * attention
    if method == "mean":
        return np.mean(attention_embedding, axis=0)
    else:
        return np.max(attention_embedding, axis=0)

