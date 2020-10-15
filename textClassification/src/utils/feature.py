import numpy as np  
import copy 
import string 
import json
import joblib
import jieba.posseg as pseg
import pandas as pd 
from src.utils.tools import wam
from src.utils import config
from __init__ import * 
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

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
    data["queryCutRMStopWords"] = data["queryCutRMStopWord"].apply(lambda x: " ".join(x))
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

    joblib.dump(w2v_label_embedding, config.root_path + '/data/w2v_label_embedding.pkl')
    
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
    data[model_name + "_label_mean"] = data[model_name].progress_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method="mean"))
    data[model_name + "_label_max"] = data[model_name].progress_apply(
        lambda x: Find_Label_embedding(x, label_embedding, method="max"))

    print("generate embedding max/mean") 
    data[model_name + "_mean"] = data[model_name].progress_apply(
        lambda x: np.mean(np.array(x), axis=0))
    data[model_name + "_max"] = data[model_name].progress_apply(
        lambda x: np.max(np.array(x), axis=0))
    
    print("generate embedding winodw max/mean")
    data[model_name + "_win_2_mean"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 2, method='mean'))
    data[model_name + "_win_3_mean"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 3, method='mean'))
    data[model_name + "_win_4_mean"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 4, method='mean'))
    data[model_name + "_win_2_max"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 2, method='max'))
    data[model_name + "_win_3_max"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 3, method='max'))
    data[model_name + "_win_4_max"] = data[model_name].progress_apply(
        lambda x: Find_embedding_with_windows(x, 4, method='max'))
    return data

def softmax(x):
    """
    x: ndarry of embedding. 
    make x to probability so that sum(x) = 1
    """
    return np.exp(x)/np.exp(x).sum(axis=0)
    

def Find_Label_embedding(example_matrix, label_embedding, method='mean'):
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

def Find_embedding_with_windows(embedding_matrix, window_size=2, method='mean'):
    """
    generate embedding use window
    @param:
        - embedding_matrix: input sentence's emebdding
        - window_size: 2, 3, 4
        - method: max/mean
    @return:
        ndarray of embedding
    """
    # 最终的词向量
    result_list = []
    embedding_len = len(embedding_matrix)
    for k in range(embedding_len):
        if int(k + window_size) > embedding_len:
            result_list.extend(embedding_matrix[k : ])
        else:
            result_list.extend(embedding_matrix[k : k + window_size])
    if method == 'mean':
        return np.mean(result_list, axis=0)
    else:
        return np.max(result_list, axis=0)


ch2en = {
    '！': '!',
    '？': '?',
    '｡': '.',
    '（': '(',
    '）': ')',
    '，': ',',
    '：': ':',
    '；': ';',
    '｀': ','
}

def tag_part_of_speech(data):
    """
    tag part of speech, then calculate the num of noun, adj, verb
    @param: data 
    @return:
        - noun_count
        - adj_count,
        - verb_count
    """
    words = [tuple(x) for x in list(pseg.cut(data))]
    noun_count = len([w for w in words if w[1] in ("NN", "NNP", "NNPS", "NNS")])
    adj_count = len([w for w in words if w[1] in ("JJ", "JJR", "JJS")])
    verb_count = len([w for w in words if w[1] in ("VB", "VBD", "VBN", "VBP", "VBZ")])
    return noun_count, adj_count, verb_count

def get_basic_feature(df):
    """
    get basic feature: length, capitals numbers, num_exclamation_marks,
                        num_punctuation, num_question_marks,num_symbols,
                        num_words, num_unique_words. etc.
    @param: df DataFrame
    @return: df
    """
    # 中文标点转英文标点
    df["queryCut"] = df["queryCut"].progress_apply(
        lambda x: [i if i not in ch2en.keys() else ch2en[i] for i in x])
    # 文本长度
    df["length"] = df["queryCut"].progress_apply(lambda x: len(x))
    # 大写个数
    df["capitals"] = df["queryCut"].progress_apply(lambda x: sum(1 for c in x if c.isuper()))
    # 大写 与 文本长度比例
    df["caps_vs_length"] = df.progress_apply(lambda row: float(row["capitals"]) / float(row["length"]), axis=1)
    # 感叹号个数
    df["num_exclamation_marks"] = df["queryCut"].progress_apply(lambda x: x.count("!"))
    # 标点符号个数
    df["num_punctuation"] = df["queryCut"].progress_apply(lambda x: sum(x.count(w) for w in string.punctuation))
    #　问号个数
    df["num_question_marks"] = df["queryCut"].progress_apply(lambda x: x.count("?"))
    # 特殊符号个数
    df["num_symbols"] = df["queryCut"].progress_apply(lambda x: sum(x.count(w) for w in "*&$%"))
    # 单词个数
    df["num_words"] = df["queryCut"].progress_apply(lambda x: len(x))
    # 唯一词的个数
    df["num_unique_words"] = df["queryCut"].progress_apply(lambda x: len(set(w for w in x)))
    # 唯一词　占比
    df["words_vs_unique"] = df["num_unique_words"] / df["num_words"]
    # 获取 名词,形容词,动词的个数,使用tag_part_of_speech函数
    df["noun"], df["adjectives"], df["verbs"] = zip(*df["text"].progress_apply(lambda x: tag_part_of_speech(x)))
    # 名词占总长度的比率
    df['nouns_vs_length'] = df['nouns'] / df['length']
    # 形容词占总长度的比率
    df['adjectives_vs_length'] = df['adjectives'] / df['length']
    # 动词占总长度的比率
    df['verbs_vs_length'] = df['verbs'] / df['length']
    # 名词占总词数的比率
    df['nouns_vs_words'] = df['nouns'] / df['num_words']
    # 形容词占总词数的比率
    df['adjectives_vs_words'] = df['adjectives'] / df['num_words']
    # 动词占总词数的比率
    df['verbs_vs_words'] = df['verbs'] / df['num_words']
    # 首字母大写其他小写的个数
    df["count_words_title"] = df["queryCut"].progress_apply(lambda x: len([w for w in x if w.istitle()]))
    # 平均词的个数
    df["mean_word_len"] = df["text"].progress_apply(lambda x: np.mean([len(w) for w in x]))
    # 标点符号占比
    df["punct_percent"] = df['num_punctuation'] * 100 / df['num_words']
    return df 


def get_pretrain_embedding(text, tokenizer, model):
    """
    get bert embedding
    @param:
        - text: input
        - tokenizer: bert tokenizer
        - model: bert model
    @return: 
        - bert embedding ndarray
    """
    # 通过bert tokenizer 来处理数据,然后使用bert model 获取bert embedding
    text_dict = tokenizer.encode_plus(
        text, # sentence  to encode
        add_special_tokens = True, # Add [CLS] and [SEP]
        max_length = 400, # pad and truncate all sentences
        ad_to_max_length = True,
        return_attention_mask = True, # Construct attn, mask
        return_tensors = 'pt')
    
    input_ids, attention_mask, token_type_ids = text_dict["input_ids"], text_dict["attention_mask"], text_dict["token_type_ids"]
    
    _, res = model(input_ids.to(config.device), 
                    attention_mask = attention_mask.to(config.device),
                    token_type_ids = token_type_ids.to(config.device))
    return res.detach().cpu().numpy()[0]

def get_lda_features(document, lda_model):
    """
    Transforms a bag of words document to features
    @param: 
        - document: list of bag of words
        - model: lda model
    @return:
        - lda features
    """
    # 基于bag of word 格式数据获取lda 特征
    topic_importances = lda_model.get_document_topics(document, minimum_probability=0)
    topic_importances = np.array(topic_importances)
    return topic_importances[:, 1]

def formate_data(data, tfidf):
    Data = pd.concat([
        data[[
            'labelIndex', 'length', 'capitals', 'caps_vs_length',
            'num_exclamation_marks', 'num_question_marks', 'num_punctuation',
            'num_symbols', 'num_words', 'num_unique_words', 'words_vs_unique',
            'nouns', 'adjectives', 'verbs', 'nouns_vs_length',
            'adjectives_vs_length', 'verbs_vs_length', 'nouns_vs_words',
            'adjectives_vs_words', 'verbs_vs_words', 'count_words_title',
            'mean_word_len', 'punct_percent'
        ]], tfidf
    ], axis=1).fillna(0.0)
    return Data
