"""
DESC: 对话模型整合
"""
import os
from intention.business import Intention 
from retrieval.hnsw_faiss import HNSW 
from ranking.ranker import RANK 
import config 
import pandas as pd 
from pathlib import Path

def retrieve(k):
    it = Intention(config.train_path,
                   config.ware_path,
                   model_path=config.ft_path,
                   kw_path=config.keyword_path)
    hnsw = HNSW(config.w2v_path,
                config.ef_construction,
                config.M, 
                config.hnsw_path,
                config.train_path)
    dev_set = pd.read_csv(config.dev_path).dropna()
    test_set = pd.read_csv(config.test_path).dropna()
    data_set = dev_set.append(test_set)

    res = pd.DataFrame()
    for query in data_set['custom']:
        query = query.strip()
        intention = it.predict(query)
        print(query, intention)
        if len(query) > 1 and intention[0][0] == '__label__1':
            res = res.append(pd.DataFrame({
                "query": [query]*k,
                "retrieved": hnsw.search(query, k)['custom']
            }))
    
    res.to_csv(os.fspath(config.result_path/'retrieved.csv'), index=False)


def rank():
    # retrieved = pd.read_csv(os.fspath(Path(config.result_path / 'retrieved.csv')))
    ranker = RANK(do_train=False)
    # ranked = pd.DataFrame()
    # ranked['question1'] = retrieved['query']
    # ranked['question2'] = retrieved['retrieved']
    # ranked.to_csv(os.fspath(Path(config.result_path / 'ranked.csv')),index=False)
    ranked = pd.read_csv(os.fspath(Path(config.result_path / 'ranked.csv')))
    rank_scores = ranker.predict(ranker.generate_feature(ranked, False)) # 2190
    print(rank_scores.shape) 
    ranked['rank_score'] = rank_scores 
    ranked.to_csv('result/ranked-ret.csv', index=False)

if __name__ == "__main__":
    # retrieve(5)
    rank()