
import os, sys
import gc
import tqdm
import numpy as np
from gensim import corpora, models, similarities
from pathlib import Path
sys.path.append('../')
import config
from gensim.models import KeyedVectors
    
w2v_model = KeyedVectors.load(os.fspath(config.retrieval_path / 'w2v'))
print(w2v_model.most_similar('人民币'))
print(w2v_model.most_similar('银行卡'))
print(w2v_model.most_similar('理财'))
print(w2v_model.most_similar('新手'))
print(w2v_model.most_similar('汇率'))

