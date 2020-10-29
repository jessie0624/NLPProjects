import torch 
import os 
from pathlib import Path 
import logging
import sys

log_level = logging.INFO
log_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
log_date_format = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=log_format, datefmt=log_date_format, level=log_level, stream=sys.stderr)

root_path = Path(__file__).parent.expanduser().resolve()

data_path = root_path / "data"
data_intention_path = data_path / "intention"
data_retrieval_path = data_path / "retrieval"
data_ranking_path = data_path / "ranking"

data_path.mkdir(parents=True, exist_ok=True)
data_intention_path.mkdir(parents=True, exist_ok=True)
data_retrieval_path.mkdir(parents=True, exist_ok=True)
data_ranking_path.mkdir(parents=True, exist_ok=True)

train_raw = data_path / "chat.txt"
dev_raw = data_path / "开发集.txt"
test_raw = data_path / "测试集.txt"
ware_path = data_path / "ware.txt"

sep = "[SEP]"

### Data ###
stop_words_path = root_path / "data" / "stopwords.txt"
# main
train_path = data_path / "train_no_blank.csv"
dev_path = data_path / "dev.csv"
test_path = data_path / "test.csv"

# intention
business_train = data_intention_path / "bussiness.train"
business_test = data_intention_path / "business.test"
keyword_path = data_intention_path / "key_word.txt"

# ranking
ranking_train = data_ranking_path / "train.csv"
ranking_test = data_ranking_path / "test.csv"
ranking_dev = data_ranking_path / "dev.csv"

### Model ###
model_path = root_path / "model"
model_path.mkdir(parents=True, exist_ok=True)

# ranking
rank_path = model_path / "ranking"
rank_path.mkdir(parents=True, exist_ok=True)
# Intention fasttext
ft_path = model_path / "intention" /"fasttext"
# Retrival embedding
w2v_path = model_path / "retrieval" / "word2vec"

# HNSW parameters
ef_construction = 3000
M = 64
hnsw_path = model_path / "retrieval" / "hnsw_index"

# BERT parameters
max_sequence_length = 103

# result folder
result_path = root_path / "result"
result_path.mkdir(parents=True, exist_ok=True)

### Config ###
is_cuda = True 
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")