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
Path(root_path/'log').mkdir(parents=True, exist_ok=True)
Path(root_path / "model" /"ranking4").mkdir(parents=True, exist_ok=True)
### Model ###
model_path = root_path / "model"
model_path.mkdir(parents=True, exist_ok=True)

# ranking
rank_path = model_path / "ranking"
rank_path.mkdir(parents=True, exist_ok=True)
retrieval_path = model_path / "retrieval"
retrieval_path.mkdir(parents=True, exist_ok=True)


train_raw = data_path / "raw_data" /"raw_train.txt"
dev_raw = data_path / "raw_data"/"raw_dev.txt"
test_raw = data_path / "raw_data"/"raw_test.txt"

# 这个是通过preprocess处理之后得到得 customer-assistance的 问题-答复 对儿数据集。
#　这里面对一些表情符号，然后网址，订单号，商品编号等　已经做了替换处理。
# 这个数据集主要用来做下一步的retrieval 和 rerank. 不过在retrieval和rerank时 还需要对问题进行处理。
pair_data_train = data_path / "pair_data"/ "train.csv"
pair_data_dev = data_path / "pair_data"/ "dev.csv"
pair_data_test = data_path / "pair_data"/ "test.csv"
pair_data_all = data_path / "pair_data" / "all.csv"

user_dict = data_path / "userdict.txt"
stop_words_path = root_path / "data" / "stopwords.txt"

clean_data = data_retrieval_path /'clean_all2.csv'
clean_data3 = data_retrieval_path /'clean_all3.csv'
retrieval_data = data_retrieval_path /'train.csv'
# retrieval_tfidf = retrieval_path /"tfidf.index"
# retrieval_lsi = retrieval_path /"lsi.index"
# retrieval_lda = retrieval_path /"lda.index"
# retrieval_dict = retrieval_path / "re_dict"
# retrieval_corpus = retrieval_path / "sampleCorpus.mm"

retrieval_tfidf = retrieval_path /"tfidf2.index"
retrieval_lsi = retrieval_path /"lsi2.index"
retrieval_lda = retrieval_path /"lda2.index"
retrieval_dict = retrieval_path / "re_dict2"
retrieval_corpus = retrieval_path / "sampleCorpus2.mm"
# Retrival embedding
w2v_path = model_path / "retrieval" / "w2v"
hnsw_path = model_path / "retrieval" / "hnsw_index"
hnsw_path_sif = model_path / "retrieval" / "hnsw_index_sif"

# ranking
ranking_raw_path =  data_path / "ranking_raw_data"
ranking_train = data_ranking_path / "train_all.csv"
ranking_train_clean = data_ranking_path / "train_all_clean.csv"

ranking_bert_train = data_ranking_path / "train.csv"
ranking_bert_dev = data_ranking_path / "dev.csv"
ranking_bert_train_clean = data_ranking_path / "train_clean.csv"
ranking_bert_dev_clean = data_ranking_path / "dev_clean.csv"
rank_model = 'bert'
# ranking_dev = data_ranking_path / "dev.csv"

ware_path = data_path / "ware.txt"

sep = "[SEP]"

### Data ###
business_train = data_intention_path / "bussiness.train"
business_test = data_intention_path / "business.test"
keyword_path = data_intention_path / "key_word.txt"



# Intention fasttext
ft_path = model_path / "intention" /"fasttext"


# HNSW parameters
ef_construction = 3000
M = 64


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

# ranking2 model
model_name_or_path = root_path/"lib"/'bert'
r2_data_dir = root_path / "data" / "ranking/0"
r2_output_dir = root_path/"model"/"ranking"
# per_gpu_train_batch_size = 32
num_train_epochs = 20.0
raw_file = root_path /"data"/ "ranking"/"pc_atec_double.csv"