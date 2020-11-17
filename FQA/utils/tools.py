import logging
import re 
# from generate.bert_seq2seq import config_distil


def create_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger

# def divide_parameters(named_parameters, lr=None):
#     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
#     decay_parameters_names = list(zip(*[(p,n) for n,p in named_parameters if not any((di in n) for di in no_decay)]))
#     no_decay_parameters_names = list(zip(*[(p,n) for n,p in named_parameters if any((di in n) for di in no_decay)]))
#     param_group = []
#     if len(decay_parameters_names)>0:
#         decay_parameters, decay_names = decay_parameters_names
#         #print ("decay:",decay_names)
#         if lr is not None:
#             decay_group = {'params':decay_parameters,   'weight_decay_rate': config_distil.args.weight_decay_rate, 'lr':lr}
#         else:
#             decay_group = {'params': decay_parameters, 'weight_decay_rate': config_distil.args.weight_decay_rate}
#         param_group.append(decay_group)

#     if len(no_decay_parameters_names)>0:
#         no_decay_parameters, no_decay_names = no_decay_parameters_names
#         #print ("no decay:", no_decay_names)
#         if lr is not None:
#             no_decay_group = {'params': no_decay_parameters, 'weight_decay_rate': 0.0, 'lr': lr}
#         else:
#             no_decay_group = {'params': no_decay_parameters, 'weight_decay_rate': 0.0}
#         param_group.append(no_decay_group)

#     assert len(param_group)>0
#     return param_group

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"\s+", "", string)
    string = re.sub(r"[^\u4e00-\u9fff]", "", string)
    # string = re.sub(r"\'s", " \'s", string)
    # string = re.sub(r"\'ve", " \'ve", string)
    # string = re.sub(r"n\'t", " n\'t", string)
    # string = re.sub(r"\'re", " \'re", string)
    # string = re.sub(r"\'d", " \'d", string)
    # string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"\?", " \? ", string)
    return string.strip()

def clean(sent, keyword_sent_only=False, sep='['):
    """
    @desc: 过滤无用符号，并对[SEP]等分割符号，假如前后空格，避免影响分词结果。
    @param:
        - sent: 句子
        - sep: 分隔符是以 [SEP] 
    @return: string 清洗后的句子
    """                        
    sent = re.sub(r"[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+",
                  "", sent)
    
    i = 0
    tmp = []
    while i < len(sent):
        if sent[i] != sep:
            tmp.append(sent[i])
            i += 1
        else:
            tmp.append(sent[i: i+5])
            i += 5
    return " ".join(tmp)

# def build_keyword(sku_path=config.ware_path, to_file=config.keyword_path, data_path=config.train_path):
#     """
#     @desc: 构建业务咨询相关关键词,并保存(读取语料中的名词和提供的sku)
#     @param:
#         - sku_path: sku文件路径
#         - to_file: 关键词保存路径
#     @return: 关键词list
#     # """
    # logger.info("Building keywords")
    # if to_file.exists():
    #     return set(to_file.open(mode='r').read().strip().split('\n'))
    
    # tokens = []
    # data = pd.read_csv(data_path)
    # tokens = data["custom"].dropna().apply(lambda x: [
    #     token for token, pos in pseg.cut(x) if pos in ['n', 'vn', 'nz']
    # ])
    # key_words = set([tk for idx, sample in tokens.iteritems() for tk in sample if len(tk) > 1])
    # logger.info("key words build.")
    # lines = sku_path.open(mode='r').read().strip().split('\n')
    # key_words |= set([item.strip().split('\t')[1] for item in lines[1:]])
    # logger.info("Sku words merged.")
    # if to_file is not None:
    #     with open(to_file, "w") as wf:
    #         for item in key_words:
    #             wf.write(item + '\n')
    # return key_words