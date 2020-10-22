import pandas as pd     
import torch 
from torch.utils.data import Dataset 
import csv                       

class DataProcessForSentence(Dataset):
    """
    对文本处理
    """
    def __init__(self, bert_tokenizer, file, max_char_len=103):
        """
        bert tokenizer:分词器, 
        file: 语料文件
        """
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_char_len
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.get_input(file)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]
    
    # 获取文本与标签
    def get_input(self, file):
        """
        @desc: 对输入文本近分词,ID化,截断,
        """
        pass