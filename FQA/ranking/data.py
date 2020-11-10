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
        @desc: 对输入文本近分词,ID化,截断,填充最后得到可用于bert输入的序列
        @param:
            - file: training file
        @return:
            - seqs, seq_masks, labels
        """
        df = pd.read_csv(file).dropna()#, sep='\t', header=None, names=['question1', 'question2', 'label']).dropna(how='any')
        df['text_a'] = df['text_a'].apply(lambda x: "".join(str(x).split()))
        df['text_b'] = df['text_b'].apply(lambda x: "".join(str(x).split()))
        labels = df['labels'].astype('int8').values 
         
        # 切词
        tokens_seq_1 = list(map(self.bert_tokenizer.tokenize, df['text_a'].values))
        tokens_seq_2 = list(map(self.bert_tokenizer.tokenize, df['text_b'].values))

        # 获取定长序列及其mask
        result = list(map(self.trunate_and_pad, tokens_seq_1, tokens_seq_2))
        seqs, seq_masks, seq_segments = [i[0] for i in result], [i[1] for i in result], [i[2] for i in result]
        
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long),\
            torch.Tensor(seq_segments).type(torch.long), torch.Tensor(labels).type(torch.long)

    def trunate_and_pad(self, tokens_seq_1, tokens_seq_2):
        """
        @decs: 对输入进行截断,padding, 添加分隔符
                如果是单序列,按照BERT序列处理方式,需要在输入序列头尾分别拼接'CLS'和'SEP'
                因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2,如果序列长度大于该值则需要进行截断.
                对输入序列 最终形成['CLS', seq, 'SEP']的序列, 该序列长度如果小于max_seq_len, 那么使用0进行padding
        @param:
            - tokens_seq_1: 文本1
            - tokens_seq_2: 文本2
        @return:
            - seqs, 
            - seq_masks, 
            - seq_segments
        """
        # 对超长序列进行截断
        if len(tokens_seq_1) > (self.max_seq_len - 3)//2:
            tokens_seq_1 = tokens_seq_1[0 : (self.max_seq_len - 3)//2]
        if len(tokens_seq_2) > (self.max_seq_len - 3)//2:
            tokens_seq_2 = tokens_seq_2[0 : (self.max_seq_len - 3)//2]
        
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)
        # ID 化
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq) # convert_tokens_to_ids

        # 根据max_seq_len 与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(seq))
        # 创建seq_mask:
        seq_mask = [1] * len(seq) + padding         
        # 创建seq_segment
        seq_segment = seq_segment + padding 
        #对seq padding
        seq += padding 
        
        assert len(seq) == self.max_seq_len
        assert len(seq_mask) == self.max_seq_len
        assert len(seq_segment) == self.max_seq_len
        return seq, seq_mask, seq_segment