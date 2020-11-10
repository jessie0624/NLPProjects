import torch 
import torch.nn as nn 
import numpy as np 
from __init__ import * 

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding =nn.Embedding(config.n_vocab, config.embed, padding_idx=0)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
    
    def forward(self, x): # x (train, mask, tokens) 这里只需要x[0] train
        out = self.embedding(x[0]) # [batch_size, seq_len, embedding] = [128, 32, 300]
        out, _ = self.lstm(out) # batch_size, seq_len, hidden_size.     
        out = self.fc(out[:, -1, :]) # 取最后一个位置 作为句向量
        return out
