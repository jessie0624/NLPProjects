"""
Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classiﬁcation

双向LSTM
前后向lstm 拼接 得到每个位置的Hiden H

attention layer: 
M = tanh(H)
a = softmax(w^T*M) # 这个是attention的权重矩阵, 它有经过tanh函数生成的M再通过softmax函数得到,其中w是可以训练的参数. 
r = H*a^T # 最终句子的表示, 由hidden H与权重相乘.

Classification:
将上一步得到的r通过tanh函数得到h*
h* = tanh(r)
将h*作为softmax classifier的变量来做预测
p(y|S) = softmax(Ws* h* + bs)
y = argmax(p(y|S))

代价函数是:
J(a) = -1/m \sum ti log(yi) + b|a|^2

"""
import torch  
import torch.nn as nn 
import torch.nn.functional as F   
import numpy as np

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=0)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(config.hidden_size * 2)) # 这个双向LSTM的对应的attention矩阵
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.fc = nn.linear(config.hidden_size, config.num_classes)
    
    def forward(self, x): # x (train, mask, tokens)
        emb = self.embedding(x[0]) # [batch_size, seq_len, embedding]
        O, _ = self.lstm(emb) # [batch_size, seq_len, hidden_size * 2]   
        # lstm的输出 out,self.hidden
        # output: (seq_len, batch, num_directions * hidden_size)
        # h_n: (num_layers*num_directions, batch, hidden_size)
        # c_n: (num_layers*num_directions, batch, hidden_size)
        M = self.tanh1(O)
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1) # attention weight [batch_size, seq_len, 1]
        out = O * alpha # 每个输出值都乘上对应的alpha [batch_size, seq_len, hidden_size*2]
        out = torch.sum(out, 1) # [batch_size, hidden_size*2] 沿着seq_len长度方向进行加和, 得到 一个句子的hidden_size*2的长度
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)
        return out 




