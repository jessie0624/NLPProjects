"""
DESC: 深度匹配模型
"""
import logging 
import os, sys 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm 
from transformers import (AdamW, BertModel, BertTokenizer,
                         get_linear_schedule_with_warmup,
                         BertConfig,
                         BertForSequenceClassification)
sys.path.append('..')
import config
from config import is_cuda, max_sequence_length, root_path,rank_model
from ranking.data import DataProcessForSentence

tqdm.pandas()

bert_config = BertConfig.from_pretrained(os.fspath(root_path / 'lib'/rank_model/"config.json"))

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# class BCEFocalLoss(torch.nn.Module):
#     """
#     二分类的Focalloss alpha 固定
#     """
#     def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = torch.Tensor([alpha,1-alpha])
#         self.reduction = reduction
 
#     def forward(self, pt, target):
#         # pt = torch.sigmoid(_input)
#         alpha = self.alpha
#         target = target.view(-1,1)
#         logpt = F.log_softmax(pt) # 这里转成log(pt)
#         logpt = logpt.gather(1,target)

#         loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
#                (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
#         if self.reduction == 'elementwise_mean':
#             loss = torch.mean(loss)
#         elif self.reduction == 'sum':
#             loss = torch.sum(loss)
#         return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # if isinstance(alpha,(float,int)): 
        self.alpha = torch.Tensor([alpha,1-alpha])
        # if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)
            input = input.transpose(1,2)
            input = input.contiguous().view(-1,input.size(2))
        target = target.view(-1,1)

        logpt_org = F.log_softmax(input)
        logpt = logpt_org.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean(), logpt_org
        else: return loss.sum(), logpt_org

# def compute_loss(outputs, labels, loss_method='focal_loss'):
#     loss = 0.
#     if loss_method == 'binary':
#         labels = labels.unsqueeze(1)
#         loss = F.binary_cross_entropy(torch.sigmoid(outputs), labels)
#     elif loss_method == 'cross_entropy':
#         loss = F.cross_entropy(outputs, labels)
#     elif loss_method == 'focal_loss':
#         loss = FocalLoss()
#     else:
#             raise Exception("loss_method {binary or cross_entropy} error. ")
#     return loss

class BertModelTrain(nn.Module):
    """
    The base model for training a matching network
    """
    def __init__(self, n_classes=2, multi_drop=3):
        super(BertModelTrain, self).__init__()
        self.bert_model = BertModel.from_pretrained(os.fspath(root_path / 'lib'/ rank_model))
        
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.multi_drop = multi_drop
        self.classifier = nn.Linear(bert_config.hidden_size * 2, n_classes)
        self.loss = FocalLoss()

        self.device = torch.device("cuda") if is_cuda else torch.device("cpu")
        for param in self.bert_model.parameters():
            param.requires_grad = True
    
    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        output = self.bert_model(input_ids=batch_seqs,
                                attention_mask=batch_seq_masks,
                                token_type_ids=batch_seq_segments)
        sequence_output, pooler_output = output[0],  output[1]
        seq_avg = torch.mean(sequence_output, dim=1) # seq_output是最后一层的hidden state
        concat_out = torch.cat((seq_avg, pooler_output), dim=1) # 2*hidden_size
        
        loss, logits, probabilities = None, None, None 

        for i in range(self.multi_drop):
            concat_out = self.dropout(concat_out)
            logits = self.classifier(concat_out) # 2个类别的
            # probabilities = nn.functional.softmax(logits, dim=-1) # 概率值
            if labels is not None:
                if i == 0:
                    # print()
                    loss, probabilities = self.loss(logits, labels)
                    loss, probabilities = loss/ self.multi_drop, probabilities / self.multi_drop
                else:
                    loss += loss / self.multi_drop
                    probabilities += probabilities / self.multi_drop
        return loss, logits, probabilities # loss, logits, probabilities


class BertModelPredict(nn.Module):
    """
    The base model for doing prediction using trained matching network
    """
    def __init__(self):
        super(BertModelPredict, self).__init__()
        bert_config = BertConfig.from_pretrained(os.fspath(root_path / 'lib'/rank_model/"config.json"))
        self.bert = BertForSequenceClassification(bert_config)
        self.device = torch.device("cuda") if is_cuda else torch.device("cpu")
    
    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments):
        output = self.bert_model(input_ids=batch_seqs,
                                attention_mask=batch_seq_masks,
                                token_type_ids=batch_seq_segments)
        sequence_output, pooler_output = output[0],  output[1]
        seq_avg = torch.mean(sequence_output, dim=1)
        concat_out = torch.cat((seq_avg, pooler_output), dim=1) # 2*hidden_size
        logits = self.classifier(concat_out) 
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities

class MatchingNN(object):
    """
    The wrapper model for doing prediction using BertModelPredict
    """
    def __init__(self,
                 model_path = os.fspath(root_path / 'model/ranking/best.pth.tar'),
                 vocab_path = os.fspath(root_path / 'lib'/config.rank_model/'vocab.txt'),
                 data_path = os.fspath(root_path / 'data/ranking/train.csv'),
                 is_cuda = is_cuda,
                 max_sequence_length = max_sequence_length):
        self.vocab_path = vocab_path
        self.model_path = model_path 
        self.data_path = data_path
        self.max_sequence_length = max_sequence_length
        self.is_cuda = is_cuda 
        self.device = torch.device('cuda') if self.is_cuda else torch.device('cpu')
        self.load_model()
    
    def load_model(self):
        self.model = BertModelPredict().to(self.device)
        checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.vocab_path, do_lower_case=True)
        self.dataPro = DataProcessForSentence(self.bert_tokenizer, self.data_path, self.max_sequence_length)   
    
    def predict(self, q1, q2):
        result = [self.dataPro.trunate_and_pad(self.bert_tokenizer.tokenize(q1), self.bert_tokenizer.tokenize(q2))]
        seqs, seq_masks, seq_segments = torch.Tensor([i[0] for i in result]).type(torch.long),\
                                        torch.Tensor([i[1] for i in result]).type(torch.long),\
                                        torch.Tensor([i[2] for i in result]).type(torch.long)
        if self.is_cuda:
            seqs = seqs.to(self.device)
            seq_masks = seq_masks.to(self.device)
            seq_segments = seq_segments.to(self.device)
        
        with torch.no_grad():
            res = self.model(seqs, seq_masks, seq_segments)[-1].cpu().detach().numpy()
            label = res.argmax()
            score = res.tolist()[0][label]
        return label, score
