"""
DESC: 训练深度匹配模型
"""
import os, sys 
import torch 
from torch.utils.data import DataLoader 
from data import DataProcessForSentence
from matchnn_utils import train, validate 
from transformers import BertTokenizer
from matchnn import BertModelTrain
from transformers.optimization import AdamW
sys.path.append('..')
from config import is_cuda, root_path, max_sequence_length

