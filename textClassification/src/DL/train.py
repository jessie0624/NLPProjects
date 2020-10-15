import time
import torch 
import numpy as np 
import pandas as pd 
from importlib import import_module 
import argparse
from torch.utils.data import DataLoader 
import joblib 
from __init__ import *
from src.data.dataset import MyDataset, collate_fn
from src.DL.train_helper import train, init_network
from src.data.dictionary import Dictionary 
from src.utils.tools import create_logger 
from src.utils import config 
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer

parser = argparse.ArgumentParser(description="Chinese Text classification")
parser.add_argument('--model', default='bert', type=str, required=False, help='choose a model: RNN, CNN, RCNN, RNN_Attn, DPCNN, Transformer')
parser.add_argument('--word', default=True, type=bool, help='True for word, False for char')
parser.add_argument('--max_length', default=400, type=int, help='max sequence length')
parser.add_argument('--dictionary', default=None, type=str, help='dictionary path')

args = parser.parse_args()

logger = create_logger(config.root_path + '/logs/dl_main.log')

if __name__ == '__main__':
    model_name = args.model 
    x = import_module('models.' + model_name)
    config.model_name = model_name
    if model_name in ["bert", 'xlnet', 'roberta']:
        config.bert_path = config.root_path + '/model/' + model_name + '/'
        if 'bert' in model_name:
            config.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        elif 'xlnet' in model_name:
            config.tokenizer = XLNetTokenizer.from_pretrained(config.bert_path)
        elif 'roberta' in model_name:
            config.tokenizer = RobertaTokenizer.from_pretrained(config.bert_path)
        else:
            raise NotImplementedError
        
        config.save_path = config.root_path + '/model/saved_dict/' + model_name + '.ckpt'
        config.log_path = config.root_path + '/logs/' + model_name
        config.hidden_size = 768
        config.eps = 1e-8
        config.gradient_accumulation_steps = 1
        config.word = True
        config.max_length = 400
    np.random.seed()
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    logger.info('Loading data...')
    logger.info('Building dictionary')
    data = pd.read_csv(config.train_file, sep='\t')
    
    if args.word:
        data = data['text'].values.tolist()
    else:
        data = data['text'].apply(lambda x: " ".join("".join(x.split())))
    
    if args.dictionary is None:
        dictionary = Dictionary()
        dictionary.build_dictionary(data)
        del data 
        joblib.dump(dictionary, config.root_path + '/model/vocab.bin')
    else:
        dictionary = joblib.load(args.dictionary)
    
    if not args.model.isupper():
        tokenizer = config.tokenizer 
    else:
        tokenizer = None
    
    logger.info("Making dataset & dataloader ...")
    train_dataset = MyDataset(config.train_file, dictionary, args.max_length, tokenizer=tokenizer, word=args.word)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    dev_dataset = MyDataset(config.dev_file, dictionary, args.max_length, tokenizer=tokenizer, word=args.word)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    test_dataset = MyDataset(config.test_file, dictionary, args.max_length, tokenizer=tokenizer, word=args.word)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    
    if config.device == torch.device("cuda"):
        torch.cuda.empty_cache()

    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)

    train(config, model, train_dataloader, dev_dataloader, test_dataloader)