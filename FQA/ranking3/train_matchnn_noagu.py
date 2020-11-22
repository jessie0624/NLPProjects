"""
DESC: 训练深度匹配模型
"""
import os, sys, logging
import torch 
from torch.utils.data import DataLoader 
from data import DataProcessForSentence
from matchnn_utils import train, validate 
from transformers import BertTokenizer
from matchnn import BertModelTrain
from transformers.optimization import AdamW
import pandas as pd 
import csv 
import itertools
from sklearn.model_selection import KFold
from pathlib import Path
import os, sys, re 
sys.path.append('..')
import config 
from utils.tools import create_logger
from config import is_cuda, root_path, max_sequence_length

# logger = logging.getLogger(__name__)
logger = create_logger(os.fspath(root_path/'log/train_matchnn'))

seed = 9
torch.manual_seed(seed)
if is_cuda: 
    torch.cuda.manual_seed_all(seed)

def main(train_file, dev_file, target_dir, \
         epochs=20, batch_size=32, lr=2e-05, \
         patience=3, max_grad_norm=10.0, checkpoint=None):
    
    bert_tokenizer = BertTokenizer.from_pretrained(os.fspath(root_path \
        / 'lib/bert/vocab.txt'), do_lower_case=True)
    device = torch.device("cuda") if is_cuda else torch.device("cpu")
    logger.info(20 * "="+ " preparing for training "+"="*20)
    # if not target_dir.exists():
    #     target_dir.mkdir(parents=True, exist_ok=True)
    
    ## Loadding data 
    logger.info("\t * Loading training data ...")
    train_data = DataProcessForSentence(bert_tokenizer,  train_file)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    logger.info("\t * Loading validation data ...")
    dev_data = DataProcessForSentence(bert_tokenizer,  dev_file)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)

    # building model
    logger.info("\t * Building model ...")
    model = BertModelTrain().to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias",  "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.85, patience=0)

    best_score = 0.0
    start_epoch = 1

    epochs_count, train_losses, valid_losses = [], [], []

    if os.path.exists(checkpoint):
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        logger.info("\t * Training will continue on existing model from epoch {}...".format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count += checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
    
    _, valid_loss, _valid_accuracy, f1 = validate(model, dev_loader)
    logger.info("\t* Validation loss before training:{:.4f}, accuracy:{:.4f}%, f1:{:.4f}"
    .format(valid_loss, (_valid_accuracy * 100), f1))

    logger.info("\n"+ 20*"="+ "Training Bert model on device:{}" + 20*"=".format(device))
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        
        logger.info("* Training epoch {}".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)

        logger.info("-> Training time:{:.4f}s, loss={:.4f}, accuracy:{:.4f}".format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        logger.info("* Validation epoch {}".format(epoch)) 
        epoch_time, epoch_loss, epoch_accuracy, epoch_f1 = validate(model, dev_loader)
        valid_losses.append(epoch_loss)       
        logger.info("-> Validate time:{:.4f}s, loss={:.4f}, accuracy:{:.4f}, f1:{}".format(epoch_time, epoch_loss, (epoch_accuracy*100), epoch_f1))

        scheduler.step(epoch_accuracy)
        
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "best_score": best_score,
                "optimizer": optimizer,
                "epochs_count": epochs_count,
                "train_losses": train_losses,
                "valid_losses": valid_losses
            }, os.fspath(target_dir) +'/'+ str(epoch)+'best.pth.tar')
        
        if patience_counter >= patience:
            logger.info("-> Early stopping: patience limit reached, stopping...")
            break

# def get_fold_data(datas, indexs):
#     result = []
#     for index in indexs:
#         result.append(datas[index])
#     return result


# def write_fold_data(datas, filename):
#     with open(filename, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f, delimiter=",")
#         writer.writerow(['text_a', 'text_b', 'labels'])
#         writer.writerows(datas)


# def gen_kfold_data(name, out_dir, k=5):
#     datas = pd.read_csv(os.fspath(name))
#     datas = datas.values.tolist()
#     kf = KFold(n_splits=k, shuffle=True, random_state=42)
#     fold = 0
#     for train_index, dev_index in kf.split(datas):
#         train_datas = get_fold_data(datas, train_index)
#         dev_datas = get_fold_data(datas, dev_index)
#         train_file = os.path.join(out_dir, 'train.csv')
#         dev_file = os.path.join(out_dir, 'dev.csv')
#         write_fold_data(train_datas, train_file)
#         write_fold_data(dev_datas, dev_file)        
#         break

def data_agu(old_path, new_path):
    datas = pd.read_csv(old_path)
    dfd = datas.copy(deep=True)
    dfd['text_c'] = dfd['text_a']
    dfd['text_a'] = dfd['text_b']
    dfd['text_b'] = dfd['text_c']
    dfd = dfd[['text_a', 'text_b', 'labels']]
    ret = pd.concat([datas, dfd])
    ret.drop_duplicates(inplace=True)
    ret.to_csv(new_path, index=False)
    return ret   

if __name__ == '__main__':
    ##　不对train data 增强
    org_data_path = os.fspath(Path(root_path/"data/ranking"))
    train_file = os.path.join(org_data_path, 'train_py.csv')
    dev_file = os.path.join(org_data_path, 'dev.csv')

    Path(root_path / "data" /"ranking3_noagu").mkdir(parents=True, exist_ok=True)
    Path(root_path / "model" /"ranking3_noagu").mkdir(parents=True, exist_ok=True)

    ranking_bert_train = os.path.join(root_path, 'data','ranking3_noagu', 'train.csv')
    ranking_bert_dev = os.path.join(root_path, 'data','ranking3_noagu', 'dev.csv')
    rank_path = os.path.join(root_path, "model","ranking3_noagu")

    # train_data = data_agu(train_file, ranking_bert_train)
    train_file = pd.read_csv(train_file)
    train_file.to_csv(ranking_bert_train, index=False)
    dev = pd.read_csv(dev_file)
    dev.to_csv(ranking_bert_dev, index=False)
    # gen_kfold_data(config.raw_file, config.data_ranking_path)
    main(ranking_bert_train, ranking_bert_dev, rank_path, checkpoint=os.path.join(rank_path,"best.pth.tar"))