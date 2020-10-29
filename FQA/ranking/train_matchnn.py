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
sys.path.append('..')
from utils.tools import create_logger
from config import is_cuda, root_path, rank_path, ranking_train, \
    ranking_test, ranking_dev, max_sequence_length

# logger = logging.getLogger(__name__)
logger = create_logger(os.fspath(root_path/'log/train_matchnn'))

seed = 9
torch.manual_seed(seed)
if is_cuda: 
    torch.cuda.manual_seed_all(seed)

def main(train_file, dev_file, target_dir, \
         epochs=10, batch_size=32, lr=2e-05, \
         patience=3, max_grad_norm=10.0, checkpoint=None):
    
    bert_tokenizer = BertTokenizer.from_pretrained(os.fspath(root_path \
        / 'lib/bert/vocab.txt'), do_lower_case=True)
    device = torch.device("cuda") if is_cuda else torch.device("cpu")
    logger.info(20 * "="+ " preparing for training "+"="*20)
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
    
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

    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        logger.info("\t * Training will continue on existing model from epoch {}...".format(start_epoch))

        model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count += checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
    
    _, valid_loss, _valid_accuracy, auc = validate(model, dev_loader)
    logger.info("\t* Validation loss before training:{:.4f}, accuracy:{:.4f}%, auc:{:.4f}"
    .format(valid_loss, (_valid_accuracy * 100), auc))

    logger.info("\n"+ 20*"="+ "Training Bert model on device:{}" + 20*"=".format(device))
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        
        logger.info("* Training epoch {}".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)

        logger.info("-> Training time:{:.4f}s, loss={:.4f}, accuracy:{:.4f}".format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        logger.info("* Validation epoch {}".format(epoch)) 
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(model, dev_loader)
        valid_losses.append(epoch_loss)       
        logger.info("-> Validate time:{:.4f}s, loss={:.4f}, accuracy:{:.4f}".format(epoch_time, epoch_loss, (epoch_accuracy*100)))

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
            }, os.fspath(target_dir / str(epoch)+'best.pth.tar'))
        
        if patience_counter >= patience:
            logger.info("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == '__main__':
    main(ranking_train, ranking_dev, rank_path, checkpoint=os.fspath(rank_path/"best.pth.tar"))