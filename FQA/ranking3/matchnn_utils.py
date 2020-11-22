"""
DECS: utils for matchnn
"""
import torch 
import torch.nn as nn 
import time 
from tqdm import tqdm 
from sklearn.metrics import roc_auc_score 

from sklearn.metrics import precision_score, accuracy_score,recall_score, roc_auc_score

from matchnn import FGM 

def generate_sent_masks(enc_hiddens, source_lengths):
    """
    @desc: generate sentence masks for encoder hidden states
    @param:
        - enc_hiddens: (Tensor) encoding of shape(b, src_len, h)
                      where b=batch_size, src_len=max_source_length, h=hidden_size
        - source_lengths: (List[int]) List of actual lengths for each of the sentences in the batch.
                        len = batch_size, 
    @return:
        - enc_masks: (Tensor) Tensor of sentence masks of shape(b, src_len), 
                    where src_len=max_source_length, b=batch_size
    """
    enc_masks = torch.zeros(enc_hiddens.size(0), 
                            enc_hiddens.size(1), 
                            dtype=torch.float)
    for e_id, src_len in enumerate(source_lengths):
        enc_masks[e_id, :src_len] = 1
    return enc_masks

def masked_softmax(tensor, mask):
    """
    @desc: apply a masked softmax on the last dimension of a tensor.
    @param:
        - tensor: (batch,*,seq_len) 
                The input tensor on which the softmax function must be applied along the last dimension.
        - mask: (batch, *, seq_len) 
                A mask of the same size as the tensor with 0 in the position of the values that must be masked.
                and 1 everywhere else.
    @return:
        - retsult: A tensor of the same size as the inputs containing the result of the softmax
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    # reshape the mask so it matches the size of the input tensor
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float() 
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)

def weighted_sum(tensor, weights, mask):
    """
    @desc: apply a weighted sum on the vectors along the last dimension of tensor and mask the veoctors in the result with mask.
    @param:
        - tensor: A tensor of vectors on which a weighted sum must be applied.
        - weights: The weights to use in the weighted sum.
        - mask: A mask to apply on the result of the weighted sum
    @return:
        - result: A new tensor containing the reuslt of the weighted sum after the mask has been applied on it
    """
    weighted_sum = weights.bmm(tensor)
    
    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expaned_as(weighted_sum).contiguous().float()
    return weighted_sum * mask 

def correct_predictions(output_probabilities, targets):
    """
    @desc: compute the number of predictions that match some target classes in the output of a model.
    @param:
        - output_probabilities: A tensor of probabilities for different output classes
        - targets: the indices of the actual target classes.
    @return:
        - the number of collect predictions in "output_probabilities"
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()

def validate(model, dataloader):
    """
    compute the loss and accuarcy of a model on some validation dataset.
    """
    model.eval()
    device = model.device 
    epoch_start = time.time() 
    running_loss = 0.0
    running_accuracy = 0.0 
    all_prob, all_labels = [], []

    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            seqs = batch_seqs.to(device) 
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)
            
            loss, logits, probabilities = model(seqs, masks, segments, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probabilities, labels)
            
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    epoch_time = time.time() - epoch_start 
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))
    # epoch_f1 = epoch_accuracy 
    return epoch_time, epoch_loss, epoch_accuracy, roc_auc_score(all_labels, all_prob),

def test(model, dataloader):
    """
    Test the accuracy of a model on some labeled test dataset.
    """
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob, all_labels = [], []
    
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            batch_start = time.time()
            seqs = batch_seqs.to(device) 
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)

            _, _, probabilities = model(seqs, masks, segments, labels)
            accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))
    return batch_time, total_time, accuracy, roc_auc_score(all_labels, all_prob)

def train(model, dataloader, optimizer, epoch_number, max_gradient_norm, adv_type='fgm'):
    """
    """
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss, correct_preds = 0.0, 0
    tqdm_batch_iterator = tqdm(dataloader)
    
    # fgm = None
    # if adv_type == 'fgm':
    #     fgm = FGM(model)

    for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments, \
        batch_labels) in enumerate(tqdm_batch_iterator):
        
        batch_start = time.time()
        seqs = batch_seqs.to(device) 
        masks = batch_seq_masks.to(device)
        segments = batch_seq_segments.to(device)
        labels = batch_labels.to(device)

        optimizer.zero_grad()
        loss, logits, probabilities = model(seqs, masks, segments, labels)
        loss.backward()

        # if adv_type == 'fgm':
        #     fgm.attack()  ##对抗训练
        #     loss_adv = model(**inputs)[0]
        #     loss_adv.backward()
        #     fgm.restore()

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probabilities, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1), running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy
       