import sentence_transformers
from torch.utils.data import DataLoader 
import math 
from sentence_transformers import models, losses 
from sentence_transformers import SentencesDataset,LoggingHandler,\
    SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import LabelAccuracyEvaluator
import logging 
from datetime import datetime
import sys, os, gzip, csv 
import pandas as pd 
from pathlib import Path
sys.path.append('..')
import config 
from config import *
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

model_name = os.fspath(Path(root_path / "lib" / "bert"))
train_batch_size = 4
model_save_path = os.fspath(Path(root_path / "model" /"ranking4" / 'flsbert')) + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),\
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

class PairProcessor:
    def load_data(self, filename):
        datas = pd.read_csv(filename).values.tolist()
        return datas 
    
    def get_labels(self):
        return ['0', '1']
    
    def get_examples(self, data_dir, set_type):
        file_map = {'train': 'train.csv',
                    'dev': 'dev.csv',
                    'test': 'test.csv'}
        file_name = os.path.join(data_dir, file_map[set_type])
        datas = self.load_data(file_name)
        examples = self.create_examples(datas, set_type)
        return examples
    
    def create_examples(self, datas, set_type):
        examples = []
        for i, data in enumerate(datas):
            guid = i 
            text_a = data[0].strip()
            text_b = data[1].strip()
            if set_type == 'test':
                label = None
            else:
                label = int(data[2])
            examples.append(InputExample(
                texts=[text_a, text_b],
                label = label
            ))
        return examples

processor = PairProcessor()

data_path = os.fspath(Path(root_path/"data/ranking4"))
# dev_file = os.fspath(Path(root_path/"data/ranking4"))
# test_file = os.fspath(Path(root_path/"data/ranking4"))

train_samples = processor.get_examples(data_path, 'train')
train_dataset = SentencesDataset(train_samples, model=model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
train_loss = losses.FocalLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)

dev_samples = processor.get_examples(data_path, 'dev')
dev_evaluator = LabelAccuracyEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='dev')

# Configure the training
num_epochs=10

warmup_steps = math.ceil(len(train_dataset) * num_epochs / train_batch_size * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=num_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=model_save_path)


test_samples = processor.get_examples(data_path, 'test')
model = SentenceTransformer(model_save_path)
test_evaluator = LabelAccuracyEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='test')
test_evaluator(model, output_path=model_save_path)