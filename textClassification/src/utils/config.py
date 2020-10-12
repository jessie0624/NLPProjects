from __future__ import (absolute_import, unicode_literals)
import torch
import os
import numpy as np

# generate config
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(os.path.split(curPath)[0])[0]

train_file = root_path + '/data/train_clean.tsv'
dev_file = root_path + '/data/dev_clean.tsv'
test_file = root_path + '/data/dev.csv'
stopWords_file = root_path + '/data/stopwords.txt'
log_dir = root_path + '/logs/'
