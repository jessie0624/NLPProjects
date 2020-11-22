import os,sys
import lightgbm as lgb 
from sklearn import datasets as ds 
import pandas as pd 
import numpy as np 
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

def split_data_from_keyword(data_read, data_group, data_feats):
    """
    利用pandas转为lightgbm需要的格式进行保存
    : param: data_read
    : param: data_save
    """
    with 
