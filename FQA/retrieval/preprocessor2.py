"""
清理数据集：
1. 替换里面的数字、网址、年龄等
2. 去除重复的内容

"""


import logging
import re 
import sys
import os
from pathlib import Path 
import logging
import numpy as np 
import pandas as pd 
from typing import List 
sys.path.append("..")
import config 
from config import data_path, clean_data

# from utils.tools import create_logger
# logger = logging.getLogger(__name__)
# logger = create_logger(os.fspath(root_path/'log/preprocessor'))

def filter_content(sentence):
    """
    特殊字段有：
    1. #E-s[数字x] #E-2[数字x] 等一系列数字—— 表情
    2. [ORDERID_10187709] —— 基金号
    3. [数字x] —— 数字
    4. [地址x] —— 地址
    5. [链接x] —— 链接
    6. [金额x] —— 金额
    7. [日期x] —— 日期
    8. [时间x] —— 时间
    9. [电话x] —— 电话
    对于表情，做法是直接删除。其他用希腊符号替换。
    """
    # sentence = re.sub(r"#E\-[\w]+\[数字x]", "α", sentence)
    sentence = re.sub("[&#6##32;|&lrm;|<U\+200E>|\^Y]", "", sentence)
    sentence = re.sub(
        r"(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?",
        "ε", sentence)
    sentence = re.sub(r"(http|ftp|https):\/\/ε", "ε", sentence)
    sentence = re.sub("ε", "[链接x]", sentence)
    sentence = re.sub(r"(http|ftp|https):\/\/[\w\.#\/=&]\/", "[链接x]", sentence)
    sentence = re.sub(r"www\.[\w#\/\.]+","[链接x]", sentence)
    sentence = re.sub('\"',"", sentence)
    sentence = re.sub("([0-9#\.,]+)([亿万百千]{0,}[英镑|加币|外币|欧元|日元|美元|韩元|港币|人民币]{1,})",
        lambda x: "[金额x]"+x.group(2), sentence)
    sentence = re.sub("([0-9#\.]+)([岁])",
        lambda x: "[年龄x]"+x.group(2), sentence)
    sentence = re.sub("([#0-9]+)([.]{,3}基金)", lambda x: "[基金号x]"+x.group(2), sentence)
    # sentence = re.sub("20#\d-[0-9]+-#[0-9]+#:\d")
    sentence = re.sub("([0-9#-]+)(.{,7}[卡])", lambda x: "[卡号x]"+x.group(2), sentence)
    sentence = re.sub("([0-9#-]+)(银行卡号)", lambda x: "[卡号x]"+x.group(2), sentence)
    sentence = re.sub("\$[0-9#\.]+", "[金额x]", sentence)
    sentence = re.sub("(人民币：)([0-9#\.]+)", lambda x: x.group(1) + "[金额x]", sentence)
    sentence = re.sub("([0-9#\s]{0,}\.[0-9#]%)", "[百分数x]", sentence)
    sentence = re.sub("([拨打|致电])([#0-9-]+)", lambda x: x.group(1) + "[电话x]", sentence)
    # sentence = re.sub("([*区|州|城|路|场|市|道|厦|叉|口]+)([的|招商银行|])", lambda x: "[地址x]" + x.group(2), sentence)
    # sentence = re.sub("**[\u4e00-\u9fff](*[\u4e00-\u9fff]{,2}){,3}", "[地址x]", sentence)
    return sentence


# data = pd.read_csv(clean_data)
# data['clean_title'] = data['best_title'].apply(lambda x: filter_content(x))



# data = "#1#17亿港币等于多少美元,19#94.1721人民币元"

# data = re.sub("([0-9#\.]+)([亿万百千]{0,}[日元|美元|韩元|港币]{0,})", lambda x: "[金额x]"+x.group(2), data)
# print(data)

data = pd.read_csv(config.clean_data)
print(data.shape)
data['title'] = data['best_title'].apply(lambda x: filter_content(str(x)))
data['reply'] = data['reply'].apply(lambda x: filter_content(str(x)))
data.drop_duplicates(subset=['title'], keep='first', inplace=True)
data[['title', 'reply']].to_csv(config.clean_data3,index=False)
print(data.shape)