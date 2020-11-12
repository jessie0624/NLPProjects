"""
@DESC: 该文件是建立retriveal index索引库
根据客服对话语料中customer的问题 建立索引

需要注意的是 原始语料中 customer有太多与业务无关的问题，比如“谢谢， 好的，没事了”等 不能直接作为问题 建立index

为了能够提取有效的客户问题， 对语料库的 customer 一列进行如下清洗：
1. 问题和答复里面 有太多比如网址链接，表情符号，商品编号，订单号等，需要对这些进行统一处理。 
    --这一步在preprocess里面完成， 目前pair_data里面的数据已经去除了这些信息。
2. 查看是否有关键词。如果有关键词存在，则保留该问题和客服的答案。
   如果没有 则将其加入nouse_df文件里，可以用于闲聊 或者 从客服答复里面随机选取作为找不到匹配问题时的答复等。
2.1 关键词建立，根据customer问题一列 进行词性分析，提取出里面的 名词 动名词等。构成关键词list.
2.2 （用户问题一般比较短，所以只要能提取到关键词 就可以认为是有效问题）

用户问题的关键词： 为什么，是什么， 哪天，哪个，哪，什么，怎么， 吗， 呢 这些都是主要的词语 不能去掉。

将retrieval 的file 存入 retrieval folder下。
"""

import sys
sys.path.append('../')
import config
import pandas as pd 
import numpy as np 
from utils.jiebaSegment import *
if not config.retrieval_data.exists():
    data = pd.read_csv(config.pair_data_all)
    print(data.shape) # 260450
    # 去重
    data.drop_duplicates(subset=['custom'], keep='first', inplace=True)
    print(data.shape) # 174961
    # 去短
    # data = pd.read_csv(config.retrieval_data)
    data['flag'] = data['custom'].apply(lambda x: True if len(str(x))>5 else False)
    data[data['flag']==True][['session_id','custom', 'assistance']].to_csv(config.retrieval_data, index=False)

# 清除冗余问题
data = pd.read_csv(config.retrieval_data)   
seg = Seg()
seg.load_userdict(os.fspath(config.user_dict))

def clean(x):
    import re 
    x = re.sub("[,，\.。!]", "",x)
    x = re.sub("好","",x)
    x = re.sub("谢谢","",x)
    x = re.sub("[哦嗯啊]","",x)
    x = re.sub("多谢","", x)
    x = re.sub("哈", "", x)
    x = re.sub("\[SEP\]","",x)
    x = re.sub("\[数字x\]","", x)
    x = re.sub('[a-zA-Z0-9]',"",x)
    
    return x.strip()

data['clean_custom'] = data['custom'].apply(lambda x: "".join(seg.cut(clean(x))))
data['clean_custom_flag'] = data['clean_custom'].apply(lambda x: True if len(x.strip())==0 else False)
print(data[data['clean_custom_flag']==True])
data[data['clean_custom_flag']==False][['session_id','custom', 'assistance']].to_csv(config.retrieval_data, index=False)

# data = pd.read_csv(config.retrieval_data) 
# print(data.iloc[817])
# data=data[data.index!=817]
# # data.to_csv(config.retrieval_data, index=False)

print(data.shape) # 161610 ->161087 ->160993 ->160953 ->159388去除短的问句后 只剩下161k的数据
