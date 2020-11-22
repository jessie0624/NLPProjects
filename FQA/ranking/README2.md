Ranking 这个文件夹主要目的是 计算retrieval 得到的内容 与query的相似度得分。即matching步骤。
然后通过matching计算出的得分，通过lightGBM进行rerank 排序。

# Step-1 matching 
## BM25
文件 bm25_self.py

问题是 基于BOW 所有如果两句话没有共同的WORD 相似度就为0. 
label==1
    count    68661.000000
    mean         3.091870
    std          2.755140
    min          0.000000
    25%          0.971987
    50%          2.694159
    75%          4.602477
    max         60.715115
    Name: score, dtype: float64

label==0
    count    133792.000000
    mean          2.341484
    std           2.429989
    min           0.000000
    25%           0.290370
    50%           1.871636
    75%           3.550688
    max          49.876873

比如label == 1但是BM25 得分为0：
202450               如何申请货款            怎样开通我微粒贷       1  0.000000
202451               多久才有贷款         凌晨以后的申请何时到账       1  0.000000

## trainLM
文件 trainLM.py, 为了解决BM25的问题，这里训练词嵌入 W2V, fasttext 根据上下文来训练。

training on a 18481360 raw words (13229622 effective words)

## train_BERT 语义匹配
train_matchnn.py  深度语义匹配
"hidden_dropout_prob": 0.1 时， train acc, eval: acc
"hidden_dropout_prob": 0.4 时， 增大dropout 防止过拟合。
 

# Step-2 rerank

目前采用的是 wam model 词向量平均模型，不包含bert的训练结果。
直接用lgbm classification 试一下结果。

## 参数1：方法 lightGBM
     w2v_cos, w2v_eucl, w2v_pearson, w2v_wmd, fast_cos, fast_eucl, fast_pearson, fast_wmd

 
