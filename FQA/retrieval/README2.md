
## 数据集 金融知道

https://github.com/SophonPlus/ChineseNlpCorpus

训练数据 100k

## run_model1.py:  TFIDF, LSI, LDA

data: 10000
数据清理前：

| method | recall | top3 score | top10 score|
| -- |-- |-- |--|--|
| TFDIF | 100% | 0.7977 | 0.458|
| LSI | 100%| 0.9058| 0.788|
| LDA | 89%| 0.9454 | 0.851|

其中LDA 召回失败的都是：需要清洗数据集

    01月21日今日新*兰*汇率多少
    002803基金能随时赎回吗
    00478下跌的怎样赎回
    004916赎回多久到账
    01月25日今日美元汇率多少 
    021-51####75是不是招行信用卡号码 
    27######55催款威胁我欠招商银行信用卡7万，他们打电话明天下午3点还，要不就转停卡部，封卡起诉，没钱还，会咋样
    027######55电话是招商银行的吗
    028-83####55是不是招商信用卡催收部门
    1####元投资理财选什么平台好？


data: 100000
数据清理后： (网址x  金额x  基金号x)

| method | recall | top3 score | top10 score|
| -- |-- |-- |--|--|
| TFDIF | 99.8% | 0.828 | 0.694|
| LSI | 99.8%| 0.9229| 0.892| 
| LDA | 92.9%| 0.94 | 0.98|

LDA 虽然召回率低，但是召回的都是非常相似的问题，所以相似度得分都很高。其实需要清洗数据集。

数据清理后 虽然tfidf/lsi的recal准确率降低，但是 召回的score 都提升了很多。



## run_model3.py: W2V + tfidf  HNSW

W2V  +  TFDIF:　recall 0.995, top3 disc: 0.695, top10 disc: 0.7986
W2V w/o TFIDF:  recall 0.993, top3 disc: 0.358, top10 disc: 0.318

## run_model4.py: W2V + SIF HNSW
10万数据集：
W2V + SIF w/o StopWords: top 10 recall 99.8%, top3 disc: 0.00369, total disc: 0.00653
                    （top30: 99.9%, top3 0.0048, total 0.01）

W2V + SIF + StopWords: top10 recall 99.8%, top3 disc: 0.0011,  total disc: 0.00167
                     （top20: 100%, top3 0.0016, total 0.0026）

问题：训练语言模型后，要评估模型好坏。
- 评估语言模型训练的好坏。 在对话数据集上训练语言模型时，dim=300，得出的大部分相似度都很高。 怀疑过拟合。降低维度到200, 相似度有了明显的差异。欧式距离。数据集比较小的时候，维度可以不用很高。

问题：SIF权重时 是否需要去除stopwords。
- 数据集比较小的时候，保留stopwords效果好。主成分不足够，去除stopwords之后，只剩下几个有代表性的词语，再去除主成分，使得这些有意义的词语的权重都变小，结果反而不好。
- 数据集比较大的时候去除的效果比较好。因为大数据集中 除了stopwords 还有其他的主成分。

问题：计算召回，查看top10召回的相似度得分，从而选择方法。
- tfidf: 虽然召回率高，基于词语的，但是整体句子的score比较低。
- lda: 虽然召回率稍微低于tfidf, 但是top10的召回结果文本相似度score很高。
- w2v+sif w/o remove stopwords: hnsw 欧式距离 相对来说比较大
- w2v+sif+remove stopwords: hnsw 欧式距离 相比没有stopwords 要好一些  enable remove topwords




