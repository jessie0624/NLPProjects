# FQA 

## Description

FQA是一个简单的基于检索的客户问答系统. 该系统包含2个模块: 业务咨询和闲聊.

- 业务咨询: 基于检索
- 闲聊: 基于生成模型

FQA的模型架构:

1. 通过分类模型,判断用户输入是业务咨询 还是 闲聊.
2. 如果是业务咨询,进入咨询模型.
3. 如果是闲聊,进入闲聊模块

## FQA模块采用技术以及未来优化空间
#### 1. 判断用户输入为咨询业务还是闲聊

folder: intention
目前采用: FastText 做二分类. 通过代码对对话数据集进行自动标注, 根据用户咨询中是否包含关键词来判断是否为业务咨询.
    
    可优化地方:

    1. 关键词提取方法: 目前根据pesg分词,提取用户咨询里面的名词,动名词等, 这是最简单粗暴的方式.  
        优化方法: 根据句法分析, 提取关键词. 可以尝试一下. TODO

    2. 分类模型为: 目前采用的FastText 做文本分类, 这个方法很简单, 效果一般.
        优化方法: 1. 调参看结果, 2.采用更复杂的模型, 或者ensemble 模型等.  TODO

    3. 数据标注方法: 目前根据是否含有关键词, 进行标注, 可能步准确. 
        优化方法: 人工标注(目前做不到)

    4. 目前只是一个简单的二分类,可以做成多分类,比如业务咨询是哪方面业务,或者 闲聊方面闲聊等. TODO 待定.
   
#### 2. 信息检索模块

folder: retrieval + ranking

该模块分为2部分, 1 召回粗排, 2 提取细排
    
##### 2.１ 召回粗排
1.句向量表示: 

目前采用 word2vec 在对话语料上训练300维词向量. 句向量通过wam(word average model)计算得到.

    优化空间: 1. 句向量可以用bert 得到, 或者通过TFIDF加权模式得到句向量等.TODO 
        
2.召回算法：

目前采用 Faiss的HNSW库,对用户的问题进行粗排,召回.

    优化空间: 
    1. 调参. 
    2. 在句向量中加上规则 权重等信息. 
    3. 降维(目前句向量是300维度,HNSW有点慢,可以用PCA降维) TODO.  
    4. 倒排索引(学一下). (可以参考: https://github.com/liqima/faiss_note 看有没有优化方案)

    粗排目标在于 快, 高召回率. top100 或者 top1000达到99%-100% 召回率.

##### 2.２　提取细排

目前采用模型: lightGBM
        
特征包含３大类: 
- 各种计算字符串间距离得分(lcs, jaccard, bm25, edit_dist), 
- 各种embedding + 相似度评估的算法计算出的 相似度.         
    - 具体embedding: embedding: w2v, fasttext, tfidf,
    - 相似度评估算法:　cos, eurl, pearson, wmd
- BERT相似度比较. q1 q2 用classification模型计算相似度得分.


将这些特征组合起来, 作为lightGBM 的训练特征, 进行训练. 

    优化空间：
        1. 调参
        2.             

#### 3. 闲聊生成模块

    还没有做




### 具体代码

####　召回粗排模块
文件夹: retrieval
1. 训练词向量 word2vec
2. 训练HNSW模型进行快速召回.

#### 细排模块
文件夹: ranking
1. 处理数据集用于做相似度匹配. ranking/data.py

2. 人工特征: bm25, train_LM 在训练集上训练几个模型: TFIDF, BM25, word2vec, FastText.

   构建相似度特征: ranking/similarity.py

3. 深度匹配:训练一个bert模型对输入的两个问题做序列相似度匹配,得到一个相似度分数. ranking/train_matchnn.py

4. 排序: 利用前面步骤的特征,使用LightGBM 来训练排序模型. ranking/ranker.py(RANK do_train=True 训练, do_train=False 预测)

#### 整合任务pipeline
整合任务型对话模块 task.py

    先做意图识别, 筛选业务性查询.

    然后对业务性查询进行召回.

    对召回结果进行排序.

### 相关知识点：

1. 分类模型 lightgbm VS xgboost VS GBDT

    lightgbm:
    https://blog.csdn.net/weixin_39807102/article/details/81912566   https://blog.csdn.net/weixin_44750583/article/details/103896854  https://blog.csdn.net/weixin_42001089/article/details/85343332


2. word2vec VS fasttext


3. bm25 VS TFIDF 


4. 相似度评估算法对比．


##### LightGBM参数

https://lightgbm.readthedocs.io/en/latest/Parameters.html

1. 控制参数

|控制参数| 意义| 用法|
|-|-|-|
|max_depth|  树的最大深度,  |      当模型过拟合时 可以考虑首先降低 max_depth|
|min_data_in_leaf| 叶子可能具有的最小记录数, |    默认为20, 过拟合时 可以调大一些|
|feature_fraction| 每次迭代中随机选择k%的特征来建树, |   boosting 为random forest时 用|
|bagging_fraction|  每次迭代时用的数据比例,     |   用于加快训练速度和减小过拟合|
|early_stopping_round| 如果一次验证数据的一个度量在最近的early_stopping_round回合中没有提高, 模型就停止训练,   | 加速分析,减少过多迭代|
|lambda| 指定正则化|  0~1|
|min_gain_to_split| 描述分裂的最小gain, | 控制树的有效分裂|
|max_cat_group| 在group边界上找到分割点, |   当类别数量很多时, 找分割点很容易过拟合时用.|

2. 核心参数

|控制参数| 意义| 用法|
|-|-|-|
|Task| 数据的用途, |选择Train或者predict|
|application | 模型的用途|     选择regression: 回归时,  binary: 二分类时, multiclass: 多分类时,
|boosting| 要用的算法,|    gbdt, rf:random forest,  dart: Dropouts meet Multiple Additive Regression Trees,  goss: Gradient-based One-side Sampling
|um_boost_round| 迭代次数,|  通常100+
|learning_rate| 如果一次验证数据的一个度量在最近的early_stopping_round 回合中没有提高, 模型将停止训练. | 常用0.1, 0.001, 0.003...|
|num_leaves||  默认为31|
|device||  cpu 或者gpu|
|metric ||mae: mean absolute error.   mse: mean squared error,   binary_logloss: loss for binary classification , multi_logloss: loss for multi classification.|

3. IO参数

|IO参数| 意义|
|-|-|-|
|max_bin| 表示feature将存入的bin的最大数量|
|categorical_feature| 如果categorical_features=0, 1, 2 则列0, 1, 2是categorical变量.|
|ignore_column| 与categorical_features类似, 只是 完全忽略 对应列|
|save_bianry| 这个参数为true时, 则数据集被保存为二进制文件, 下次读数据时速度会变快.|

4. 调参常用参数

|IO参数| 意义|
|-|-|-|
|num_leaves|取值应 <= 2^(max_depth)  超过此值会导致过拟合.|
|min_data_in_leaf| 将它设置为较大的值,可以避免生长太深的树, 但可能会导致 欠拟合, 在大型数据集时 就设置为数百或者数千|
|max_depth| 这个也是可以限制树的深度.|

5. 对于Faster Speed, better accuracy, over-fitting 三种目的时, 可以调整的参数

|Faster Speed| better accuracy | over-fitting|
|-|-|-|
|将max_bin 设置小一些| 用较大的max_bin| max_bin设置小一些|
|| num_leaves 大一些| num_leaves 小一些|
|用feature_fraction 来做sub-sampling|| 用feature_fraction|
|用bagging-fraction 和 bagging_freq||设定bagging_fraction 和bagging_freq|
||train data 多一些| training data 多一些|
|用save_binary来加速数据加载|直接用categorical features| 用gmin_data_in_leaf和min_sum_hessian_in_leaf|
|用parallel learning| 用dart| 用lambda_l1, lambda_l2, min_gain_to_spilt做正则化|
||num_iterations 大一些，learning_rate 小一些|用max_depth控制树的深度|

6. 参数boosting 对应的算法:
- gbdt: gbdt
- rf: random forest
- dart: Dropouts meet Multiple Additive Regression Trees.
- goss: Gradient-based One-side Sampling

7. lightGBM 缺失值处理: 通过设置缺失值是否为0 或者nan 等方式 告诉程序哪些是缺失值. 在做树的分裂时,通过从左到右侧遍历一遍, 再通过从右到左遍历一遍 得到当前的左子树或者右子树 然后通过根节点减去左/右子树获得 另外一个分支(缺失值就在另外一个分支里面), 为什么呢?因为遍历时 只遍历有值的情况进行计算, 所以另外一个分支通过根节点减去 计算出的分支 得到的结果里面 就包含了缺失值的信息.
8. 对类别特征处理 没有将他们one-hot, 好处是 可以实现many-vs-many,  如果是one-hot的化 只能one-vs-many(这样树的深度容易深)
9. lightGBM EFB特征捆绑算法 (其实就是降维)

   （1）到底那些特征需要合并到一起（2）怎么合并到一起

   (答: (1)互相排斥的特征可以捆绑(可以想象成 one-hot的逆过程). (2) 假如A特征的范围是[0,10),B特征的范围是[0,20),那么就给B特征加一个偏值，比如10，那么B的范围就变为[10,30)，所以捆绑为一个特征后范围就是[0,30].  所以结合两个问题来看其完成的任务不但是简简单单的捆绑，而且要达到捆绑后还能从取值上区分出特征之间的关系。)

   EFB大概就是先计算当前特征和当前bundle冲突，冲突小就将当前特征捆绑到当前bundle中，否则就再重新建一个bundle。需要注意的是该过程只需要在最开始做一次就好了，后面就都用捆绑好的bundle，其算法复杂度显而易见是#feature*#feature，但是当特征纬度过高，这显然也是不好的，于是乎对算法进行了改进，其不再建立图了，而是统计非零的个数，非零个数越多就说明冲突越大，互相排斥越小，越不能捆绑到一起。

10. lightGBM GOSS算法 (其实是采样)

    对于稀疏数据集来说,  首先, GBDT如果采用pre-sorted方式进行分裂可以通过忽略掉大部分值为0 特征 来减少复杂度, 但是GBDT如果使用了histogram-based 的形式, 则没有了相关的稀疏优化方法, 因为histogram-based需要遍历所有的数据的bin值,而不会管其是不是0, 同时, 传统的Adaboost 算法 数据集是有一个权重的,用来衡量其重要程度, 没有被训练好的样本权重加大, 以便下一个基学习器对其多加训练,于是可以依据该权重进行采样,做到采样利用部分数据集.  但是GBDT里面没有权重这一说法, 每次都是利用整个数据集, 这些数据集权重都是一样的,所以怎么办呢?

    lightGBM提出GOSS:  抽样肯定还是要抽的，毕竟减少了样本减少了复杂度嘛！没有权值我们根据什么抽呢？其发现可以将一阶导数看做权重，一阶导数大的说明离最优解还远，这部分样本带来的增益大，或者说这部分样本还没有被好好训练，下一步我们应该重点训练他们。

11. LightGBM 速度快,省内存 的几个原因总结:
    - 减少内存和时间: histogram bin  (时间复杂度降低 O(data*feature) -> O(bin * feature), 内存占用降低很多)以及 histogram 加速,计算 另外节点时 通过相减直接得到.
    - 快速高效生成决策树: 带深度限制的叶子节点增长策略. 每次分裂 增益最大的叶子节点.
    - 减少特征维度(降维): 互斥特征捆绑 EFB 
    - 减少样本(采样): 基于梯度单侧采样GOSS, 重点关注梯度较大的样本. 
    (降维和采样 加速和省内存)