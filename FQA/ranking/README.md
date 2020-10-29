
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