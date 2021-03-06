
# 方法:

bert 最后一层 隐层avg + pooler 拼接 + dropout(多次dropout) + 二分类 + FocalLoss

数据: pycorrector + q1<->q2 互换 ===>>data/ranking_raw_data/pc_atec_double.csv 


## 数据预处理

原始数据集合是 102477， 切分训练集和验证集,然对训练集进行数据增强, pycorrector 增强 + q1-q2交换.

特点：
- 数据不均衡，正样本: 负样本 = 18685 : 83792   正:负=1:4.48
- 存在错别字 比如花呗--花被--花杯
- 很多数据被**遮掉。

### step-1 数据增强: 

参考: https://github.com/yongzhuo/nlp_xiaojiang/tree/master/AugmentText


1. 数据增强，因为是判断句子对儿，调换顺序后也是相似的文本，所以可以通过调换q1和q2来增强，后的数据由204951条记录. 231713: 正:负 = 42292: 189421 

    注意: 
        
        对于bert 来说 调换前后顺序 有意义,因为bert 是word-embedding + pos-embeding + seg-embedding, 所以输入前后顺序会因为pos-embedding + seg-embedding 不同而产生影响. 
        
        但是对与孪生网络来说是没有意义的. 因为孪生网络只是各自产生embedding,然后计算距离, 前后顺序没有影响.


2. 数据增强尝试:

- pycorrect 增强 在该数据集上, 增强后  总共: 115863  正样本: 21149   负样本: 94714   正:负=1:4.47

    比如: 
        
        刚消费的，扣卡里的钱，没扣花呗的钱，怎么回事 
        -> 港消费的，扣卡里的钱，美扣花呗的钱，怎么回事

        花呗怎么付款不鸟了
        --> 花呗怎么付款不鸣了


        上个月都用花呗，这个月也还款了，怎么还不能提额？是不是每个月都要用了一定的额度，才能提额 
        --> 上个月都用花呗，这个月也还款了，怎么还不能是额？是不是每个月都要用了一定的额度，才能使额   

        怎么感觉蚂蚁花呗钱越还欠的越多 
        --> 怎么感觉蚂蚁花呗钱说还欠的越多

        我的花呗是以前的手机号码，怎么更改成现在的支付宝的号码手机号 
        -->我的花呗是以前的手机号码，怎么更改呈现在的支付宝的号码手机号

        我上个月已经还过的一笔，现在退回来的钱怎么还直接退回花呗 
        --> 我上个月已经换过的一笔，现在退回来的钱怎么还直接退回花呗

        这几天用花呗交了三次话费都没充上 怎么回事 
        -->这几天用花呗交了三次话费都没冲上 怎么回事

        花呗绑定的是另一个号码，蜜码和帐号忘记了，手机号码也注消了，怎么办
        -->花呗绑定的是另一个号码，密码和帐号忘记了，手机号码也注消了，怎么办

        双***花呗有木有临时额度	 
        --> 双***花呗有模有临时额度

        花呗提额***到多少 
        --> 花呗市额***到多少

- 使用EDA增强

    注意:

        EDA 增强是通过jieba分词后,随机交换词的顺序,或者随机删除来进行增强的,所以增强后的文本有些语义不通顺,有些有可能会语义发生变化. 所以通过EDA增强时,一对儿句子,我们只选择1个来做增强. 两个都做增强时 label不一定正确.



### step-2: 数据不均衡： FocalLoss来处理。

    模型： BERT 作为embedding. 输出可以通过拼接。

