# 方法与数据

模型: 孪生网络
数据: data/ranking 
其中train.csv dev.csv 是原始切分出来的数据
train_py.csv是通过pycorrector增强后的数据。
训练时应该用train_py.csv 135130条
bert训练时 还需要对这个py文件进行double --270k训练样本。

数据是通过pycorrector 增强后的数据集，总共12w左右(原始数据10w)。数据不均衡 正：负 = 1:4.47

bert encoder ->hidden

方法一： 孪生网络 + bert encoder + 余弦距离 + CE:  ACC: 85.85， F1: 61.19, Prec: 56.54, Recall: 66.67, average prec: 61.45
方法二： 孪生网络 + bert encoder + 余弦距离 + FL:  F1: 61.55%
方法三： 孪生网络 + bert encoder + 余弦距离 + FL + 数据增强？： TODO
方法三中 数据增强的方法：

sent-bert：效果不如单独的bert输入两句话直接计算出相似度得分 好。 但是因为可以直接预先存数database里面的embedding 然后通过计算余弦距离来计算两者的相似度，速度比较快。

sent-bert: FL明显比CE要好很多。

# 非孪生网络：ranking3
方法一： 单纯的bert， f1: 89.2%
方法二： bert + FL + multi-sample-dropout: (folder: ranking3, saving:ranking3_noagu)
方法三： bert + FL + multi-sample-dropout  对train数据增强q1->q2:  (folder: ranking3, saving:ranking3)

# 非孪生网络：rank5
方法四： maskbert + FL + 对train数据增强q1->q2:


# 非bert孪生网络：
方法一：孪生网络 





https://zhuanlan.zhihu.com/p/138061003


https://zhuanlan.zhihu.com/p/165991279
数据增强方案: https://www.biendata.xyz/models/category/3870/

参考: https://github.com/WenRichard/DIAC2019-Adversarial-Attack-Share
https://zhuanlan.zhihu.com/p/101039989
https://www.zhihu.com/collection/563173965
https://www.zhihu.com/question/358069078
https://tianchi.aliyun.com/competition/entrance/231762/forum
https://zhuanlan.zhihu.com/p/61725100
https://zhuanlan.zhihu.com/p/47580077
