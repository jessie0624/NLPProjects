import jieba
import re 
import os, sys, re 
sys.path.append('..')
import config 
from config import *

def num_to_ch(num):
    """
    功能说明：将阿拉伯数字 ===> 转换成中文数字（适用于[0, 10000)之间的阿拉伯数字 ）
    """
    if len(num) == 5: return '手机号'
    num = int(num)
    _MAPPING = (u'零', u'一', u'二', u'三', u'四', u'五', u'六', u'七', u'八', u'九', ) 
    _P0 = (u'', u'十', u'百', u'千', ) 
    _S4 = 10 ** 4
    if num < 0 or num >= _S4:
        return None
    if num < 10: 
        return _MAPPING[num] 
    else: 
        lst = []
        while num >= 10: 
            lst.append(num % 10) 
            num = num // 10
        lst.append(num) 
        c = len(lst)    # 位数
        result = u'' 
        for idx, val in enumerate(lst): 
            if val != 0: 
                result += _P0[idx] + _MAPPING[val] 
            if idx < c - 1 and lst[idx + 1] == 0: 
                result += u'零'
        result = result[::-1]
        if result[:2] == u"一十":
            result = result[1:]
        if result[-1:] == u"零":
            result = result[:-1]
        return result
    
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff' or ch =='\ufeff':
            return True
    return False

def is_ustr(in_str): # 去除非中文和数字的
    out_str=''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str=out_str+in_str[i]
        else:
            out_str=out_str
    return out_str

def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True        
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return False
    if uchar in ('-',',','，','。','.','>','?'):
            return False
    return False 

def clean(x):# 两个字符间插入空白字符
    jieba.add_word('花呗')
    jieba.add_word('借呗')
    jieba.add_word('闲鱼')
    # step-1: 去除英文和标点符号
    r1 = '[a-zA-Z’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~：| ☆；．（）—～]+'
    x = re.sub(r1, '', x)
    # step-2: 去除非中文和数字的
    x = is_ustr(x)
    # step-3: 分词并且将数字改成大写
    # step-4: 将句子用空格拼接起来
    try:
        y = [num_to_ch(a)  if not is_Chinese(a) else a for a in jieba.lcut(x)]
        return " ".join(y)
    except:
        print(jieba.lcut(x))

def cor(x):
    ret, bias = pycorrector.correct(x) 
    if len(bias)>0:
        return ret
    else: 
        return ''


# def data_agu(datas, new_path):
#     dfd = datas.copy(deep=True)
#     dfd['text_c'] = dfd['text_a']
#     dfd['text_a'] = dfd['text_b']
#     dfd['text_b'] = dfd['text_c']
#     dfd = dfd[['id','text_a', 'text_b', 'labels']]
#     print(dfd.columns)
#     print(datas.columns)
#     ret = pd.concat([datas, dfd])
#     ret.drop_duplicates(inplace=True)
#     ret.to_csv(new_path, index=False)
#     return ret   

if __name__=='__main__':
    # reverse = True 
    # atec_file = ['atec_nlp_sim_train.csv', 'atec_nlp_sim_train_add.csv']
    # columns = ['id', 'text_a','text_b', 'labels']
    # df = read_file('ranking_raw_data', atec_file, columns)
    # print('before agu',df.shape)
    # if reverse:
    #     df = data_agu(df, data_path/"ranking"/"train_all.csv")
    #     print('after agu', df.shape)
    import pandas as pd 
    import pycorrector
    df = pd.read_csv(os.fspath(data_path/"ranking"/"train.csv"))
    pycorrector.set_custom_confusion_dict('/home/jie/Documents/JIE_NLP/NLPProjects/FQA/mycc.txt')
    from pycorrector import Corrector
    lm_path = '/home/jie/Documents/JIE_NLP/NLPProjects/FQA/data/ranking_raw_data/atec.klm'
    model = Corrector(language_model_path=lm_path)

    def cor(x, model):
        ret, bias = model.correct(x) 
        if len(bias)>0:
            return ret
        else: 
            return ''

    df['cor_a'] = df['text_a'].apply(lambda x: cor(x, model))
    df['cor_b'] = df['text_b'].apply(lambda x: cor(x, model))

    cor_a = df[df.cor_a!=''][['cor_a', 'text_b', 'labels']]
    cor_b = df[df.cor_b!=''][['text_a', 'cor_b', 'labels']]
    print(cor_a[:20])
    print(cor_b[:20])
    cor_a.rename(columns={'cor_a':'text_a'}, inplace = True)
    cor_b.rename(columns={'cor_b':'text_b'}, inplace = True)

    df_all = pd.concat([df[['text_a', 'text_b', 'labels']], cor_a, cor_b])
    df_all.to_csv(os.fspath(data_path/"ranking"/"train_py.csv"))