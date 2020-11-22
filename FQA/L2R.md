
# Query-dependent models and Query-independent model

## Query-dependent models

- Boolean models 

    只能判断query 与 doc是否相关，无法得出相关性有多高的等。

- Vector Space model (VSM)

    query 和 doc 都表示为向量 欧式空间的向量。 然后通过向量内积来计算相似度。

    TF-IDF 常用的向量表示方法。tf*idf = f(t) * log(N/n(t)) = count(t)/total_words * log(total_doc/count_doc(t))
    (VSM 基于 BOW 形式的，假设词与词之间是独立的)

- LSI 
    (tries to avoid the assumption on the indepandence between terms)

    SVD 