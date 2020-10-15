from collections import Counter

class Dictionary(object):
    def __init__(self, max_vocab_size=50000, min_count=None, start_end_tokens=False, 
                wordvec_mode=None, embedding_size=300):
        self.max_vocab_size = max_vocab_size
        self.min_count = min_count
        self.start_end_tokens = start_end_tokens
        self.wordvec_mode = wordvec_mode
        self.embedding_size = embedding_size
        self.PAD_TOKEN = '<PAD>'
    
    def build_dictionary(self, data):
        self.vocab_words, self.word2idx, self.idx2word, self.idx2word = self._build_dictionary(data)
        self.vocabulary_size = len(self.vocab_words)
        if self.wordvec_mode is None:
            self.embedding = None 
        elif self.wordvec_mode == 'word2vec':
            self.embedding = self._load_word2vec()
        
    def indexer(self, word):
        try:
            return self.word2idx[word]
        except:
            return self.word2idx['<unk>']
        
    def _build_dictionary(self, data):
        # 加入UNK, 按照需要加入EOS, 
        vocab_words = [self.PAD_TOKEN, '<UNK>']
        vocab_size = 2
        if self.start_end_tokens:
            vocab_words += ['<S0S>','<EOS>']
            vocab_size += 2
        counter = Counter([word for sentence in data for word in sentence.split()])
        # 按照最大的词典个数进行筛选
        if self.max_vocab_size:
            counter = {word: freq for word, freq in counter.most_common(self.max_vocab_size - vocab_size)}
            print(len(counter))
        # 过滤掉低频词
        if self.min_count:
            counter = {word: freq for word, freq in counter.items() if freq >= self.min_count}
        
        # 按照出现次数进行排序,并加到vocab_words 当中
        vocab_words += list(sorted(counter.keys()))
        idx2count = [counter.get(word, 0) for word in vocab_words]
        word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        idx2word = vocab_words
        return vocab_words, word2idx, idx2word, idx2count
