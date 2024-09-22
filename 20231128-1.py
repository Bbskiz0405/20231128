# 使用gensim库的Word2Vec
from gensim.models import Word2Vec

sentences = ["今天天氣真好", "不想上班想請假玩", "地鐵又要罷工", "網絡很差總是斷線"]

# 将句子转换为单个字符的列表，因为中文经常以字为单位进行分词
tokenized_sentences = [[char for char in sentence] for sentence in sentences]

# 初始化Word2Vec模型
word2vec_model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# 训练模型
word2vec_model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=10)


# 使用Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# 将句子标记化，以便Doc2Vec可以使用
tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(tokenized_sentences)]

# 初始化Doc2Vec模型
doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=1, workers=4)

# 建立词汇表
doc2vec_model.build_vocab(tagged_data)

# 训练模型
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)
