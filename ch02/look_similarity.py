from ch02.create_co_matrix import create_co_matrix
from ch02.preprocess import preprocess
from ch02.similarity import cos_similarity, most_similar

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]
c1 = C[word_to_id['i']]
print(cos_similarity(c0, c1))

most_similar('you', word_to_id, id_to_word, C, top=5)