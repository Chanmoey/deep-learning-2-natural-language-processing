import numpy as np
from ch02.ppmi import ppmi
from ch02.preprocess import preprocess
from ch02.create_co_matrix import create_co_matrix

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)
np.set_printoptions(precision=3)  # 有效位数为3位
print('covariance matrix')
print(C)
print('-' * 50)
print('PPMI')
print(W)
