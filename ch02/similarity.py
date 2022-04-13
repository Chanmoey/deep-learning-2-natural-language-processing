import numpy as np


def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(np.power(x, 2))) + eps)
    ny = y / (np.sqrt(np.sum(np.power(y, 2))) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # 取出单词
    if query not in word_to_id:
        print("{} is not found".format(query))
        return

    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 计算余弦相似性
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 基于余弦相似性，按降序输出值
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print("{}: {}".format(id_to_word[i], similarity[i]))
        count += 1
        if count >= top:
            return
