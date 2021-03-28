# 从数据库取出数据，利用词向量转变为训练xgboost模型的材料

import gensim
import numpy
import pandas
import pickle
import math


model = gensim.models.KeyedVectors.load_word2vec_format('D:/vector_stopwords.bin')
vocab = model.vocab.keys()
with open('./wv_material/stop_char.dat', 'r', encoding='utf-8') as f:   # 两个一起改！
    stopwords = [i.strip('\n') for i in f.readlines()]
savepath = 'D:/train_data_stopwords.csv'
with open('idfs_dict.bin', 'rb') as f:
    idf_dict = pickle.load(f)


def judge(word):
    if word == '':
        return False
    if word in stopwords:
        return False
    if word not in vocab:
        return False
    return True


def words_to_vector(words):
    vector = numpy.zeros(300)
    word_count = 0
    word_dict = {}
    tf_idfs = {}
    for w in words:
        word = w.strip()
        if judge(word):
            word_count += 1
            # vector += model[word]
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
    if word_count == 0:
        return None
    summary = 0
    for w in word_dict:
        idf = idf_dict[w] if w in idf_dict else 10
        tf_idfs[w] = word_dict[w] * idf
        summary += math.pow(tf_idfs[w], 2)
    summary = math.sqrt(summary)
    for w in word_dict:
        idf = tf_idfs[w] / summary
        vector += model[w] * idf
    # vector /= word_count
    return vector


def transform_binary():
    all_vectors = []
    for i in range(2):
        # count = 0
        with open('./wv_material/training/idf/content_%d.txt' % i, 'r', encoding='utf-8') as f:
            for a in f.readlines():
                vector = words_to_vector(a.strip('\n').split(' '))
                if vector is not None:
                    vector = numpy.append(vector, numpy.array([i]))
                    all_vectors.append(vector)
                #     count += 1
                # if count == 70000:
                #     break
    data = pandas.DataFrame(all_vectors)
    data.to_csv(savepath, header=False, index=False)


if __name__ == '__main__':
    transform_binary()
