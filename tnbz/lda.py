import jieba
import pandas as pd
import pymysql
import re
from gensim import corpora
import gensim
import pickle
import os
import numpy as np


class LDA:

    def __init__(self):
        with open('./material/stop_char.dat', 'r', encoding='utf-8') as f:
            self.stopwords = [i.strip('\n') for i in f.readlines()]
        jieba.initialize()
        jieba.load_userdict('./material/mydict.txt')
        self.pattern = [
            [re.compile(r'&\w+;'), '[emo]'],
            [re.compile(r'{:soso\w+:}'), '[soso]'],
            [re.compile(r'https?://(www\.)?[\w/%=@#!.?&\-]+/?'), '[url]'],
            [re.compile(r'[\d.]+'), '']
        ]
        if os.path.exists('./model/LDA_model.dat') and os.path.exists('./model/dictionary.dat'):
            self.dictionary = pickle.load(open('./model/dictionary.dat', 'rb'))
            self.model = gensim.models.ldamodel.LdaModel.load("./model/LDA_model.dat")
        else:
            self.dictionary, self.model = self.generate_model()

    def generate_model(self):
        db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
        cursor = db.cursor()
        data = []
        cursor.execute("select content from posts")
        contents = cursor.fetchall()
        for c in contents:
            words = self.__seg(c[0])
            if len(words) > 0:
                data.append(words)
        print("finish segmentation")

        dictionary = corpora.Dictionary(data)
        pickle.dump(dictionary, open('./model/dictionary.dat', 'wb'))
        print("Dictionary constructed.")
        corpus = [dictionary.doc2bow(doc) for doc in data]

        print("Start to learn LDA Model.")
        model = gensim.models.ldamodel.LdaModel(corpus, num_topics=20)
        model.save('./model/LDA_model.dat')
        return dictionary, model

    def lda_feature(self, sentence: str):
        bow = self.dictionary.doc2bow(self.__seg(sentence))
        topic_distribution = self.model.get_document_topics(bow)
        topic_distribution.reverse()
        next_topic = topic_distribution.pop() if len(topic_distribution) > 0 else None
        topic_feature = []
        for i in range(20):
            if next_topic is not None and i == next_topic[0]:
                topic_feature.append(next_topic[1])
                next_topic = topic_distribution.pop() if len(topic_distribution) > 0 else None
            else:
                topic_feature.append(0)
        return topic_feature

    def __seg(self, content: str):
        for p in self.pattern:
            content = re.sub(p[0], p[1], content)
        content = content.strip()
        words = []
        try:
            sentence = jieba.cut(content)
        except UnicodeDecodeError:
            return []
        for w in sentence:
            if self.__judge(w):
                words.append(w)
        return words

    def __judge(self, word):
        if word == '' or word == ' ':
            return False
        if word in self.stopwords:
            return False
        if re.match(r'\w', word) is not None and len(word) == 1:
            return False
        return True


if __name__ == '__main__':
    a = LDA()
    a.generate_model()
