import jieba
import pandas as pd
import pymysql
import re
from gensim import corpora
import gensim
import pickle
import os

N_TOPICS = 20


class LDA:

    def __init__(self):
        self.stop_words = self.load_stop_word()
        jieba.initialize()
        jieba.load_userdict('./material/mydict.txt')
        self.pattern = [
            [re.compile(r'&\w+;'), 'emo'],
            [re.compile(r'{:soso\w+:}'), 'soso'],
            [re.compile(r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'), 'url']
        ]
        self._word = re.compile(r'[\u4e00-\u9fa5\dA-Za-z]+')
        if os.path.exists('./LDA/LDA_model.dat') and os.path.exists('./LDA/LDA_dictionary.dat'):
            self.dictionary = pickle.load(open('./LDA/LDA_dictionary.dat', 'rb'))
            self.model = gensim.models.ldamodel.LdaModel.load("./LDA/LDA_model.dat")
        else:
            self.dictionary, self.model = self.generate_model()

    @staticmethod
    def load_stop_word():
        with open('./material/stopwords.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            stop_words = set()
            for i in lines:
                i = i.replace('\n', "")
                stop_words.add(i)

        return stop_words

    def generate_model(self):
        db = pymysql.connect(host='localhost',
                             user='root',
                             password='651133439a',
                             database='rec_sys')
        cursor = db.cursor()
        cursor.execute(
            "SELECT content FROM `posts` where LENGTH(content) > 0 AND tid in "
            "(select tid from threads where forum=1 and publish_date BETWEEN '2018-01-01' AND '2018-12-31')"
        )
        data = pd.DataFrame(cursor.fetchall(), columns=['content'])
        data["content_cut"] = data.content.apply(self.__seg)

        dictionary = corpora.Dictionary(data.content_cut)
        pickle.dump(dictionary, open('./LDA/LDA_dictionary.dat', 'wb'))
        print(r"Dictionary is constructed and saved at '%s\LDA\LDA_dictionary.dat'" % os.getcwd())
        corpus = [dictionary.doc2bow(doc) for doc in data.content_cut]

        print("Start to learn LDA Model.")
        model = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=N_TOPICS)
        model.save('./LDA/LDA_model.dat')
        print(r"Training Model is completed. The model is saved at '%s\LDA\LDA_model.dat'" % os.getcwd())
        return dictionary, model

    def lda_feature(self, sentence: str):
        bow = self.dictionary.doc2bow(self.__seg(sentence))
        topic_distribution = self.model.get_document_topics(bow)
        topic_distribution.reverse()
        next_topic = topic_distribution.pop() if len(topic_distribution) > 0 else None
        topic_feature = []
        for i in range(N_TOPICS):
            if next_topic is not None and i == next_topic[0]:
                topic_feature.append(next_topic[1])
                next_topic = topic_distribution.pop() if len(topic_distribution) > 0 else None
            else:
                topic_feature.append(0)
        return topic_feature

    def __seg(self, content: str):
        for p, w in self.pattern:
            content = p.sub(w, content)
        content = self._word.findall(content)
        new_text = " ".join(content)
        seg_list_exact = jieba.cut(new_text, cut_all=True)
        result_list = []

        for word in seg_list_exact:
            if len(word) > 1 and word not in self.stop_words:
                result_list.append(word)
        return result_list


if __name__ == '__main__':
    a = LDA()
