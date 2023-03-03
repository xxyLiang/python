import pandas as pd
import pymysql
from gensim import corpora
import gensim
import pickle
import os
from auxiliary import CutWord

N_TOPICS = 20


class LDA:

    def __init__(self):
        self.cut_tool = CutWord(cut_all=True)
        if os.path.exists('LDA/LDA_model.dat') and os.path.exists('LDA/LDA_dictionary.dat'):
            self.dictionary = pickle.load(open('LDA/LDA_dictionary.dat', 'rb'))
            self.model = gensim.models.ldamodel.LdaModel.load("./LDA/LDA_model.dat")
        else:
            self.dictionary, self.model = self.generate_model()

    def generate_model(self):
        db = pymysql.connect(host='localhost',
                             user='root',
                             password='651133439a',
                             database='rec_sys')
        cursor = db.cursor()
        cursor.execute(
            "SELECT content FROM `posts` where LENGTH(content) > 2 AND tid in "
            "(select tid from threads where forum=1 and publish_date BETWEEN '2018-01-01' AND '2018-12-31')"
        )
        data = pd.DataFrame(cursor.fetchall(), columns=['content'])
        data["content_cut"] = data.content.apply(self.cut_tool.cut)

        dictionary = corpora.Dictionary(data.content_cut)
        pickle.dump(dictionary, open('LDA/LDA_dictionary.dat', 'wb'))
        print(r"Dictionary is constructed and saved at '%s/LDA/LDA_dictionary.dat'" % os.getcwd())
        corpus = [dictionary.doc2bow(doc) for doc in data.content_cut]

        print("Start to learn LDA Model.")
        model = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=N_TOPICS)
        model.save('./LDA/LDA_model.dat')
        print(r"Training Model is completed. The model is saved at '%s/LDA/LDA_model.dat'" % os.getcwd())
        return dictionary, model

    def lda_feature(self, sentence: str):
        bow = self.dictionary.doc2bow(self.cut_tool.cut(sentence))
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


if __name__ == '__main__':
    a = LDA()
