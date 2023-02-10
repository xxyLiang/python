# 从数据库取出数据，清洗后形成训练word2vec的材料

import pymysql
import jieba
import re
# import pynlpir


class Clean:

    def __init__(self):
        self.db = pymysql.connect("localhost", "root", "admin", "weibo", charset='utf8mb4')
        self.cursor = self.db.cursor()
        # pynlpir.open()
        # pynlpir.nlpir.ImportUserDict(b"./wv_material/user_dict2.txt")
        with open('./wv_material/stop_char.dat', 'r', encoding='utf-8') as f:
            self.stopwords = [i.strip('\n') for i in f.readlines()]
        self.savepath = './wv_material/training/with_stopwords/'
        jieba.initialize()
        jieba.load_userdict('./wv_material/user_dict.txt')

    def judge(self, word):
        if word == '':
            return False
        if word in self.stopwords:
            return False
        # if re.match(r'@.+', word) is not None:  # 去@某人
        #     return False
        if re.match(r'[a-zA-Z]', word) is not None and len(word) == 1:
            return False
        return True

    def seg(self, content):
        words = []
        try:
            sentence = jieba.cut(content)
        except UnicodeDecodeError:
            return []
        for w in sentence:
            word = w.strip()
            if self.judge(word):
                words.append(word)
        return words

    def clean_my_text(self):
        sql = "select content_words,content_bq from comment_info"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        all_words = []
        for i in result:
            content = i[0].strip() + i[1]
            words = self.seg(content)
            if len(words) == 0:
                continue
            all_words.append(words)
        with open(self.savepath + 'word2vec_mytext.txt', 'w', encoding='utf-8') as f:
            for a in all_words:
                f.writelines(' '.join(a) + '\n')
        print('finish clean_my_text')

    def clean_NLPCC_text(self):
        sql = "select weibo_id from train_weibo"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        all_words = []
        for i in result:
            content = ""
            try:
                sql = "select content from train_sentence where weibo_id=%s"
                self.cursor.execute(sql, i[0])
                content_list = self.cursor.fetchall()
                for c in content_list:
                    content += (' ' + c[0])
            except:
                continue
            content = self.clean_pattern(content)
            words = self.seg(content)
            if len(words) == 0:
                continue
            all_words.append(words)
        with open(self.savepath + 'word2vec_NLPCC.txt', 'w', encoding='utf-8') as f:
            for a in all_words:
                f.writelines(' '.join(a) + '\n')
        print('finish clean_NLPCC_text')

    def clean_500w(self, start=0, end=10):
        db = pymysql.connect("localhost", "root", "admin", "test", charset='utf8mb4')
        cursor = db.cursor()

        for t in range(start, end):
            all_sentences = []
            for i in range(t*5, t*5+5):
                sql = "select text from weibo where weiboId>=%d and weiboId<%d" % (i*100000, i*100000+100000)
                cursor.execute(sql)
                result = cursor.fetchall()
                for r in result:
                    text = self.clean_pattern(r[0])
                    words = self.seg(text)
                    if len(words) != 0:
                        all_sentences.append(words)
                print("i=%d text finished" % i)
            with open(self.savepath + 'word2vec_500w_%d.txt' % t, 'w', encoding='utf-8') as f:
                for s in all_sentences:
                    f.writelines(' '.join(s) + '\n')
        print('finish clean_500w')

    def clean_10w(self):
        db = pymysql.connect("localhost", "root", "admin", "test", charset='utf8mb4')
        cursor = db.cursor()
        sql = "select review from weibo_senti_100k"
        cursor.execute(sql)
        result = cursor.fetchall()
        all_sentences = []
        for r in result:
            text = self.clean_pattern(r[0])
            words = self.seg(text)
            if len(words) != 0:
                all_sentences.append(words)
        with open(self.savepath + 'word2vec_10w.txt', 'w', encoding='utf-8') as f:
            for s in all_sentences:
                f.writelines(' '.join(s) + '\n')
        print('finish clean_10w')

    def clean_news(self):
        import json
        list = []
        with open(r'D:\news2016zh_valid.json', 'r', encoding='utf-8') as f:
            for i in f.readlines():
                list.append(json.loads(i))

        all_sentences = []
        for i in list:
            text = i['content']
            text = re.sub(re.compile(r'(https?://)?t.cn/\w{6}\w?'), '', text)  # 去短链接
            text = re.sub(re.compile(r'(https?://)?(www\.)?[\w/%=@#!\.\?\&\-]+\.(com|cn|html?)'), '', text)  # 去url
            text = re.sub(re.compile(r'微信(ID|号)[:|：|\s][a-zA-Z0-9]+'), '', text)
            text = re.sub(re.compile(r'\d+'), '', text)  # 去数字
            text = re.sub(re.compile(r'[，。；!！：:"?？…\-~～@()（）【】＜＞<>“”\'《》、/\\+-=]+'), '', text).strip()
            words = self.seg(text)
            if len(words) != 0:
                all_sentences.append(words)
        with open(self.savepath + 'word2vec_news.txt', 'w', encoding='utf-8') as f:
            for s in all_sentences:
                f.writelines(' '.join(s) + '\n')
        print('finish clean_news')

    def clean_baike(self):
        import json
        list = []
        with open(r'D:\baike_qa_train.json', 'r', encoding='utf-8') as f:
            count = 0
            for i in f.readlines():
                list.append(json.loads(i))
                count += 1
                if count > 300000:
                    break
        all_sentences = []
        for i in list:
            text = i['desc'] + i['answer']
            text = re.sub(re.compile(r'\d+'), '', text)  # 去数字
            text = re.sub(re.compile(r'[\r|\n]+'), '', text)  # 去空白
            words = self.seg(text)
            if len(words) != 0:
                all_sentences.append(words)
        with open(self.savepath + 'word2vec_baike.txt', 'w', encoding='utf-8') as f:
            for s in all_sentences:
                f.writelines(' '.join(s) + '\n')
        print('finish clean_baike')

    def clean_binary(self):
        content_list = [[], []]
        for i in range(2):
            sql = "select content from train_sentence where weibo_id!=23002 and emotion_int_multi=%d" % i
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            for r in result:
                content = self.clean_pattern(r[0])
                content_list[i].append(content)

            sql = "select review from test.weibo_senti_100k where label=%d" % i
            self.cursor.execute(sql)
            result = self.cursor.fetchall()
            for r in result:
                content = self.clean_pattern(r[0])
                content_list[i].append(content)

        for i in range(2):
            all_words = []
            for r in content_list[i]:
                words = self.seg(r)
                if len(words) == 0:
                    continue
                all_words.append(words)
            with open('./wv_material/training/idf/content_%d.txt' % i, 'w', encoding='utf-8') as f:
                for a in all_words:
                    f.writelines(' '.join(a) + '\n')

    @staticmethod
    def clean_pattern(text):
        try:
            text = re.sub(re.compile(r'#(.*?)#'), '', text)  # 去话题标签
            text = re.sub(re.compile(r'【(.*?)】'), '', text)  # 去方括号标签
            text = re.sub(re.compile(r'(https?://)?t.cn/\w{6}\w?'), '', text)  # 去短链接
            text = re.sub(re.compile(r'(https?://)?(www\.)?[\w/%=@#!\.\?\&\-]+\.(com|cn|jpg|html?)'), '', text)  # 去url
            text = re.sub(re.compile(r'微信(ID|号)[:|：|\s][a-zA-Z0-9]+'), '', text)
            text = re.sub(re.compile(r'[a-zA-Z]+[0-9]+'), '', text)
            text = re.sub(re.compile(r'\d+'), '', text)  # 去数字
            text = re.sub(re.compile(r'//@.+[:|：]'), '', text)  # 去转发内容头
            text = re.sub(re.compile(r'回复@.+[:|：]'), '', text)  # 去回复头
            text = re.sub(re.compile(r'@.+\s'), '', text)  # 去@某人
            text = re.sub(re.compile(r'[（|(]\s?分享自.+[）|)]'), '', text)  # 去分享信息
            text = re.sub(re.compile(r'{%(.+)?%}'), '', text)
            text = re.sub(re.compile(r'[，。；!！：:"?？…\-~～@()（）【】＜＞<>“”\'《》、/\\+-=]+'), '', text).strip()
        except:
            pass
        return text

    def generate_vector(self):
        from gensim.models.word2vec import PathLineSentences
        from gensim.models import Word2Vec
        model = Word2Vec(PathLineSentences(self.savepath), size=300, window=10, min_count=10, sg=1, workers=4)
        model.wv.save_word2vec_format('D:/vector_stopwords.bin')
        print('finish generate_vector')

    @staticmethod
    def generate_idfs():
        from gensim.models.word2vec import PathLineSentences
        import math
        import pickle
        idf_dict = {}
        sentence_count = 0
        # for sentence in PathLineSentences('./wv_material/training/idf/'):
        #         sentence_count += 1
        #         word = []
        #         for w in sentence:
        #             if w not in word:
        #                 word.append(w)
        #         for w in word:
        #             if w in idf_dict:
        #                 idf_dict[w] += 1
        #             else:
        #                 idf_dict[w] = 1
        # for item in idf_dict:
        #     idf_dict[item] = math.log(sentence_count/idf_dict[item], 10)
        emotion_dict = [{}, {}]
        for i in range(2):
            for sentence in PathLineSentences('./wv_material/training/idf/content1_%d.txt' % i):
                sentence_count += 1
                word = []
                for w in sentence:
                    if w not in word:
                        word.append(w)
                for w in word:
                    if w in emotion_dict[i]:
                        emotion_dict[i][w] += 1
                    else:
                        emotion_dict[i][w] = 1
        for item in set(emotion_dict[0]) | set(emotion_dict[1]):
            a = emotion_dict[0][item] if item in emotion_dict[0] else 0
            b = emotion_dict[1][item] if item in emotion_dict[1] else 0
            x = math.sqrt(math.pow((a-b)/2, 2) + math.pow((b-a)/2, 2) + 1) / 2
            idf_dict[item] = math.log(sentence_count/(a+b), 10) * x
        with open('idfs_dict1.bin', 'wb') as f:
            pickle.dump(idf_dict, f)

    def predict(self):
        import transform_to_traindata
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model('./result_model/train_result')
        sql = "select id, content_words, content_bq from predictions"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        for r in result:
            content = r[2] + r[1].strip()
            content = re.sub(re.compile(r'//:.+'), '', content)
            words = self.seg(content)
            if len(words) == 0:
                continue
            vector = transform_to_traindata.words_to_vector(words)
            if vector is not None:
                y_pred = model.predict(xgb.DMatrix(vector.reshape(1, 300)))
                sql2 = "update predictions set pred=%f where id=%d" % (y_pred[0], r[0])
                try:
                    self.cursor.execute(sql2)
                    self.db.commit()
                except:
                    self.db.rollback()

    def nCov(self):
        import pandas
        data = pandas.read_csv(r'C:\Users\65113\Desktop\nCov_100k_train.labled.csv',
                               usecols=['微博id', '微博中文内容', '情感倾向']).values
        all_words = [[], []]
        for i in data:
            if i[2] in ['-1', '1']:
                if isinstance(i[1], float):
                    continue
                words = self.seg(self.clean_pattern(i[1]))
                if len(words) != 0:
                    if i[2] == '-1':
                        all_words[0].append(words)
                    else:
                        all_words[1].append(words)
        for i in range(2):
            with open('./wv_material/training/'
                      'idf/content1_%d.txt' % i, 'w', encoding='utf-8') as f:
                for a in all_words[i]:
                    f.writelines(' '.join(a) + '\n')


if __name__ == '__main__':
    a = Clean()
    # a.clean_my_text()
    #
    # a.clean_NLPCC_text()
    #
    # a.clean_10w()
    #
    # a.clean_news()
    # a.clean_baike()
    # a.clean_500w(0, 4)
    # Clean.generate_idfs()
    # a.generate_vector()
    a.clean_binary()
    # a.predict()
    # a.nCov()

