import pymysql
import clean
import numpy
import pandas


int_emo = {0: 'none', 1: 'disgust', 2: 'sadness', 3: 'like'}
emo_int = {'anger': 1, 'disgust': 1, 'sadness': 2, 'none': 0, 'like': 3, 'happiness': 3}

Clean = clean.Clean()

def to_file():
    for i in range(4):
        sql = "select content from train_sentence " \
              "where emotion_int_multi=%d order by rand() limit 7000"
        Clean.cursor.execute(sql % i)
        result = Clean.cursor.fetchall()

        all_words = []
        for r in result:
            content = r[0]
            content = Clean.clean_pattern(content)
            words = Clean.seg(content)
            if len(words) == 0:
                continue
            all_words.append(words)
        with open('./training/content_%d.txt' % i, 'w', encoding='utf-8') as f:
            for a in all_words:
                f.writelines(' '.join(a) + '\n')


def para(mode):     # 0-training, 1-test
    import transform_to_traindata
    if mode == 0:
        savepath = 'D:/train_data_multi.csv'
        sql = "select weibo_id from train_weibo " \
              "where weibo_id not in (select weibo_id from train_weibo_test) " \
              "and emotion_int_multi=%d"
    else:
        savepath = 'D:/test_data_multi.csv'
        sql = "select weibo_id from train_weibo_test where emotion_int_multi=%d"

    all_vectors = []
    for i in range(4):
        Clean.cursor.execute(sql % i)
        result = Clean.cursor.fetchall()

        for r in result:
            content = ''
            sql2 = "select content from train_sentence where weibo_id=%d" % r[0]
            try:
                Clean.cursor.execute(sql2)
                sentences = Clean.cursor.fetchall()
                for s in sentences:
                    content += s[0]
            except:
                continue
            content = Clean.clean_pattern(content)
            words = Clean.seg(content)
            if len(words) == 0:
                continue
            vector = transform_to_traindata.words_to_vector(words)
            if vector is None:
                continue
            vector = numpy.append(vector, numpy.array([i]))
            all_vectors.append(vector)
    data = pandas.DataFrame(all_vectors)
    data.to_csv(savepath, header=False, index=False, mode='a')


def sentence():
    import transform_to_traindata
    savepath = 'D:/train_data_multi.csv'
    sql = "select content from train_sentence_test " \
          "where emotion_int_multi=%d"

    all_vectors = []
    for i in range(4):
        Clean.cursor.execute(sql % i)
        result = Clean.cursor.fetchall()

        for r in result:
            try:
                content = r[0]
                content = Clean.clean_pattern(content)
                words = Clean.seg(content)
                if len(words) == 0:
                    continue
                vector = transform_to_traindata.words_to_vector(words)
                if vector is None:
                    continue
                vector = numpy.append(vector, numpy.array([i]))
                all_vectors.append(vector)
            except:
                continue
    data = pandas.DataFrame(all_vectors)
    data.to_csv(savepath, header=False, index=False)
    # all_vectors = []
    # for i in range(4):
    #     with open('./training/content_%d.txt' % i, 'r', encoding='utf-8') as f:
    #         for a in f.readlines():
    #             vector = transform_to_traindata.words_to_vector(a.strip('\n').split(' '))
    #             if vector is not None:
    #                 vector = numpy.append(vector, numpy.array([i]))
    #                 all_vectors.append(vector)
    # data = pandas.DataFrame(all_vectors)
    # data.to_csv(savepath, header=False, index=False)


def generate_idfs():
    from gensim.models.word2vec import PathLineSentences
    import math
    import pickle
    idf_dict = {}
    sentence_count = 0
    emotion_dict = [{}, {}, {}, {}]
    for i in range(4):
        for sentence in PathLineSentences('./training/content_%d.txt' % i):
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
    for item in set(emotion_dict[0]) | set(emotion_dict[1]) | set(emotion_dict[2]) | set(emotion_dict[3]):
        a = emotion_dict[0][item] if item in emotion_dict[0] else 0
        b = emotion_dict[1][item] if item in emotion_dict[1] else 0
        c = emotion_dict[2][item] if item in emotion_dict[2] else 0
        d = emotion_dict[3][item] if item in emotion_dict[3] else 0
        sum = (a + b + c + d) / 4
        x = math.sqrt(
            math.pow(sum-a, 2) +
            math.pow(sum-b, 2) +
            math.pow(sum-c, 2) +
            math.pow(sum-d, 2) + 1
        ) / 4
        idf_dict[item] = math.log(sentence_count/sum, 10) * x
    with open('idfs_dict.bin', 'wb') as f:
        pickle.dump(idf_dict, f)


def predict():
    import transform_to_traindata
    import xgboost as xgb
    import re
    model = xgb.Booster()
    model.load_model('train_result')
    sql = "select id, content_words, content_bq from predictions"
    Clean.cursor.execute(sql)
    result = Clean.cursor.fetchall()

    for r in result:
        content = r[2] + r[1].strip()
        content = re.sub(re.compile(r'//:.+'), '', content)
        words = Clean.seg(content)
        if len(words) == 0:
            continue
        vector = transform_to_traindata.words_to_vector(words)
        if vector is not None:
            y_pred = round(model.predict(xgb.DMatrix(vector.reshape(1, 300)))[0])
            sql2 = "update predictions set multiclass=%d, emotion='%s' where id=%d" % (y_pred, int_emo[y_pred], r[0])
            try:
                Clean.cursor.execute(sql2)
                Clean.db.commit()
            except:
                Clean.db.rollback()


def sadness():
    count = 1
    sql = "select content from train_sentence where emotion_int_multi=1 order by length(content) desc limit 3300"
    Clean.cursor.execute(sql)
    li = Clean.cursor.fetchall()

    for i in li:
        content = i[0]
        sql = "insert into train_sentence values(23002, %s, %s, 'Y', 'sadness', 'none', 2)"
        Clean.cursor.execute(sql, [count, content])
        count += 1
    try:
        Clean.db.commit()
    except:
        Clean.db.rollback()


def sadness2():
    import pandas
    import transform_to_traindata
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model('train_result')
    data = pandas.read_csv(r'C:\Users\65113\Desktop\nCov_100k_train.labled.csv',
                           usecols=['微博中文内容']).values
    all_sentence = []
    for i in data:
        if isinstance(i[0], float):
            continue
        words = Clean.seg(Clean.clean_pattern(i[0]))
        if len(words) != 0:
            vector = transform_to_traindata.words_to_vector(words)
            if vector is not None:
                y_pred = round(model.predict(xgb.DMatrix(vector.reshape(1, 300)))[0])
                if y_pred == 1:
                    all_sentence.append(i[0])
    with open(r'C:\Users\65113\Desktop\sadness.txt', 'w', encoding='utf-8') as f:
        for i in all_sentence:
            f.writelines(i + '\n')


def sadness3():
    import traceback
    with open(r'C:\Users\65113\Desktop\sadness.txt', 'r', encoding='utf-8') as f:
        sql = "select sentence_id from train_sentence where weibo_id=23001 ORDER BY sentence_id desc limit 1"
        Clean.cursor.execute(sql)
        count = Clean.cursor.fetchall()[0][0] + 1
        sql = "insert into train_sentence values(23001, %s, %s, 'Y', 'sadness', 'none', 1)"
        for i in f.readlines():
            try:
                Clean.cursor.execute(sql, [count, i.strip('\n')])
                Clean.db.commit()
                count += 1
            except:
                Clean.db.rollback()
                traceback.print_exc()


if __name__ == '__main__':
    sentence()
    # generate_idfs()
    # predict()
    # sadness()
    # to_file()
