from xml.etree import ElementTree
import pymysql
import traceback


def train2013():
    try:
        tree = ElementTree.parse("./train_data/2013train/data.xml")
        db = pymysql.connect("localhost", "root", "admin", "weibo", charset='utf8mb4')
        cursor = db.cursor()
        count = cursor.execute("select * from train_weibo") + 1
    except:
        traceback.print_exc()

    weibo = tree.findall('weibo')
    for i in weibo:
        emotion_type = i.attrib['emotion-type']
        document_id = int(i.attrib['id'])
        try:
            sql = "insert into train_weibo values(%s,%s,%s,%s)"
            cursor.execute(sql, [count, emotion_type, '2013train', document_id])
            db.commit()
        except:
            db.rollback()
            with open('log.txt', 'a') as f:
                f.writelines('dataset 2013train id=%d error\n' % document_id)
            continue

        sentence_list = i.findall('sentence')
        for s in sentence_list:
            id = int(s.attrib['id'])
            emotion_tag = s.attrib['emotion_tag']
            if emotion_tag == 'Y':
                emotion_type1 = s.attrib['emotion-1-type']
                emotion_type2 = s.attrib['emotion-2-type']
            else:
                emotion_type1 = 'none'
                emotion_type2 = 'none'
            try:
                sql = "insert into train_sentence values(%s,%s,%s,%s,%s,%s)"
                cursor.execute(sql, [count, id, s.text, emotion_tag, emotion_type1, emotion_type2])
                db.commit()
            except:
                db.rollback()
                traceback.print_exc()
        count += 1


def test2013():
    trans_dict = {"": "none", "D": "N", "无": "none", '悲伤': "sadness",
                  "喜好": "like", "愤怒": "anger", "高兴": "happiness",
                  "厌恶": "disgust", "恐惧": "fear", "惊讶": "surprise"}
    try:
        tree = ElementTree.parse("./train_data/2013test/data.xml")
        db = pymysql.connect("localhost", "root", "admin", "weibo", charset='utf8mb4')
        cursor = db.cursor()
        count = cursor.execute("select * from train_weibo") + 1
    except:
        traceback.print_exc()

    weibo = tree.findall('weibo')
    for i in weibo:
        if "emotion-type" in i.attrib.keys():
            emotion_type = i.attrib['emotion-type']
        else:
            emotion_type = i.attrib['emotion-type1']
        document_id = int(i.attrib['id'])
        try:
            sql = "insert into train_weibo values(%s,%s,%s,%s)"
            cursor.execute(sql, [count, trans_dict[emotion_type], '2013test', document_id])
            db.commit()
        except:
            db.rollback()
            with open('log.txt', 'a') as f:
                f.writelines('dataset 2013test id=%d error\n' % document_id)
            continue

        sentence_list = i.findall('sentence')
        for s in sentence_list:
            id = int(s.attrib['id'])
            emotion_tag = s.attrib['opinionated']
            if emotion_tag == 'Y':
                emotion_type1 = trans_dict[s.attrib['emotion-1-type']]
                emotion_type2 = trans_dict[s.attrib['emotion-2-type']]
            else:
                emotion_type1 = 'none'
                emotion_type2 = 'none'
            try:
                sql = "insert into train_sentence values(%s,%s,%s,%s,%s,%s)"
                cursor.execute(sql, [count, id, s.text, emotion_tag, emotion_type1, emotion_type2])
                db.commit()
            except:
                db.rollback()
                traceback.print_exc()
        count += 1


def train2014():
    try:
        tree = ElementTree.parse("./train_data/2014train/data.xml")
        db = pymysql.connect("localhost", "root", "admin", "weibo", charset='utf8mb4')
        cursor = db.cursor()
        count = cursor.execute("select * from train_weibo") + 1
    except:
        traceback.print_exc()

    weibo = tree.findall('weibo')
    for i in weibo:
        emotion_type = i.attrib['emotion-type1']
        document_id = int(i.attrib['id'])
        try:
            sql = "insert into train_weibo values(%s,%s,%s,%s)"
            cursor.execute(sql, [count, emotion_type, '2014train', document_id])
            db.commit()
        except:
            db.rollback()
            with open('log.txt', 'a') as f:
                f.writelines('dataset 2014train id=%d error\n' % document_id)
            continue

        sentence_list = i.findall('sentence')
        for s in sentence_list:
            id = int(s.attrib['id'])
            emotion_tag = s.attrib['opinionated']
            if emotion_tag == 'Y':
                emotion_type1 = s.attrib['emotion-1-type']
                emotion_type2 = s.attrib['emotion-2-type']
            else:
                emotion_type1 = 'none'
                emotion_type2 = 'none'
            try:
                sql = "insert into train_sentence values(%s,%s,%s,%s,%s,%s)"
                cursor.execute(sql, [count, id, s.text, emotion_tag, emotion_type1, emotion_type2])
                db.commit()
            except:
                db.rollback()
                traceback.print_exc()
        count += 1


def test2014_1():   # EmotionClassficationTest
    try:
        tree = ElementTree.parse("./train_data/2014test/data1.xml")
        db = pymysql.connect("localhost", "root", "admin", "weibo", charset='utf8mb4')
        cursor = db.cursor()
        count = cursor.execute("select * from train_weibo") + 1
    except:
        traceback.print_exc()

    weibo = tree.findall('weibo')
    for i in weibo:
        emotion_type = i.attrib['emotion-type1']
        document_id = int(i.attrib['id'])
        try:
            sql = "insert into train_weibo values(%s,%s,%s,%s)"
            cursor.execute(sql, [count, emotion_type, '2014test_1', document_id])
            db.commit()
        except:
            db.rollback()
            with open('log.txt', 'a') as f:
                f.writelines('dataset 2014test_1 id=%d error\n' % document_id)
            continue

        sentence_list = i.findall('sentence')
        for s in sentence_list:
            id = int(s.attrib['id'])
            emotion_tag = s.attrib['opinionated']
            if emotion_tag == 'Y':
                emotion_type1 = s.attrib['emotion-1-type']
                emotion_type2 = s.attrib['emotion-2-type']
            else:
                emotion_type1 = 'none'
                emotion_type2 = 'none'
            try:
                sql = "insert into train_sentence values(%s,%s,%s,%s,%s,%s)"
                cursor.execute(sql, [count, id, s.text, emotion_tag, emotion_type1, emotion_type2])
                db.commit()
            except:
                db.rollback()
                traceback.print_exc()
        count += 1


def test2014_2():   # ExpressionTest
    try:
        tree = ElementTree.parse("./train_data/2014test/data2.xml")
        db = pymysql.connect("localhost", "root", "admin", "weibo", charset='utf8mb4')
        cursor = db.cursor()
        count = cursor.execute("select * from train_weibo") + 1
    except:
        traceback.print_exc()

    weibo = tree.findall('weibo')
    for i in weibo:
        emotion_type = i.attrib['emotion-type1']
        document_id = int(i.attrib['id'])
        try:
            sql = "insert into train_weibo values(%s,%s,%s,%s)"
            cursor.execute(sql, [count, emotion_type, '2014test_2', document_id])
            db.commit()
        except:
            db.rollback()
            with open('log.txt', 'a') as f:
                f.writelines('dataset 2014test_2 id=%d error\n' % document_id)
            continue

        sentence_list = i.findall('sentence')
        for s in sentence_list:
            id = int(s.attrib['id'])
            emotion_tag = s.attrib['opinionated']
            if emotion_tag == 'Y':
                emotion_type1 = s.attrib['emotion-1-type']
                emotion_type2 = s.attrib['emotion-2-type']
            else:
                emotion_type1 = 'none'
                emotion_type2 = 'none'
            try:
                sql = "insert into train_sentence values(%s,%s,%s,%s,%s,%s)"
                cursor.execute(sql, [count, id, s.text, emotion_tag, emotion_type1, emotion_type2])
                db.commit()
            except:
                db.rollback()
                traceback.print_exc()
        count += 1


def NLPIR():
    try:
        tree = ElementTree.parse("D:/weibo_content_corpus/data.xml")
        db = pymysql.connect("localhost", "root", "admin", "test", charset='utf8mb4')
        cursor = db.cursor()
    except:
        traceback.print_exc()

    record = tree.findall('RECORD')
    for r in record:
        id = int(r.find('id'))
        content = r.find('article')
        print('%d: %s' % (id, content))


if __name__ == '__main__':
    train2013()
    test2013()
    train2014()
    test2014_1()
    test2014_2()