import pymysql
import re


class dbase:

    def __init__(self):
        self.db = pymysql.connect("localhost", "root", "admin", "weibo", charset='utf8mb4')
        self.cursor = self.db.cursor()

    def func1(self):
        sql = "select pred, multiclass, cid from predictions where cid in (select cid from julei)"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        sql = "update julei set emotion_bin=%s, emotion_multi=%s where cid=%s "
        for i in result:
            try:
                self.cursor.execute(sql, i)
                self.db.commit()
            except:
                self.db.rollback()

    # julei 和 user_info 结合
    def func2(self):
        sql = "select a.cid, b.gender, b.doctor, b.location, b.following, b.follower, b.weibo, b.intro, b.birthday, b.tag, b.job, b.university from " \
              "(select cid, user_page from comment_info where cid in (select cid from julei)) a " \
              "inner join user_info b " \
              "on a.user_page=b.user_page"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        sql = "update julei set gender=%s, doctor=%s, following=%s, follower=%s, " \
              "weibo=%s, location=%s, intro=%s, age=%s, tag=%s, job=%s, education=%s " \
              "where cid = %s"
        for r in result:
            gender = r[1]
            doctor = r[2]
            # 其他0 海外1 一线2 新一线3 二线4
            l_1 = ['北京', '上海', '广州', '深圳']
            l_2 = ['成都', '杭州', '重庆', '天津', '南京', '长沙', '郑州', '东莞', '青岛', '沈阳', '宁波', '昆明']
            location = -1
            a = r[3].split()
            for l in a:
                if l in l_1:
                    location = 2
                    break
                if l in l_2:
                    location = 3
                    break
            if location == -1:
                if '海外' in a[0]:
                    location = 1
                elif '其他' in a[0]:
                    location = 0
                else:
                    location = 4

            if r[4] < 150:
                following = 1
            elif r[4] < 350:
                following = 2
            else:
                following = 3

            if r[5] < 80:
                follower = 1
            elif r[5] < 250:
                follower = 2
            elif r[5] < 1000:
                follower = 3
            else:
                follower = 4

            if r[6] < 100:
                weibo = 1
            elif r[6] < 800:
                weibo = 2
            else:
                weibo = 3

            intro = 0 if r[7] is None else 1

            if r[8] is None:
                age = None
            else:
                s = re.search(r'(\d{4})年', r[8])
                age = 2020 - int(s.group(1)) if s is not None else None
                if age is not None and (age > 70 or age < 10):
                    age = None
                if age is not None:
                    if age < 20:
                        age = 1
                    elif age < 30:
                        age = 2
                    else:
                        age = 3

            tag = 0 if r[9] is None else len(r[9].split('/'))
            job = 0 if r[10] is None else 1
            education = 0 if r[11] is None else 1

            item = (gender, doctor, following, follower, weibo, location, intro, age, tag, job, education, r[0])

            self.cursor.execute(sql, item)
        try:
            self.db.commit()
        except:
            self.db.rollback()

    def func3(self):
        import jieba
        import gensim
        import numpy
        import pandas
        from sklearn.metrics.pairwise import cosine_similarity as cs
        model = gensim.models.KeyedVectors.load_word2vec_format('D:/vector_stopwords.bin')
        vocab = model.vocab.keys()
        jieba.initialize()
        jieba.load_userdict('./wv_material/user_dict.txt')
        sql = "select tag from user_info where tag is not null"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        tag_dict = {}
        for r in result:
            tags = r[0].split('/')
            for t in tags:
                count = 0
                for w in jieba.cut(t):
                    count += 1
                if count <= 2:
                    if t in tag_dict:
                        tag_dict[t] += 1
                    else:
                        tag_dict[t] = 1
        pop_list = []
        for t in tag_dict:
            if tag_dict[t] <= 3:
                pop_list.append(t)
        for p in pop_list:
            tag_dict.pop(p)

        keyword = ['时尚娱乐', '影视音乐', '游戏动漫', '文化艺术', '科学科技数码',
                   '体育健身', '美食旅游购物', '军事政治', '新闻时事', '生活个性']
        keyword_vector = []
        for k in keyword:
            vector = numpy.zeros(300)
            count = 0
            for w in jieba.cut(k):
                if w in vocab:
                    vector += model[w]
                    count += 1
            vector /= count
            keyword_vector.append(vector.reshape(1, -1))

        all_tags = []
        for t in tag_dict:
            count = 0
            vector = numpy.zeros(300)
            for w in jieba.cut(t):
                if w in vocab:
                    vector += model[w]
                    count += 1
            if count == 0:
                continue
            vector /= count
            vector = vector.reshape(1, -1)

            similarity = 0
            index = 0
            for i in range(10):
                s = cs(keyword_vector[i], vector)[0][0]
                if s > similarity:
                    similarity = s
                    index = i
            all_tags.append((t, tag_dict[t], keyword[index], similarity))
        data = pandas.DataFrame(all_tags)
        data.to_csv(r'C:\Users\65113\Desktop\tag2.csv', header=False, index=False)

    def func4(self):
        import numpy
        import traceback
        topic_dict = {'时尚娱乐购物美食旅游': 0, '新闻时事': 1, '军事政治经济': 2, '影视音乐': 3,
                      '科学科技数码': 4, '文化艺术': 5, '体育健身': 6, '游戏动漫': 7,
                      '职业': 8, '生活个性': 9}
        keyword_dict = {}
        sql = "select keyword, topic from tag"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        for r in result:
            keyword_dict[r[0]] = topic_dict[r[1].strip()]

        sql = "select a.cid, b.tag, b.gender from " \
              "(select cid, user_page from comment_info where cid in (select cid from test)) a " \
              "INNER JOIN " \
              "(select user_page, tag, gender from user_info where tag is not null) b " \
              "on a.user_page=b.user_page"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        sql = "update test set tag1=%s, tag2=%s, tag3=%s, tag4=%s, tag5=%s, " \
              "tag6=%s, tag7=%s, tag8=%s, tag9=%s, tag10=%s, gender=%s where cid=%s"
        for r in result:
            tags = r[1].split('/')
            tag_count = 0
            arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, r[2], r[0]]
            for t in tags:
                if t in keyword_dict:
                    arr[keyword_dict[t]] += 1
                    tag_count += 1
            if tag_count != 0:
                try:
                    self.cursor.execute(sql, arr)
                    self.db.commit()
                except:
                    self.db.rollback()
                    traceback.print_exc()

    def func5(self):
        from numpy import loadtxt
        from xgboost import XGBClassifier
        from xgboost import plot_importance
        from matplotlib import pyplot

        dataset = loadtxt(r'C:\Users\65113\Desktop\test.csv', delimiter=',')
        X = dataset[:, 0:-1]
        Y = dataset[:, -1]

        model = XGBClassifier()
        model.fit(X, Y)

        plot_importance(model)
        pyplot.show()

    地图
    def func6(self):
        sql = "select location from user_info where user_page in " \
              "(select user_page from comment_info where k_event=2)"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        from pyecharts.charts import Map
        from pyecharts import options as opts
        dict = {}
        for r in result:
            location = r[0].split()[0]
            if location not in ['海外', '其他']:
                if location in dict:
                    dict[location] += 1
                else:
                    dict[location] = 1
        province = list(dict.keys())
        values = list(dict.values())

        map = Map()
        map.add("地图", [list(z) for z in zip(province, values)], "china")
        map.set_global_opts(visualmap_opts=opts.VisualMapOpts())
        map.render(path="C:/Users/65113/Desktop/map2.html")

    # 时间序列
    def func7(self):
        import math
        sql = "select a.dat, b.pred, a.likes from (select * from comment_info where k_event=2 and dat<'2020-01-03') a " \
              "inner join predictions b on a.cid=b.cid "
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        times = {}
        w = {}
        count = {}
        for r in result:
            t = re.match(r'\d{4}-\d+-\d+ \d+', str(r[0])).group()
            weight = math.log2(r[2] + 2)
            if t in times:
                times[t] += r[1] j* weight
                w[t] += weight
                count[t] += 1
            else:
                times[t] = r[1] * weight
                w[t] = weight
                count[t] = 1

        t = 0
        ww = 0
        count2 = 0
        flag = 0
        with open("C:/Users/65113/Desktop/minhang.csv", 'w') as f:
            for item in times:
                if re.match(r'\d{4}-\d+-\d+ \d[02468]', item) is not None and flag == 0:
                    t = times[item]
                    ww = w[item]
                    count2 = count[item]
                    flag = 1
                else:
                    times[item] = (times[item] + t) / (w[item] + ww)
                    f.writelines('%s:00:00,%f,%d\n' % (item, times[item], count[item]+count2))
                    t = 0
                    ww = 0
                    count2 = 0
                    flag = 0

    # wordcloud
    def func8(self):
        import jieba
        jieba.initialize()
        jieba.load_userdict('./wv_material/user_dict.txt')
        with open('./wv_material/stopwords.dat', 'r', encoding='utf-8') as f:
            stopwords = [i.strip('\n') for i in f.readlines()]
        sql = "select content_words from comment_info where doctor=0 and k_event<3"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()

        word_dict = {}
        for r in result:
            content = r[0].strip()
            content = re.sub(re.compile(r'//:.+'), '', content)
            words = jieba.cut(content)
            for w in words:
                if w not in stopwords:
                    if w in word_dict:
                        word_dict[w] += 1
                    else:
                        word_dict[w] = 1

        with open("C:/Users/65113/Desktop/ndoctor.csv", 'w', encoding='utf-8') as f:
            for w in word_dict:
                if word_dict[w] > 350:
                    f.writelines("%s,%d\n" % (w, word_dict[w]))


    def func9(self):
        import pyecharts.options as opts
        from pyecharts.charts import WordCloud

        data = []
        with open("C:/Users/65113/Desktop/ndoctor.csv", 'r', encoding='utf-8') as f:
            for i in f.readlines():
                data.append(i.strip('\n').split(','))

        w = WordCloud()
        w.add("词云", data_pair=data, word_size_range=[24, 66])
        w.set_global_opts(tooltip_opts=opts.TooltipOpts(is_show=True))
        w.render(path="C:/Users/65113/Desktop/ndoctor.html")


if __name__ == '__main__':
    a = dbase()
    a.func8()
    a.func9()