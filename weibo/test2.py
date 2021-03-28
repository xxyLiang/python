import pymysql
from pyecharts.charts import Map
from pyecharts import options as opts
import jieba
import re
import pandas
import math

db = pymysql.connect("localhost", "root", "admin", "weibo", charset='utf8mb4')
cursor = db.cursor()

sql = "select * from julei where multiclass is not null"
cursor.execute(sql)
result = cursor.fetchall()

li = [['gender', 'following', 'follower', 'weibo', 'location1', 'location2', 'location3', 'location4',
       'age', 'register', 'intro', 'tag', 'education', 'pred']]
l_1 = ['北京', '上海', '广州', '深圳']
l_2 = ['成都', '杭州', '重庆', '天津', '南京', '长沙', '郑州', '东莞', '青岛', '沈阳', '宁波', '昆明']
for r in result:
    gender = r[1]   # 女1男2
    # if r[2] < 150:
    #     following = 1
    # elif r[2] < 350:
    #     following = 2
    # else:
    #     following = 3
    #
    # if r[3] < 80:
    #     follower = 1
    # elif r[3] < 250:
    #     follower = 2
    # else:
    #     follower = 3
    #
    # if r[4] < 100:
    #     weibo = 1
    # elif r[4] < 800:
    #     weibo = 2
    # else:
    #     weibo = 3
    following = math.log10(r[2] + 1)
    follower = math.log10(r[3] + 1)
    weibo = math.log10(r[4] + 1)

    # True=1 False=2
    location1 = 0   # 海外
    location2 = 0   # 一线城市
    location3 = 0   # 新一线城市
    location4 = 0   # 其他城市
    a = r[5].split()
    for l in a:
        if l in l_1:
            location2 = 1
            break
        if l in l_2:
            location3 = 1
            break
    if location2 == 0 and location3 == 0:
        if '海外' in a[0]:
            location1 = 1
        elif '其他' in a[0]:
            pass
        else:
            location4 = 1

    age = r[6]
    regis = 2020 - int(re.match(r'\d{4}', str(r[7])).group())
    if regis > 6:
        register = 3
    elif regis > 3:
        register = 2
    else:
        register = 1
    intro = 1 if r[8] is not None else 0
    tag_num = len(r[9].split('/')) if r[9] is not None else 0
    if tag_num > 7:
        tag = 3
    elif tag_num > 3:
        tag = 2
    else:
        tag = 1
    education = 1 if r[10] is not None else 0
    if r[-2] > 0.7:
        pred = 1
    elif r[-2] < 0.3:
        pred = 0
    else:
        pred = -1

    li.append([gender, following, follower, weibo, location1, location2, location3, location4,
               age, register, intro, tag, education, pred])

data = pandas.DataFrame(li)
data.to_csv("C:/Users/65113/Desktop/julei2.csv", header=False, index=False)
