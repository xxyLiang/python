import pymysql
import os
import jieba

db = pymysql.connect("localhost", "root", "admin", "weibo", charset='utf8mb4')
cursor = db.cursor()
sql = "select content_words from comment_info limit 10,20"
cursor.execute(sql)
content = cursor.fetchall()