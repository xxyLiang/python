import pymysql
import traceback

db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
cursor = db.cursor()
sql = "update `threads` set will_crawl = 1 where author_id=%s and (TO_DAYS(publish_date) - TO_DAYS(%s) <= 30)"


try:
    cursor.execute("select author_id, publish_date from threads where new_user=1")
    ts = cursor.fetchall()
except:
    pass
for t in ts:
    try:
        cursor.execute(sql, (t[0], t[1]))
        db.commit()
    except:
        traceback.print_exc()
        db.rollback()
