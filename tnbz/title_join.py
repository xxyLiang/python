import pymysql
import traceback

db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
cursor = db.cursor()

cursor.execute("select pid, tid, content from posts where is_initiate_post=1")
result = cursor.fetchall()

for r in result:
    try:
        cursor.execute("select title from threads where tid=%s", r[1])
        title = cursor.fetchall()[0][0]
        new_content = "【%s】 %s" % (title, r[2])
        cursor.execute("update posts set content=%s where pid=%s", (new_content, r[0]))
        db.commit()
    except:
        db.rollback()
        traceback.print_exc()
