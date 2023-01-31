import pymysql
import pandas as pd
import pickle
import math

db = pymysql.connect(host='localhost',
                     user='root',
                     password='651133439a',
                     database='rec_sys')
cursor = db.cursor()

THREAD_CNT_LOW = 5
THREAD_CNT_HIGH = 10000


class FileMaker:

    def __init__(self):
        cursor.execute("with "
                       "t1 as (SELECT uid FROM user_list WHERE thread_cnt BETWEEN %d AND %d),"
                       "t2 as (SELECT distinct tid from posts p "
                       "where publish_time BETWEEN '2018-01-01' AND '2018-12-31' "
                       "AND EXISTS (select 1 from t1 where p.author_id=t1.uid)), "
                       "t3 as (SELECT tid from threads where forum=1) "
                       "SELECT tid, `rank`, content, author_id, publish_time from posts p2 "
                       "where EXISTS (select 1 from t1, t2, t3 where p2.tid = t2.tid and p2.tid=t3.tid and p2.author_id=t1.uid) "
                       "AND publish_time BETWEEN '2018-01-01' AND '2018-12-31' order by 5 asc" %
                       (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        self.data = pd.DataFrame(cursor.fetchall(), columns=['tid', 'rank', 'content', 'uid', 'publish_time'])

    def make_train_test_file(self):
        id2user, id2thread, user2id, thread2id = self.make_id_transfer()

        train_dict = dict()
        test_dict = dict()

        user_group = self.data[['uid', 'tid']].groupby('uid')
        for uid, df in user_group:
            thread = df.drop_duplicates(keep='first')['tid']
            test_thread_cnt = math.ceil(len(thread) / 10)

            train_thread_id = thread.iloc[:-test_thread_cnt]
            train_dict[user2id[uid]] = train_thread_id.to_list()

            test_thread_id = thread.iloc[-test_thread_cnt:]
            test_dict[user2id[uid]] = test_thread_id.to_list()

        with open('./data/Train_pairs.txt', 'w') as f:
            for k, v in train_dict.items():
                f.writelines("%d %s\n" % (k, " ".join(v).strip()))

        with open('./data/Test_pairs.txt', 'w') as f:
            for k, v in test_dict.items():
                f.writelines("%d %s\n" % (k, " ".join(v).strip()))

    def make_id_transfer(self):
        users = self.data['uid'].drop_duplicates()
        threads = self.data['tid'].drop_duplicates()

        users.reset_index(drop=True, inplace=True)
        threads.reset_index(drop=True, inplace=True)

        id2user = users.to_dict()
        id2thread = threads.to_dict()

        user2id = dict(zip(id2user.values(), id2user.keys()))
        thread2id = dict(zip(id2thread.values(), id2thread.keys()))

        with open('./data/id2user.pickle', 'wb') as f:
            pickle.dump(id2user, f)
        with open('./data/id2thread.pickle', 'wb') as f:
            pickle.dump(id2thread, f)
        with open('./data/user2id.pickle', 'wb') as f:
            pickle.dump(user2id, f)
        with open('./data/thread2id.pickle', 'wb') as f:
            pickle.dump(thread2id, f)

        return id2user, id2thread, user2id, thread2id

