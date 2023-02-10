import pymysql
import pandas as pd
import numpy as np
import pickle
import math
import os
import sys
from bert_serving.client import BertClient
from sklearn.decomposition import PCA
from tqdm import tqdm
sys.path.append('..')
from lda import LDA, N_TOPICS


HISTORY_THREAD_TAKEN_CNT = 10
PCA_DIM = 50
THREAD_CNT_LOW = 5
THREAD_CNT_HIGH = 10000

db = pymysql.connect(host='localhost',
                     user='root',
                     password='651133439a',
                     database='rec_sys')
cursor = db.cursor()


if os.name == 'nt':
    prefix = 'C:/Users/65113/Desktop/Recsys_data/'
elif os.name == 'posix':
    prefix = '/Users/liangxiyi/Files/Recsys_data'
filetype = '.pickle'


def read_data(filename):
    address = prefix + filename + '.pickle'
    with open(address, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data(data, filename):
    address = prefix + filename + '.pickle'
    with open(address, 'wb') as f:
        pickle.dump(data, f)
    return


def check_data(filename):
    address = prefix + filename + '.pickle'
    return os.path.exists(address)


class FileMaker:

    def __init__(self):

        cursor.execute("with "
                       "t1 as (SELECT uid FROM user_list WHERE thread_cnt BETWEEN %d AND %d),"
                       "t2 as (SELECT distinct tid from posts p "
                       "where publish_time BETWEEN '2018-01-01' AND '2018-12-31' "
                       "AND EXISTS (select 1 from t1 where p.author_id=t1.uid)) "
                       "SELECT tid, `rank`, content, author_id, publish_time from posts p2 "
                       "where EXISTS (select 1 from t1, t2 where p2.tid = t2.tid and p2.author_id=t1.uid) "
                       "AND publish_time BETWEEN '2018-01-01' AND '2018-12-31' order by 5 asc" %
                       (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        self.data = pd.DataFrame(cursor.fetchall(), columns=['tid', 'rank', 'content', 'uid', 'publish_time'])

        self.thread_bert = read_data('thread_bert') if check_data('thread_bert') else self.make_thread_LDA_file()
        self.thread_lda = read_data('thread_lda') if check_data('thread_lda') else self.make_thread_LDA_file()
        self.user_seq = read_data('user_sequence') if check_data('user_sequence') else self.make_user_sequence()
        self.id2uid, self.uid2id = read_data('uid_transfer') if check_data('uid_transfer') else self.make_id_transfer()

    def make_user_sequence(self):
        """
        pattern of User_seq：
        {
          user1: [ [tid_1, tid_2, ..., tid_n], [tid_1_time, tid_2_time, ..., tid_n_time] ],
          user2: [ [ thread_list ], [ time_list ] ]
        }
        """
        id2user, user2id = self.make_id_transfer()
        user_seq = dict()

        user_group = self.data[['uid', 'tid', 'publish_time']].groupby('uid')
        for uid, df in user_group:
            thread = df.drop_duplicates(subset=['tid'], keep='first')
            t_list = thread['tid'].to_list()
            time_list = thread['publish_time'].to_list()

            user_seq[user2id[uid]] = [t_list, time_list]

        save_data(user_seq, 'user_sequence')

        return user_seq

    def make_id_transfer(self):
        users = self.data['uid'].drop_duplicates()
        users.reset_index(drop=True, inplace=True)
        id2uid = users.to_dict()
        uid2id = dict(zip(id2uid.values(), id2uid.keys()))

        save_data((id2uid, uid2id), 'uid_transfer')

        return id2uid, uid2id

    def make_user_info_file(self):
        pass

    @staticmethod
    def make_thread_LDA_file():
        lda = LDA()
        cursor.execute("with "
                       "t1 as (SELECT uid FROM user_list WHERE thread_cnt BETWEEN %d AND %d),"
                       "t2 as (SELECT distinct tid from posts p "
                       "where publish_time BETWEEN '2018-01-01' AND '2018-12-31' "
                       "AND EXISTS (select 1 from t1 where p.author_id=t1.uid)) "
                       "SELECT p2.tid, threads.title, p2.content from posts p2 "
                       "LEFT JOIN threads ON p2.tid = threads.tid "
                       "WHERE EXISTS (select 1 from t2 where p2.tid=t2.tid) and `rank`=1" %
                       (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        data = pd.DataFrame(cursor.fetchall(), columns=['tid', 'title', 'content'])
        threads = data['tid'].drop_duplicates()

        features = pd.DataFrame(
            data=np.zeros((len(threads), N_TOPICS)),
            dtype=float,
            index=threads
        )

        for _, row in data.iterrows():
            f = pd.Series(lda.lda_feature(row.title + ' ' + row.content))
            features.loc[row.tid] = f

        save_data(features, 'thread_lda')

        return features

    @staticmethod
    def make_thread_BERT_file(dim=PCA_DIM):
        print("start BERT embedding initiating...")
        bc = BertClient(check_length=False)
        cursor.execute("with "
                       "t1 as (SELECT uid FROM user_list WHERE thread_cnt BETWEEN %d AND %d),"
                       "t2 as (SELECT distinct tid from posts p "
                       "where publish_time BETWEEN '2018-01-01' AND '2018-12-31' "
                       "AND EXISTS (select 1 from t1 where p.author_id=t1.uid)) "
                       "SELECT p2.tid, threads.title, p2.content from posts p2 "
                       "LEFT JOIN threads ON p2.tid = threads.tid "
                       "WHERE EXISTS (select 1 from t2 where p2.tid=t2.tid) and `rank`=1" %
                       (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        data = pd.DataFrame(cursor.fetchall(), columns=['tid', 'title', 'content'])
        threads = data['tid'].drop_duplicates()
        t_cnt = len(threads)

        features = pd.DataFrame(
            data=np.zeros((t_cnt, 768)),
            dtype=float,
            index=threads
        )

        pbar = tqdm(total=len(threads))
        batchSize = 64
        for i in range(0, len(threads), batchSize):
            end = min(i+batchSize, t_cnt)
            df = data.iloc[i: end]
            content = df['title'] + " " + df['content']
            content = content.to_list()
            f = pd.DataFrame(bc.encode(content))
            features.iloc[i: end] = f
            pbar.update(end-i)
        pbar.close()

        print("lowering dimension with PCA...")
        if not os.path.exists(prefix + 'pca.pickle'):
            pca = PCA(n_components=dim)
            pca_feature = pca.fit_transform(features)
        else:
            pca = read_data('pca')
            pca_feature = pca.transform(features)
        pca_feature = pd.DataFrame(pca_feature, index=threads)
        save_data(pca, 'pca')
        save_data(pca_feature, 'thread_bert')

        return pca_feature

    def user_history_feature(self, user_seq, idx):

        t_list, time_list = user_seq

        item = {
            'lda': np.zeros((HISTORY_THREAD_TAKEN_CNT, N_TOPICS)),
            'bert': np.zeros((HISTORY_THREAD_TAKEN_CNT, PCA_DIM)),
            'timeDelta': np.array([0] * HISTORY_THREAD_TAKEN_CNT)
        }

        if idx == 0:
            return item
        elif idx >= HISTORY_THREAD_TAKEN_CNT:
            start = idx - HISTORY_THREAD_TAKEN_CNT
        else:
            start = 0

        item['lda'] = self.thread_lda.loc[t_list[start:idx]].to_numpy()
        item['bert'] = self.thread_bert.loc[t_list[start:idx]].to_numpy()
        item['timeDelta'] = np.array([(time_list[idx] - time_list[i]).total_seconds() / 2592000 for i in range(start, idx)])

        if item['lda'].shape[0] < HISTORY_THREAD_TAKEN_CNT:
            item['lda'] = np.concatenate((item['lda'],
                                          np.zeros((HISTORY_THREAD_TAKEN_CNT - item['lda'].shape[0], N_TOPICS))))
            item['bert'] = np.concatenate((item['bert'],
                                           np.zeros((HISTORY_THREAD_TAKEN_CNT - item['bert'].shape[0], PCA_DIM))))
            item['timeDelta'] = np.concatenate((item['timeDelta'],
                                                np.zeros(HISTORY_THREAD_TAKEN_CNT - len(item['timeDelta']))))

        return item

    def make_train_test_file(self):
        print('Generating train and test file...')
        train_list = list()
        test_list = list()

        pbar = tqdm(total=len(self.user_seq.keys()))
        for user, seq in self.user_seq.items():
            t_cnt = len(seq[0])
            train_cnt = t_cnt - math.ceil(t_cnt / 10)

            for i in range(t_cnt):
                item = dict()
                item['user'] = np.array(user).astype(np.int64)
                item['tid'] = seq[0][i]
                item['item_lda'] = self.thread_lda.loc[seq[0][i]].to_numpy()
                item['item_bert'] = self.thread_bert.loc[seq[0][i]].to_numpy()
                f = self.user_history_feature(seq, i)
                item['hist_lda'] = f['lda']
                item['hist_bert'] = f['bert']
                item['timeDelta'] = f['timeDelta']

                # more features here

                if i < train_cnt:
                    train_list.append(item)
                else:
                    test_list.append(item)
            pbar.update(1)
        pbar.close()

        save_data(train_list, 'train_data')
        save_data(test_list, 'test_data')

        return


if __name__ == '__main__':
    fileMaker = FileMaker()
    fileMaker.make_train_test_file()

