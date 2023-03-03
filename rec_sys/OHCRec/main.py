import pymysql
import numpy as np
import pandas as pd
import time
from lda import LDA, N_TOPICS
import os
import math
from tqdm import tqdm
import pickle

MAX = float('inf')
THREAD_CNT_LOW = 20
THREAD_CNT_HIGH = 100
DEBUG = False


def timeit(func):

    def wrapper(*args, **kwargs):
        print("Starting %s function..." % func.__name__)
        t = time.time()
        ret = func(*args, **kwargs)
        print("Function %s spend %.1f seconds." % (func.__name__, time.time()-t))
        return ret

    return wrapper


class OHCRec:
    def __init__(self, c=0.8, simRank_k=2):
        self.db = pymysql.connect(host='localhost',
                                  user='root',
                                  password='651133439a',
                                  database='rec_sys')
        self.cursor = self.db.cursor()
        self.uid2id = None
        self.tid2id = None
        self.c = c
        self.simRank_k = simRank_k
        self.data = None
        self.data_train = None
        self.y_test = None
        self.rec_thread_time_limit = None
        self.user_cnt = 0
        self.thread_cnt = 0
        self.user_list = None
        self.user_lda_feature = None
        self.threads = None
        self.thread_lda_feature = None
        self.Bn = None          # user behavior networks Matrix: n*n
        self.S = None           # User influence relationships Matrix: n*n
        self.P = None           # User Post Matrix: n*m
        self.R = None           # User Rating Matrix: n*m
        self.load_all_posts()

    def load_thread_rank1(self):
        self.cursor.execute("with "
                            "t1 as (SELECT uid FROM user_list WHERE thread_cnt BETWEEN %d AND %d),"
                            "t2 as (SELECT distinct tid from posts p "
                            "where publish_time BETWEEN '2018-01-01' AND '2018-12-31' "
                            "AND EXISTS (select 1 from t1 where p.author_id=t1.uid)) "
                            "SELECT p2.tid, threads.title, p2.content from posts p2 "
                            "LEFT JOIN threads ON p2.tid = threads.tid "
                            "WHERE EXISTS (select 1 from t2 where p2.tid=t2.tid) and `rank`=1" %
                            (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        data = pd.DataFrame(self.cursor.fetchall(), columns=['tid', 'title', 'content'])
        data['tid'] = data['tid'].apply(lambda x: self.tid2id[x])
        return data

    def load_all_posts(self, initiate=False):
        self.cursor.execute("with "
                            "t1 as (SELECT uid FROM user_list WHERE thread_cnt BETWEEN %d AND %d),"
                            "t2 as (SELECT distinct tid from posts p "
                            "where publish_time BETWEEN '2018-01-01' AND '2018-12-31' "
                            "AND EXISTS (select 1 from t1 where p.author_id=t1.uid)) "
                            "SELECT tid, `rank`, content, author_id, publish_time from posts p2 "
                            "where EXISTS (select 1 from t1, t2 where p2.tid = t2.tid and p2.author_id=t1.uid) "
                            "AND publish_time BETWEEN '2018-01-01' AND '2018-12-31' order by 5 asc" %
                            (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        data = pd.DataFrame(self.cursor.fetchall(), columns=['tid', 'rank', 'content', 'uid', 'publish_time'])

        # transfer id from char to nrange
        users = data['uid'].drop_duplicates()
        users.reset_index(drop=True, inplace=True)
        id2uid = users.to_dict()
        self.uid2id = dict(zip(id2uid.values(), id2uid.keys()))
        threads = data['tid'].drop_duplicates()
        threads.reset_index(drop=True, inplace=True)
        id2tid = threads.to_dict()
        self.tid2id = dict(zip(id2tid.values(), id2tid.keys()))

        data['tid'] = data['tid'].apply(lambda x: self.tid2id[x])
        data['uid'] = data['uid'].apply(lambda x: self.uid2id[x])
        data['test_flag'] = False
        test_data_idx = list()
        for uid, df in data.groupby('uid'):
            engage_thread = df['tid'].drop_duplicates()
            test_cnt = math.ceil(engage_thread.shape[0] / 10)
            test_data_idx.extend(df[df['tid'].isin(engage_thread.iloc[-test_cnt:])].index.to_list())
        data.loc[test_data_idx, 'test_flag'] = True
        self.data = data
        self.threads = self.data.tid.drop_duplicates()
        self.thread_cnt = self.threads.shape[0]

        user_attr = ['uid', 'thread_cnt', 'post_cnt', 'level', 'user_group', 'total_online_hours',
                     'regis_time', 'latest_login_time', 'latest_active_time', 'latest_pub_time',
                     'prestige', 'points', 'wealth', 'visitors', 'friends', 'records', 'logs',
                     'albums', 'total_posts', 'total_threads', 'shares', 'diabetes_type', 'treatment_type',
                     'gender', 'birthdate', 'habitation']
        self.user_cnt = self.cursor.execute('SELECT * FROM user_list WHERE thread_cnt BETWEEN %d AND %d' %
                                            (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        rs = self.cursor.fetchall()
        self.user_list = pd.DataFrame(rs, columns=user_attr)
        self.user_list['uid'] = self.user_list['uid'].apply(lambda x: self.uid2id[x])
        self.user_list.set_index('uid', inplace=True)
        self.user_list.sort_index(inplace=True)
        self.user_cnt = self.user_list.shape[0]
        print("Total user number: %d\nTotal thread number: %d" % (self.user_cnt, self.thread_cnt))
        return data

    # Implicit user behavior networks. 用户-用户n*n矩阵
    @timeit
    def IUBN(self):
        data = self.data_train if self.data_train is not None else self.data

        self.Bn = self.init_user_matrix()

        threads_users = data[['tid', 'uid']]

        # equation 1
        for _, df in threads_users.groupby('tid'):
            user_in_tid = df['uid'].drop_duplicates().to_list()
            NU_p = len(user_in_tid)
            for i in user_in_tid:
                self.Bn[i, user_in_tid] += 1 / NU_p

        self.Bn[np.diag_indices(self.Bn.shape[0])] = 0

        # equation 2
        self.Bn = np.divide(self.Bn, np.array(self.user_list['thread_cnt']).reshape(-1, 1))

        # equation 3
        self.Bn /= np.max(self.Bn)

        # 转化为对角为0，不连接结点为inf的矩阵
        self.Bn[np.where(self.Bn == 0)] = MAX     # 0转为inf
        self.Bn[np.diag_indices(self.Bn.shape[0])] = 0

    # User influence relationships
    def UIRs(self):
        ws = self.cal_WS()
        sr = self.cal_SR()
        ps = self.cal_PS()
        cs = self.cal_CS()
        us = 0.374 * sr + 0.229 * ps + 0.397 * cs
        self.S = ws * us

    def URM(self):
        """
        User Rating Matrix
        :return:
        """
        self.P = self.user_post_matrix()
        self.cal_thread_lda_feature()

        # Equation 16
        R1 = np.matmul(self.S.T, self.P)

        # Equation 17
        thread_feature_T = self.thread_lda_feature.T
        M = np.matmul(self.user_lda_feature, thread_feature_T)
        user_feature_mod = np.sqrt(np.sum(self.user_lda_feature * self.user_lda_feature, axis=1))
        thread_feature_mod = np.sqrt(np.sum(self.thread_lda_feature * self.thread_lda_feature, axis=1))
        M = M / user_feature_mod.reshape(-1, 1) / thread_feature_mod

        # Equation 18
        R2 = R1 * M

        # Equation 19
        H = 1 - self.P

        # Equation 20
        self.R = R2 * H

        if self.rec_thread_time_limit is not None:
            self.R = self.R * self.rec_thread_time_limit

    @timeit
    def cal_WS(self):
        """
        Calculate the WS matrix as shown in Equation 4.\n
        WS_ij is social influence of user i on user j
        :return: WS matrix
        """
        path = './model/ws_%d_%d.pickle' % (THREAD_CNT_LOW, THREAD_CNT_HIGH)
        if not DEBUG and os.path.exists(path):
            ws = pickle.load(open(path, 'rb'))
            if len(ws) == self.user_list.shape[0]:
                print("Using the Pre-train ws model")
                return ws

        ws = self.init_user_matrix()

        pbar = tqdm(total=self.user_cnt)
        for i in range(self.user_cnt):
            ws_i = self.dijkstra(self.Bn, i)
            ws[i] = np.array(ws_i)
            pbar.update(1)
        pbar.close()

        # Equation 8
        ws = np.divide(ws, np.sum(ws, axis=0))

        pickle.dump(ws, open(path, 'wb'))
        return ws

    @timeit
    def cal_SR(self):
        """
        SimRank for structural similarity between any two nodes in a network
        :return: None
        """
        path = './model/sr_%d_%d.pickle' % (THREAD_CNT_LOW, THREAD_CNT_HIGH)
        if not DEBUG and os.path.exists(path):
            sr = pickle.load(open(path, 'rb'))
            if len(sr) == self.user_list.shape[0]:
                print("Using the Pre-train sr model")
                return sr

        # Initiate SR as shown in Equation 9
        sr = self.init_user_matrix(diagonal=1)

        neighbor = (self.Bn > 0) & (self.Bn < MAX)

        pbar = tqdm(total=self.user_cnt * self.simRank_k)
        # start SimRank iteration as Equation 11
        for k in range(self.simRank_k):

            sr_old = np.copy(sr)

            for i in range(self.user_cnt):
                i_neighbors = neighbor[i]
                try:
                    i_neighbors_cnt = i_neighbors.astype(int).sum()
                except:
                    continue
                sr_i = sr_old[i_neighbors, :]
                w_i = self.Bn[i_neighbors, i]

                for j in range(i+1):
                    j_neighbors = neighbor[j]
                    try:
                        j_neighbors_cnt = j_neighbors.astype(int).sum()
                    except:
                        continue
                    sr_ij = sr_i[:, j_neighbors]
                    w_j = self.Bn[j_neighbors, j]

                    sim_ij = np.multiply(sr_ij, w_i.reshape(-1, 1))
                    sim_ij = np.multiply(sim_ij, w_j).sum()

                    s = self.c * sim_ij / i_neighbors_cnt / j_neighbors_cnt
                    sr[[i, j], [j, i]] = s

                pbar.update(1)

        pbar.close()
        sr = (sr - np.min(sr)) / (np.max(sr) - np.min(sr))
        pickle.dump(sr, open(path, 'wb'))

        return sr

    @timeit
    def cal_PS(self):

        total_sim = self.init_user_matrix()

        numeric_attrs = [
            'thread_cnt', 'post_cnt', 'level', 'user_group', 'total_online_hours',
            'prestige', 'points', 'wealth', 'visitors', 'friends', 'records', 'logs',
            'albums', 'total_posts', 'total_threads', 'shares'
        ]

        nominal_attrs = ['diabetes_type', 'gender']

        for attr in numeric_attrs:
            diff = []
            attr_col = self.user_list[attr]
            for i in range(self.user_cnt):
                temp = (attr_col-attr_col.iloc[i]).abs()
                if attr not in ['level', 'user_group']:     # if not ordinal attribute
                    temp = temp.apply(lambda x: np.log2(x+1))
                temp.name = temp.index[i]
                diff.append(temp)
            diff_matrix = pd.DataFrame(diff)
            min_diff = diff_matrix.min().min()
            max_diff = diff_matrix.max().max()
            diff_matrix = diff_matrix.applymap(lambda x: 1-(x-min_diff)/(max_diff-min_diff), na_action='ignore')
            diff_matrix.fillna(0, inplace=True)
            total_sim += np.array(diff_matrix)

        for attr in nominal_attrs:
            diff = []
            attr_col = self.user_list[attr]
            attr_col = attr_col.fillna('Null')
            for i in range(self.user_cnt):
                temp = attr_col.eq(attr_col.iloc[i]).astype(int)     # Compare the whole column with the i_th element
                temp.name = temp.index[i]
                diff.append(temp)
            diff_matrix = pd.DataFrame(diff)
            total_sim += np.array(diff_matrix)

        total_sim /= (len(numeric_attrs) + len(nominal_attrs))

        total_sim[np.diag_indices(total_sim.shape[0])] = 1

        return total_sim

    @timeit
    def cal_CS(self):
        if self.user_lda_feature is None:
            self.cal_user_lda_feature()

        # Equation 15
        cs = np.matmul(self.user_lda_feature, self.user_lda_feature.T)
        user_feature_mod = np.sqrt(np.sum(self.user_lda_feature * self.user_lda_feature, axis=1))
        cs = cs / user_feature_mod / user_feature_mod.reshape(-1, 1)

        return cs

    def cal_user_lda_feature(self):
        lda = LDA()
        data = self.data_train if self.data_train is not None else self.data
        data = data[['uid', 'content']]

        features = np.zeros((self.user_cnt, N_TOPICS))

        for uid, df in data.groupby('uid'):
            if uid not in self.user_list.index:
                continue

            # text = " ".join(df.content.tolist())
            # feature = np.array(lda.lda_feature(text))

            feature = df.content.apply(lda.lda_feature)
            feature = pd.DataFrame(feature.values.tolist(), index=df.content.index)
            feature = feature.mean().to_numpy()

            features[uid] = feature

        self.user_lda_feature = features

    def cal_thread_lda_feature(self):
        lda = LDA()
        data = self.load_thread_rank1()

        features = np.zeros((len(self.threads), N_TOPICS))

        for _, row in data.iterrows():
            ft = np.array(lda.lda_feature(row.title + " " + row.content))
            features[row.tid] = ft

        features += 1e-3        # 修正features=0时出错

        self.thread_lda_feature = features

    def init_user_matrix(self, diagonal: int = 0):
        matrix = np.zeros((self.user_cnt, self.user_cnt))
        if diagonal:
            matrix[np.diag_indices(self.user_cnt)] = diagonal
        return matrix

    def user_post_matrix(self):
        data = self.data_train if self.data_train is not None else self.data

        threads_users = data[['tid', 'uid']].drop_duplicates()

        p = np.zeros((len(self.user_list.index), len(self.threads)))

        user_set = self.user_list.index
        for uid, df in threads_users.groupby('uid'):
            if uid in user_set:
                p[uid, df.tid.tolist()] = 1

        return p

    @staticmethod
    def __tuple_to_dict(tup):
        """
        将(key, value)二元组转化为 {key: {value1, value2, ...}}的字典
        :param tup: 二元组
        :return: A dict whose value is set type
        """
        result = {}
        for t, u in tup:
            if t in result:
                result[t].add(u)
            else:
                result[t] = {u}
        return result

    @staticmethod
    def dijkstra(matrix: np.ndarray, start_node):
        """
        根据Equation6，利用dijkstra算法计算从start_node开始，到其他结点路径权重，规则如下：
            1、选择步数最短路径\n
            2、如果步数相同，选择权重最大的路径
        此路径权重对应Equation 7的 WS_ij
        :param matrix: 邻接矩阵，对角为0，不连通的结点间为inf
        :param start_node: 起点结点的index
        :return: 数组ws，ws[j]代表从start_node到j的路径权重
        """
        matrix_length = len(matrix)
        used_node = [False] * matrix_length
        shortest_path = [MAX] * matrix_length
        shortest_path[start_node] = 0       # 路径步数，越少越好
        ws = [0] * matrix_length    # 路径权重，越大越好，对应equation 7
        ws[start_node] = 1

        while used_node.count(False):
            min_value = MAX
            min_value_index = -1
            flag = 0

            for index in range(matrix_length):
                if not used_node[index] and shortest_path[index] < min_value:
                    min_value = shortest_path[index]
                    min_value_index = index
                    flag = 1

            if flag == 0:   # 所有与start_node连通的结点都已遍历过
                break

            used_node[min_value_index] = True

            adjacent_node = matrix[min_value_index] < MAX

            for index in range(matrix_length):
                if not adjacent_node[index] or used_node[index]:
                    continue
                new_path = shortest_path[min_value_index] + 1
                if new_path < shortest_path[index]:
                    shortest_path[index] = new_path
                    ws[index] = ws[min_value_index] * matrix[min_value_index, index]
                elif new_path == shortest_path[index]:
                    new_path_weight = ws[min_value_index] * matrix[min_value_index, index]
                    if new_path_weight > ws[index]:
                        shortest_path[index] = new_path
                        ws[index] = new_path_weight
        ws[start_node] = 0
        return ws

    @staticmethod
    def __path_weight(matrix: pd.DataFrame, previous: list, end_node):
        weight = 1.0
        pre = previous[end_node]
        while pre is not None:
            weight *= matrix.iat[pre, end_node]
            end_node = pre
            pre = previous[end_node]
        if weight == 0 or weight == MAX:
            raise ValueError("weight == %f, previous:" % weight, previous)
        return weight

    def train_test_split(self):
        test_flag = pd.Series([False] * self.data.shape[0], dtype=bool)
        user_group = self.data.groupby('uid')

        thread_time = self.data[['tid', 'publish_time']].drop_duplicates(['tid'])
        thread_time = thread_time.set_index('tid')
        thread_time.sort_index(inplace=True)
        # 根据user最后一次post的时间，只推荐在此时间之前的帖子
        rec_thread_time_limit = np.zeros((self.user_cnt, self.data.tid.nunique()), dtype=np.int32)
        y_test = np.zeros((self.user_cnt, self.data.tid.nunique()), dtype=np.int32)

        for uid, df in user_group:
            thread = df.drop_duplicates(['tid'], keep='first')
            test_thread_cnt = math.ceil(thread.tid.nunique() / 10)
            test_thread_id = thread.iloc[-test_thread_cnt:]['tid']
            test_row = df.tid.isin(test_thread_id)
            test_row_index = test_row[test_row].index.tolist()
            test_flag.iloc[test_row_index] = True

            latest_time = thread.iloc[-1].publish_time
            rec_thread_time_limit[uid] = np.array(thread_time.publish_time.le(latest_time).astype(int))

            y_test[uid, test_thread_id.tolist()] = 1

        self.data_train = self.data.loc[~test_flag].reset_index()
        self.y_test = y_test
        self.rec_thread_time_limit = rec_thread_time_limit

    def run(self):
        self.train_test_split()
        self.IUBN()
        self.UIRs()
        self.URM()


if __name__ == '__main__':
    rec = OHCRec()
    rec.run()
    # engage = rec.user_engage_matrix()
    with open('model.pickle', 'wb') as f:
        pickle.dump((rec.R, rec.y_test), f)
