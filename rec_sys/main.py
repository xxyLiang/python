import pymysql
import numpy as np
import pandas as pd
import time
from lda import *

MAX = float('inf')
THREAD_CNT_LOW = 20
THREAD_CNT_HIGH = 100


def timeit(func):

    def wrapper(*args, **kwargs):
        t = time.time()
        ret = func(*args, **kwargs)
        print("%s use %.1fs." % (func.__name__, time.time()-t))
        return ret

    return wrapper


class OHCRec:
    def __init__(self, c=0.8, simRank_k=2):
        self.db = pymysql.connect(host='localhost',
                                  user='root',
                                  password='651133439a',
                                  database='rec_sys')
        self.cursor = self.db.cursor()
        self.c = c
        self.simRank_k = simRank_k
        self.data = None
        self.user_cnt = 0
        self.user_list = None
        self.user_lda_feature = None
        self.threads = None
        self.thread_lda_feature = None
        self.Bn = None          # user behavior networks Matrix: n*n
        self.S = None           # User influence relationships Matrix: n*n
        self.P = None           # User Post Matrix: n*m
        self.R = None           # User Rating Matrix: n*m
        self.load_data_from_db()

    def load_data_from_db(self):
        self.cursor.execute("with "
                            "t1 as (SELECT uid FROM user_list WHERE thread_cnt BETWEEN %d AND %d),"
                            "t2 as (SELECT distinct tid from posts p "
                            "where publish_time BETWEEN '2018-01-01' AND '2018-12-31' "
                            "AND EXISTS (select 1 from t1 where p.author_id=t1.uid)), "
                            "t3 as (SELECT tid from threads where forum=1) "
                            "SELECT tid, `rank`, content, author_id from posts p2 "
                            "where EXISTS (select 1 from t2, t3 where p2.tid = t2.tid and p2.tid=t3.tid)" %
                            (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        self.data = pd.DataFrame(self.cursor.fetchall(), columns=['tid', 'rank', 'content', 'uid'])
        self.threads = self.data.tid.drop_duplicates()

        user_attr = ['uid', 'thread_cnt', 'post_cnt', 'level', 'user_group', 'total_online_hours',
                     'regis_time', 'latest_login_time', 'latest_active_time', 'latest_pub_time',
                     'prestige', 'points', 'wealth', 'visitors', 'friends', 'records', 'logs',
                     'albums', 'total_posts', 'total_threads', 'shares', 'diabetes_type', 'treatment_type',
                     'gender', 'birthdate', 'habitation']
        self.user_cnt = self.cursor.execute('SELECT * FROM user_list WHERE thread_cnt BETWEEN %d AND %d' %
                                            (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        rs = self.cursor.fetchall()
        self.user_list = pd.DataFrame(rs, columns=user_attr)
        self.user_list.set_index('uid', inplace=True)
        self.user_cnt = self.user_list.shape[0]

    # Implicit user behavior networks. 用户-用户n*n矩阵
    @timeit
    def IUBN(self, data: pd.DataFrame = None):
        if data is None:
            data = self.data

        self.Bn = self.init_user_matrix()

        threads_users = data[['tid', 'uid']]

        # equation 1
        users_set = set(self.user_list.index)
        for _, df in threads_users.groupby('tid'):
            NU_p = df.shape[0]
            user_in_tid = set(df['uid'])
            intersect = user_in_tid & users_set
            for i in intersect:
                for j in intersect:
                    self.Bn.at[i, j] += 1 / NU_p
        for i in range(len(self.Bn)):
            self.Bn.iat[i, i] = 0

        # equation 2
        self.Bn.div(self.user_list['thread_cnt'], axis=0)

        # equation 3
        self.Bn = self.Bn / self.Bn.max().max()

        # 转化为对角为0，不连接结点为inf的矩阵
        self.Bn = self.Bn.applymap(lambda x: x if x > 0 else MAX)   # 0转为inf
        for i in range(len(self.Bn)):
            self.Bn.iat[i, i] = 0

    # User influence relationships
    def UIRs(self):
        ws = self.cal_WS()
        sr = self.cal_SR()
        ps = self.cal_PS()
        cs = self.cal_CS()
        us = 0.374 * sr + 0.229 * ps + 0.397 * cs
        self.S = ws.mul(us)
        pass

    def URM(self):
        """
        User Rating Matrix
        :return:
        """
        self.P = self.user_post_matrix()
        self.cal_thread_lda_feature()

        # Equation 16
        R1 = self.S.T.dot(self.P)

        # Equation 17
        thread_feature_T = self.thread_lda_feature.T
        M = self.user_lda_feature.dot(thread_feature_T)
        user_feature_mod = self.user_lda_feature.apply(lambda x: np.sqrt(x.dot(x)), axis=1)
        thread_feature_mod = self.thread_lda_feature.apply(lambda x: np.sqrt(x.dot(x)), axis=1)
        M = M.div(user_feature_mod, axis=0).div(thread_feature_mod, axis=1)

        # Equation 18
        R2 = R1.mul(M)

        # Equation 19
        H = 1 - self.P

        # Equation 20
        self.R = R2.mul(H)

    @timeit
    def cal_WS(self):
        """
        Calculate the WS matrix as shown in Equation 4.\n
        WS_ij is social influence of user i on user j
        :return: WS matrix
        """
        ws = self.init_user_matrix()

        for i in range(self.user_cnt):
            ws_i = self.dijkstra(self.Bn, i)
            ws.iloc[i] = pd.Series(ws_i)

        # Equation 8
        ws = ws.div(ws.sum())
        return ws

    @timeit
    def cal_SR(self):
        """
        SimRank for structural similarity between any two nodes in a network
        :return: None
        """
        # Initiate SR as shown in Equation 9
        sr = self.init_user_matrix(diagonal=1)

        neighbor = self.Bn.applymap(lambda x: True if 0 < x < MAX else False)

        # start SimRank iteration as Equation 11
        for k in range(self.simRank_k):

            sr_old = sr.copy()

            for i in range(self.user_cnt):
                i_neighbors = neighbor.iloc[i]
                i_neighbors_cnt = i_neighbors.value_counts()[True]
                if i_neighbors_cnt == 0:
                    continue
                sr_i = sr_old.loc[i_neighbors, :]
                w_i = self.Bn.loc[i_neighbors, self.Bn.index[i]]
                for j in range(i+1):
                    j_neighbors = neighbor.iloc[j]
                    j_neighbors_cnt = j_neighbors.value_counts()[True]
                    if j_neighbors_cnt == 0:
                        continue
                    sr_ij = sr_i.loc[:, j_neighbors]
                    w_j = self.Bn.loc[j_neighbors, self.Bn.index[j]]

                    sim_ij = sr_ij.mul(w_i, axis=0).mul(w_j, axis=1).sum().sum()

                    s = self.c * sim_ij / i_neighbors_cnt / j_neighbors_cnt
                    sr.iat[i, j] = s
                    sr.iat[j, i] = s

        sr = (sr - sr.min().min()) / (sr.max().max() - sr.min().min())

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
                temp = attr_col.diff(periods=i+1).abs()
                if attr not in ['level', 'user_group']:     # if not ordinal attribute
                    temp = temp.apply(lambda x: np.log2(x+1))
                temp.name = temp.index[i]
                diff.append(temp)
            diff_matrix = pd.DataFrame(diff)
            min_diff = diff_matrix.min().min()
            max_diff = diff_matrix.max().max()
            diff_matrix = diff_matrix.applymap(lambda x: 1-(x-min_diff)/(max_diff-min_diff), na_action='ignore')
            diff_matrix.fillna(0, inplace=True)
            total_sim = total_sim.add(diff_matrix)

        total_sim = total_sim.add(total_sim.T)      # diff_matrix是上三角矩阵，这一句将total_sim转为对称矩阵

        for attr in nominal_attrs:
            diff = []
            attr_col = self.user_list[attr]
            attr_col = attr_col.fillna('Null')
            for i in range(self.user_cnt):
                temp = attr_col.eq(attr_col[i]).astype(int)     # Compare the whole column with the i_th element
                temp.name = temp.index[i]
                diff.append(temp)
            diff_matrix = pd.DataFrame(diff)
            total_sim = total_sim.add(diff_matrix)

        total_sim /= (len(numeric_attrs) + len(nominal_attrs))

        for i in range(self.user_cnt):
            total_sim.iat[i, i] = 1

        return total_sim

    @timeit
    def cal_CS(self):
        if self.user_lda_feature is None:
            self.cal_user_lda_feature()

        # Equation 15
        cs = self.user_lda_feature.dot(self.user_lda_feature.T)
        user_feature_mod = self.user_lda_feature.apply(lambda x: np.sqrt(x.dot(x)), axis=1)
        cs = cs.div(user_feature_mod, axis=0).div(user_feature_mod, axis=1)

        return cs

    def cal_user_lda_feature(self):
        lda = LDA()
        data = self.data[['uid', 'content']]

        features = pd.DataFrame(
            data=np.zeros((self.user_cnt, N_TOPICS)),
            dtype=float,
            index=self.user_list.index
        )

        for uid, df in data.groupby('uid'):
            if uid not in self.user_list.index:
                continue

            text = " ".join(df.content.tolist())
            feature = pd.Series(lda.lda_feature(text))

            # feature = df.content.apply(lda.lda_feature)
            # feature = pd.DataFrame(feature.values.tolist(), index=df.content.index)
            # feature = feature.mean()

            features.loc[uid] = feature

        self.user_lda_feature = features

    def cal_thread_lda_feature(self):
        lda = LDA()
        data = self.data[self.data['rank'] == 1][['tid', 'content']]

        features = pd.DataFrame(
            data=np.zeros((len(self.threads), N_TOPICS)),
            dtype=float,
            index=self.threads
        )

        for _, row in data.iterrows():
            f = pd.Series(lda.lda_feature(row.content))
            features.loc[row.tid] = f

        self.thread_lda_feature = features

    def init_user_matrix(self, diagonal: int = 0):
        matrix = pd.DataFrame(
            data=np.zeros((self.user_cnt, self.user_cnt)),
            dtype=float,
            index=self.user_list.index,
            columns=self.user_list.index
        )
        if diagonal:
            for i in range(self.user_cnt):
                matrix.iat[i, i] = diagonal
        return matrix

    def user_post_matrix(self, data: pd.DataFrame = None):
        if data is None:
            data = self.data

        threads_users = data[['tid', 'uid']].drop_duplicates()

        p = pd.DataFrame(
            data=np.zeros((len(self.user_list.index), len(self.threads))),
            dtype=int,
            index=self.user_list.index,
            columns=self.threads.tolist()
        )

        user_set = self.user_list.index
        for uid, df in threads_users.groupby('uid'):
            if uid in user_set:
                p.loc[uid, df.tid.tolist()] = 1

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
    def dijkstra(matrix: pd.DataFrame, start_node):
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

            adjacent_node = matrix.iloc[min_value_index] < MAX

            for index in range(matrix_length):
                if not adjacent_node[index] or used_node[index]:
                    continue
                new_path = shortest_path[min_value_index] + 1
                if new_path < shortest_path[index]:
                    shortest_path[index] = new_path
                    ws[index] = ws[min_value_index] * matrix.iat[min_value_index, index]
                elif new_path == shortest_path[index]:
                    new_path_weight = ws[min_value_index] * matrix.iat[min_value_index, index]
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

    @staticmethod
    def train_test_split(x):
        pass

    def run(self):
        self.IUBN()
        self.UIRs()


if __name__ == '__main__':
    rec = OHCRec()
    rec.run()
