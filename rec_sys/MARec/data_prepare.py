import pandas as pd
import numpy as np
import pickle
import math
from bert_serving.client import BertClient
from sklearn.decomposition import PCA
from tqdm import tqdm
from lda import LDA
from random import sample
import networkx as nx
from settings import *


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


def data_exists(filename):
    address = prefix + filename + '.pickle'
    return os.path.exists(address)


class FileMaker:

    def __init__(self):
        self.uid2id, self.tid2id = read_data('id_transfer') if data_exists('id_transfer') else self.make_id_transfer()
        self.thread_cnt = len(self.tid2id)
        self.user_cnt = len(self.uid2id)
        self.user_seq = read_data('user_sequence') if data_exists('user_sequence') else self.make_user_sequence()
        self.thread_vector = read_data('thread_vector') if data_exists('thread_vector') else self.make_thread_BERT_file()
        self.thread_lda = read_data('thread_lda') if data_exists('thread_lda') else self.make_thread_LDA_file()

    @staticmethod
    def load_thread_rank1(thread_map=None):
        cursor.execute("with "
                       "t1 as (SELECT uid FROM user_list WHERE thread_cnt BETWEEN %d AND %d),"
                       "t2 as (SELECT distinct tid from posts p "
                       "where publish_time BETWEEN '2018-01-01' AND '2018-12-31' "
                       "AND EXISTS (select 1 from t1 where p.author_id=t1.uid)) "
                       "SELECT p2.tid, threads.title, p2.content, p2.img_num from posts p2 "
                       "LEFT JOIN threads ON p2.tid = threads.tid "
                       "WHERE EXISTS (select 1 from t2 where p2.tid=t2.tid) and `rank`=1" %
                       (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        data = pd.DataFrame(cursor.fetchall(), columns=['tid', 'title', 'content', 'img_num'])
        if thread_map is not None:
            data['tid'] = data['tid'].apply(lambda x: thread_map[x])
        return data

    @classmethod
    def load_all_posts(cls, user_map=None, thread_map=None):
        cursor.execute("with "
                       "t1 as (SELECT uid FROM user_list WHERE thread_cnt BETWEEN %d AND %d),"
                       "t2 as (SELECT distinct tid from posts p "
                       "where publish_time BETWEEN '2018-01-01' AND '2018-12-31' "
                       "AND EXISTS (select 1 from t1 where p.author_id=t1.uid)) "
                       "SELECT tid, `rank`, content, author_id, publish_time from posts p2 "
                       "where EXISTS (select 1 from t1, t2 where p2.tid = t2.tid and p2.author_id=t1.uid) "
                       "AND publish_time BETWEEN '2018-01-01' AND '2018-12-31' order by 5 asc" %
                       (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        data = pd.DataFrame(cursor.fetchall(), columns=['tid', 'rank', 'content', 'uid', 'publish_time'])
        cls.train_test_split(data)
        if user_map is not None and thread_map is not None:
            data['tid'] = data['tid'].apply(lambda x: thread_map[x])
            data['uid'] = data['uid'].apply(lambda x: user_map[x])

        return data

    @classmethod
    def load_posts_and_reply(cls):
        cursor.execute("with "
                       "t1 as (SELECT uid FROM user_list WHERE thread_cnt BETWEEN %d AND %d),"
                       "t2 as (SELECT distinct tid from posts p "
                       "where publish_time BETWEEN '2018-01-01' AND '2018-12-31' "
                       "AND EXISTS (select 1 from t1 where p.author_id=t1.uid)) "
                       "SELECT tid, `rank`, content, p2.author_id, p3.author_id as reply_to_uid from posts p2 "
                       "LEFT JOIN (select pid, author_id from posts) p3 on p2.reply_to_pid=p3.pid "
                       "where EXISTS (select 1 from t1, t2 where p2.tid = t2.tid and p2.author_id=t1.uid) "
                       "AND publish_time BETWEEN '2018-01-01' AND '2018-12-31' " %
                       (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        data = pd.DataFrame(cursor.fetchall(), columns=['tid', 'rank', 'content', 'uid', 'reply_to'])
        cls.train_test_split(data)
        return data

    @classmethod
    def load_user_info(cls, user_map=None):
        cursor.execute("SELECT uid, level, user_group, points FROM user_list WHERE thread_cnt BETWEEN %d AND %d" %
                       (THREAD_CNT_LOW, THREAD_CNT_HIGH))
        data = pd.DataFrame(cursor.fetchall(), columns=['uid', 'level', 'user_group', 'points'])
        if user_map is not None:
            data['uid'] = data['uid'].apply(lambda x: user_map[x])
        data = data.set_index(keys='uid', drop=True).sort_index().fillna(0)
        data['points'] = data['points'].apply(lambda x: x if x > 0 else 0)
        return data

    @staticmethod
    def train_test_split(data):
        data['test_flag'] = False
        test_data_idx = list()
        for uid, df in data.groupby('uid'):
            engage_thread = df['tid'].drop_duplicates()
            test_cnt = math.ceil(engage_thread.shape[0] / 10)
            test_data_idx.extend(df[df['tid'].isin(engage_thread.iloc[-test_cnt:])].index.to_list())
        data.loc[test_data_idx, 'test_flag'] = True
        return

    def make_id_transfer(self, data=None):
        if data is None:
            data = self.load_all_posts()
        users = data['uid'].drop_duplicates()
        users.reset_index(drop=True, inplace=True)
        id2uid = users.to_dict()
        uid2id = dict(zip(id2uid.values(), id2uid.keys()))
        threads = data['tid'].drop_duplicates()
        threads.reset_index(drop=True, inplace=True)
        id2tid = threads.to_dict()
        tid2id = dict(zip(id2tid.values(), id2tid.keys()))

        save_data((uid2id, tid2id), 'id_transfer')

        return uid2id, tid2id

    def make_user_sequence(self, data=None):
        """
        以用户为单位，生成每个用户的参与帖子序列，包括：帖子id、参与时间、是否为测试集；以及用户没有参与，且在最后活跃时间已经有的其他帖子。
        """
        if data is None:
            data = self.load_all_posts(user_map=self.uid2id, thread_map=self.tid2id)

        user_seq = dict()

        user_group = data.groupby('uid')
        user_cnt = user_group.size().shape[0]
        pbar = tqdm(total=user_cnt)

        for uid, df in user_group:
            # 查询帖子序列及其时间序列
            thread = df.drop_duplicates(subset=['tid'], keep='first')
            t_list = thread['tid'].to_list()
            time_list = thread['publish_time'].to_list()
            test_flag = thread['test_flag'].to_list()

            # 生成负采样所需的帖子列表
            thread_time = data[['tid', 'publish_time']].drop_duplicates(subset=['tid'])
            thread_before_last_activity = thread_time[thread_time['publish_time'] < time_list[-1]]['tid']
            neg_thread = set(thread_before_last_activity) - set(t_list)
            test_neg_thread = sample(list(neg_thread), 200)
            train_neg_thread = neg_thread - set(test_neg_thread)

            user_seq[uid] = {
                'thread': t_list,
                'time': time_list,
                'test_flag': test_flag,
                'test_neg_thread': test_neg_thread,
                'train_neg_thread': list(train_neg_thread)
            }
            pbar.update(1)

        pbar.close()
        save_data(user_seq, 'user_sequence')

        return user_seq

    def make_social_network_file(self):
        """
        根据回帖关系，生成用户间的邻接矩阵、最短路径、交互指数的【字典】
        :return:
        """
        data = self.load_posts_and_reply()
        data = data[~data['test_flag']]
        data['tid'] = data['tid'].apply(lambda x: self.tid2id[x])
        data['uid'] = data['uid'].apply(lambda x: self.uid2id[x])
        adjacency_matrix = np.zeros((self.user_cnt+1, self.user_cnt+1))
        print('Creating adjacency matrix...')
        for tid, df in data.groupby('tid'):
            first_thread = df[df['rank'] == 1]
            if len(first_thread) == 0:
                continue
            thread_initiator = first_thread['uid']
            for _, row in df.iterrows():
                adjacency_matrix[row['uid'], thread_initiator] += 1
                if row['reply_to'] and row['reply_to'] in self.uid2id.keys():
                    adjacency_matrix[row['uid'], self.uid2id[row['reply_to']]] += 1
        adjacency_matrix[np.diag_indices(self.user_cnt)] = 0

        # graph = nx.DiGraph(adjacency_matrix)

        adjacency_matrix = 1 / (1 + np.exp(-adjacency_matrix)) * 2 - 1      # 通过sigmoid转化至（0，1）

        # 计算两两用户间的最短路径长度，并取倒数
        # print("Calculating shortest path...")
        # shortest_path_length = np.zeros((self.user_cnt+1, self.user_cnt+1))
        # for k, v in nx.shortest_path_length(graph, weight=1):
        #     shortest_path_length[k, list(v.keys())] = list(v.values())
        # shortest_path_length = np.reciprocal(shortest_path_length.astype(np.float32), where=shortest_path_length != 0)

        # 计算两两用户间的交互指数
        print('Calculating interact index...')
        user_cnt = data['uid'].nunique()
        interact_index = np.zeros((user_cnt+1, user_cnt+1)).astype(np.float32)

        for tid, df in data.groupby('tid'):
            user_in_tid = df['uid'].drop_duplicates().to_list()
            nu = len(user_in_tid)
            for i in user_in_tid:
                interact_index[i, user_in_tid] += 1 / nu

        interact_index[np.diag_indices(interact_index.shape[0])] = 0  # 对角线（即用户对自己的相关度）赋值为0

        for uid, df in data.groupby('uid'):
            user_train_thread_cnt = df['tid'].drop_duplicates().shape[0]
            interact_index[uid] /= user_train_thread_cnt            # feature除以用户参与的帖子数，以消除用户活跃度差异的影响。

        interact_index /= np.max(interact_index)                    # 归一化

        social_network = {
            'adjacency_matrix': adjacency_matrix,
            # 'shortest_path': shortest_path_length,
            'interact': interact_index
        }
        save_data(social_network, 'social_network')
        return social_network

    def make_user_info_file(self):
        data = self.load_user_info(user_map=self.uid2id)
        data['level'] /= 10
        data['user_group'] /= 10
        data['points'] = data['points'].apply(lambda x: 1 / (1 + np.exp(-math.log(x+1, 100))) * 2 - 1)

        if data.shape[0] == self.user_cnt:
            feature = data.to_numpy()
        else:
            feature = np.zeros((self.user_cnt, data.shape[1]), dtype=np.float32)
            feature[data.index] = data.to_numpy()

        save_data(feature, 'user_info')
        return feature

    def make_thread_LDA_file(self):
        lda = LDA()

        data = self.load_thread_rank1(thread_map=self.tid2id)
        threads = data['tid'].drop_duplicates()

        features = np.zeros((len(threads), N_TOPICS), dtype=np.float32)

        for _, row in data.iterrows():
            f = np.array(lda.lda_feature(row.title + ' ' + row.content), dtype=np.float32)
            features[row.tid] = f

        features = np.concatenate((features, np.zeros((1, N_TOPICS))), dtype=np.float32)      # 为历史帖子小于10个的填充做准备
        save_data(features, 'thread_lda')

        return features

    def make_thread_BERT_file(self, dim=VECTOR_DIM):
        print("start BERT embedding initiating...")
        bc = BertClient(ip='192.168.2.15', check_length=False)

        data = self.load_thread_rank1(thread_map=self.tid2id)
        threads = data['tid'].drop_duplicates()
        t_cnt = len(threads)

        features = np.zeros((t_cnt, 768), dtype=np.float32)

        pbar = tqdm(total=len(threads))
        batchSize = 64
        for i in range(0, len(threads), batchSize):
            end = min(i+batchSize, t_cnt)
            df = data.iloc[i: end]
            content = df['title'] + " " + df['content']
            content = content.to_list()
            f = bc.encode(content)
            features[df['tid'].to_list()] = f
            pbar.update(end-i)
        pbar.close()

        print("lowering dimension with PCA...")
        if not data_exists('pca'):
            pca = PCA(n_components=dim)
            pca_feature = pca.fit_transform(features)
            save_data(pca, 'pca')
        else:
            pca = read_data('pca')
            pca_feature = pca.transform(features)
        pca_feature = np.concatenate((pca_feature, np.zeros((1, dim))), dtype=np.float32)     # 为历史帖子小于10个的填充做准备
        save_data(pca_feature, 'thread_vector')

        return pca_feature

    def make_thread_info_file(self, data=None):
        """
        thread_info【字典】中存储帖子相关的统计性数据，包括：
            1. statistics: np.array (m, 3)，文本长度、图片数量、信息增益
            2. initiator: list (m, ) ，发帖者
            3. participants: dict[int, list]，主题帖中出现的所有用户（训练集）
            4. participant_feature: np.array (m, 20)，根据帖子参与用户的lda偏好，以lda形式计算帖子的参与者特征
        :return:
        """
        thread_info = dict()

        if data is None:
            data = self.load_all_posts(user_map=self.uid2id, thread_map=self.tid2id)
        user_lda_dist = read_data('user_dist')['lda_dist']
        thread_user = self._thread_user(data)
        participant_feature = self._thread_participants_feature(thread_user['participants'], user_lda_dist)

        thread_info['statistics'] = self._thread_statistic()
        thread_info['initiator'] = thread_user['initiator']
        thread_info['participants'] = thread_user['participants']
        thread_info['participant_feature'] = participant_feature

        save_data(thread_info, 'thread_info')
        return thread_info

    def _thread_statistic(self):
        # 描述统计：text_len: (0.52 +- 0.29), img_num:(0.26 +- 0.31), info_quan: (0.47 += 0.20)
        from auxiliary import CutWord
        cut_tool = CutWord()
        data = self.load_thread_rank1(thread_map=self.tid2id)
        data['text'] = data['title'] + " " + data['content']

        # 首先统计text_len和img_num
        text_len_array = np.zeros(self.thread_cnt + 1)
        img_num_array = np.zeros(self.thread_cnt + 1)
        text_len_array[data['tid']] = data['text'].apply(lambda x: len(x) / 50)
        img_num_array[data['tid']] = data['img_num'].apply(lambda x: np.log2(x + 1))

        # 计算信息增益
        topic_word_freq = []
        for i in range(N_TOPICS):
            topic_word_freq.append(dict())

        for _, row in data.iterrows():
            topic_class = self.thread_lda[row.tid].argmax()
            for i in cut_tool.cut(row['text']):
                freq_dict = topic_word_freq[topic_class]
                if i in freq_dict:
                    freq_dict[i] += 1
                else:
                    freq_dict[i] = 1

        # 整理，词频>=3才最终加入词典
        for i in range(N_TOPICS):
            d = {'max_freq': 0, 'word_freq': {}}
            for k, v in topic_word_freq[i].items():
                if v >= 3:
                    d['word_freq'][k] = v
                    d['max_freq'] = max(d['max_freq'], v)
            topic_word_freq[i] = d

        info_quan_array = np.zeros(data['tid'].nunique() + 1, dtype=np.float32)
        for _, row in data.iterrows():
            info_quan = 0
            topic_class = self.thread_lda[row.tid].argmax()
            content = row['title'] + ' ' + row['content']
            wf = topic_word_freq[topic_class]
            for i in cut_tool.cut(content):
                if i in wf['word_freq'].keys():
                    info_quan += np.log10(wf['max_freq'] / wf['word_freq'][i])
            info_quan_array[row.tid] = np.log10(1 + info_quan)

        # sigmoid转化为（0, 1），由于原本值域为[0, MAX)，直接转化后值域变为[0.5, 1)，所以进行*2-1
        text_len_array = 1 / (1 + np.exp(-text_len_array)) * 2 - 1
        img_num_array = 1 / (1 + np.exp(-img_num_array)) * 2 - 1
        info_quan_array = 1 / (1 + np.exp(-info_quan_array)) * 2 - 1

        thread_stat = np.vstack((text_len_array, img_num_array, info_quan_array)).T

        return thread_stat

    @staticmethod
    def _thread_user(data):
        # 初始化
        thread_cnt = data['tid'].nunique()

        # 发帖者
        initiator = np.zeros(thread_cnt+1, dtype=np.int32)
        initiator[-1] = -1       # 空白
        first_post = data[data['rank'] == 1]
        initiator[first_post['tid'].to_numpy()] = first_post['uid'].to_numpy()

        # 所有用户
        participants = dict()
        for i in data['tid'].drop_duplicates().to_list():
            participants[i] = []
        participants[len(participants)] = []    # 空白
        data_train = data[~data['test_flag']][['tid', 'uid']].drop_duplicates()
        for tid, df in data_train.groupby('tid'):
            participants[tid] = df['uid'].to_list()

        thread_user = {
            'initiator': initiator,
            'participants': participants,
        }

        return thread_user

    @staticmethod
    def _thread_participants_feature(thread_participants, user_lda_dist):
        """
        根据帖子参与用户的lda偏好，以lda形式计算帖子的参与者特征
        :return: m*20的numpy矩阵
        """
        feature = np.zeros((len(thread_participants), N_TOPICS)).astype(np.float32)
        for tid, participants in thread_participants.items():
            if len(participants) != 0:
                feature[tid] = user_lda_dist[participants].mean(axis=0)

        return feature

    def make_train_test_file(self):
        print('Generating train and test file...')

        user_cnt = len(self.user_seq.keys())
        train_data = list()
        test_data = list()

        user_lda_dist = np.zeros((user_cnt, N_TOPICS), dtype=np.float32)
        user_vector_dist = np.zeros((user_cnt, VECTOR_DIM), dtype=np.float32)

        pbar = tqdm(total=user_cnt)
        for user, seq in self.user_seq.items():
            user: int
            for idx in range(len(seq['thread'])):
                if idx <= 1:
                    continue
                item = dict()
                item['user'] = np.array(user).astype(np.int32)
                item['item_id'] = np.array(seq['thread'][idx]).astype(np.int32)

                if idx >= HISTORY_THREAD_TAKEN_CNT:
                    start = idx - HISTORY_THREAD_TAKEN_CNT
                else:
                    start = 0

                item['hist_item'] = np.array(seq['thread'][start: idx], dtype=np.int32)
                item['timeDelta'] = np.array(
                    [(seq['time'][idx] - seq['time'][i]).total_seconds() / 2592000 for i in range(start, idx)],
                    dtype=np.float32
                )

                if idx < HISTORY_THREAD_TAKEN_CNT:
                    item['hist_item'] = np.concatenate((
                        item['hist_item'],
                        np.zeros(HISTORY_THREAD_TAKEN_CNT - item['hist_item'].shape[0])-1
                    )).astype(np.int32)
                    item['timeDelta'] = np.concatenate((
                        item['timeDelta'],
                        np.zeros(HISTORY_THREAD_TAKEN_CNT - item['timeDelta'].shape[0])
                    )).astype(np.float32)
                # more features here

                if seq['test_flag'][idx]:
                    test_data.append(item)
                else:
                    train_data.append(item)

            user_lda_dist[user] = self.thread_lda[seq['thread'][: seq['test_flag'].index(True)]].mean(axis=0)
            user_vector_dist[user] = self.thread_vector[seq['thread'][: seq['test_flag'].index(True)]].mean(axis=0)
            pbar.update(1)
        pbar.close()

        save_data(train_data, 'train_data')
        save_data(test_data, 'test_data')
        save_data({'lda_dist': user_lda_dist, 'vector_dist': user_vector_dist}, 'user_dist')

        return

    def make_all_files(self):
        data = self.load_all_posts()
        self.uid2id, self.tid2id = self.make_id_transfer()
        data['tid'] = data['tid'].apply(lambda x: self.tid2id[x])
        data['uid'] = data['uid'].apply(lambda x: self.uid2id[x])
        self.thread_cnt = len(self.tid2id)
        self.user_cnt = len(self.uid2id)
        self.user_seq = self.make_user_sequence(data)
        # self.thread_vector = self.make_thread_BERT_file()
        # self.thread_lda = self.make_thread_LDA_file()
        self.make_train_test_file()
        self.make_thread_info_file(data)
        self.make_user_info_file()
        self.make_social_network_file()


if __name__ == '__main__':
    fileMaker = FileMaker()
    # seq = fileMaker.make_user_sequence()
    fileMaker.make_train_test_file()
    # time.sleep(1)
