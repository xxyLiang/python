from tqdm import tqdm
import pandas as pd
import numpy as np


class DataPrepare:
    def load_all_posts(self):
        return pd.DataFrame()
        pass

    # 根据用户交互网络，计算某一帖子已回复的人与当前用户的交互指数和
    def make_user_sequence(self):
        """
        pattern of User_seq：
        {
          user1: {
                'thread': [tid_1, tid_2, ..., tid_n],
                'time': [tid_1_time, tid_2_time, ..., tid_n_time],
                'test_flag': [test_flag1, ...],
          },
          user2: ...
        }
        """
        data = self.load_all_posts()
        data_i = data.set_index(['tid', 'rank'])

        user_seq = dict()

        user_group = data[['uid', 'tid', 'publish_time', 'test_flag']].groupby('uid')
        user_cnt = user_group.size().shape[0]
        pbar = tqdm(total=user_cnt)

        final_interact_matrix = np.zeros((user_cnt, user_cnt)).astype(np.float64)

        for uid, df in user_group:
            # 查询帖子序列及其时间序列
            thread = df.drop_duplicates(subset=['tid'], keep='first')
            t_list = thread['tid'].to_list()
            time_list = thread['publish_time'].to_list()
            test_flag = thread['test_flag'].to_list()

            # 计算每个帖子的交互指数
            user_interact_index = list()
            interact_array = np.zeros(user_cnt).astype(np.float64)

            for idx in range(len(t_list)):
                if idx == 0:
                    user_interact_index.append(0)
                    continue

                # 首先算以前和哪些人有过交集，及其交互指数
                interact_post = data_i.loc[t_list[idx-1]]
                interact_post = interact_post[interact_post['publish_time'] < time_list[idx]].drop_duplicates(
                    subset=['uid'])
                interact_array[interact_post['uid']] += 1 / max(interact_post.shape[0], 1)

                # 然后算待预测帖子中，特定时间前与已跟帖者的交互指数的和
                curr_thread_post = data_i.loc[t_list[idx]]
                curr_thread_post = curr_thread_post[curr_thread_post['publish_time'] < time_list[idx]]
                interact_user = curr_thread_post['uid'].drop_duplicates()
                if uid in interact_user:
                    interact_user.drop(uid)
                interaction_index = interact_array[interact_user].sum() / idx
                user_interact_index.append(interaction_index)

            final_interact_matrix[uid] = interact_array / len(t_list)

            user_seq[uid] = {
                'thread': t_list,
                'time': time_list,
                'test_flag': test_flag,
                # 'interact': user_interact_index
            }
            pbar.update(1)

        pbar.close()
        # save_data(user_seq, 'user_sequence')
        # save_data(final_interact_matrix, 'interact_matrix')

        return user_seq

    # 计算用户与用户之间的交互矩阵，并根据这个交互矩阵，通过平均值计算帖子的参与用户特征
    def community_feature(self):
        data = self.load_all_posts()
        user_cnt = data['uid'].nunique()
        thread_cnt = data['tid'].nunique()
        user_feature = np.zeros((user_cnt, user_cnt)).astype(np.float64)

        data = data[~data['test_flag']]

        for tid, df in data.groupby('tid'):
            user_in_tid = df['uid'].drop_duplicates().to_list()
            nu = len(user_in_tid)
            for i in user_in_tid:
                user_feature[i, user_in_tid] += 1/nu

        user_feature[np.diag_indices(user_feature.shape[0])] = 0        # 对角线（即用户对自己的相关度）赋值为0
        # feature除用户参与的帖子数，以消除用户活跃度差异的影响。
        for k, v in self.user_seq.items():
            user_train_thread_cnt = max(v['test_flag'].index(True), 1)
            user_feature[k] /= user_train_thread_cnt

        user_feature /= np.max(user_feature)

        thread_feature = np.zeros((thread_cnt, user_cnt)).astype(np.float64)
        for tid, df in data.groupby('tid'):
            user_in_tid = df['uid'].drop_duplicates().to_list()
            thread_feature[tid] = user_feature[user_in_tid].mean(axis=0)

        save_data((user_feature, thread_feature), 'community_feature')

        return user_feature, thread_feature

    def make_thread_vector(self):
        filename = prefix + 'doc2vec.model'
        cuter = CutWord()
        if os.path.exists(filename):
            model = Doc2Vec.load(filename)
        else:
            print('Training Doc2Vec model...')
            cursor.execute("select content from posts where CHAR_LENGTH(content)>10")
            data = cursor.fetchall()
            documents = list()
            for i, text in enumerate(data):
                words = cuter.cut(text[0])
                documents.append(TaggedDocument(words, [i]))
            model = Doc2Vec(documents, vector_size=VECTOR_DIM)
            model.save(filename)
            del documents

        print('Convert text to vector...')
        data = self.load_thread_rank1(thread_map=self.tid2id)
        data['text'] = data['title'] + " " + data['content']
        pbar = tqdm(total=data.shape[0])

        features = np.zeros((self.thread_cnt + 1, VECTOR_DIM), dtype=np.float32)
        for _, row in data.iterrows():
            words = cuter.cut(row['text'])
            features[row.tid] = model.infer_vector(words)
            pbar.update(1)
        pbar.close()

        save_data(features, 'thread_vector')
        return features