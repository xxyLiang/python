import time
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from random import sample
from copy import deepcopy
from data_prepare import read_data, FileMaker

if os.name == 'nt':
    prefix = 'C:/Users/65113/Desktop/Recsys_data/'
elif os.name == 'posix':
    prefix = '/Users/liangxiyi/Files/Recsys_data/'
filetype = '.pickle'


class UserData(Dataset):
    def __init__(self, data):
        super(UserData, self).__init__()
        self.data = data
        self.user_seq = read_data('user_sequence')
        self.thread_lda = read_data('thread_lda')
        self.thread_vector = read_data('thread_vector')
        self.thread_info = read_data('thread_info')         # (m, 3)
        self.thread_user = read_data('thread_user')         # {'initiator': [], 'participants': []}
        self.thread_participants_feature = read_data('thread_participants_feature')     # (m, 20)
        self.social_network = read_data('social_network')
        self.thread_cnt = self.thread_lda.shape[0] - 1
        self.thread_set = set(range(self.thread_cnt))
        self.L = len(data)
        self.negNum = 2
        # out_degree = self.social_network['adjacency_matrix']
        in_degree = self.social_network['adjacency_matrix'].T
        # shortest_path = self.social_network['shortest_path']
        interact = self.social_network['interact']
        self.social_network = np.stack((in_degree, interact), axis=2)       # (n, n, 4)
        # del out_degree, in_degree, shortest_path, interact

    def __len__(self):
        return self.L

    def __getitem__(self, idx):
        item = deepcopy(self.data[idx])
        user = item['user']
        item_id = item['item_id']
        # data中原来就有的key：
        # 'user', 'timeDelta'
        item['item_lda'] = self.thread_lda[item_id]
        item['item_vector'] = self.thread_vector[item_id]
        item['item_info'] = self.thread_info[item_id]
        item['item_participants'] = self.thread_participants_feature[item_id]
        item['item_interact'] = self.social_network[user, self.thread_user['initiator'][item_id]]

        hist_item = item['hist_item']
        item['hist_lda'] = self.thread_lda[hist_item]
        item['hist_vector'] = self.thread_vector[hist_item]
        item['hist_info'] = self.thread_info[hist_item]
        item['hist_participants'] = self.thread_participants_feature[hist_item]
        item['hist_interact'] = self.social_network[user, self.thread_user['initiator'][hist_item]]

        neg = sample(self.user_seq[int(item['user'])]['neg_thread'], k=self.negNum)
        item['negItem_lda'] = self.thread_lda[neg]
        item['negItem_vector'] = self.thread_vector[neg]
        item['negItem_info'] = self.thread_info[neg]
        item['negItem_participants'] = self.thread_participants_feature[neg]
        item['negItem_interact'] = self.social_network[user, self.thread_user['initiator'][neg]]
        return item

    def set_negN(self, neg):
        self.negNum = neg


class TestData(UserData):

    def __getitem__(self, idx):
        item = deepcopy(self.data[idx])
        user = item['user']
        item_id = item['item_id']
        # data中原来就有的key：
        # 'user', 'timeDelta'
        item['item_lda'] = self.thread_lda[item_id]
        item['item_vector'] = self.thread_vector[item_id]
        item['item_info'] = self.thread_info[item_id]
        item['item_participants'] = self.thread_participants_feature[item_id]
        item['item_interact'] = self.social_network[user, self.thread_user['initiator'][item_id]]

        hist_item = item['hist_item']
        item['hist_lda'] = self.thread_lda[hist_item]
        item['hist_vector'] = self.thread_vector[hist_item]
        item['hist_info'] = self.thread_info[hist_item]
        item['hist_participants'] = self.thread_participants_feature[hist_item]
        item['hist_interact'] = self.social_network[user, self.thread_user['initiator'][hist_item]]

        return item


if __name__ == '__main__':
    train = read_data('train_data')
    trainset = UserData(train)

    trainLoader = DataLoader(trainset, batch_size=2, shuffle=True)
    i, data = next(enumerate(trainLoader))

    time.sleep(0)
