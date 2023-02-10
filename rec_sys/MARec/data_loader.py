from torch.utils.data import Dataset, DataLoader
import os
import pickle
import numpy as np
import random
from copy import deepcopy
from data_prepare import read_data

if os.name == 'nt':
    prefix = 'C:/Users/65113/Desktop/Recsys_data/'
elif os.name == 'posix':
    prefix = '/Users/liangxiyi/Files/Recsys_data'
filetype = '.pickle'


class UserData(Dataset):
    def __init__(self, data, user_hist, thread_lda, thread_bert):
        super(UserData, self).__init__()
        self.data = data
        self.user_hist = user_hist
        self.thread_lda = thread_lda
        self.thread_bert = thread_bert
        self.tid = list(thread_lda.index)
        self.L = len(data)
        self.negNum = 2

    def __len__(self):
        return self.L

    def __getitem__(self, idx):
        item = deepcopy(self.data[idx])
        neg = self.get_neg(int(item['user']))
        item['negItem_lda'] = self.thread_lda.loc[neg].to_numpy()
        item['negItem_bert'] = self.thread_bert.loc[neg].to_numpy()
        return item

    def get_neg(self, user):
        hist = self.user_hist[user][0]
        neg = []
        for i in range(self.negNum):
            while True:
                negThread = random.choice(self.tid)
                if negThread not in hist and negThread not in neg:
                    neg.append(negThread)
                    break
        return neg

    def set_negN(self, neg):
        self.negNum = neg


if __name__ == '__main__':
    train = read_data('train_data')
    user_seq = read_data('user_sequence')
    lda = read_data('thread_lda')
    trainset = UserData(train, user_seq, lda)

    trainLoader = DataLoader(trainset, batch_size=2, shuffle=True)

