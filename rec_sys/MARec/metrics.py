import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_prepare import HISTORY_THREAD_TAKEN_CNT
import data_loader
from tqdm import tqdm
import pandas as pd
import random
from mdl import *


class BaseEval:
    def __init__(self, neg_num=100):
        self.negNum = neg_num
        self.pos_rating = dict()
        self.neg_rating = dict()

    def eval(self, k=10):
        cnt = 0
        hits = 0
        miss = 0
        for user, rating in self.pos_rating.items():
            neg_k = self.neg_rating[user][-k]
            for r in rating:
                if r >= neg_k:
                    hits += 1
                else:
                    miss += 1
                cnt += (k+1)

        p = hits / cnt
        recall = hits / (hits + miss)
        f1 = 2 * p * recall / (p + recall)

        return p, recall, f1

    def random_eval(self, k=10):
        cnt = 0
        hits = 0
        miss = 0
        for user, rating in self.pos_rating.items():
            for _ in rating:
                if 0 in random.sample([i for i in range(self.negNum+1)], k):
                    hits += 1
                else:
                    miss += 1
                cnt += (k + 1)

        p = hits / cnt
        recall = hits / (hits + miss)
        f1 = 2 * p * recall / (p + recall)

        return p, recall, f1


class MARecEval(BaseEval):

    def __init__(self, model, testdata, neg_num=100):
        super(MARecEval, self).__init__(neg_num)
        self.model = model
        self.data = data_loader.TestData(testdata)
        self.testLoader = DataLoader(self.data, batch_size=128, shuffle=False)
        self.cal_rating()

    def cal_rating(self):
        with torch.no_grad():
            neg_item_hist = dict()
            pbar = tqdm(total=len(self.testLoader.dataset))
            for i, batchData in enumerate(self.testLoader):
                r = np.array(self.model.forward(batchData))
                users = batchData['user']
                for j in range(len(users)):
                    if int(users[j]) in self.pos_rating:
                        self.pos_rating[int(users[j])].append(r[j])
                    else:
                        self.pos_rating[int(users[j])] = [r[j]]
                    if int(users[j]) not in neg_item_hist:
                        neg_item_hist[int(users[j])] = {
                            'user': users[j],
                            'hist_lda': batchData['hist_lda'][j],
                            'hist_vector': batchData['hist_vector'][j],
                            'hist_info': batchData['hist_info'][j],
                            'hist_participants': batchData['hist_participants'][j],
                            'hist_interact': batchData['hist_interact'][j],
                            'timeDelta': batchData['timeDelta'][j]
                        }
                pbar.update(len(users))
            pbar.close()

            seq = self.data.user_seq
            pbar = tqdm(total=len(seq))
            for user, s in seq.items():

                hist_topic_gain = self.model.hist_topic_gain(neg_item_hist[user])  # 历史收益部分
                topic_gain_diff = torch.sub(self.model.lda_gain_ref_user(neg_item_hist[user]['user']), hist_topic_gain)  # 历史收益与用户预期的差异

                neg_thread = random.sample(s['neg_thread'], self.negNum)
                neg_item = dict()
                neg_item['user'] = torch.LongTensor([user] * len(neg_thread))
                neg_item['item_lda'] = torch.FloatTensor(self.data.thread_lda[neg_thread])
                neg_item['item_vector'] = torch.FloatTensor(self.data.thread_vector[neg_thread])
                neg_item['item_info'] = torch.FloatTensor(self.data.thread_info[neg_thread])
                neg_item['item_participants'] = torch.FloatTensor(self.data.thread_participants_feature[neg_thread])
                neg_item['item_interact'] = torch.FloatTensor(self.data.social_network[user, self.data.thread_user['initiator'][neg_thread]])

                neg_topic_gain = self.model.curr_topic_gain(neg_item)
                dot_ = torch.mul(topic_gain_diff, neg_topic_gain).sum(1).view(-1)

                self.neg_rating[user] = np.sort(dot_)
                pbar.update(1)
            pbar.close()


class OHCRecEval(BaseEval):
    def __init__(self, rating, y_test, neg_num=100):
        super(OHCRecEval, self).__init__(neg_num)
        self.rating_matrix = rating
        self.y_test = y_test
        self.cal_rating()

    def cal_rating(self):
        test_rating = self.rating_matrix * self.y_test
        neg_ = self.rating_matrix * (1-self.y_test)

        for i in range(test_rating.shape[0]):
            row = test_rating[i]
            row = row[row>0]
            self.pos_rating[i] = row
        for i in range(neg_.shape[0]):
            row = neg_[i]
            row = row[row > 0]
            self.neg_rating[i] = sorted(random.sample(list(row), self.negNum))


if __name__ == '__main__':
    model = data_loader.read_data('model')
    test = data_loader.read_data('test_data')
    e = MARecEval(model, test)
    # with open('../OHCRec/model.pickle', 'rb') as f:
    #     rating, y_test = pickle.load(f)
    # e = OHCRecEval(rating, y_test)
    print(e.random_eval())
    print(e.eval())
    print('OK')


