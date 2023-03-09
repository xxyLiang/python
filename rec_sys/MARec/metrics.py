import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
import data_loader
from tqdm import tqdm
import random
import os
from auxiliary import prefix
from mdl import PT
from DIN import DIN_PT


class BaseEval:
    def __init__(self, neg_num=100):
        self.negNum = neg_num
        self.pos_rating = dict()
        self.neg_rating = dict()
        self.cal_rating()

    def cal_rating(self):
        pass

    def eval(self, k=10):
        hits = 0
        miss = 0
        dcg = 0
        for user, rating in self.pos_rating.items():
            for p_rating in rating:
                for idx, n_rating in enumerate(self.neg_rating[user]):
                    if idx == k-1:          # 在neg_rating中抽9个，和pos_rating一共10个
                        miss += 1
                        break
                    if p_rating >= n_rating:
                        hits += 1
                        dcg += (1 / np.log2(idx + 2))
                        break

        p = hits / (hits + miss) / k
        recall = hits / (hits + miss)
        f1 = 2 * p * recall / (p + recall)
        ndcg = dcg / (hits + miss)

        print("Model  result (k=%d):\tp: %.4f, recall: %.4f, F1: %.4f, NDCG: %.4f" % (k, p, recall, f1, ndcg))

    def random_eval(self, k=10):
        hits = 0
        miss = 0
        dcg = 0
        s = [i for i in range(self.negNum+1)]
        for user, rating in self.pos_rating.items():
            for _ in rating:
                if 0 in random.sample(s, k):
                    hits += 1
                    dcg += 1 / np.log2(random.randint(2, k+1))
                else:
                    miss += 1

        p = hits / (hits + miss) / k
        recall = hits / (hits + miss)
        f1 = 2 * p * recall / (p + recall)
        ndcg = dcg / (hits + miss)

        print("Random result (k=%d):\tp: %.4f, recall: %.4f, F1: %.4f, NDCG: %.4f" % (k, p, recall, f1, ndcg))


class MARecEval(BaseEval):

    def __init__(self, model, testdata, neg_num=100):
        self.model = model
        self.data = data_loader.BaseData(testdata)
        self.testLoader = DataLoader(self.data, batch_size=128, shuffle=False)
        super(MARecEval, self).__init__(neg_num)

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
                            'hist_lda': batchData['hist_lda'][j],
                            'hist_vector': batchData['hist_vector'][j],
                            'hist_info': batchData['hist_info'][j],
                            'hist_authority': batchData['hist_authority'][j],
                            'hist_participants': batchData['hist_participants'][j],
                            'hist_interact': batchData['hist_interact'][j],
                            'timeDelta': batchData['timeDelta'][j].view(1, -1)
                        }
                pbar.update(len(users))
            pbar.close()

            seq = self.data.user_seq
            pbar = tqdm(total=len(seq))
            for user, s in seq.items():
                data = {}
                if user not in neg_item_hist:
                    continue
                for k, v in neg_item_hist[user].items():
                    if k == 'timeDelta':
                        data[k] = v.expand(self.negNum, -1)
                    else:
                        data[k] = v.expand(self.negNum, -1, -1)

                neg_thread = random.sample(s['test_neg_thread'], self.negNum)
                data['user'] = torch.LongTensor([user] * len(neg_thread))
                data['user_profile'] = torch.LongTensor(self.data.user_profile[user]).expand(self.negNum, -1)
                data['item_lda'] = torch.FloatTensor(self.data.thread_lda[neg_thread])
                data['item_vector'] = torch.FloatTensor(self.data.thread_vector[neg_thread])
                data['item_info'] = torch.FloatTensor(self.data.thread_stat[neg_thread])
                data['item_authority'] = torch.FloatTensor(self.data.user_info[self.data.thread_initiator[neg_thread]])
                data['item_participants'] = torch.FloatTensor(self.data.thread_participants_feature[neg_thread])
                data['item_interact'] = torch.FloatTensor(
                    self.data.social_network[user, self.data.thread_initiator[neg_thread]])

                rs = self.model.forward(data)
                self.neg_rating[user] = np.sort(rs)[::-1]
                pbar.update(1)
            pbar.close()


class DINEval(BaseEval):
    def __init__(self, model, testdata, neg_num=100):
        self.model = model
        self.data = data_loader.BaseData(testdata)
        self.testLoader = DataLoader(self.data, batch_size=128, shuffle=False)
        super().__init__(neg_num)

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
                            'hist_lda': batchData['hist_lda'][j],
                            'hist_vector': batchData['hist_vector'][j],
                            'hist_info': batchData['hist_info'][j],
                            'hist_authority': batchData['hist_authority'][j],
                            'hist_participants': batchData['hist_participants'][j],
                            'hist_interact': batchData['hist_interact'][j],
                        }
                pbar.update(len(users))
            pbar.close()

            seq = self.data.user_seq
            pbar = tqdm(total=len(seq))
            for user, s in seq.items():
                data = {}
                if user not in neg_item_hist:
                    continue
                for k, v in neg_item_hist[user].items():
                    data[k] = v.expand(self.negNum, -1, -1)

                neg_thread = random.sample(s['test_neg_thread'], self.negNum)
                data['user'] = torch.LongTensor([user] * len(neg_thread))
                data['user_profile'] = torch.FloatTensor(self.data.user_profile[data['user']])
                data['item_lda'] = torch.FloatTensor(self.data.thread_lda[neg_thread])
                data['item_vector'] = torch.FloatTensor(self.data.thread_vector[neg_thread])
                data['item_info'] = torch.FloatTensor(self.data.thread_stat[neg_thread])
                data['item_authority'] = torch.FloatTensor(self.data.user_info[self.data.thread_initiator[neg_thread]])
                data['item_participants'] = torch.FloatTensor(self.data.thread_participants_feature[neg_thread])
                data['item_interact'] = torch.FloatTensor(self.data.social_network[user, self.data.thread_initiator[neg_thread]])

                rs = self.model.forward(data).reshape(-1)

                self.neg_rating[user] = np.sort(rs)[::-1]
                pbar.update(1)
            pbar.close()


class OHCRecEval(BaseEval):
    def __init__(self, rating, y_test, neg_num=100):
        self.rating_matrix = rating
        self.y_test = y_test
        super(OHCRecEval, self).__init__(neg_num)

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
            self.neg_rating[i] = sorted(random.sample(list(row), self.negNum), reverse=True)


if __name__ == '__main__':
    # model = torch.load(prefix+"model.pickle")
    # test = data_loader.read_data('test_data')
    # e = MARecEval(model, test)
    #
    # with open(os.path.expanduser("~") + '/Files/OHCRec_data/model.pickle', 'rb') as f:
    #     rating, y_test = pickle.load(f)
    # e = OHCRecEval(rating, y_test)

    model = torch.load(prefix+"DIN_model.pickle")
    test = data_loader.read_data('test_data')
    e = DINEval(model, test)

    e.random_eval(5)
    e.random_eval(10)
    e.random_eval(20)
    e.eval(5)
    e.eval(10)
    e.eval(20)


