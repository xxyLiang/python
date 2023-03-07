import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from settings import *
from tqdm import tqdm
import math
import datetime
from torch.utils.data import DataLoader
from data_prepare import save_data, read_data
from data_loader import UserData


def user_profile_file():
    uid2id, tid2id = read_data('id_transfer')
    cursor.execute("select * from user_list where thread_cnt BETWEEN %d AND %d" % (THREAD_CNT_LOW, THREAD_CNT_HIGH))
    data = pd.DataFrame(
        cursor.fetchall(),
        columns=['uid', 'thread_cnt', 'post_cnt', 'level', 'user_group', 'total_online_hours', 'regis_time',
                 'latest_login_time', 'latest_active_time', 'latest_pub_time', 'prestige', 'points',
                 'wealth', 'visitors', 'friends', 'records', 'logs', 'albums', 'total_posts', 'total_threads',
                 'shares', 'diabetes_type', 'treatment_type', 'gender', 'birthdate', 'habitation']
    )
    data = data[['uid', 'thread_cnt', 'post_cnt', 'level', 'user_group', 'prestige', 'points',
                 'wealth', 'visitors', 'friends', 'records', 'logs', 'albums', 'total_posts', 'total_threads',
                 'shares', 'diabetes_type', 'gender', 'birthdate']]
    data['uid'] = data['uid'].apply(lambda x: uid2id[x])
    data = data.set_index(keys='uid', drop=True).sort_index()
    data['birthdate'].fillna(datetime.date(1900, 1, 1), inplace=True)
    data['birthdate'] = data['birthdate'].apply(lambda x: math.floor(x.year / 10))
    data.fillna(0, inplace=True)

    numeric_attrs = ['prestige', 'thread_cnt', 'post_cnt', 'points', 'wealth', 'visitors', 'friends',
                     'records', 'logs', 'albums',
                     'total_posts', 'total_threads', 'shares']
    nominal_attrs = ['level', 'user_group']
    other_attrs = ['diabetes_type', 'gender', 'birthdate']
    for col in numeric_attrs:
        data[col] = data[col].apply(lambda x: np.log10(x+1) if x >= 0 else 0)
        data[col] /= data[col].max()
    for col in nominal_attrs:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    for col in other_attrs:
        col_values = data[col].drop_duplicates().to_list()
        for k, v in enumerate(col_values):
            data[col + "_" + str(k)] = data[col].eq(v).astype(int)
    data.drop(columns=other_attrs, inplace=True)

    data = data.to_numpy()
    save_data(data, 'DIN_user_profile')
    return data


class DINData(UserData):
    def __init__(self, data, test=False):
        super().__init__(data, test)
        self.user_profile = read_data('DIN_user_profile')

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item['user_profile'] = self.user_profile[item['user']]
        return item


class DIN_PT(nn.Module):
    def __init__(self, user_len, param):
        super().__init__()
        self.userNum = user_len
        self.params = param

        # if torch.cuda.is_available():
        #     self.device = 'cuda'

        self.device = 'cpu'

        self.zero_ = torch.tensor(0).to(self.device)

        self.relu_alpha = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
        self.activate_fc_1 = torch.nn.Linear(294, 36)
        self.PReLU_1 = torch.nn.PReLU()
        self.activate_fc_2 = torch.nn.Linear(36, 1)

        self.fc_1 = torch.nn.Linear(235, 200)
        self.g_PReLu_1 = torch.nn.PReLU()
        self.fc_2 = torch.nn.Linear(200, 80)
        self.g_PReLu_2 = torch.nn.PReLU()
        self.fc_3 = torch.nn.Linear(80, 1)

        self.to(self.device)
        self.grads = {}

    def forward(self, data):
        user_profile = data['user_profile']     # (n, 39)
        hist_lda = data['hist_lda']             # (n, 10, 20)
        hist_vector = data['hist_vector']       # (n, 10, 50)
        hist_stat = data['hist_info']           # (n, 10, 3)
        hist_auth = data['hist_authority']      # (n, 10, 3)
        hist_participants = data['hist_participants']   # (n, 10, 20)
        hist_interact = data['hist_interact']       # (n, 10, 2)
        hist_feature = torch.cat([hist_lda, hist_vector, hist_stat, hist_auth,    # (n, 10, 98)
                                  hist_participants, hist_interact], dim=2).to(torch.float)

        item_lda = data['item_lda']             # (n, 20)
        item_vector = data['item_vector']       # (n, 50)
        item_stat = data['item_info']           # (n, 3)
        item_auth = data['item_authority']      # (n, 3)
        item_participants = data['item_participants']  # (n, 20)
        item_interact = data['item_interact']           # (n, 2)
        item_feature = torch.cat([item_lda, item_vector, item_stat, item_auth,
                                  item_participants, item_interact], dim=1).to(torch.float)        # (n, 98)

        behavior_pool = self.self_activate(hist_feature, item_feature)

        flatten = torch.cat([user_profile, behavior_pool, item_feature], dim=1).to(torch.float)
        fc_1 = self.fc_1(flatten)
        relu_1 = self.g_PReLu_1(fc_1)
        fc_2 = self.fc_2(relu_1)
        relu_2 = self.g_PReLu_2(fc_2)
        fc_3 = self.fc_3(relu_2)
        return fc_3.reshape(-1)

    def self_activate(self, behavior, candidate):
        # candidate_ = candidate.view(-1, 1, candidate.shape[-1]).expand(-1, behavior.shape[1], -1)
        # out_product = torch.einsum('mni,mni->mni', behavior, candidate_)       # (n, 10, 98)
        # concat_ = torch.cat([behavior, out_product, candidate_], dim=2)
        # fc_1 = self.activate_fc_1(concat_)
        # relu_1 = self.PReLU_1(fc_1)
        # fc_2 = self.activate_fc_2(relu_1)
        # weighted_behavior = torch.mul(behavior, fc_2)
        #
        # return weighted_behavior.sum(1)
        return behavior.mean(1)

    def loss(self, data):
        pos_out = self.forward(data).reshape(-1, 1)
        neg_data = self.neg_sample(data, self.params['negNum_train'])
        neg_out = self.forward(neg_data).reshape(-1, self.params['negNum_train'])

        Out = torch.cat((pos_out, neg_out), dim=1)

        criterion = nn.LogSoftmax(dim=1)
        res = criterion(Out)[:, 0]
        loss = torch.mean(res)
        return -loss

    def neg_sample(self, data, negNum):
        neg_data = dict()
        neg_data['user'] = self.__duplicates(data['user'], times=negNum)
        neg_data['user_profile'] = self.__duplicates(data['user_profile'], times=negNum)
        neg_data['hist_lda'] = self.__duplicates(data['hist_lda'], times=negNum)
        neg_data['hist_vector'] = self.__duplicates(data['hist_vector'], times=negNum)
        neg_data['hist_info'] = self.__duplicates(data['hist_info'], times=negNum)
        neg_data['hist_authority'] = self.__duplicates(data['hist_authority'], times=negNum)
        neg_data['hist_participants'] = self.__duplicates(data['hist_participants'], times=negNum)
        neg_data['hist_interact'] = self.__duplicates(data['hist_interact'], times=negNum)
        neg_data['item_lda'] = data['negItem_lda'].reshape((-1, N_TOPICS))
        neg_data['item_vector'] = data['negItem_vector'].reshape((-1, VECTOR_DIM))
        neg_data['item_info'] = data['negItem_info'].reshape(-1, data['negItem_info'].size(-1))
        neg_data['item_authority'] = data['negItem_authority'].reshape(-1, data['negItem_authority'].size(-1))
        neg_data['item_participants'] = data['negItem_participants'].reshape(-1, data['negItem_participants'].size(-1))
        neg_data['item_interact'] = data['negItem_interact'].reshape(-1, data['negItem_interact'].size(-1))
        return neg_data

    @staticmethod
    def __duplicates(arr: torch.tensor, times):
        shape = list(arr.shape)
        dim = len(shape)
        if dim == 1:
            arr_ = arr.reshape(-1, 1).expand(shape[0], times).reshape(-1)
        else:
            shape[0] = -1
            arr_ = arr.tile((times, 1)).reshape(shape)
        return arr_

    def get_grads(self):
        return self.grads

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad

        return hook

    def test_precision(self, testset):
        print('testing...')
        pbar = tqdm(total=len(testset.dataset))
        total_cnt = 0
        success = 0
        with torch.no_grad():
            for i, batchData in enumerate(testset):
                score_pos = self.forward(batchData).reshape(-1)
                neg_batch = self.neg_sample(batchData, self.params['negNum_test'])
                score_neg = self.forward(neg_batch).reshape(-1, self.params['negNum_test'])
                score_neg_max = score_neg.max(dim=1).values

                rs = torch.gt(score_pos, score_neg_max).int()

                success += rs.sum()
                total_cnt += len(score_pos)
                pbar.update(len(score_pos))
        pbar.close()
        print('precision = %.2f %%' % (success / total_cnt * 100))


if __name__ == '__main__':
    params = {
        'lr': 1e-3,
        'w_decay': 0,
        'batch_size': 128,
        'negNum_train': 2,
        'negNum_test': 10,
        'epoch_limit': 5,
    }

    train = read_data('train_data')
    test = read_data('test_data')
    user_seq = read_data('user_sequence')

    trainset = DINData(train)
    trainLoader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True)
    testset = DINData(test, test=True)
    testset.set_negN(params['negNum_test'])
    testLoader = DataLoader(testset, batch_size=16, shuffle=False)

    model = DIN_PT(
        user_len=len(user_seq),
        param=params
    )
    # model = read_data('DIN_model')
    model.to(model.device)
    print('initialization')
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['w_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)

    epoch = 0
    print('start training...')
    while epoch < params['epoch_limit']:
        epoch += 1
        print('Epoch ', str(epoch), ' training...')
        L = len(trainLoader.dataset)
        pbar = tqdm(total=L)

        total_loss = 0
        for i, batchData in enumerate(trainLoader):
            optimizer.zero_grad()

            batch_loss = model.loss(batchData)
            batch_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += batch_loss.clone()
            pbar.update(batchData['user'].shape[0])
        pbar.close()
        scheduler.step()

        print('epoch loss', total_loss)
        model.test_precision(testLoader)

    save_data(model, 'DIN_model')
