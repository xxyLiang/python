import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from auxiliary import ScaledEmbedding, ZeroEmbedding
from lda import N_TOPICS
from data_prepare import HISTORY_THREAD_TAKEN_CNT, PCA_DIM
import torch.nn.functional as F
import data_loader
import data_prepare
from tqdm import tqdm

"""
动机：
1. 个人收益/学习/效能/
2. 成就感/利他主义/互助/声望： 他人的评价
3. 享乐/参与/共鸣/兴趣：
"""


class PT(nn.Module):
    def __init__(self, userLen, params):
        super(PT, self).__init__()
        self.userNum = userLen
        self.params = params

        # if torch.cuda.is_available():
        #     self.device = 'cuda'

        self.device = 'cpu'

        self.cossim = nn.CosineSimilarity(dim=1)

        self.global_tDecay_lamda = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.global_tDecay_lamda.weight.data += 0.8
        self.global_tDecay_lamda.requires_grad_ = False
        self.user_tDecay_lamda = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.user_tDecay_lamda.weight.data.uniform_(-0.5, 0.5)

        self.global_topicSim_ref = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.global_topicSim_ref.weight.data += 0.05
        self.user_topicSim_ref = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.user_topicSim_ref.weight.data.uniform_(-0.05, 0.05)

        self.global_topicSim_weight = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.global_topicSim_weight.weight.data += 0.33
        self.global_topicSim_weight.requires_grad_ = False
        self.user_topicSim_weight = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.user_topicSim_weight.weight.data.uniform_(-0.2, 0.2)

        self.global_x_lamda = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.global_x_lamda.weight.data += 1.5
        self.global_x_lamda.requires_grad_ = False
        self.user_x_lamda = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.user_x_lamda.weight.data.uniform_(-0.5, 0.5)

        self.global_x_alpha = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.global_x_alpha.weight.data += 0.6
        self.global_x_alpha.requires_grad_ = False
        self.user_x_alpha = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.user_x_alpha.weight.data.uniform_(-0.1, 0.1)

        self.global_x_beta = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.global_x_beta.weight.data += 0.55
        self.global_x_beta.requires_grad_ = False
        self.user_x_beta = ZeroEmbedding(userLen, 1).to(self.device).to(torch.float)
        self.user_x_beta.weight.data.uniform_(-0.1, 0.1)

        self.topic_fc_1 = nn.Linear(2 * PCA_DIM, 2 * PCA_DIM, dtype=torch.float64).to(self.device)
        self.topic_fc_2 = nn.Linear(2 * PCA_DIM, 1, dtype=torch.float64).to(self.device)

        self.wide_weight = ZeroEmbedding(1, 1).to(self.device).to(torch.float)
        self.wide_weight.weight.data += 0.5

        self.to(self.device)
        self.grads = {}

    def forward(self, data):
        # return self.__deep_forward(data)
        user = data['user']             # (n, 1)
        hist_lda = data['hist_lda']     # (n, 10, 20)
        timeDelta = data['timeDelta']   # (n, 10)
        item_lda = data['item_lda']     # (n, 20)

        # Deep model
        # deep_model_output = self.__deep_forward(data)

        # Wide model
        tDecay_lamda = self.global_tDecay_lamda(torch.tensor(0).to(self.device)) + self.user_tDecay_lamda(user)
        weight = timeDelta.mul(-tDecay_lamda).exp().reshape((-1, HISTORY_THREAD_TAKEN_CNT, 1))
        weighted_hist = hist_lda.mul(weight).sum(1)
        weighted_hist = F.softmax(weighted_hist, dim=1)

        topicSim = self.cossim(weighted_hist, item_lda).reshape((-1, 1))
        topicSim_ref = self.global_topicSim_ref(torch.tensor(0)).to(self.device) + self.user_topicSim_ref(user)
        topicSim_weight = self.global_topicSim_weight(torch.tensor(0).to(self.device)) + self.user_topicSim_weight(user)

        x = torch.abs(torch.sub(topicSim, topicSim_ref).mul(topicSim_weight))

        lamda = self.global_x_lamda(torch.tensor(0).to(self.device)) + self.user_x_lamda(user)
        alpha = self.global_x_alpha(torch.tensor(0).to(self.device)) + self.user_x_alpha(user)
        beta = self.global_x_beta(torch.tensor(0).to(self.device)) + self.user_x_beta(user)

        x_binary_pos = torch.gt(x, torch.FloatTensor([0]).to(self.device)).to(torch.float)
        x_binary_neg = torch.ones_like(x).to(self.device) - x_binary_pos

        v_exp = torch.mul(alpha, x_binary_pos) + torch.mul(beta, x_binary_neg)
        v = x.pow(v_exp)
        v_coef = x_binary_pos - torch.mul(lamda, x_binary_neg)
        value = torch.mul(v, v_coef).to(self.device)

        wide_model_output = value.mul(self.wide_weight(torch.tensor(0).to(self.device)))

        return wide_model_output

    def __deep_forward(self, data):
        hist_bert = data['hist_bert']    # (n, 10, 50)
        item_bert = data['item_bert']   # (n, 50)

        mean_hist_bert = torch.mean(hist_bert, dim=1)       # (n, 50)

        bert_ = torch.cat([mean_hist_bert, item_bert], dim=1)
        fc_1_output = self.topic_fc_1(bert_)
        fc_2_output = self.topic_fc_2(fc_1_output)

        deep_model_output = fc_2_output.mul(1 - self.wide_weight(torch.tensor(0).to(self.device)))

        return deep_model_output

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
        neg_data['hist_lda'] = self.__duplicates(data['hist_lda'], times=negNum)
        neg_data['hist_bert'] = self.__duplicates(data['hist_bert'], times=negNum)
        neg_data['timeDelta'] = self.__duplicates(data['timeDelta'], times=negNum)
        neg_data['item_lda'] = data['negItem_lda'].reshape((-1, N_TOPICS))
        neg_data['item_bert'] = data['negItem_bert'].reshape((-1, PCA_DIM))
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


if __name__ == '__main__':
    params = {
        'lr': 0.1,
        'w_decay': 1e-2,
        'batch_size': 256,
        'negNum_train': 2,
        'negNum_test': 10,
        'epoch_limit': 3,
    }

    train = data_loader.read_data('train_data')
    test = data_loader.read_data('test_data')
    user_seq = data_loader.read_data('user_sequence')
    lda = data_loader.read_data('thread_lda')
    bert = data_loader.read_data('thread_bert')

    trainset = data_loader.UserData(train, user_seq, lda, bert)
    trainLoader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True)
    testset = data_loader.UserData(test, user_seq, lda, bert)
    testset.set_negN(10)
    testLoader = DataLoader(testset, batch_size=1, shuffle=False)

    model = PT(userLen=len(user_seq), params=params)
    print('initialization', model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['w_decay'])

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

            if i == 0:
                total_loss = batch_loss.clone()
            else:
                total_loss += batch_loss.clone()
            pbar.update(batchData['user'].shape[0])
        pbar.close()
        print('testing...')
        L = len(testLoader.dataset)
        pbar = tqdm(total=L)
        total_cnt = 0
        success = 0
        with torch.no_grad():
            for i, batchData in enumerate(testLoader):
                score_pos = model.forward(batchData)[0][0]
                neg_batch = model.neg_sample(batchData, params['negNum_test'])
                neg_ = model.forward(neg_batch).reshape(-1, params['negNum_test']).reshape(-1)

                gt = False
                for j in neg_:
                    if j > score_pos:
                        gt = True
                        break

                if gt is False:
                    success += 1
                total_cnt += 1
                pbar.update(1)
        pbar.close()
        print('precision = %f' % (success/total_cnt))

        print('epoch loss', total_loss)

    data_prepare.save_data(model, 'model')


