import torch
import torch.nn as nn
from settings import *
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from data_prepare import save_data, read_data
from data_loader import UserData
from auxiliary import EarlyStopping


class DIN_PT(nn.Module):
    def __init__(self, user_len, param):
        super().__init__()
        self.userNum = user_len
        self.params = param

        self.device = 'cpu'
        # if torch.cuda.is_available():
        #     self.device = 'cuda'

        self.zero_ = torch.tensor(0).to(self.device)

        self.activate_fc_1 = torch.nn.Linear(294, 36, dtype=torch.float64)
        self.PReLU_1 = torch.nn.PReLU(dtype=torch.float64)
        self.activate_fc_2 = torch.nn.Linear(36, 1, dtype=torch.float64)

        self.fc_1 = torch.nn.Linear(231, 200, dtype=torch.float64)
        self.g_PReLu_1 = torch.nn.PReLU(dtype=torch.float64)
        self.fc_2 = torch.nn.Linear(200, 80, dtype=torch.float64)
        self.g_PReLu_2 = torch.nn.PReLU(dtype=torch.float64)
        self.fc_3 = torch.nn.Linear(80, 1, dtype=torch.float64)

        self.dice_alpha = torch.tensor(0.2, requires_grad=True)

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
                                  hist_participants, hist_interact], dim=2).to(torch.float64)

        item_lda = data['item_lda']             # (n, 20)
        item_vector = data['item_vector']       # (n, 50)
        item_stat = data['item_info']           # (n, 3)
        item_auth = data['item_authority']      # (n, 3)
        item_participants = data['item_participants']  # (n, 20)
        item_interact = data['item_interact']           # (n, 2)
        item_feature = torch.cat([item_lda, item_vector, item_stat, item_auth,
                                  item_participants, item_interact], dim=1).to(torch.float64)        # (n, 98)

        behavior_pool = self.self_activate(hist_feature, item_feature)

        flatten = torch.cat([user_profile, behavior_pool, item_feature], dim=1).to(torch.float64)
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
        # # relu_1 = self.dice(fc_1)
        # fc_2 = self.activate_fc_2(relu_1)
        # weighted_behavior = torch.mul(behavior, fc_2).sum(1)

        weighted_behavior = behavior.sum(1)
        return weighted_behavior
        # return behavior.sum(1)

    def dice(self, tensor):
        m = tensor.mean().detach()
        s = tensor.std().detach()
        ps = 1/(1+torch.exp(-(tensor-m)/torch.sqrt(s+1e-8)))
        a = tensor * (ps + self.dice_alpha * (1-ps))
        return a

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

    def test_precision(self, testset, verbose=False):
        total_cnt = 0
        success = 0
        ndcg = 0
        negNum = self.params['negNum_test']
        with torch.no_grad():
            for i, batchData in enumerate(testset):
                score_pos = self.forward(batchData).view(-1, 1)
                neg_batch = self.neg_sample(batchData, negNum)
                score_neg = self.forward(neg_batch).reshape(-1, negNum)

                rs = torch.lt(score_pos, score_neg).int().sum(1)
                success += rs.eq(0).int().sum()
                rs = 1 / torch.log2(rs+2)
                ndcg += rs.sum()
                total_cnt += len(score_pos)
            p = success / total_cnt
            ndcg = ndcg/total_cnt
        if verbose:
            print('precision = %.4f, NDCG=%.4f' % (p, ndcg))
        return p, ndcg


if __name__ == '__main__':
    params = {
        'lr': 1e-2,
        'w_decay': 0,
        'batch_size': 128,
        'negNum_train': 2,
        'negNum_test': 10,
        'epoch_limit': 7,
    }
    print('initialization')

    train = read_data('train_data')
    validate = read_data('validate_data')
    user_seq = read_data('user_sequence')

    trainSet = UserData(train)
    trainSet.set_negN((params['negNum_train']))
    trainLoader = DataLoader(trainSet, batch_size=params['batch_size'], shuffle=True)

    validateSet = UserData(validate, test=True)
    validateSet.set_negN(params['negNum_test'])
    validateLoader = DataLoader(validateSet, batch_size=16, shuffle=True)

    model = DIN_PT(
        user_len=len(user_seq),
        param=params
    )
    model.to(model.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['w_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    epoch = 0
    es = EarlyStopping(patience=3, delta=0.002, path=prefix+"DIN_model.pickle")
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
            if (i + 1) % 350 == 0:
                p, ndcg = model.test_precision(validateLoader)
                if es(ndcg, model):
                    exit(0)
        pbar.close()
        scheduler.step()

        print('epoch loss', total_loss)
        model.test_precision(validateLoader, verbose=True)
        if es(ndcg, model):
            exit(0)

    torch.save(model, prefix + "DIN_model.pickle")
