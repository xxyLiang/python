import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from auxiliary import KEmbedding, UniformEmbedding, EarlyStopping
from settings import *
from data_loader import *
import data_prepare
from tqdm import tqdm

"""
动机：
1. 个人收益/学习/效能/
2. 成就感/利他主义/互助/声望： 跟帖人数、发帖者的权威性（显性、用户能感受到的指标）
3. 享乐/参与/共鸣/兴趣：用户网络、与发帖者的关系（出入度、最短距离、交互指数）
"""


class PT(nn.Module):
    def __init__(self, user_len, param, lda_dist, vector_dist):
        super(PT, self).__init__()
        self.userNum = user_len
        self.params = param

        self.device = 'cpu'
        # if torch.cuda.is_available():
        #     self.device = 'cuda'

        self.cossim_1 = nn.CosineSimilarity(dim=1).to(self.device)
        self.cossim_2 = nn.CosineSimilarity(dim=2).to(self.device)
        self.zero_ = torch.tensor(0).to(self.device)
        self.criterion = nn.LogSoftmax(dim=1)

        # 用户的lda/vector偏好特征
        self.know_lda_pref_user = nn.Embedding(user_len, N_TOPICS).to(self.device).to(torch.float64)
        self.know_lda_pref_user.weight.data = torch.FloatTensor(lda_dist).to(self.device)

        self.know_vector_pref_user = nn.Embedding(user_len, VECTOR_DIM).to(self.device).to(torch.float64)
        self.know_vector_pref_user.weight.data = torch.FloatTensor(vector_dist).to(self.device)
        self.lda_gain_ref_user = nn.Embedding(user_len, N_TOPICS).to(self.device).to(torch.float64)
        self.lda_gain_ref_user.weight.data = torch.FloatTensor(lda_dist * 5).to(self.device)

        # 总体的时间衰减参数
        self.time_decay_lamda_global = KEmbedding(1, 1, 0.8).to(self.device).to(torch.float64)
        self.time_decay_lamda_global.requires_grad_ = False
        self.time_decay_lamda_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)

        # 信息收益部分
        self.know_info_part_weight_global = KEmbedding(1, 3, 0.33).to(self.device).to(torch.float64)
        self.know_info_part_weight_user = UniformEmbedding(user_len, 3, -0.05, 0.05).to(self.device).to(torch.float64)

        self.know_topicSim_weight_global = KEmbedding(1, 1, 0.33).to(self.device).to(torch.float64)
        self.know_topicSim_weight_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)
        self.know_contentSim_weight_global = KEmbedding(1, 1, 0.33).to(self.device).to(torch.float64)
        self.know_contentSim_weight_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)
        self.know_info_weight_user = KEmbedding(1, 1, 0.34).to(self.device).to(torch.float64)
        self.know_info_weight_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)

        # 社区感部分
        self.com_participant_pref_user = nn.Embedding(user_len, N_TOPICS).to(torch.float64).to(self.device)
        self.com_participant_pref_user.weight.data = torch.FloatTensor(lda_dist)

        self.com_interact_apart_weight_global = KEmbedding(1, 2, 0.5).to(self.device).to(torch.float64)
        self.com_interact_apart_weight_user = UniformEmbedding(user_len, 2, -0.05, 0.05).to(self.device).to(torch.float64)
        self.com_auth_apart_weight_global = KEmbedding(1, 3, 0.33).to(self.device).to(torch.float64)
        self.com_auth_apart_weight_user = UniformEmbedding(user_len, 3, -0.05, 0.05).to(self.device).to(torch.float64)

        self.com_participant_weight_global = KEmbedding(1, 1, 0.33).to(self.device).to(torch.float64)
        self.com_participant_weight_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)
        self.com_interact_weight_global = KEmbedding(1, 1, 0.33).to(self.device).to(torch.float64)
        self.com_interact_weight_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)
        self.com_auth_weight_global = KEmbedding(1, 1, 0.33).to(self.device).to(torch.float64)
        self.com_auth_weight_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)

        # 收益函数参数
        self.x_ref_global = KEmbedding(1, 1, 0.8).to(self.device).to(torch.float64)
        self.x_ref_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)

        self.x_lamda_global = KEmbedding(1, 1, 1.5).to(self.device).to(torch.float64)
        self.x_lamda_user = UniformEmbedding(user_len, 1, -0.3, 0.3).to(self.device).to(torch.float64)
        self.x_alpha_global = KEmbedding(1, 1, 0.6).to(self.device).to(torch.float64)
        self.x_alpha_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)
        self.x_beta_global = KEmbedding(1, 1, 0.55).to(self.device).to(torch.float64)
        self.x_beta_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)
        self.x_lamda_global.requires_grad_ = False
        self.x_alpha_global.requires_grad_ = False
        self.x_beta_global.requires_grad_ = False

        self.x_bias = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)

        self.fc1 = nn.Linear(95, 16, dtype=torch.float64)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2 = nn.Linear(16, 1, dtype=torch.float64)
        self.fc2.weight.data.uniform_(-0.1, 0.1)

        self.relu = nn.ELU()

        self.to(self.device)
        self.grads = {}

    def forward(self, data):

        hist_topic_gain = self.hist_topic_gain(data)  # 历史收益部分
        lda_gain_ref = self.lda_gain_ref_user(data['user'])
        gain_lda_diff = torch.sub(lda_gain_ref, hist_topic_gain)  # 历史收益与用户预期的差异
        curr_item_topic_gain = self.curr_topic_gain(data)  # 当前帖子的收益

        cross = torch.mul(gain_lda_diff, curr_item_topic_gain)
        dot_ = self.fc1(torch.cat([data['user_profile'], gain_lda_diff, cross, curr_item_topic_gain], dim=1))
        # dot_ = self.relu(dot_)
        dot_ = self.fc2(dot_).view(-1)
        return dot_

    def hist_topic_gain(self, data):
        user = data['user'].to(self.device)                 # (n, 1)
        hist_lda = data['hist_lda'].to(self.device)         # (n, 10, 20)
        hist_vector = data['hist_vector'].to(self.device)   # (n, 10, 50)
        hist_info = data['hist_info'].to(self.device)       # (n, 10)
        hist_auth = data['hist_authority'].to(self.device)
        hist_participants = data['hist_participants'].to(self.device)
        hist_interact = data['hist_interact'].to(self.device)
        timeDelta = data['timeDelta'].to(self.device)       # (n, 10)

        total_hist_gain = self.total_gain(user, hist_lda, hist_vector, hist_info, hist_participants, hist_interact, hist_auth)

        time_decay_lamda = self.time_decay_lamda_global(self.zero_) + self.time_decay_lamda_user(user)
        weight = timeDelta.mul(-time_decay_lamda).exp()
        weighted_hist_gain = torch.mul(total_hist_gain, weight)
        hist_topic_gain = torch.mul(hist_lda, weighted_hist_gain.view((-1, hist_lda.size(1), 1)))
        hist_topic_gain_sum = hist_topic_gain.sum(1)

        return hist_topic_gain_sum

    def curr_topic_gain(self, data):
        user = data['user'].to(self.device)                 # (n, 1)
        item_lda = data['item_lda'].to(self.device)         # (n, 20)
        item_vector = data['item_vector'].to(self.device)   # (n, 50)
        item_info = data['item_info'].to(self.device)       # (n, 3)
        item_auth = data['item_authority'].to(self.device)
        item_participants = data['item_participants'].to(self.device)
        item_interact = data['item_interact'].to(self.device)

        curr_item_gain = self.total_gain(
            user,
            item_lda.view(-1, 1, N_TOPICS),
            item_vector.view(-1, 1, VECTOR_DIM),
            item_info.view(-1, 1, 3),
            item_participants.view(-1, 1, N_TOPICS),
            item_interact.view(-1, 1, 2),
            item_auth.view(-1, 1, 3)
        )

        curr_item_topic_gain = torch.mul(curr_item_gain, item_lda)

        return curr_item_topic_gain

    def total_gain(self, user, lda, vector, info, participants, interact, authority):
        user_lda_pref = self.know_lda_pref_user(user)  # shape: (n, 20)
        lda_gain = self.cossim_2(user_lda_pref.view((-1, 1, N_TOPICS)), lda)

        user_vector_pref = self.know_vector_pref_user(user)
        vector_gain = self.cossim_2(user_vector_pref.view((-1, 1, VECTOR_DIM)), vector)

        info_part_weight = self.know_info_part_weight_global(self.zero_) + self.know_info_part_weight_user(user)
        info_gain = torch.mul(info_part_weight.view(-1, 1, 3), info).sum(2)

        user_participant_pref = self.com_participant_pref_user(user)
        participant_similarity = self.cossim_2(user_participant_pref.view((-1, 1, N_TOPICS)), participants)

        interact_apart_weight = self.com_interact_apart_weight_global(self.zero_) + self.com_interact_apart_weight_user(
            user)
        interact_gain = torch.mul(interact, interact_apart_weight.view(-1, 1, 2)).sum(2)

        auth_apart_weight = self.com_auth_apart_weight_global(self.zero_) + self.com_auth_apart_weight_user(user)
        auth_gain = torch.mul(authority, auth_apart_weight.view(-1, 1, 3)).sum(2)

        topic_sim_weight = self.know_topicSim_weight_global(self.zero_) + self.know_topicSim_weight_user(user)
        content_sim_weight = self.know_contentSim_weight_global(self.zero_) + self.know_contentSim_weight_user(user)
        info_weight = self.know_info_weight_user(self.zero_) + self.know_info_weight_user(user)
        auth_weight = self.com_auth_weight_global(self.zero_) + self.com_auth_weight_user(user)
        participant_weight = self.com_participant_weight_global(self.zero_) + self.com_participant_weight_user(user)
        interact_weight = self.com_interact_weight_global(self.zero_) + self.com_interact_weight_user(user)

        x_ref = self.x_ref_global(self.zero_) + self.x_ref_user(user)

        x = torch.mul(lda_gain, topic_sim_weight) \
            + torch.mul(vector_gain, content_sim_weight) \
            + torch.mul(info_gain, info_weight) \
            + torch.mul(participant_similarity, participant_weight) \
            + torch.mul(interact_gain, interact_weight) \
            + torch.mul(auth_gain, auth_weight) \
            - x_ref

        x_binary_pos = torch.gt(x, self.zero_).to(torch.float64)
        x_binary_neg = torch.ones_like(x).to(self.device) - x_binary_pos

        x = torch.abs(x)

        lamda = self.x_lamda_global(self.zero_) + self.x_lamda_user(user)
        alpha = self.x_alpha_global(self.zero_) + self.x_alpha_user(user)
        beta = self.x_beta_global(self.zero_) + self.x_beta_user(user)

        v_exp = torch.mul(alpha, x_binary_pos) + torch.mul(beta, x_binary_neg)
        v = x.pow(v_exp)
        v_coef = x_binary_pos - torch.mul(lamda, x_binary_neg)
        value = torch.mul(v, v_coef) + self.x_bias(user)

        return value

    def loss(self, data):
        pos_out = self.forward(data).reshape(-1, 1)
        neg_data = self.neg_sample(data, self.params['negNum_train'])
        neg_out = self.forward(neg_data).reshape(-1, self.params['negNum_train'])

        Out = torch.cat((pos_out, neg_out), dim=1)
        res = self.criterion(Out)[:, 0]
        loss = torch.mean(res)
        return -loss

    def neg_sample(self, data, negNum):
        neg_data = dict()
        neg_data['user'] = self.__duplicates(data['user'], times=negNum)
        neg_data['hist_lda'] = self.__duplicates(data['hist_lda'], times=negNum)
        neg_data['hist_vector'] = self.__duplicates(data['hist_vector'], times=negNum)
        neg_data['hist_info'] = self.__duplicates(data['hist_info'], times=negNum)
        neg_data['hist_authority'] = self.__duplicates(data['hist_authority'], times=negNum)
        neg_data['hist_participants'] = self.__duplicates(data['hist_participants'], times=negNum)
        neg_data['hist_interact'] = self.__duplicates(data['hist_interact'], times=negNum)
        neg_data['timeDelta'] = self.__duplicates(data['timeDelta'], times=negNum)
        neg_data['item_lda'] = data['negItem_lda'].reshape((-1, N_TOPICS))
        neg_data['item_vector'] = data['negItem_vector'].reshape((-1, VECTOR_DIM))
        neg_data['item_info'] = data['negItem_info']
        neg_data['item_authority'] = data['negItem_authority']
        neg_data['item_participants'] = data['negItem_participants']
        neg_data['item_interact'] = data['negItem_interact']
        neg_data['user_profile'] = self.__duplicates(data['user_profile'], times=negNum)
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
        'epoch_limit': 8,
    }
    print('initialization')
    train = read_data('train_data')
    validate = read_data('validate_data')
    user_seq = read_data('user_sequence')
    user_dist = read_data('user_dist')

    trainSet = UserData(train)
    trainSet.set_negN((params['negNum_train']))
    trainLoader = DataLoader(trainSet, batch_size=params['batch_size'], shuffle=True)

    validateSet = UserData(validate, test=True)
    validateSet.set_negN((params['negNum_test']))
    validateLoader = DataLoader(validateSet, batch_size=16, shuffle=True)

    model = PT(
        user_len=len(user_seq),
        param=params,
        lda_dist=user_dist['lda_dist'],
        vector_dist=user_dist['vector_dist']
    )
    model.to(model.device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'], weight_decay=params['w_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    epoch = 0
    es = EarlyStopping(delta=0.01)
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
            if (i+1) % 350 == 0:
                p, ndcg = model.test_precision(validateLoader)
                if es(ndcg, model):
                    exit(0)
        pbar.close()
        scheduler.step()

        print('epoch loss', total_loss)
        p, ndcg = model.test_precision(validateLoader, verbose=True)
        if es(ndcg, model):
            exit(0)

    #     model.test_precision(validateLoader)
    torch.save(model, prefix+"model.pickle")
