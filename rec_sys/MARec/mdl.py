import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from auxiliary import KEmbedding, UniformEmbedding
from lda import N_TOPICS
from data_prepare import HISTORY_THREAD_TAKEN_CNT, VECTOR_DIM
import data_loader
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

        # if torch.cuda.is_available():
        #     self.device = 'cuda'

        self.device = 'cpu'

        self.cossim_1 = nn.CosineSimilarity(dim=1).to(self.device)
        self.cossim_2 = nn.CosineSimilarity(dim=2).to(self.device)
        self.zero_ = torch.tensor(0).to(self.device)

        # 三类收益的权重
        self.know_weight_global = KEmbedding(1, 1, 1).to(self.device).to(torch.float64)
        self.know_weight_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)
        self.rep_weight_global = KEmbedding(1, 1, 0.5).to(self.device).to(torch.float64)
        self.rep_weight_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)
        self.com_weight_global = KEmbedding(1, 1, 1).to(self.device).to(torch.float64)
        self.com_weight_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)

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
        self.time_decay_lamda_user = UniformEmbedding(user_len, 1, -0.5, 0.5).to(self.device).to(torch.float64)

        # 信息收益部分
        self.know_info_part_weight_global = KEmbedding(1, 3, 0.33).to(self.device).to(torch.float64)
        self.know_info_part_weight_user = UniformEmbedding(user_len, 3, -0.1, 0.1).to(self.device).to(torch.float64)

        self.know_topicSim_weight_global = KEmbedding(1, 1, 0.33).to(self.device).to(torch.float64)
        self.know_topicSim_weight_user = UniformEmbedding(user_len, 1, -0.2, 0.2).to(self.device).to(torch.float64)
        self.know_contentSim_weight_global = KEmbedding(1, 1, 0.33).to(self.device).to(torch.float64)
        self.know_contentSim_weight_user = UniformEmbedding(user_len, 1, -0.2, 0.2).to(self.device).to(torch.float64)
        self.know_info_weight_user = KEmbedding(1, 1, 0.34).to(self.device).to(torch.float64)
        self.know_info_weight_user = UniformEmbedding(user_len, 1, -0.2, 0.2).to(self.device).to(torch.float64)

        self.know_x_ref_global = KEmbedding(1, 1, 0.3).to(self.device).to(torch.float64)
        self.know_x_ref_global.requires_grad_ = False
        self.know_x_ref_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)

        self.know_x_lamda_global = KEmbedding(1, 1, 1.5).to(self.device).to(torch.float64)
        self.know_x_lamda_user = UniformEmbedding(user_len, 1, -0.5, 0.5).to(self.device).to(torch.float64)
        self.know_x_alpha_global = KEmbedding(1, 1, 0.6).to(self.device).to(torch.float64)
        self.know_x_alpha_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)
        self.know_x_beta_global = KEmbedding(1, 1, 0.55).to(self.device).to(torch.float64)
        self.know_x_beta_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)
        self.know_x_lamda_global.requires_grad_ = False
        self.know_x_alpha_global.requires_grad_ = False
        self.know_x_beta_global.requires_grad_ = False

        # 社区感部分
        self.com_participant_pref_user = nn.Embedding(user_len, N_TOPICS).to(torch.float64).to(self.device)
        self.com_participant_pref_user.weight.data = torch.FloatTensor(lda_dist)

        self.com_participant_weight_global = KEmbedding(1, 1, 0.5).to(self.device).to(torch.float64)
        self.com_participant_weight_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)
        self.com_interact_apart_weight_global = KEmbedding(1, 2, 0.5).to(self.device).to(torch.float64)
        self.com_interact_apart_weight_user = UniformEmbedding(user_len, 2, -0.05, 0.05).to(self.device).to(torch.float64)
        self.com_interact_weight_global = KEmbedding(1, 1, 0.5).to(self.device).to(torch.float64)
        self.com_interact_weight_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)

        self.com_x_ref_global = KEmbedding(1, 1, 0.3).to(self.device).to(torch.float64)
        self.com_x_ref_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)

        self.com_x_lamda_global = KEmbedding(1, 1, 1.5).to(self.device).to(torch.float64)
        self.com_x_lamda_user = UniformEmbedding(user_len, 1, -0.5, 0.5).to(self.device).to(torch.float64)
        self.com_x_alpha_global = KEmbedding(1, 1, 0.6).to(self.device).to(torch.float64)
        self.com_x_alpha_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)
        self.com_x_beta_global = KEmbedding(1, 1, 0.55).to(self.device).to(torch.float64)
        self.com_x_beta_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)
        self.com_x_lamda_global.requires_grad_ = False
        self.com_x_alpha_global.requires_grad_ = False
        self.com_x_beta_global.requires_grad_ = False

        # 声望部分
        # self.rep_SN_weight_global = KEmbedding(1, 4, 0)
        # self.rep_SN_weight_global.weight.data = torch.FloatTensor([[0.2, 0.15, 0.1, 0.6]])
        # self.rep_SN_weight_user = UniformEmbedding(user_len, 4, -0.05, 0.05)
        #
        # self.rep_x_lamda_global = KEmbedding(1, 1, 1.5).to(self.device).to(torch.float64)
        # self.rep_x_lamda_global.requires_grad_ = False
        # self.rep_x_lamda_user = UniformEmbedding(user_len, 1, -0.5, 0.5).to(self.device).to(torch.float64)
        # self.rep_x_alpha_global = KEmbedding(1, 1, 0.6).to(self.device).to(torch.float64)
        # self.rep_x_alpha_global.requires_grad_ = False
        # self.rep_x_alpha_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)
        # self.rep_x_beta_global = KEmbedding(1, 1, 0.55).to(self.device).to(torch.float64)
        # self.rep_x_beta_global.requires_grad_ = False
        # self.rep_x_beta_user = UniformEmbedding(user_len, 1, -0.1, 0.1).to(self.device).to(torch.float64)
        #
        # self.rep_x_ref_global = KEmbedding(1, 1, k=0.3).to(self.device).to(torch.float64)
        # self.rep_x_ref_user = UniformEmbedding(user_len, 1, -0.05, 0.05).to(self.device).to(torch.float64)

        self.topic_fc_1 = nn.Linear(2 * (VECTOR_DIM + N_TOPICS), 200, dtype=torch.float64).to(self.device)
        self.topic_fc_2 = nn.Linear(200, 1, dtype=torch.float64).to(self.device)

        self.wide_weight = KEmbedding(1, 1, 0.5).to(self.device).to(torch.float64)

        self.to(self.device)
        self.grads = {}

    def forward(self, data):
        # return self.__deep_forward(data)

        hist_topic_gain = self.hist_topic_gain(data)  # 历史收益部分
        gain_lda_diff = torch.sub(self.lda_gain_ref_user(data['user'].to(self.device)), hist_topic_gain)  # 历史收益与用户预期的差异
        curr_item_topic_gain = self.curr_topic_gain(data)  # 当前帖子的收益

        dot_ = torch.mul(gain_lda_diff, curr_item_topic_gain).sum(1)

        return dot_

    def hist_topic_gain(self, data):
        user = data['user'].to(self.device)  # (n, 1)
        hist_lda = data['hist_lda'].to(self.device)  # (n, 10, 20)
        hist_vector = data['hist_vector'].to(self.device)  # (n, 10, 50)
        hist_info = data['hist_info'].to(self.device)  # (n, 10)
        hist_participants = data['hist_participants'].to(self.device)
        hist_interact = data['hist_interact'].to(self.device)
        timeDelta = data['timeDelta'].to(self.device)  # (n, 10)

        hist_knowledge_gain = self.knowledge_gain(user, hist_lda, hist_vector, hist_info)  # (n, 10)
        hist_com_gain = self.com_gain(user, hist_participants, hist_interact)

        knowledge_weight = self.know_weight_global(self.zero_) + self.know_weight_user(user)
        com_weight = self.com_weight_global(self.zero_) + self.com_weight_user(user)
        total_hist_gain = torch.mul(hist_knowledge_gain, knowledge_weight) \
            + torch.mul(hist_com_gain, com_weight)

        time_decay_lamda = self.time_decay_lamda_global(self.zero_) + self.time_decay_lamda_user(user)
        weight = timeDelta.mul(-time_decay_lamda).exp()
        weighted_hist_gain = torch.mul(total_hist_gain, weight)
        hist_topic_gain = torch.mul(hist_lda, weighted_hist_gain.view((-1, HISTORY_THREAD_TAKEN_CNT, 1)))
        hist_topic_gain_sum = hist_topic_gain.sum(1)

        return hist_topic_gain_sum

    def curr_topic_gain(self, data):
        user = data['user'].to(self.device)  # (n, 1)
        item_lda = data['item_lda'].to(self.device)  # (n, 20)
        item_vector = data['item_vector'].to(self.device)  # (n, 50)
        item_info = data['item_info'].to(self.device)  # (n, 3)
        item_participants = data['item_participants'].to(self.device)
        item_interact = data['item_interact'].to(self.device)

        curr_item_knowledge_gain = self.knowledge_gain(
            user,
            item_lda.view(-1, 1, N_TOPICS),
            item_vector.view(-1, 1, VECTOR_DIM),
            item_info.view(-1, 1, 3)
        )
        curr_item_com_gain = self.com_gain(
            user,
            item_participants.view(-1, 1, N_TOPICS),
            item_interact.view(-1, 1, 2)
        )

        knowledge_weight = self.know_weight_global(self.zero_) + self.know_weight_user(user)
        com_weight = self.com_weight_global(self.zero_) + self.com_weight_user(user)

        curr_item_gain = torch.mul(curr_item_knowledge_gain, knowledge_weight) \
            + torch.mul(curr_item_com_gain, com_weight)

        curr_item_topic_gain = torch.mul(curr_item_gain, item_lda)

        return curr_item_topic_gain

    def knowledge_gain(self, user, lda, vector, info):
        user_lda_pref = self.know_lda_pref_user(user)  # shape: (n, 20)
        lda_gain = self.cossim_2(user_lda_pref.view((-1, 1, N_TOPICS)), lda)

        user_vector_pref = self.know_vector_pref_user(user)
        vector_gain = self.cossim_2(user_vector_pref.view((-1, 1, VECTOR_DIM)), vector)

        info_part_weight = self.know_info_part_weight_global(self.zero_) + self.know_info_part_weight_user(user)
        info_gain = torch.mul(info_part_weight.view(-1, 1, 3), info).sum(2)

        topic_sim_weight = self.know_topicSim_weight_global(self.zero_) + self.know_topicSim_weight_user(user)
        content_sim_weight = self.know_contentSim_weight_global(self.zero_) + self.know_contentSim_weight_user(user)
        info_weight = self.know_info_weight_user(self.zero_) + self.know_info_weight_user(user)

        x_ref = self.know_x_ref_global(self.zero_) + self.know_x_ref_user(user)

        # 用于padding的lda和vector是np.zeros()，到这里计算出来的x就是0，backward时就会出现nan，所以加上eps
        x = torch.mul(lda_gain, topic_sim_weight) \
            + torch.mul(vector_gain, content_sim_weight) \
            + torch.mul(info_gain, info_weight) \
            - x_ref

        x_binary_pos = torch.gt(x, self.zero_).to(torch.float64)
        x_binary_neg = torch.ones_like(x).to(self.device) - x_binary_pos

        x = torch.abs(x)

        lamda = self.know_x_lamda_global(self.zero_) + self.know_x_lamda_user(user)
        alpha = self.know_x_alpha_global(self.zero_) + self.know_x_alpha_user(user)
        beta = self.know_x_beta_global(self.zero_) + self.know_x_beta_user(user)

        v_exp = torch.mul(alpha, x_binary_pos) + torch.mul(beta, x_binary_neg)
        v = x.pow(v_exp)
        v_coef = x_binary_pos - torch.mul(lamda, x_binary_neg)
        value = torch.mul(v, v_coef)

        return value

    def rep_gain(self, user, centrality):
        # rep_sn_weight = self.rep_SN_weight_global(self.zero_) + self.rep_SN_weight_user(user)
        # rep_x_ref = self.rep_x_ref_global(self.zero_) + self.rep_x_ref_user(user)
        #
        # sn = self.cossim_2(centrality, rep_sn_weight.view(-1, 1, 4))
        # x = sn - rep_x_ref
        # # x = torch.mul(sn, self.rep_SN_weight_user(user)) - rep_x_ref
        #
        # x_binary_pos = torch.gt(x, self.zero_).to(torch.float64)
        # x_binary_neg = torch.ones_like(x).to(self.device) - x_binary_pos
        #
        # x = torch.abs(x)
        #
        # lamda = self.rep_x_lamda_global(self.zero_) + self.rep_x_lamda_user(user)
        # alpha = self.rep_x_alpha_global(self.zero_) + self.rep_x_alpha_user(user)
        # beta = self.rep_x_beta_global(self.zero_) + self.rep_x_beta_user(user)
        #
        # v_exp = torch.mul(alpha, x_binary_pos) + torch.mul(beta, x_binary_neg)
        # v = x.pow(v_exp)
        # v_coef = x_binary_pos - torch.mul(lamda, x_binary_neg)
        # value = torch.mul(v, v_coef)
        #
        # return value
        pass

    def com_gain(self, user, participant, interact):
        user_participant_pref = self.com_participant_pref_user(user)
        participant_similarity = self.cossim_2(user_participant_pref.view((-1, 1, N_TOPICS)), participant)
        #
        interact_apart_weight = self.com_interact_apart_weight_global(self.zero_) + self.com_interact_apart_weight_user(user)
        interact_gain = torch.mul(interact, interact_apart_weight.view(-1, 1, 2)).sum(2)

        participant_weight = self.com_participant_weight_global(self.zero_) + self.com_participant_weight_user(user)
        interact_weight = self.com_interact_weight_global(self.zero_) + self.com_interact_weight_user(user)
        x_ref = self.com_x_ref_global(self.zero_) + self.com_x_ref_user(user)
        x = torch.mul(participant_similarity, participant_weight) - x_ref \
            + torch.mul(interact_gain, interact_weight)

        x_binary_pos = torch.gt(x, self.zero_).to(torch.float64)
        x_binary_neg = torch.ones_like(x).to(self.device) - x_binary_pos

        x = torch.abs(x)

        lamda = self.com_x_lamda_global(self.zero_) + self.com_x_lamda_user(user)
        alpha = self.com_x_alpha_global(self.zero_) + self.com_x_alpha_user(user)
        beta = self.com_x_beta_global(self.zero_) + self.com_x_beta_user(user)

        v_exp = torch.mul(alpha, x_binary_pos) + torch.mul(beta, x_binary_neg)
        v = x.pow(v_exp)
        v_coef = x_binary_pos - torch.mul(lamda, x_binary_neg)
        value = torch.mul(v, v_coef)

        return value

    def __deep_forward(self, data):
        hist_vector = data['hist_vector'].to(self.device)  # (n, 10, 50)
        hist_lda = data['hist_lda'].to(self.device)
        item_vector = data['item_vector'].to(self.device)  # (n, 50)
        item_lda = data['item_lda'].to(self.device)

        mean_hist_vector = torch.mean(hist_vector, dim=1)  # (n, 50)
        mean_hist_lda = torch.mean(hist_lda, dim=1)

        vector_ = torch.cat([mean_hist_vector, item_vector, mean_hist_lda, item_lda], dim=1)
        fc_1_output = self.topic_fc_1(vector_)
        fc_2_output = self.topic_fc_2(fc_1_output).view(-1)

        return fc_2_output

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
        neg_data['hist_vector'] = self.__duplicates(data['hist_vector'], times=negNum)
        neg_data['hist_info'] = self.__duplicates(data['hist_info'], times=negNum)
        neg_data['hist_participants'] = self.__duplicates(data['hist_participants'], times=negNum)
        neg_data['hist_interact'] = self.__duplicates(data['hist_interact'], times=negNum)
        neg_data['timeDelta'] = self.__duplicates(data['timeDelta'], times=negNum)
        neg_data['item_lda'] = data['negItem_lda'].reshape((-1, N_TOPICS))
        neg_data['item_vector'] = data['negItem_vector'].reshape((-1, VECTOR_DIM))
        neg_data['item_info'] = data['negItem_info']
        neg_data['item_participants'] = data['negItem_participants']
        neg_data['item_interact'] = data['negItem_interact']
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
                score_pos = self.forward(batchData)
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
        'lr': 1e-2,
        'w_decay': 1e-2,
        'batch_size': 128,
        'negNum_train': 2,
        'negNum_test': 10,
        'epoch_limit': 3,
    }

    train = data_loader.read_data('train_data')
    test = data_loader.read_data('test_data')
    user_seq = data_loader.read_data('user_sequence')
    user_dist = data_loader.read_data('user_dist')
    user_lda_dist, user_vector_dist = user_dist['lda_dist'], user_dist['vector_dist']

    trainset = data_loader.UserData(train)
    trainLoader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True)
    testset = data_loader.UserData(test)
    testset.set_negN(params['negNum_test'])
    testLoader = DataLoader(testset, batch_size=16, shuffle=False)

    model = PT(
        user_len=len(user_seq),
        param=params,
        lda_dist=user_lda_dist,
        vector_dist=user_vector_dist
    )
    # model = data_prepare.read_data('model')
    model.to(model.device)
    print('initialization')
    optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['w_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

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

    data_prepare.save_data(model, 'model')
