import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

        if 'gpu' in params and params['gpu'] == True:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        pass

