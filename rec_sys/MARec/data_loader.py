from torch.utils.data import Dataset, DataLoader
import os


class UserData(Dataset):
    def __init__(self, data):
        super(UserData, self).__init__()
        self.data = data
        

