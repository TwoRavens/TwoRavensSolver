import torch
from torch.utils.data import Dataset


'''Placeholder dataset for torch model training'''


class Dataset_PH(Dataset):
    def __init__(self, x, y):
        super(Dataset_PH, self).__init__()
        # Convert DataFrame to Numpy array
        self.x, self.y = x.to_numpy(), y.to_numpy()

    def __getitem__(self, index):
        features = torch.from_numpy(self.x[index])
        label = self.y[index]
        return features, label

    def __len__(self):
        return len(self.x)
