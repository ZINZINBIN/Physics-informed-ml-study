import torch
import numpy as np
from torch.utils.data import Dataset

class DfpDataset(Dataset):
    def __init__(self, x : np.ndarray, y:np.ndarray):
        super(DfpDataset, self).__init__()
        self.inputs = x
        self.targets = y

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.targets[idx]).float()
        return x,y
    
    def __len__(self):
        return len(self.inputs)