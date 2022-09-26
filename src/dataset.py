import torch
from torch.utils.data import Dataset
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, data_fold):
        # print(data_fold.loc[data_fold['ENSG00000121410_A1BG']=='911991c08e7a', :5])
        # print(data_fold.iloc[:5, :5])
        # print(data_fold.shape)
        self.X_data = data_fold.iloc[:, :-141].values.astype(np.float32) # -141 is column 'fold'
        self.Y_data = data_fold.iloc[:, -140:].values.astype(np.float32)

    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, item):
        inputs = torch.tensor(self.X_data[item], dtype=torch.float)
        labels = torch.tensor(self.Y_data[item], dtype=torch.float)
        return inputs, labels

