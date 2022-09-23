from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, X_data_fold, Y_data_fold):
        self.X_data = X_data_fold
        self.Y_data = Y_data_fold

    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, item):
        return self.X_data[item], self.Y_data[item]

