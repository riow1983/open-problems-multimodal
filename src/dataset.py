from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, X_data_fold, y_data_fold):
        self.X_data = X_data_fold
        self.y_data = y_data_fold

    def __getitem__(self, item):
        return self.X_data[item], self.y_data[item]

    def __len__(self):
        return len(self.X_data)