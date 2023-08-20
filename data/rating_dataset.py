import pandas as pd
from torch.utils.data import Dataset


class RatingDataset(Dataset):
    def __init__(self, fpath):
        self.df = pd.read_csv(fpath, sep="\t")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[[index]].values[0][:2].tolist()