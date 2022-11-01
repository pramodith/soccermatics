import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from constants import *
from ast import literal_eval
from torch.nn.utils.rnn import pad_sequence
import torch

class DeepXGDataset(Dataset):

    def __init__(self, data_dir: str = "./", mode=TRAIN):
        self.df = pd.read_csv(data_dir+f"{mode}.csv")
        self.df.fillna(-1, inplace=True)
        self.labels = self.df[IS_GOAL].astype("int")
        self.pass_sequence = self.df[PASS_ZONES].apply(literal_eval).to_numpy()
        self.shot_zone = self.df[SHOT_ZONES].to_numpy()
        self.df.drop(columns=[PASS_ZONES], inplace=True)
        self.cat_var_df = self.df[CATEGORICAL_VARIABLES].to_numpy()
        self.cont_var_df = self.df[CONTINUOUS_VARIABLES+BOOLEAN_VARIABLES].astype("float").to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return torch.LongTensor(self.pass_sequence[item]), self.shot_zone[item],\
               torch.LongTensor(self.cat_var_df[item]), torch.FloatTensor(self.cont_var_df[item]), self.labels[item]


class DeepXGDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.deepxg_test = None
        self.deepxg_val = None
        self.deepxg_train = None
        self.data_dir = data_dir
        self.deepxg_train = DeepXGDataset(self.data_dir, TRAIN)
        self.deepxg_val = DeepXGDataset(self.data_dir, VAL)
        self.deepxg_test = DeepXGDataset(self.data_dir, TEST)

    def custom_collate(self, features):
        pass_seq, shot_zones, cat_features, cont_features, labels = list(zip(*features))
        pass_seq = pad_sequence(pass_seq, padding_value=80, batch_first=True)
        return pass_seq, torch.LongTensor(shot_zones), torch.stack(cat_features), torch.stack(cont_features), torch.LongTensor(labels)

    def train_dataloader(self):
        return DataLoader(self.deepxg_train, batch_size=64, collate_fn=self.custom_collate)

    def val_dataloader(self):
        return DataLoader(self.deepxg_val, batch_size=256, collate_fn=self.custom_collate)

    def test_dataloader(self):
        return DataLoader(self.deepxg_test, batch_size=256, collate_fn=self.custom_collate)


if __name__ == "__main__":
    pass