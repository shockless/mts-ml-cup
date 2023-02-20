import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class TargetDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 agg_column: str,
                 time_column: str,
                 target_column: str,
                 cat_features: list,
                 cont_features: list,
                 max_len: int,
                 padding_side: str = "left"):
        self.df = df
        self.agg_column = agg_column
        self.time_column = time_column
        self.target_column = target_column
        self.cat_features = cat_features
        self.cont_features = cont_features
        self.max_len = max_len
        self.padding_side = padding_side

        self.__get_sequences()

    def __len__(self) -> int:
        return len(self.cat_sequences)

    def __getitem__(self, idx: int) -> tuple:
        cat_features = self.cat_sequences[idx]
        cont_features = self.cont_sequences[idx]
        target = torch.tensor(self.targets[idx]).long()

        cat_features, attention_mask = self.__pad_and_truncate(cat_features)
        cont_features, attention_mask = self.__pad_and_truncate(cont_features)

        cat_features = torch.tensor(cat_features).long()
        cont_features = torch.tensor(cont_features).float()
        attention_mask = torch.tensor(attention_mask).bool()

        return cat_features, cont_features, attention_mask, target

    def __get_sequences(self):
        self.df = self.df.sort_values(
            by=[self.agg_column, self.time_column], ascending=[True, True]
        )

        self.cat_sequences = []
        self.cont_sequences = []
        self.targets = []
        self.agg_col_array = []

        self.agg_col = self.df[self.agg_column].to_numpy()
        target = self.df[self.target_column].to_numpy()

        curr_ind = 0
        curr_val = self.agg_col[0]
        for i in tqdm(range(self.agg_col.shape[0])):
            if self.agg_col[i] != curr_val:
                self.cat_sequences.append(self.df.iloc[curr_ind:i][self.cat_features].to_numpy().astype("int32"))
                self.cont_sequences.append(self.df.iloc[curr_ind:i][self.cont_features].to_numpy().astype("float32"))
                self.targets.append(target[curr_ind:i][0])
                self.agg_col_array.append(curr_val)

                curr_ind = i
                curr_val = self.agg_col[i]

        self.cat_sequences.append(self.df.iloc[curr_ind:][self.cat_features].to_numpy().astype("int32"))
        self.cont_sequences.append(self.df.iloc[curr_ind:][self.cont_features].to_numpy().astype("float32"))
        self.targets.append(target[curr_ind:][0])
        self.agg_col_array.append(self.agg_col[i])

        del self.df
        gc.collect()

    def __right_pad_and_truncate(self, sequence: np.array) -> tuple:
        if sequence.shape[0] < self.max_len:
            sequence = np.concatenate([
                sequence,
                np.zeros((self.max_len - sequence.shape[0], sequence.shape[1])).astype("int")
            ], axis=0)

            attention_mask = np.concatenate([
                np.ones((sequence.shape[0])),
                np.zeros((self.max_len - sequence.shape[0]))
            ], axis=0).astype("bool")

        elif sequence.shape[0] >= self.max_len:
            sequence = sequence[-self.max_len:]
            attention_mask = np.ones((sequence.shape[0])).astype("bool")

        return sequence, attention_mask

    def __left_pad_and_truncate(self, sequence: np.array) -> tuple:
        if sequence.shape[0] < self.max_len:
            sequence = np.concatenate([
                np.zeros((self.max_len - sequence.shape[0], sequence.shape[1])).astype("int"),
                sequence
            ], axis=0)

            attention_mask = np.concatenate([
                np.zeros((self.max_len - sequence.shape[0])),
                np.ones((sequence.shape[0]))
            ], axis=0).astype("bool")

        elif sequence.shape[0] >= self.max_len:
            sequence = sequence[-self.max_len:]
            attention_mask = np.ones((sequence.shape[0])).astype("bool")

        return sequence, attention_mask

    def __pad_and_truncate(self, sequence: np.array) -> tuple:
        if self.padding_side.lower() == "right":
            sequence, attention_mask = self.__right_pad_and_truncate(sequence)
        elif self.padding_side.lower() == "left":
            sequence, attention_mask = self.__left_pad_and_truncate(sequence)
        else:
            raise ValueError("Expected padding_side to be either 'right' of 'left'")

        return sequence, attention_mask

    def get_agg_col(self) -> np.ndarray:
        return np.array(self.agg_col_array)


class TestDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 agg_column: str,
                 time_column: str,
                 cat_features: list,
                 cont_features: list,
                 max_len: int,
                 padding_side: str = "left"):
        self.df = df
        self.agg_column = agg_column
        self.time_column = time_column
        self.cat_features = cat_features
        self.cont_features = cont_features
        self.max_len = max_len
        self.padding_side = padding_side

        self.__get_sequences()

    def __len__(self) -> int:
        return len(self.cat_sequences)

    def __getitem__(self, idx: int) -> tuple:
        cat_features = self.cat_sequences[idx]
        cont_features = self.cont_sequences[idx]

        cat_features, attention_mask = self.__pad_and_truncate(cat_features)
        cont_features, attention_mask = self.__pad_and_truncate(cont_features)

        cat_features = torch.tensor(cat_features).long()
        cont_features = torch.tensor(cont_features).float()
        attention_mask = torch.tensor(attention_mask).bool()

        return cat_features, cont_features, attention_mask

    def __get_sequences(self):
        self.df = self.df.sort_values(
            by=[self.agg_column, self.time_column], ascending=[True, True]
        )

        self.cat_sequences = []
        self.cont_sequences = []
        self.agg_column_array = self.df[self.agg_column].to_numpy()

        self.agg_col_array = []
        self.agg_col = self.df[self.agg_column].to_numpy()

        curr_ind = 0
        curr_val = self.agg_col[0]
        for i in tqdm(range(self.agg_col.shape[0])):
            if self.agg_col[i] != curr_val:
                self.cat_sequences.append(self.df.iloc[curr_ind:i][self.cat_features].to_numpy().astype("int32"))
                self.cont_sequences.append(self.df.iloc[curr_ind:i][self.cont_features].to_numpy().astype("float32"))
                self.agg_col_array.append(curr_val)

                curr_ind = i
                curr_val = self.agg_col[i]

        self.cat_sequences.append(self.df.iloc[curr_ind:][self.cat_features].to_numpy().astype("int32"))
        self.cont_sequences.append(self.df.iloc[curr_ind:][self.cont_features].to_numpy().astype("float32"))
        self.agg_col_array.append(self.agg_col[i])

        del self.df
        gc.collect()

    def __right_pad_and_truncate(self, sequence: np.array) -> tuple:
        if sequence.shape[0] < self.max_len:
            sequence = np.concatenate([
                sequence,
                np.zeros((self.max_len - sequence.shape[0], sequence.shape[1])).astype("int")
            ], axis=0)

            attention_mask = np.concatenate([
                np.ones((sequence.shape[0])),
                np.zeros((self.max_len - sequence.shape[0]))
            ], axis=0).astype("bool")

        elif sequence.shape[0] >= self.max_len:
            sequence = sequence[-self.max_len:]
            attention_mask = np.ones((sequence.shape[0])).astype("bool")

        return sequence, attention_mask

    def __left_pad_and_truncate(self, sequence: np.array) -> tuple:
        if sequence.shape[0] < self.max_len:
            sequence = np.concatenate([
                np.zeros((self.max_len - sequence.shape[0], sequence.shape[1])).astype("int"),
                sequence
            ], axis=0)

            attention_mask = np.concatenate([
                np.zeros((self.max_len - sequence.shape[0])),
                np.ones((sequence.shape[0]))
            ], axis=0).astype("bool")

        elif sequence.shape[0] >= self.max_len:
            sequence = sequence[-self.max_len:]
            attention_mask = np.ones((sequence.shape[0])).astype("bool")

        return sequence, attention_mask

    def __pad_and_truncate(self, sequence: np.array) -> tuple:
        if self.padding_side.lower() == "right":
            sequence, attention_mask = self.__right_pad_and_truncate(sequence)
        elif self.padding_side.lower() == "left":
            sequence, attention_mask = self.__left_pad_and_truncate(sequence)
        else:
            raise ValueError("Expected padding_side to be either 'right' of 'left'")

        return sequence, attention_mask

    def get_agg_col(self) -> np.ndarray:
        return np.array(self.agg_col_array)


class CoLESDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 agg_column: str,
                 time_column: str,
                 cat_features: list,
                 cont_features: list,
                 num_splits: int,
                 max_len: int,
                 padding_side: str = "left"):
        self.df = df
        self.agg_column = agg_column
        self.time_column = time_column
        self.cat_features = cat_features
        self.cont_features = cont_features
        self.num_splits = num_splits
        self.max_len = max_len
        self.padding_side = padding_side

        self.__get_sequences()

    def __len__(self) -> int:
        return len(self.cat_sequences)

    def __getitem__(self, idx: int) -> tuple:
        cat_features = self.cat_sequences[idx]
        cont_features = self.cont_sequences[idx]
        target = self.targets[idx]

        cat_features, attention_mask = self.__pad_and_truncate(cat_features)
        cont_features, attention_mask = self.__pad_and_truncate(cont_features)

        cat_features = torch.tensor(cat_features).long()
        cont_features = torch.tensor(cont_features).float()
        attention_mask = torch.tensor(attention_mask).bool()

        return cat_features, cont_features, attention_mask, target

    def __get_sequences(self):
        self.df = self.df.sort_values(
            by=[self.agg_column, self.time_column], ascending=[True, True]
        )

        self.targets = []
        self.cat_sequences = []
        self.cont_sequences = []
        self.agg_col = self.df[self.agg_column].to_numpy()

        target_id = 0
        curr_ind = 0
        curr_val = self.agg_col[0]

        for i in tqdm(range(self.agg_col.shape[0])):
            if self.agg_col[i] != curr_val:
                if i - curr_ind >= self.num_splits:
                    self.cat_sequences.extend(
                        np.split(self.df.iloc[curr_ind:i][self.cat_features].to_numpy().astype("int32"),
                                 range(0, i - curr_ind, self.num_splits)
                                 )
                    )

                    self.cont_sequences.extend(
                        np.split(self.df.iloc[curr_ind:i][self.cont_features].to_numpy().astype("float32"),
                                 range(0, i - curr_ind, self.num_splits)
                                 )
                    )

                    self.targets.extend([target_id] * (i - curr_ind))

                target_id += 1
                curr_ind = i
                curr_val = self.agg_col[i]

        del self.df
        gc.collect()

    def __right_pad_and_truncate(self, sequence: np.array) -> tuple:
        if sequence.shape[0] < self.max_len:
            sequence = np.concatenate([
                sequence,
                np.zeros((self.max_len - sequence.shape[0], sequence.shape[1])).astype("int")
            ], axis=0)

            attention_mask = np.concatenate([
                np.ones((sequence.shape[0])),
                np.zeros((self.max_len - sequence.shape[0]))
            ], axis=0).astype("bool")

        elif sequence.shape[0] >= self.max_len:
            sequence = sequence[-self.max_len:]
            attention_mask = np.ones((sequence.shape[0])).astype("bool")

        return sequence, attention_mask

    def __left_pad_and_truncate(self, sequence: np.array) -> tuple:
        if sequence.shape[0] < self.max_len:
            sequence = np.concatenate([
                np.zeros((self.max_len - sequence.shape[0], sequence.shape[1])).astype("int"),
                sequence
            ], axis=0)

            attention_mask = np.concatenate([
                np.zeros((self.max_len - sequence.shape[0])),
                np.ones((sequence.shape[0]))
            ], axis=0).astype("bool")

        elif sequence.shape[0] >= self.max_len:
            sequence = sequence[-self.max_len:]
            attention_mask = np.ones((sequence.shape[0])).astype("bool")

        return sequence, attention_mask

    def __pad_and_truncate(self, sequence: np.array) -> tuple:
        if self.padding_side.lower() == "right":
            sequence, attention_mask = self.__right_pad_and_truncate(sequence)
        elif self.padding_side.lower() == "left":
            sequence, attention_mask = self.__left_pad_and_truncate(sequence)
        else:
            raise ValueError("Expected padding_side to be either 'right' of 'left'")

        return sequence, attention_mask
