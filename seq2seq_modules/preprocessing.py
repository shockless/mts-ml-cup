import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class TargetPandasPreprocessor:
    def __init__(self,
                 agg_column: str,
                 time_column: str,
                 target_column: str,
                 features: list,
                 max_len: int,
                 padding_side: str = "left"):
        self.agg_column = agg_column
        self.time_column = time_column
        self.target_column = target_column
        self.features = features
        self.max_len = max_len
        self.padding_side = padding_side

    def get_sequences(self, dataset: pd.DataFrame) -> tuple:
        dataset = dataset.sort_values(
            by=[self.agg_column, self.time_column], ascending=[True, True]
        )

        sequences = []
        targets = []

        agg_col = dataset[self.agg_column].to_numpy()
        target = dataset[self.target_column].to_numpy()
        dataset = dataset[self.features].to_numpy().astype("float32")

        curr_ind = 0
        curr_val = agg_col[0]
        for i in tqdm(range(agg_col.shape[0])):
            if agg_col[i] != curr_val:
                sequences.append(dataset[curr_ind:i])
                targets.append(target[curr_ind:i][0])

                curr_ind = i
                curr_val = agg_col[i]

        return sequences, targets

    def to_tensor(self, sequences: list) -> list:
        return [torch.tensor(sequence) for sequence in tqdm(sequences)]

    def add_batch(self, sequences: list) -> list:
        return [sequence.unsqueeze(dim=0) for sequence in tqdm(sequences)]

    def concat(self, sequences: list) -> torch.tensor:
        return torch.cat(sequences, dim=0)

    def right_pad_and_truncate(self, sequences: list) -> tuple:
        attention_masks = []

        for i in tqdm(range(len(sequences))):
            if sequences[i].shape[0] < self.max_len:
                sequences[i] = np.concatenate([
                    sequences[i],
                    np.zeros((self.max_len - sequences[i].shape[0], sequences[i].shape[1]))
                ], axis=0)

                attention_masks.append(
                    np.concatenate([
                        np.ones((sequences[i].shape[0])),
                        np.zeros((self.max_len - sequences[i].shape[0]))
                    ], axis=0)
                )

            elif sequences[i].shape[0] >= self.max_len:
                sequences[i] = sequences[i][-self.max_len:]
                attention_masks.append(np.ones((sequences[i].shape[0])))

        return sequences, attention_masks

    def left_pad_and_truncate(self, sequences: list) -> tuple:
        attention_masks = []

        for i in tqdm(range(len(sequences))):
            if sequences[i].shape[0] < self.max_len:
                sequences[i] = np.concatenate([
                    np.zeros((self.max_len - sequences[i].shape[0], sequences[i].shape[1])),
                    sequences[i]
                ], axis=0)

                attention_masks.append(
                    np.concatenate([
                        np.zeros((self.max_len - sequences[i].shape[0])),
                        np.ones((sequences[i].shape[0]))
                    ], axis=0)
                )

            elif sequences[i].shape[0] >= self.max_len:
                sequences[i] = sequences[i][-self.max_len:]
                attention_masks.append(np.ones((sequences[i].shape[0])))

        return sequences, attention_masks

    def transform(self, dataset: pd.DataFrame) -> tuple:
        sequences, target = self.get_sequences(dataset)
        target = torch.tensor(target).long()
        if self.padding_side.lower() == "right":
            sequences, attention_masks = self.right_pad_and_truncate(sequences)
        elif self.padding_side.lower() == "left":
            sequences, attention_masks = self.left_pad_and_truncate(sequences)
        sequences = self.concat(self.add_batch(self.to_tensor(sequences)))
        attention_masks = self.concat(self.add_batch(self.to_tensor(attention_masks)))

        return sequences, attention_masks.bool(), target


class NSPPandasPreprocessor(TargetPandasPreprocessor):
    def __init__(self, agg_column: str, time_column: str, max_len: int, padding_side: str = "left"):
        super().__init__(agg_column, time_column, max_len, padding_side)

    def get_sequences(self, dataset: pd.DataFrame) -> tuple:
        dataset = dataset.sort_values(
            by=[self.agg_column, self.time_column], ascending=[True, True]
        )

        sequences = []
        targets = []

        agg_col = dataset[self.agg_column].to_numpy()
        target_col = dataset[self.target_column].to_numpy()
        dataset = dataset[self.features].to_numpy().astype("float32")

        curr_ind = 0
        curr_val = agg_col[0]
        for i in tqdm(range(agg_col.shape[0])):
            if agg_col[i] != curr_val:
                sequence = dataset[curr_ind:i - 1]
                target = target_col[i - 1]
                sequences.append(sequence)
                targets.append(target)

                curr_ind = i
                curr_val = agg_col[i]

        return sequences, targets
