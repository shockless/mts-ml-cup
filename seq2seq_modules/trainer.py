from copy import deepcopy

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import compute_class_weight
from transformers import get_constant_schedule

from seq2seq_modules.errors import NotFittedError
from seq2seq_modules.loops import eval_epoch, train_epoch, predict
from seq2seq_modules.utils import fix_random_state


class CVTrainer:
    def __init__(
            self,
            model_name,
            model,
            n_folds: int = 4,
    ):
        self.model_name = model_name
        self.model = model
        self.n_folds = n_folds

        self.fold_models = []
        self.fitted = False

    def fit_transform(self,
                      dataset,
                      loss_function,
                      metric_func,
                      optimizer,
                      get_scheduler,
                      target_name: str = None,
                      user_ids: np.ndarray = None,
                      strat_array: torch.tensor = None,
                      shuffle: bool = True,
                      epochs: int = 5,
                      lr: float = 1e-3,
                      weight_decay: float = 1e-2,
                      num_warmup_steps: int = 0,
                      batch_size: int = 32,
                      random_state: int = 69,
                      device: str = "cuda"
                      ):
        self.fold_models = []

        if type(strat_array) != type(None):
            kfold = StratifiedKFold(self.n_folds, shuffle=shuffle, random_state=random_state)
            split = kfold.split(dataset, strat_array)
        else:
            kfold = KFold(self.n_folds, shuffle=shuffle, random_state=random_state)
            split = kfold.split(dataset)

        fold_embeddings = []
        fold_logits = []
        fold_targets = []

        train_fold_metrics = []
        eval_fold_metrics = []
        eval_fold_indexes = []

        for fold, (train_ids, eval_ids) in enumerate(split):
            eval_fold_indexes.append(eval_ids)
            print(f"FOLD {fold}")
            print("--------------------------------")

            fold_model = deepcopy(self.model)
            fold_model.to(device)

            fold_loss_function = loss_function(
                weight=torch.tensor(
                    compute_class_weight(
                        class_weight="balanced",
                        classes=sorted(strat_array.unique().numpy()),
                        y=strat_array.numpy()[train_ids]
                    )
                )
            ).to(device)

            fold_optimizer = optimizer(
                fold_model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

            fix_random_state(random_state)
            train_subsampler = torch.utils.data.Subset(dataset, train_ids)
            train_loader = torch.utils.data.DataLoader(
                train_subsampler, batch_size=batch_size, shuffle=False
            )

            fix_random_state(random_state)
            eval_subsampler = torch.utils.data.Subset(dataset, eval_ids)
            eval_loader = torch.utils.data.DataLoader(
                eval_subsampler, batch_size=batch_size, shuffle=False
            )

            total_steps = len(train_loader) * epochs

            if get_scheduler != get_constant_schedule:
                scheduler = get_scheduler(
                    fold_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps
                )
            else:
                scheduler = get_scheduler(
                    fold_optimizer,
                )

            train_epoch_metrics = []
            eval_epoch_metrics = []

            for epoch_i in range(epochs):
                train_metrics, train_embeddings, train_logits, train_targets = train_epoch(
                    fold_model,
                    train_loader,
                    fold_loss_function,
                    fold_optimizer,
                    scheduler,
                    device,
                    metric_func,
                )
                eval_metrics, eval_embeddings, eval_logits, eval_targets = eval_epoch(
                    fold_model,
                    eval_loader,
                    fold_loss_function,
                    device,
                    metric_func
                )

                train_epoch_metrics.append(train_metrics)
                eval_epoch_metrics.append(eval_metrics)

                print(f"EPOCH: {epoch_i}")
                print(train_metrics)
                print(eval_metrics)

            train_fold_metrics.append(train_epoch_metrics)
            eval_fold_metrics.append(eval_epoch_metrics)

            self.fold_models.append(fold_model)
            fold_embeddings.append(eval_embeddings)

            if eval_logits.shape[1] <= 2:
                eval_logits = torch.sigmoid(eval_logits)
            else:
                eval_logits = torch.softmax(eval_logits, dim=1)
            fold_logits.append(eval_logits)
            fold_targets.append(eval_targets)

        self.fitted = True

        embeddings_array = np.zeros((strat_array.shape[0], eval_embeddings.shape[1]))
        logits_array = np.zeros((strat_array.shape[0], eval_logits.shape[1]))
        targets_array = np.zeros((strat_array.shape[0], 1))

        for i in range(self.n_folds):
            embeddings_array[eval_fold_indexes[i]] = fold_embeddings[i]
            logits_array[eval_fold_indexes[i]] = fold_logits[i]
            targets_array[eval_fold_indexes[i]] = fold_targets[i]

        embeddings_df = pd.DataFrame(embeddings_array, columns=[f"embedding_feature_{i}" for i in range(eval_embeddings.shape[1])])
        logits_df = pd.DataFrame(logits_array, columns=[f"target_feature_{i}" for i in range(eval_logits.shape[1])])
        embeddings_df["user_id"] = user_ids
        logits_df["user_id"] = user_ids
        embeddings_df[target_name] = targets_array
        logits_df[target_name] = targets_array

        return train_fold_metrics, eval_fold_metrics, embeddings_df, logits_df

    def transform(self,
                  dataset,
                  user_ids: np.ndarray,
                  batch_size: int = 32,
                  device: str = "cuda"
                  ) -> tuple:
        if not self.fitted:
            raise NotFittedError()

        fold_embeddings, fold_logits = [], []

        for fold_model in self.fold_models:
            embeddings, logits = predict(
                model=fold_model,
                dataset=dataset,
                device=device,
                batch_size=batch_size
            )
            fold_embeddings.append(embeddings)
            fold_logits.append(logits)

        embeddings_array = sum(fold_embeddings) / self.n_folds
        logits_array = sum(fold_logits) / self.n_folds
        if logits_array.shape[1] <= 2:
            logits_array = torch.sigmoid(logits_array)
        else:
            logits_array = torch.softmax(logits_array, dim=1)

        embeddings_df = pd.DataFrame(embeddings_array, columns=[f"embedding_feature_{i}" for i in range(embeddings_array.shape[1])])
        logits_df = pd.DataFrame(logits_array, columns=[f"target_feature_{i}" for i in range(logits_array.shape[1])])
        embeddings_df["user_id"] = user_ids
        logits_df["user_id"] = user_ids

        return embeddings_df, logits_df

    def get_models(self) -> list:
        return self.fold_models

    def save_model(self, model_name: str = "model"):
        joblib.dump(self, f"{model_name}.joblib")
