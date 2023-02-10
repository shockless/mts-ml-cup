import os
import random
from copy import deepcopy

import numpy as np
import torch
from transformers import get_constant_schedule
# import wandb
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm

from utils import save_model


def train_epoch(model, data_loader, loss_function, optimizer, scheduler, device, metric_func):
    model.train()

    dl_size = len(data_loader)
    total_train_loss = 0

    outputs = []
    targets = []

    for batch in tqdm(data_loader):
        inputs, attention_mask, target = batch
        inputs, attention_mask, target = (
            inputs.to(device),
            attention_mask.to(device),
            target.to(device),
        )

        optimizer.zero_grad()
        logits = model(inputs, attention_mask)
        outputs.append(logits.detach().cpu())
        targets.append(target.cpu())
        
        loss = loss_function(logits.double(), target)
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    metrics = metric_func(outputs, targets)
    metrics["loss"] = total_train_loss / dl_size

    return metrics


def eval_epoch(model, data_loader, loss_function, device, metric_func):
    model.eval()

    dl_size = len(data_loader)
    total_train_loss = 0

    outputs = []
    targets = []

    for batch in tqdm(data_loader):
        inputs, attention_mask, target = batch
        inputs, attention_mask, target = (
            inputs.to(device),
            attention_mask.to(device),
            target.to(device),
        )

        with torch.no_grad():
            logits = model(inputs, attention_mask)
            outputs.append(logits.detach().cpu())
            targets.append(target.cpu())
        
        loss = loss_function(logits.double(), target)
        total_train_loss += loss.item()

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)

    metrics = metric_func(outputs, targets)
    metrics["loss"] = total_train_loss / dl_size

    return metrics


def cross_validation(
        project_name,
        model,
        dataset,
        loss_function,
        metric_func,
        optimizer,
        get_scheduler,
        strat_array=None,
        device=torch.device("cuda"),
        random_state: int = 69,
        shuffle: bool = True,
        dataloader_shuffle=False, 
        n_folds: int = 4,
        epochs: int = 5,
        lr: float = 1e-6,
        weight_decay: float = 1e-2,
        num_warmup_steps: int = 0,
        start_fold: int = 0,
        batch_size: int = 32,
):
    random.seed(random_state),
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    loss_function.to(device)
    if type(strat_array) != type(None):
        kfold = StratifiedKFold(n_folds, shuffle=shuffle, random_state=random_state)
        split = kfold.split(dataset, strat_array)
    else:
        kfold = KFold(n_folds, shuffle=shuffle, random_state=random_state)
        split = kfold.split(dataset)

    for fold, (train_ids, eval_ids) in enumerate(split):
        if fold >= start_fold:
            print(f"FOLD {fold}")
            print("--------------------------------")

#             run = wandb.init(
#                 name=f"fold_{fold}",
#                 project=f"{project_name}_fold_{fold}",
#                 config={
#                     "random_state": random_state,
#                     "shuffle": shuffle,
#                     "epochs": epochs,
#                     "learning_rate": lr,
#                     "batch_size": batch_size,
#                     # "iters_to_accumulate": iters_to_accumulate
#                 },
#             )

            fold_model = deepcopy(model)
            fold_model.to(device)

            fold_optimizer = optimizer(
                fold_model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )

            train_subsampler = torch.utils.data.Subset(dataset, train_ids)
            train_loader = torch.utils.data.DataLoader(
                train_subsampler, batch_size=batch_size, shuffle=dataloader_shuffle
            )

            eval_subsampler = torch.utils.data.Subset(dataset, eval_ids)
            eval_loader = torch.utils.data.DataLoader(
                eval_subsampler, batch_size=batch_size, shuffle=dataloader_shuffle
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

            for epoch_i in range(epochs):
                train_metrics = train_epoch(
                    fold_model,
                    train_loader,
                    loss_function,
                    fold_optimizer,
                    scheduler,
                    device,
                    metric_func,
                )
                eval_metrics = eval_epoch(
                    fold_model,
                    eval_loader,
                    loss_function,
                    device,
                    metric_func
                )

                print(f"EPOCH: {epoch_i}")
                print(train_metrics)
                print(eval_metrics)

#                 run.log(train_metrics)
#                 run.log(eval_metrics)

#             run.finish()


def single_model_training(
        model,
        dataset,
        loss_function,
        metric_func,
        optimizer,
        get_scheduler,
        save_folder,
        device=torch.device("cuda"),
        random_state: int = 69,
        shuffle: bool = True,
        epochs: int = 15,
        lr: float = 1e-6,
        batch_size: int = 32,
        start_epoch: int = 0,
):
    random.seed(random_state),
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)

    loss_function.to(device)
    model.to(device)

    optimizer = optimizer(
        model.parameters(),
        lr=lr
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    total_steps = len(data_loader) * epochs

    scheduler = get_scheduler(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    os.mkdir(save_folder)

    for epoch_i in range(0, epochs):
        if epoch_i >= start_epoch:
            train_metrics = train_epoch(
                model,
                data_loader,
                loss_function,
                optimizer,
                scheduler,
                device,
                metric_func,
            )

            save_model(model, save_folder, f"epoch_{epoch_i}")

            print("EPOCH", epoch_i)
            print(train_metrics)
