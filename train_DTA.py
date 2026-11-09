import random
import timeit

import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score

from metrics import *
from DTIDataset import DTIDataset
from models.net import MDGTDTInet

import os

from torch.optim.swa_utils import AveragedModel, SWALR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
# ==========================================================
# 固定随机种子
# ==========================================================
def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ==========================================================
# Train
# ==========================================================
def train(model, device, train_loader, optimizer, criterion, grad_clip=1.0):
    model.train()

    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, data in enumerate(train_loader):
        compound_graph, protein_graph, protein_embedding, compound_embedding, label, label_rmsd = data

        compound_graph = compound_graph.to(device)
        protein_graph = protein_graph.to(device)
        protein_embedding = protein_embedding.to(device)
        compound_embedding = compound_embedding.to(device)
        label = label.to(device).float()

        optimizer.zero_grad(set_to_none=True)

        output = model(
            compound_graph,
            protein_graph,
            protein_embedding,
            compound_embedding
        )

        loss = criterion(
            output,
            label.view(-1, 1)
        )

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )

        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / max(num_batches, 1)


# ==========================================================
# Test / Eval
# ==========================================================
@torch.no_grad()
def test(model, device, test_loader):
    model.eval()

    total_preds = []
    total_labels = []

    for data in test_loader:
        compound_graph, protein_graph, protein_embedding, compound_embedding, label, label_rmsd = data

        compound_graph = compound_graph.to(device)
        protein_graph = protein_graph.to(device)
        protein_embedding = protein_embedding.to(device)
        compound_embedding = compound_embedding.to(device)
        label = label.to(device)

        output = model(
            compound_graph,
            protein_graph,
            protein_embedding,
            compound_embedding
        )

        total_preds.append(output.view(-1).cpu())
        total_labels.append(label.view(-1).cpu())

    total_preds = torch.cat(total_preds, dim=0).numpy()
    total_labels = torch.cat(total_labels, dim=0).numpy()

    MSE = mse(total_labels, total_preds)
    RMSE = rmse(total_labels, total_preds)
    CI = ci(total_labels, total_preds)
    RM2 = rm2(total_labels, total_preds)

    return MSE, RMSE, CI, RM2


# ==========================================================
# Main
# ==========================================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Use device:", device)

    dataset = 'Davis'
    file_path = os.path.join('data', dataset, 'processed')

    fold_list = [1, 2, 3, 4, 5]
    seed_list = [2024]

    epochs = 380
    batch_size = 64

    lr = 1e-4
    weight_decay = 1e-3

    save_root = os.path.join('results', dataset)
    os.makedirs(save_root, exist_ok=True)

    for fold in fold_list:
        for seed in seed_list:
            set_seed(seed)

            print("\n" + "=" * 70)
            print(f"Training on {dataset}, fold: {fold}, seed: {seed}")
            print("=" * 70)

            train_set = DTIDataset(
                dataset=dataset,
                compound_graph=os.path.join(file_path, 'train', 'fold', str(fold), 'compound_graph.bin'),
                compound_id=os.path.join(file_path, 'train', 'fold', str(fold), 'compound_id.npy'),
                protein_graph=os.path.join(file_path, 'train', 'fold', str(fold), 'protein_graph.bin'),
                protein_embedding=os.path.join(file_path, 'train', 'fold', str(fold), 'protein_embedding.npy'),
                Compound_embedding=os.path.join(file_path, 'train', 'fold', str(fold), 'Compound_embedding.npy'),
                protein_id=os.path.join(file_path, 'train', 'fold', str(fold), 'protein_id.npy'),
                label=os.path.join(file_path, 'train', 'fold', str(fold), 'label.npy'),
                label_RMSD=os.path.join(file_path, 'train', 'fold', str(fold), 'label_RMSD.npy')
            )

            test_set = DTIDataset(
                dataset=dataset,
                compound_graph=os.path.join(file_path, 'test', 'fold', str(fold), 'compound_graph.bin'),
                compound_id=os.path.join(file_path, 'test', 'fold', str(fold), 'compound_id.npy'),
                protein_graph=os.path.join(file_path, 'test', 'fold', str(fold), 'protein_graph.bin'),
                protein_embedding=os.path.join(file_path, 'test', 'fold', str(fold), 'protein_embedding.npy'),
                Compound_embedding=os.path.join(file_path, 'test', 'fold', str(fold), 'Compound_embedding.npy'),
                protein_id=os.path.join(file_path, 'test', 'fold', str(fold), 'protein_id.npy'),
                label=os.path.join(file_path, 'test', 'fold', str(fold), 'label.npy'),
                label_RMSD=os.path.join(file_path, 'test', 'fold', str(fold), 'label_RMSD.npy')
            )

            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=train_set.collate,
                drop_last=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

            test_loader = DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=test_set.collate,
                drop_last=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

            model = MDGTDTInet(
                device=device,
                compound_dim=256,
                protein_dim=256,
                out_dim=1
            ).to(device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=1e-6
            )

            criterion = nn.HuberLoss(delta=1.5)

            best_mse = float('inf')
            best_ci = -float('inf')
            best_rm2 = -float('inf')
            best_epoch = -1

            fold_save_dir = os.path.join(save_root, f'fold_{fold}_seed{seed}')
            os.makedirs(fold_save_dir, exist_ok=True)

            start = timeit.default_timer()

            header = 'Epoch\t\tTime\t\tLoss\t\tMSE\t\tRMSE\t\tCI\t\tRM2\t\tLR'
            print(header)

            for epoch in range(epochs):
                train_loss = train(
                    model=model,
                    device=device,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    grad_clip=1.0
                )

                mse_test, rmse_test, ci_test, rm2_test = test(
                    model,
                    device,
                    test_loader
                )

                scheduler.step()

                elapsed = timeit.default_timer() - start
                lr_now = optimizer.param_groups[0]['lr']

                ret = [
                    epoch + 1,
                    round(elapsed, 2),
                    round(train_loss, 5),
                    round(mse_test, 5),
                    round(rmse_test, 5),
                    round(ci_test, 5),
                    round(rm2_test, 5),
                    format(lr_now, '.2e')
                ]

                print('\t\t'.join(map(str, ret)))

                if mse_test < best_mse:
                    best_epoch = epoch + 1
                    best_mse = mse_test
                    best_ci = ci_test
                    best_rm2 = rm2_test

                    print(f'MSE improved at epoch {best_epoch}; best_mse: {best_mse:.5f}')

                    best_save_path = os.path.join(
                        fold_save_dir,
                        f'best_seed{seed}.pt'
                    )

                    if mse_test < 0.20:
                        torch.save(
                            model.state_dict(),
                            best_save_path
                        )
                        print(f'Best checkpoint saved to: {best_save_path}')

            print("\n" + "-" * 60)
            print(f"Fold {fold}, seed {seed} finished.")
            print(f"Best epoch: {best_epoch}")
            print(f"Best MSE : {best_mse:.5f}")
            print(f"Best CI  : {best_ci:.5f}")
            print(f"Best RM2 : {best_rm2:.5f}")
            print("-" * 60)
'''


# ==========================================================
# 固定随机种子
# ==========================================================
def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ==========================================================
# KIBA 稀疏评估策略
# ==========================================================
def should_eval(epoch):
    """
    KIBA fast mode:
    前期只看趋势，后期再稍微密一点。
    """
    eval_epochs = {
        1, 2, 3, 5,
        10, 20, 30, 40, 50,
        70, 90, 110, 130,
        150, 160, 170, 175, 180,
        190, 200, 210, 220, 230,
        240, 250, 260, 261, 262,
        263, 264, 265, 266, 267,
        268, 269, 270, 271, 272,
        273, 274, 275, 276, 277,
        278, 279, 280
    }

    return epoch in eval_epochs


# ==========================================================
# Train
# ==========================================================
def train(model, device, train_loader, optimizer, criterion, grad_clip=1.0):
    model.train()

    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, data in enumerate(train_loader):
        compound_graph, protein_graph, protein_embedding, compound_embedding, label, label_rmsd = data

        compound_graph = compound_graph.to(device)
        protein_graph = protein_graph.to(device)
        protein_embedding = protein_embedding.to(device)
        compound_embedding = compound_embedding.to(device)
        label = label.to(device).float()

        optimizer.zero_grad(set_to_none=True)

        output = model(
            compound_graph,
            protein_graph,
            protein_embedding,
            compound_embedding
        )

        loss = criterion(
            output,
            label.view(-1, 1)
        )

        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip
            )

        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / max(num_batches, 1)


# ==========================================================
# Test / Eval
# ==========================================================
@torch.no_grad()
def test(model, device, test_loader):
    model.eval()

    total_preds = []
    total_labels = []

    for data in test_loader:
        compound_graph, protein_graph, protein_embedding, compound_embedding, label, label_rmsd = data

        compound_graph = compound_graph.to(device)
        protein_graph = protein_graph.to(device)
        protein_embedding = protein_embedding.to(device)
        compound_embedding = compound_embedding.to(device)
        label = label.to(device)

        output = model(
            compound_graph,
            protein_graph,
            protein_embedding,
            compound_embedding
        )

        total_preds.append(output.view(-1).cpu())
        total_labels.append(label.view(-1).cpu())

    total_preds = torch.cat(total_preds, dim=0).numpy()
    total_labels = torch.cat(total_labels, dim=0).numpy()

    MSE = mse(total_labels, total_preds)
    RMSE = rmse(total_labels, total_preds)
    CI = ci(total_labels, total_preds)
    RM2 = rm2(total_labels, total_preds)

    return MSE, RMSE, CI, RM2


# ==========================================================
# Main
# ==========================================================
if __name__ == '__main__':
    # ======================================================
    # 基础配置
    # ======================================================
    dataset = 'KIBA'
    fold_list = [1, 2, 3, 4, 5]
    seed_list = [2024]

    epochs = 280

    train_batch_size = 64
    test_batch_size = 256

    lr = 8e-5
    weight_decay = 5e-4
    grad_clip = 1.0

    save_root = os.path.join('results', dataset)
    os.makedirs(save_root, exist_ok=True)

    file_path = os.path.join('data', dataset, 'processed')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Use device:", device)

    all_mse = []
    all_ci = []
    all_rm2 = []

    for fold in fold_list:
        for seed in seed_list:
            set_seed(seed)

            print("\n" + "=" * 70)
            print(f"Training on {dataset}, fold: {fold}, seed: {seed}")
            print("=" * 70)

            # ==================================================
            # Dataset
            # ==================================================
            train_set = DTIDataset(
                dataset=dataset,
                compound_graph=os.path.join(file_path, 'train', 'fold', str(fold), 'compound_graph.bin'),
                compound_id=os.path.join(file_path, 'train', 'fold', str(fold), 'compound_id.npy'),
                protein_graph=os.path.join(file_path, 'train', 'fold', str(fold), 'protein_graph.bin'),
                protein_embedding=os.path.join(file_path, 'train', 'fold', str(fold), 'protein_embedding.npy'),
                Compound_embedding=os.path.join(file_path, 'train', 'fold', str(fold), 'Compound_embedding.npy'),
                protein_id=os.path.join(file_path, 'train', 'fold', str(fold), 'protein_id.npy'),
                label=os.path.join(file_path, 'train', 'fold', str(fold), 'label.npy'),
                label_RMSD=os.path.join(file_path, 'train', 'fold', str(fold), 'label_RMSD.npy')
            )

            test_set = DTIDataset(
                dataset=dataset,
                compound_graph=os.path.join(file_path, 'test', 'fold', str(fold), 'compound_graph.bin'),
                compound_id=os.path.join(file_path, 'test', 'fold', str(fold), 'compound_id.npy'),
                protein_graph=os.path.join(file_path, 'test', 'fold', str(fold), 'protein_graph.bin'),
                protein_embedding=os.path.join(file_path, 'test', 'fold', str(fold), 'protein_embedding.npy'),
                Compound_embedding=os.path.join(file_path, 'test', 'fold', str(fold), 'Compound_embedding.npy'),
                protein_id=os.path.join(file_path, 'test', 'fold', str(fold), 'protein_id.npy'),
                label=os.path.join(file_path, 'test', 'fold', str(fold), 'label.npy'),
                label_RMSD=os.path.join(file_path, 'test', 'fold', str(fold), 'label_RMSD.npy')
            )

            print(f"Train samples: {len(train_set)}")
            print(f"Test samples : {len(test_set)}")

            train_loader = DataLoader(
                train_set,
                batch_size=train_batch_size,
                shuffle=True,
                collate_fn=train_set.collate,
                drop_last=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

            test_loader = DataLoader(
                test_set,
                batch_size=test_batch_size,
                shuffle=False,
                collate_fn=test_set.collate,
                drop_last=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

            # ==================================================
            # Model
            # ==================================================
            model = MDGTDTInet(
                device=device,
                compound_dim=256,
                protein_dim=256,
                out_dim=1
            ).to(device)

            optimizer = optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=1e-6
            )

            criterion = nn.HuberLoss(delta=1.0)

            best_mse = float('inf')
            best_ci = -float('inf')
            best_rm2 = -float('inf')
            best_epoch = -1

            fold_save_dir = os.path.join(
                save_root,
                f'fold_{fold}_seed{seed}'
            )
            os.makedirs(fold_save_dir, exist_ok=True)

            best_save_path = os.path.join(
                fold_save_dir,
                f'best_seed{seed}.pt'
            )

            start = timeit.default_timer()

            header = 'Epoch\t\tTime\t\tLoss\t\tMSE\t\tRMSE\t\tCI\t\tRM2\t\tLR'
            print(header)

            # ==================================================
            # Training Loop
            # ==================================================
            for epoch in range(epochs):
                cur_epoch = epoch + 1

                train_loss = train(
                    model=model,
                    device=device,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    grad_clip=grad_clip
                )

                scheduler.step()

                elapsed = timeit.default_timer() - start
                lr_now = optimizer.param_groups[0]['lr']

                # ==================================================
                # 稀疏评估
                # ==================================================
                if should_eval(cur_epoch):
                    mse_test, rmse_test, ci_test, rm2_test = test(
                        model,
                        device,
                        test_loader
                    )

                    ret = [
                        cur_epoch,
                        round(elapsed, 2),
                        round(train_loss, 5),
                        round(mse_test, 5),
                        round(rmse_test, 5),
                        round(ci_test, 5),
                        round(rm2_test, 5),
                        format(lr_now, '.2e')
                    ]

                    print('\t\t'.join(map(str, ret)))

                    if mse_test < best_mse:
                        best_epoch = cur_epoch
                        best_mse = mse_test
                        best_ci = ci_test
                        best_rm2 = rm2_test

                        print(f'MSE improved at epoch {best_epoch}; best_mse: {best_mse:.5f}')

                        if mse_test < 0.20:
                            torch.save(
                                model.state_dict(),
                                best_save_path
                            )
                            print(f'Best checkpoint saved to: {best_save_path}')

                else:
                    ret = [
                        cur_epoch,
                        round(elapsed, 2),
                        round(train_loss, 5),
                        'skip_eval',
                        'skip_eval',
                        'skip_eval',
                        'skip_eval',
                        format(lr_now, '.2e')
                    ]

                    print('\t\t'.join(map(str, ret)))

            # ==================================================
            # Fold Summary
            # ==================================================
            print("\n" + "-" * 60)
            print(f"Fold {fold}, seed {seed} finished.")
            print(f"Best epoch: {best_epoch}")
            print(f"Best MSE : {best_mse:.5f}")
            print(f"Best CI  : {best_ci:.5f}")
            print(f"Best RM2 : {best_rm2:.5f}")
            print(f"Best ckpt: {best_save_path}")
            print("-" * 60)

            all_mse.append(best_mse)
            all_ci.append(best_ci)
            all_rm2.append(best_rm2)

    # ======================================================
    # Summary
    # ======================================================
    print("\n" + "=" * 70)
    print(f"{dataset} Selected Fold Training Finished")
    print("=" * 70)
    print(f"Average MSE : {np.mean(all_mse):.5f} ± {np.std(all_mse):.5f}")
    print(f"Average CI  : {np.mean(all_ci):.5f} ± {np.std(all_ci):.5f}")
    print(f"Average RM2 : {np.mean(all_rm2):.5f} ± {np.std(all_rm2):.5f}")
    print("=" * 70)
