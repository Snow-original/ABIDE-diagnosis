"""
train-2.py — Two-Stage Training (v2)
======================================
Phase 1: CL loss only (pre-train encoders)
Phase 3: CE + λ·CL joint loss (公式 4-7, 4-23)
"""

from __future__ import annotations

import argparse
import os
import copy
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from importlib import import_module
model_mod = import_module("model-2")
ABIDEDataset = model_mod.ABIDEDataset
ASDDiagnosisModel = model_mod.ASDDiagnosisModel
PopulationGraphBuilder = model_mod.PopulationGraphBuilder
ContrastiveLoss = model_mod.ContrastiveLoss
PHENO_FEATURES = model_mod.PHENO_FEATURES
_TRIU_IDX = model_mod._TRIU_IDX


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",        type=str,   default=".")
    p.add_argument("--phase1_epochs",   type=int,   default=50)
    p.add_argument("--phase1_lr",       type=float, default=1e-3)
    p.add_argument("--phase1_patience", type=int,   default=7)
    p.add_argument("--temperature",     type=float, default=0.5)
    p.add_argument("--epochs",          type=int,   default=150)
    p.add_argument("--lr",              type=float, default=5e-4)
    p.add_argument("--lambda_contrast", type=float, default=0.5)
    p.add_argument("--encoder_lr_scale",type=float, default=0.1)
    p.add_argument("--patience",        type=int,   default=20)
    p.add_argument("--batch_size",      type=int,   default=512)
    p.add_argument("--weight_decay",    type=float, default=5e-4)
    p.add_argument("--n_splits",        type=int,   default=10)
    p.add_argument("--K",               type=int,   default=3)
    p.add_argument("--gcn_hidden",      type=int,   default=256)
    p.add_argument("--gcn_out",         type=int,   default=128)
    p.add_argument("--clf_hidden",      type=int,   default=64)
    p.add_argument("--dropout",         type=float, default=0.3)
    p.add_argument("--alpha",           type=float, default=0.9)
    p.add_argument("--beta",            type=float, default=0.9)
    p.add_argument("--ckpt_dir",        type=str,   default="checkpoints")
    p.add_argument("--log_dir",         type=str,   default="logs")
    return p.parse_args()


def load_all_fc(dataset):
    """Load all FC matrices once (cached for per-fold graph building)."""
    print("[train] Loading all FC matrices for graph building ...")
    all_fmri = torch.stack([dataset._load_fc_matrix(r[1]) for r in dataset.records])
    all_pheno = torch.stack([
        torch.tensor(r[0][dataset.pheno_features].values.astype(np.float32))
        for r in dataset.records
    ])
    all_pheno = (all_pheno - dataset.pheno_mean) / dataset.pheno_std
    return all_fmri, all_pheno


def build_A_fold(all_fmri, all_pheno, train_idx, alpha, beta):
    """Build population graph using ONLY training subjects to avoid data leakage.

    For all N subjects, edge weights are computed using train-set-derived
    sigma (bandwidth), so validation subjects' features do not influence
    the graph structure.
    """
    idx = _TRIU_IDX
    N = all_fmri.shape[0]

    X_full = all_fmri[:, idx[0], idx[1]]  # (N, 6670)
    P_full = all_pheno                     # (N, M)

    # Compute sigmas from training set only
    X_train = X_full[train_idx]
    P_train = P_full[train_idx]

    D_pheno_train = torch.cdist(P_train, P_train, p=2)
    sigma_pheno = D_pheno_train.mean().clamp(min=1e-8)

    X_train_norm = F.normalize(X_train, p=2, dim=1)
    cos_train = X_train_norm @ X_train_norm.T
    D_func_train = 1.0 - cos_train
    sigma_func = D_func_train.mean().clamp(min=1e-8)

    # Build full N×N graph using train-derived sigmas
    D_pheno_sq = torch.cdist(P_full, P_full, p=2).pow(2)
    A_pheno = torch.exp(-D_pheno_sq / (2 * sigma_pheno ** 2))

    X_full_norm = F.normalize(X_full, p=2, dim=1)
    cos_full = X_full_norm @ X_full_norm.T
    D_func_sq = (1.0 - cos_full).pow(2)
    A_func = torch.exp(-D_func_sq / (2 * sigma_func ** 2))

    A_init = alpha * A_pheno + (1 - alpha) * A_func
    A_final = beta * A_pheno + (1 - beta) * A_init
    A_final.fill_diagonal_(0.0)

    print(f"  [graph] A_fold built: {A_final.shape}, sigma_pheno={sigma_pheno:.4f}, sigma_func={sigma_func:.4f}")
    return A_final


# ── Phase 1 ──────────────────────────────────────────────────────────────────

def train_phase1_epoch(model, loader, optimizer, criterion_cl, device):
    model.pheno_encoder.train()
    model.fmri_encoder.train()
    total_loss, total = 0, 0

    for fmri_flat, pheno, label, batch_idx in loader:
        fmri_flat, pheno, batch_idx = fmri_flat.to(device), pheno.to(device), batch_idx.to(device)

        A_batch = model.classifier.A_global[batch_idx][:, batch_idx]
        Z_pheno = model.pheno_encoder(pheno)
        Z_fmri = model.fmri_encoder(fmri_flat, A_batch)
        loss = criterion_cl(Z_pheno, Z_fmri)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * label.size(0)
        total += label.size(0)

    return {'cl_loss': total_loss / total}


@torch.no_grad()
def eval_phase1(model, loader, criterion_cl, device):
    model.eval()
    total_loss, total = 0, 0

    for fmri_flat, pheno, label, batch_idx in loader:
        fmri_flat, pheno, batch_idx = fmri_flat.to(device), pheno.to(device), batch_idx.to(device)

        A_batch = model.classifier.A_global[batch_idx][:, batch_idx]
        Z_pheno = model.pheno_encoder(pheno)
        Z_fmri = model.fmri_encoder(fmri_flat, A_batch)
        loss = criterion_cl(Z_pheno, Z_fmri)

        total_loss += loss.item() * label.size(0)
        total += label.size(0)

    return {'cl_loss': total_loss / total}


# ── Phase 3 (CE + λ·CL) ─────────────────────────────────────────────────────

def train_phase3_epoch(model, loader, optimizer, criterion_ce, criterion_cl, lambda_cl, device):
    model.train()
    total_loss, total_ce, total_cl, correct, total = 0, 0, 0, 0, 0

    for fmri, pheno, label, batch_idx in loader:
        fmri, pheno, label, batch_idx = (
            fmri.to(device), pheno.to(device), label.to(device), batch_idx.to(device)
        )

        logits, z_pheno, z_fmri = model(fmri, pheno, batch_idx)
        loss_ce = criterion_ce(logits, label)
        loss_cl = criterion_cl(z_pheno, z_fmri)
        loss = loss_ce + lambda_cl * loss_cl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = label.size(0)
        total_loss += loss.item() * bs
        total_ce += loss_ce.item() * bs
        total_cl += loss_cl.item() * bs
        correct += (logits.argmax(1) == label).sum().item()
        total += bs

    return {'loss': total_loss / total, 'ce': total_ce / total, 'cl': total_cl / total, 'acc': correct / total}


@torch.no_grad()
def eval_phase3(model, loader, criterion_ce, criterion_cl, lambda_cl, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []

    for fmri, pheno, label, batch_idx in loader:
        fmri, pheno, label, batch_idx = (
            fmri.to(device), pheno.to(device), label.to(device), batch_idx.to(device)
        )

        logits, z_pheno, z_fmri = model(fmri, pheno, batch_idx)
        loss_ce = criterion_ce(logits, label)
        loss_cl = criterion_cl(z_pheno, z_fmri)
        loss = loss_ce + lambda_cl * loss_cl

        bs = label.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == label).sum().item()
        total += bs

        probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) >= 2 else 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)  # = sensitivity
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'loss': total_loss / total, 'acc': correct / total, 'auc': auc,
        'f1': f1, 'precision': precision, 'recall': recall,
        'sensitivity': sensitivity, 'specificity': specificity,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_phase1(logs, fold, log_dir):
    epochs = [l['epoch'] for l in logs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [l['train_cl'] for l in logs], label='Train CL Loss')
    ax.plot(epochs, [l['val_cl'] for l in logs], label='Val CL Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Contrastive Loss')
    ax.set_title(f'Phase 1 — Contrastive Alignment (Fold {fold})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, f'phase1_fold{fold}.png'), dpi=150)
    plt.close(fig)


def plot_phase3(logs, fold, log_dir):
    epochs = [l['epoch'] for l in logs]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, [l['train_loss'] for l in logs], label='Train Loss')
    ax.plot(epochs, [l['train_ce'] for l in logs], label='Train CE', linestyle='--')
    ax.plot(epochs, [l['train_cl'] for l in logs], label='Train CL', linestyle='--')
    ax.plot(epochs, [l['val_loss'] for l in logs], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ACC + AUC
    ax = axes[0, 1]
    ax.plot(epochs, [l['train_acc'] for l in logs], label='Train ACC')
    ax.plot(epochs, [l['val_acc'] for l in logs], label='Val ACC')
    ax.plot(epochs, [l['val_auc'] for l in logs], label='Val AUC')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('ACC & AUC')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F1, Precision, Recall
    ax = axes[1, 0]
    ax.plot(epochs, [l['val_f1'] for l in logs], label='F1')
    ax.plot(epochs, [l['val_precision'] for l in logs], label='Precision')
    ax.plot(epochs, [l['val_recall'] for l in logs], label='Recall')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('F1 / Precision / Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sensitivity, Specificity
    ax = axes[1, 1]
    ax.plot(epochs, [l['val_sensitivity'] for l in logs], label='Sensitivity')
    ax.plot(epochs, [l['val_specificity'] for l in logs], label='Specificity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Sensitivity / Specificity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Phase 3 — Classification (Fold {fold})', fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, f'phase3_fold{fold}.png'), dpi=150)
    plt.close(fig)


def plot_fold_summary(fold_metrics, log_dir):
    metrics = list(fold_metrics.keys())
    means = [np.mean(v) for v in fold_metrics.values()]
    stds = [np.std(v) for v in fold_metrics.values()]
    n_folds = len(list(fold_metrics.values())[0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart with error bars
    ax = axes[0]
    x = np.arange(len(metrics))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(f'{n_folds}-Fold CV Results (mean +/- std)')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{m:.4f}', ha='center', va='bottom', fontsize=9)

    # Per-fold line chart
    ax = axes[1]
    folds_x = np.arange(1, n_folds + 1)
    for metric_name, values in fold_metrics.items():
        ax.plot(folds_x, values, marker='o', label=metric_name)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title('Per-Fold Metrics')
    ax.set_xticks(folds_x)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, 'fold_summary.png'), dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    print(f"Device: {device}")
    print(f"Phase 1: epochs={args.phase1_epochs}, lr={args.phase1_lr}, tau={args.temperature}")
    print(f"Phase 3: epochs={args.epochs}, lr={args.lr}, λ_CL={args.lambda_contrast}, "
          f"encoder_lr_scale={args.encoder_lr_scale}")

    dataset = ABIDEDataset(
        pheno_csv=os.path.join(args.data_dir, "final_pheno_for_fusion.csv"),
        fmri_dir=os.path.join(args.data_dir, "ABIDE_pcp", "cpac", "nofilt_noglobal"),
    )

    all_fmri, all_pheno = load_all_fc(dataset)
    labels = np.array([int(dataset[i][2].item()) for i in range(len(dataset))])
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    fold_acc, fold_auc, fold_f1, fold_sen, fold_spe, fold_prec, fold_rec = [], [], [], [], [], [], []
    all_fold_logs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*60}\nFold {fold+1}/{args.n_splits}\n{'='*60}")

        A_fold = build_A_fold(all_fmri, all_pheno, train_idx, args.alpha, args.beta)

        train_loader = DataLoader(dataset, batch_size=args.batch_size,
                                  sampler=SubsetRandomSampler(train_idx), num_workers=0, drop_last=False)
        val_loader = DataLoader(dataset, batch_size=args.batch_size,
                                sampler=SubsetRandomSampler(val_idx), num_workers=0)

        model = ASDDiagnosisModel(
            A_global=A_fold, pheno_dim=len(PHENO_FEATURES),
            fmri_dim=dataset.f_num,
            gcn_hidden=args.gcn_hidden, gcn_out=args.gcn_out,
            K=args.K, clf_hidden=args.clf_hidden,
            dropout=args.dropout,
        ).to(device)

        # ── Phase 1 ──────────────────────────────────────────────────────────
        print(f"\n  --- Phase 1: Contrastive Alignment ---")
        phase1_params = list(model.pheno_encoder.parameters()) + list(model.fmri_encoder.parameters())
        optimizer_p1 = torch.optim.Adam(phase1_params, lr=args.phase1_lr, weight_decay=args.weight_decay)
        scheduler_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, mode='min', factor=0.5, patience=5)
        criterion_cl = ContrastiveLoss(temperature=args.temperature)

        best_cl_loss, no_improve_p1, best_encoder_state = float('inf'), 0, None
        p1_logs = []

        for epoch in range(args.phase1_epochs):
            train_m = train_phase1_epoch(model, train_loader, optimizer_p1, criterion_cl, device)
            val_m = eval_phase1(model, val_loader, criterion_cl, device)
            scheduler_p1.step(val_m['cl_loss'])
            lr = optimizer_p1.param_groups[0]['lr']

            p1_logs.append({'epoch': epoch + 1, 'train_cl': train_m['cl_loss'],
                            'val_cl': val_m['cl_loss'], 'lr': lr})

            print(f"  [P1 | Fold {fold+1} | Ep {epoch+1}/{args.phase1_epochs}] "
                  f"train_CL={train_m['cl_loss']:.4f} | val_CL={val_m['cl_loss']:.4f} | lr={lr:.2e}")

            if val_m['cl_loss'] < best_cl_loss:
                best_cl_loss = val_m['cl_loss']
                no_improve_p1 = 0
                best_encoder_state = {
                    'pheno_encoder': copy.deepcopy(model.pheno_encoder.state_dict()),
                    'fmri_encoder': copy.deepcopy(model.fmri_encoder.state_dict()),
                }
            else:
                no_improve_p1 += 1
                if no_improve_p1 >= args.phase1_patience:
                    print(f"  Phase 1 early stopping at epoch {epoch+1}")
                    break

        if best_encoder_state:
            model.pheno_encoder.load_state_dict(best_encoder_state['pheno_encoder'])
            model.fmri_encoder.load_state_dict(best_encoder_state['fmri_encoder'])
        print(f"  Phase 1 best val CL loss: {best_cl_loss:.4f}")
        plot_phase1(p1_logs, fold + 1, args.log_dir)

        # ── Phase 3 (CE + λ·CL) ─────────────────────────────────────────────
        print(f"\n  --- Phase 3: Classification (L=CE+λ·CL) ---")
        optimizer_p3 = torch.optim.Adam([
            {'params': model.pheno_encoder.parameters(), 'lr': args.lr * args.encoder_lr_scale},
            {'params': model.fmri_encoder.parameters(),  'lr': args.lr * args.encoder_lr_scale},
            {'params': model.fusion.parameters(),        'lr': args.lr},
            {'params': model.classifier.parameters(),    'lr': args.lr},
        ], weight_decay=args.weight_decay)
        scheduler_p3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p3, mode='max', factor=0.5, patience=5)
        criterion_ce = nn.CrossEntropyLoss()
        criterion_cl_p3 = ContrastiveLoss(temperature=args.temperature)

        best_auc, no_improve_p3, best_state = 0.0, 0, None
        ckpt_path = os.path.join(args.ckpt_dir, f"best_fold{fold+1}.pt")
        p3_logs = []

        for epoch in range(args.epochs):
            train_m = train_phase3_epoch(model, train_loader, optimizer_p3,
                                         criterion_ce, criterion_cl_p3, args.lambda_contrast, device)
            val_m = eval_phase3(model, val_loader, criterion_ce, criterion_cl_p3, args.lambda_contrast, device)
            scheduler_p3.step(val_m['auc'])
            lr = optimizer_p3.param_groups[2]['lr']

            p3_logs.append({
                'epoch': epoch + 1,
                'train_loss': train_m['loss'], 'train_ce': train_m['ce'],
                'train_cl': train_m['cl'], 'train_acc': train_m['acc'],
                'val_loss': val_m['loss'], 'val_acc': val_m['acc'], 'val_auc': val_m['auc'],
                'val_f1': val_m['f1'], 'val_precision': val_m['precision'],
                'val_recall': val_m['recall'], 'val_sensitivity': val_m['sensitivity'],
                'val_specificity': val_m['specificity'], 'lr': lr,
            })

            print(f"  [P3 | Fold {fold+1} | Ep {epoch+1}/{args.epochs}] "
                  f"loss={train_m['loss']:.4f} (CE={train_m['ce']:.4f} CL={train_m['cl']:.4f}) | "
                  f"acc={train_m['acc']:.4f} | val_acc={val_m['acc']:.4f} | val_auc={val_m['auc']:.4f} | "
                  f"F1={val_m['f1']:.4f} | Sen={val_m['sensitivity']:.4f} | Spe={val_m['specificity']:.4f} | lr={lr:.2e}")

            if val_m['auc'] > best_auc:
                best_auc = val_m['auc']
                no_improve_p3 = 0
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, ckpt_path)
            else:
                no_improve_p3 += 1
                if no_improve_p3 >= args.patience:
                    print(f"  Phase 3 early stopping at epoch {epoch+1}")
                    break

        if best_state:
            model.load_state_dict(best_state)
        final = eval_phase3(model, val_loader, criterion_ce, criterion_cl_p3, args.lambda_contrast, device)
        fold_acc.append(final['acc'])
        fold_auc.append(final['auc'])
        fold_f1.append(final['f1'])
        fold_sen.append(final['sensitivity'])
        fold_spe.append(final['specificity'])
        fold_prec.append(final['precision'])
        fold_rec.append(final['recall'])
        print(f"  Fold {fold+1} best: ACC={final['acc']:.4f} | AUC={final['auc']:.4f} | "
              f"F1={final['f1']:.4f} | Sen={final['sensitivity']:.4f} | Spe={final['specificity']:.4f} | "
              f"Prec={final['precision']:.4f} | Rec={final['recall']:.4f}")

        plot_phase3(p3_logs, fold + 1, args.log_dir)
        all_fold_logs.append({'fold': fold + 1, 'phase1': p1_logs, 'phase3': p3_logs,
                              'best': {k: float(v) for k, v in final.items()}})

    # ── Summary ────────────────────────────────────────────────────────────
    fold_metrics = {
        'ACC': fold_acc, 'AUC': fold_auc, 'F1': fold_f1,
        'Sensitivity': fold_sen, 'Specificity': fold_spe,
        'Precision': fold_prec, 'Recall': fold_rec,
    }

    print(f"\n{'='*60}\nFinal Results ({args.n_splits}-Fold CV)\n{'='*60}")
    for name, vals in fold_metrics.items():
        print(f"  {name:>12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")

    plot_fold_summary(fold_metrics, args.log_dir)

    # Save full training log
    log_path = os.path.join(args.log_dir, 'training_log.json')
    log_data = {
        'args': vars(args),
        'summary': {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                    for k, v in fold_metrics.items()},
        'folds': all_fold_logs,
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"Training log saved to {log_path}")
    print(f"Plots saved to {args.log_dir}/")


if __name__ == "__main__":
    main()
