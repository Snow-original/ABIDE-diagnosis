"""
train_parameter.py — Hyperparameter Search
============================================
对关键超参数进行网格搜索, 用 3-fold 快速评估, 找到最佳组合后可用 10-fold 正式训练。

Usage:
    python train_parameter.py
    python train_parameter.py --quick    # 只跑 2 折, 更快
"""

from __future__ import annotations

import argparse
import os
import copy
import time
import itertools
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


# ══════════════════════════════════════════════════════════════════════════════
# Search space
# ══════════════════════════════════════════════════════════════════════════════

PARAM_GRID = {
    'phase1_lr':        [5e-4, 1e-3, 2e-3],
    'temperature':      [0.3, 0.5, 0.7],
    'lr':               [3e-4, 5e-4, 1e-3],
    'lambda_contrast':  [0.3, 0.5, 1.0],
    'dropout':          [0.2, 0.3, 0.5],
    'encoder_lr_scale': [0.05, 0.1, 0.2],
    'batch_size':       [256, 512, 1024],
}

# Fixed params (not searched)
FIXED = dict(
    phase1_epochs=30,
    phase1_patience=7,
    epochs=80,
    patience=15,
    weight_decay=5e-4,
    K=3,
    gcn_hidden=256,
    gcn_out=128,
    clf_hidden=64,
    alpha=0.9,
    beta=0.9,
)


def load_all_fc(dataset):
    all_fmri = torch.stack([dataset._load_fc_matrix(r[1]) for r in dataset.records])
    all_pheno = torch.stack([
        torch.tensor(r[0][dataset.pheno_features].values.astype(np.float32))
        for r in dataset.records
    ])
    all_pheno = (all_pheno - dataset.pheno_mean) / dataset.pheno_std
    return all_fmri, all_pheno


def build_A_fold(all_fmri, all_pheno, train_idx, alpha, beta):
    idx = _TRIU_IDX
    X_full = all_fmri[:, idx[0], idx[1]]
    P_full = all_pheno

    X_train, P_train = X_full[train_idx], P_full[train_idx]
    sigma_pheno = torch.cdist(P_train, P_train, p=2).mean().clamp(min=1e-8)
    X_tn = F.normalize(X_train, p=2, dim=1)
    sigma_func = (1.0 - X_tn @ X_tn.T).mean().clamp(min=1e-8)

    D_pheno_sq = torch.cdist(P_full, P_full, p=2).pow(2)
    A_pheno = torch.exp(-D_pheno_sq / (2 * sigma_pheno ** 2))
    X_fn = F.normalize(X_full, p=2, dim=1)
    D_func_sq = (1.0 - X_fn @ X_fn.T).pow(2)
    A_func = torch.exp(-D_func_sq / (2 * sigma_func ** 2))

    A_init = alpha * A_pheno + (1 - alpha) * A_func
    A_final = beta * A_pheno + (1 - beta) * A_init
    A_final.fill_diagonal_(0.0)
    return A_final


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
    return total_loss / total


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
    return total_loss / total


def train_phase3_epoch(model, loader, optimizer, criterion_ce, criterion_cl, lambda_cl, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for fmri, pheno, label, batch_idx in loader:
        fmri, pheno, label, batch_idx = (
            fmri.to(device), pheno.to(device), label.to(device), batch_idx.to(device))
        logits, z_pheno, z_fmri = model(fmri, pheno, batch_idx)
        loss = criterion_ce(logits, label) + lambda_cl * criterion_cl(z_pheno, z_fmri)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        bs = label.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == label).sum().item()
        total += bs
    return total_loss / total, correct / total


@torch.no_grad()
def eval_phase3(model, loader, criterion_ce, criterion_cl, lambda_cl, device):
    model.eval()
    correct, total = 0, 0
    all_probs, all_labels = [], []
    for fmri, pheno, label, batch_idx in loader:
        fmri, pheno, label, batch_idx = (
            fmri.to(device), pheno.to(device), label.to(device), batch_idx.to(device))
        logits, _, _ = model(fmri, pheno, batch_idx)
        bs = label.size(0)
        correct += (logits.argmax(1) == label).sum().item()
        total += bs
        probs = torch.softmax(logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) >= 2 else 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall_val = recall_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'acc': correct / total, 'auc': auc, 'f1': f1,
        'precision': precision, 'recall': recall_val,
        'sensitivity': sensitivity, 'specificity': specificity,
    }


def run_one_config(config, dataset, all_fmri, all_pheno, labels, device, n_folds=3):
    """Train with one hyperparameter config, return mean val AUC across folds."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        A_fold = build_A_fold(all_fmri, all_pheno, train_idx,
                              config.get('alpha', FIXED['alpha']),
                              config.get('beta', FIXED['beta']))

        train_loader = DataLoader(dataset, batch_size=config['batch_size'],
                                  sampler=SubsetRandomSampler(train_idx), num_workers=0, drop_last=False)
        val_loader = DataLoader(dataset, batch_size=config['batch_size'],
                                sampler=SubsetRandomSampler(val_idx), num_workers=0)

        model = ASDDiagnosisModel(
            A_global=A_fold, pheno_dim=len(PHENO_FEATURES),
            fmri_dim=dataset.f_num,
            gcn_hidden=config['gcn_hidden'], gcn_out=config['gcn_out'],
            K=config['K'], clf_hidden=config['clf_hidden'],
            dropout=config['dropout'],
        ).to(device)

        # Phase 1
        phase1_params = list(model.pheno_encoder.parameters()) + list(model.fmri_encoder.parameters())
        opt_p1 = torch.optim.Adam(phase1_params, lr=config['phase1_lr'], weight_decay=config['weight_decay'])
        sch_p1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_p1, mode='min', factor=0.5, patience=5)
        cl_loss_fn = ContrastiveLoss(temperature=config['temperature'])

        best_cl, no_imp, best_enc = float('inf'), 0, None
        for ep in range(config['phase1_epochs']):
            train_phase1_epoch(model, train_loader, opt_p1, cl_loss_fn, device)
            val_cl = eval_phase1(model, val_loader, cl_loss_fn, device)
            sch_p1.step(val_cl)
            if val_cl < best_cl:
                best_cl = val_cl
                no_imp = 0
                best_enc = {
                    'pheno': copy.deepcopy(model.pheno_encoder.state_dict()),
                    'fmri': copy.deepcopy(model.fmri_encoder.state_dict()),
                }
            else:
                no_imp += 1
                if no_imp >= config['phase1_patience']:
                    break

        if best_enc:
            model.pheno_encoder.load_state_dict(best_enc['pheno'])
            model.fmri_encoder.load_state_dict(best_enc['fmri'])

        # Phase 3
        opt_p3 = torch.optim.Adam([
            {'params': model.pheno_encoder.parameters(), 'lr': config['lr'] * config['encoder_lr_scale']},
            {'params': model.fmri_encoder.parameters(),  'lr': config['lr'] * config['encoder_lr_scale']},
            {'params': model.fusion.parameters(),        'lr': config['lr']},
            {'params': model.classifier.parameters(),    'lr': config['lr']},
        ], weight_decay=config['weight_decay'])
        sch_p3 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_p3, mode='max', factor=0.5, patience=5)
        ce_loss_fn = nn.CrossEntropyLoss()
        cl_p3 = ContrastiveLoss(temperature=config['temperature'])

        best_auc, no_imp, best_metrics = -1.0, 0, None
        for ep in range(config['epochs']):
            train_phase3_epoch(model, train_loader, opt_p3, ce_loss_fn, cl_p3, config['lambda_contrast'], device)
            val_m = eval_phase3(model, val_loader, ce_loss_fn, cl_p3, config['lambda_contrast'], device)
            sch_p3.step(val_m['auc'])
            if val_m['auc'] > best_auc:
                best_auc = val_m['auc']
                best_metrics = val_m.copy()
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= config['patience']:
                    break

        if best_metrics is None:
            best_metrics = eval_phase3(model, val_loader, ce_loss_fn, cl_p3, config['lambda_contrast'], device)
        fold_results.append(best_metrics)

    metric_keys = ['acc', 'auc', 'f1', 'precision', 'recall', 'sensitivity', 'specificity']
    summary = {}
    for k in metric_keys:
        vals = [float(r[k]) for r in fold_results]
        summary[f'mean_{k}'] = float(np.mean(vals))
        summary[f'std_{k}'] = float(np.std(vals))
    return summary


def sample_configs(param_grid, n_samples=30, seed=42):
    """Random sample from grid (faster than full grid search)."""
    rng = np.random.RandomState(seed)
    keys = sorted(param_grid.keys())
    configs = []
    for _ in range(n_samples):
        cfg = {}
        for k in keys:
            val = rng.choice(param_grid[k])
            cfg[k] = int(val) if isinstance(val, (np.integer,)) else float(val)
        configs.append(cfg)
    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default=".")
    parser.add_argument("--n_samples",  type=int, default=30, help="Number of random configs to try")
    parser.add_argument("--n_folds",    type=int, default=3,  help="CV folds for search (3=fast, 5=more reliable)")
    parser.add_argument("--quick",      action="store_true",  help="Quick mode: 2 folds, 15 samples")
    parser.add_argument("--output",     type=str, default="param_search_results.json")
    parser.add_argument("--log_dir",   type=str, default="logs_param_search")
    args = parser.parse_args()

    if args.quick:
        args.n_folds = 2
        args.n_samples = 15

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Search: {args.n_samples} configs × {args.n_folds}-fold CV")

    dataset = ABIDEDataset(
        pheno_csv=os.path.join(args.data_dir, "final_pheno_for_fusion.csv"),
        fmri_dir=os.path.join(args.data_dir, "ABIDE_pcp", "cpac", "nofilt_noglobal"),
    )

    print("[search] Loading FC matrices ...")
    all_fmri, all_pheno = load_all_fc(dataset)
    labels = np.array([int(dataset[i][2].item()) for i in range(len(dataset))])

    configs = sample_configs(PARAM_GRID, n_samples=args.n_samples)
    results = []
    best_auc, best_config = 0.0, None
    t0 = time.time()

    for i, search_params in enumerate(configs):
        config = {**FIXED, **search_params}
        config_str = " | ".join(f"{k}={v}" for k, v in sorted(search_params.items()))

        print(f"\n{'─'*60}")
        print(f"Config {i+1}/{len(configs)}: {config_str}")
        print(f"{'─'*60}")

        try:
            summary = run_one_config(config, dataset, all_fmri, all_pheno, labels, device, args.n_folds)
        except Exception as e:
            print(f"  FAILED: {e}")
            summary = {f'{p}_{k}': 0.0 for p in ('mean', 'std')
                       for k in ('acc', 'auc', 'f1', 'precision', 'recall', 'sensitivity', 'specificity')}

        result = {**search_params, **{k: round(v, 4) for k, v in summary.items()}}
        results.append(result)

        elapsed = (time.time() - t0) / 60
        print(f"  AUC={summary['mean_auc']:.4f}±{summary['std_auc']:.4f} | "
              f"ACC={summary['mean_acc']:.4f} | F1={summary['mean_f1']:.4f} | "
              f"Sen={summary['mean_sensitivity']:.4f} | Spe={summary['mean_specificity']:.4f} | "
              f"{elapsed:.1f} min")

        if summary['mean_auc'] > best_auc:
            best_auc = summary['mean_auc']
            best_config = search_params
            print(f"  ★ NEW BEST ★")

    # Sort by AUC
    results.sort(key=lambda x: x['mean_auc'], reverse=True)

    print(f"\n{'='*60}")
    print(f"Search Complete ({(time.time()-t0)/60:.1f} min)")
    print(f"{'='*60}")
    search_keys = sorted(PARAM_GRID.keys())

    print(f"\nTop 5 configs:")
    for i, r in enumerate(results[:5]):
        params = " | ".join(f"{k}={r[k]}" for k in search_keys)
        print(f"  {i+1}. AUC={r['mean_auc']:.4f}±{r['std_auc']:.4f} | "
              f"ACC={r['mean_acc']:.4f} | F1={r['mean_f1']:.4f} | "
              f"Sen={r['mean_sensitivity']:.4f} | Spe={r['mean_specificity']:.4f}")
        print(f"     {params}")

    print(f"\nBest config for train-2.py:")
    if best_config:
        cmd_args = " ".join(f"--{k} {v}" for k, v in best_config.items())
        print(f"  python train-2.py {cmd_args}")

    # ── Save results JSON ────────────────────────────────────────────────
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_results = [{k: to_native(v) for k, v in r.items()} for r in results]
    os.makedirs(args.log_dir, exist_ok=True)
    output_path = os.path.join(args.log_dir, args.output)
    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    print(f"\nFull results saved to {output_path}")

    # ── Generate plots ───────────────────────────────────────────────────
    if len(results) < 2:
        return

    metric_names = ['auc', 'acc', 'f1', 'sensitivity', 'specificity']
    n_show = min(len(results), 15)
    top = results[:n_show]

    # 1) Top configs comparison (grouped bar)
    fig, ax = plt.subplots(figsize=(max(10, n_show * 0.8), 6))
    x = np.arange(n_show)
    width = 0.15
    for j, m in enumerate(metric_names):
        vals = [r[f'mean_{m}'] for r in top]
        ax.bar(x + j * width, vals, width, label=m.upper())
    ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels([f'#{i+1}' for i in range(n_show)], fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title(f'Top {n_show} Configs — All Metrics')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(args.log_dir, 'top_configs_comparison.png'), dpi=150)
    plt.close(fig)

    # 2) Per-hyperparameter impact (one subplot per search param)
    fig, axes = plt.subplots(2, (len(search_keys) + 1) // 2, figsize=(5 * ((len(search_keys) + 1) // 2), 10))
    axes = axes.flatten()
    for idx, param in enumerate(search_keys):
        ax = axes[idx]
        param_vals = sorted(set(r[param] for r in results))
        for m in ['auc', 'acc', 'f1']:
            means_by_val = []
            for pv in param_vals:
                subset = [r[f'mean_{m}'] for r in results if r[param] == pv]
                means_by_val.append(np.mean(subset) if subset else 0)
            ax.plot([str(v) for v in param_vals], means_by_val, marker='o', label=m.upper())
        ax.set_xlabel(param)
        ax.set_ylabel('Mean Score')
        ax.set_title(f'Impact of {param}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    for idx in range(len(search_keys), len(axes)):
        axes[idx].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.log_dir, 'param_impact.png'), dpi=150)
    plt.close(fig)

    # 3) Best config radar chart
    best = results[0]
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    values = [best[f'mean_{m}'] for m in metric_names]
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, color='steelblue')
    ax.fill(angles, values, alpha=0.25, color='steelblue')
    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], [m.upper() for m in metric_names])
    ax.set_ylim(0, 1)
    ax.set_title('Best Config — Metric Radar', pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(args.log_dir, 'best_config_radar.png'), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {args.log_dir}/")


if __name__ == "__main__":
    main()
