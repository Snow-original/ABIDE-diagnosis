"""
model-2.py — Two-Stage ASD Diagnosis Model (v2)
===================================================
改动: 用 2 层 GCN + MLP 替换 HierarchicalAttentionGCN
原因: 层次化注意力 + 动态邻接在 871 人小图上过拟合/坍塌
"""
from __future__ import annotations

import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PHENO_FEATURES = [
    'ADOS_SOCIAL', 'ADI_R_ONSET_TOTAL_D', 'ADI_RRB_TOTAL_C',
    'ADI_R_VERBAL_TOTAL_BV', 'ADOS_STEREO_BEHAV', 'ADOS_TOTAL',
    'VIQ', 'FIQ', 'ADI_R_SOCIAL_TOTAL_A', 'SEX', 'SITE_ID', 'AGE_AT_SCAN',
]

_TRIU_IDX = torch.triu_indices(116, 116, offset=1)
FC_FLAT_DIM = 116 * 115 // 2


# ══════════════════════════════════════════════════════════════════════════════
# Dataset (same as model.py)
# ══════════════════════════════════════════════════════════════════════════════

class ABIDEDataset(Dataset):
    def __init__(self, pheno_csv: str, fmri_dir: str, pheno_features: list = None, f_num: int = 2505):
        from sklearn.ensemble import RandomForestClassifier

        self.fmri_dir = fmri_dir
        self.pheno_features = pheno_features or PHENO_FEATURES
        self.f_num = f_num

        pheno_df = pd.read_csv(pheno_csv)
        pheno_df['label'] = (pheno_df['DX_GROUP'] == 1).astype(int)

        records, missing = [], 0
        for _, row in pheno_df.iterrows():
            path = self._resolve_fmri_path(int(row['SUB_ID']))
            if path is not None:
                records.append((row, path))
            else:
                missing += 1

        if not records:
            raise RuntimeError(f"No fMRI .1D files in '{fmri_dir}'")

        self.records = records
        n_asd = sum(r[0]['label'] for r in records)
        print(f"[ABIDEDataset] {len(records)} subjects (ASD={n_asd}, TD={len(records)-n_asd}), {missing} skipped")

        all_pheno = np.array([r[0][self.pheno_features].values.astype(np.float32) for r in records])
        self.pheno_mean = torch.tensor(all_pheno.mean(axis=0), dtype=torch.float32)
        self.pheno_std = torch.tensor(all_pheno.std(axis=0).clip(min=1e-8), dtype=torch.float32)

        print("[ABIDEDataset] Loading all FC matrices ...")
        idx = _TRIU_IDX
        all_fc_flat = []
        for _, path in records:
            fc = self._load_fc_matrix(path)
            all_fc_flat.append(fc[idx[0], idx[1]].numpy())
        all_fc_flat = np.array(all_fc_flat)
        labels = np.array([int(r[0]['label']) for r in records])

        print(f"[ABIDEDataset] Random forest feature selection: 6670 → {f_num} ...")
        rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        rf.fit(all_fc_flat, labels)
        importances = rf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:f_num]
        self.feature_indices = torch.tensor(np.sort(top_idx), dtype=torch.long)
        self.all_fc_selected = torch.tensor(all_fc_flat[:, np.sort(top_idx)], dtype=torch.float32)
        print(f"[ABIDEDataset] Feature selection done. Selected shape: {self.all_fc_selected.shape}")

    def _resolve_fmri_path(self, sub_id: int) -> str | None:
        matches = glob.glob(os.path.join(self.fmri_dir, f"*_{sub_id:07d}_rois_aal.1D"))
        return matches[0] if matches else None

    @staticmethod
    def _load_fc_matrix(path: str) -> torch.Tensor:
        ts = np.loadtxt(path, comments='#')
        assert ts.ndim == 2 and ts.shape[1] == 116, f"Bad shape {ts.shape} in {path}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fc = np.corrcoef(ts.T)
        fc = np.nan_to_num(fc, nan=0.0)
        np.fill_diagonal(fc, 0.0)
        fc = np.arctanh(np.clip(fc, -0.9999, 0.9999))
        return torch.tensor(fc, dtype=torch.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row, _ = self.records[idx]
        fmri_flat = self.all_fc_selected[idx]
        pheno = torch.tensor(row[self.pheno_features].values.astype(np.float32))
        pheno = (pheno - self.pheno_mean) / self.pheno_std
        label = torch.tensor(int(row['label']), dtype=torch.long)
        return fmri_flat, pheno, label, torch.tensor(idx, dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
# Population Graph Builder (same)
# ══════════════════════════════════════════════════════════════════════════════

class PopulationGraphBuilder(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(self, all_fmri: torch.Tensor, all_pheno: torch.Tensor) -> torch.Tensor:
        idx = _TRIU_IDX.to(all_fmri.device)
        X = all_fmri[:, idx[0], idx[1]]
        D_pheno_sq = torch.cdist(all_pheno, all_pheno, p=2).pow(2)
        sigma_pheno = D_pheno_sq.sqrt().mean().clamp(min=1e-8)
        A_pheno = torch.exp(-D_pheno_sq / (2 * sigma_pheno ** 2))
        X_norm = F.normalize(X, p=2, dim=1)
        cos_sim = X_norm @ X_norm.T
        D_func = 1.0 - cos_sim
        D_func_sq = D_func.pow(2)
        sigma_func = D_func.mean().clamp(min=1e-8)
        A_func = torch.exp(-D_func_sq / (2 * sigma_func ** 2))
        A_init = self.alpha * A_pheno + (1 - self.alpha) * A_func
        A_final = self.beta * A_pheno + (1 - self.beta) * A_init
        A_final.fill_diagonal_(0.0)
        return A_final


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def sym_normalize(A: torch.Tensor) -> torch.Tensor:
    A_hat = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    deg = A_hat.sum(dim=1).clamp(min=1e-8)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * A_hat * d_inv_sqrt.unsqueeze(0)


class ChebyshevConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, K: int = 3):
        super().__init__()
        self.K = K
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(in_dim, out_dim)) for _ in range(K + 1)
        ])
        for w in self.W:
            nn.init.xavier_uniform_(w)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        L_hat = -A_norm
        T = [X]
        if self.K >= 1:
            T.append(L_hat @ X)
        for _ in range(2, self.K + 1):
            T.append(2.0 * L_hat @ T[-1] - T[-2])
        return sum(T[k] @ self.W[k] for k in range(self.K + 1))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Encoders + Contrastive Loss 
# ══════════════════════════════════════════════════════════════════════════════

class PhenotypicEncoder(nn.Module):
    def __init__(self, input_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
        )

    def forward(self, pheno: torch.Tensor) -> torch.Tensor:
        return self.net(pheno)


class FMRIEncoder(nn.Module):
    def __init__(self, in_dim: int = 2505, hidden_dim: int = 256, out_dim: int = 128, K: int = 3):
        super().__init__()
        self.cheb = ChebyshevConv(in_dim, hidden_dim, K)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, X: torch.Tensor, A_batch: torch.Tensor) -> torch.Tensor:
        A_norm = sym_normalize(A_batch)
        H = F.relu(self.bn(self.cheb(X, A_norm)))
        return F.relu(self.proj(H))


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        B = z1.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z1.device)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        sim = z1 @ z2.T / self.temperature
        targets = torch.arange(B, device=z1.device)
        return 0.5 * (F.cross_entropy(sim, targets) + F.cross_entropy(sim.T, targets))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Graph-Aware Cross-Attention Fusion 
# ══════════════════════════════════════════════════════════════════════════════

class GraphCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.W_O = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.graph_scale = nn.Parameter(torch.ones(1))

    def forward(self, Z_pheno, H_gcn, A_batch):
        B, D = Z_pheno.shape
        H = self.num_heads
        d = self.head_dim

        Q = self.W_Q(Z_pheno).view(B, H, d)
        K = self.W_K(H_gcn).view(B, H, d)
        V = self.W_V(H_gcn).view(B, H, d)

        attn = torch.einsum('ihd,jhd->hij', Q, K) * self.scale
        attn = attn + self.graph_scale * A_batch.unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('hij,jhd->ihd', attn, V).reshape(B, D)
        out = self.W_O(out)
        return self.norm(out + Z_pheno)


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Classifier 
# ══════════════════════════════════════════════════════════════════════════════

class GCNClassifier(nn.Module):
    """
    Simple 2-layer Chebyshev GCN + MLP head on population graph.
    No dynamic adjacency, no attention layers — just straightforward graph convolution.

    128 → GCN → 64 → GCN → 64 → MLP → 2
    """

    def __init__(
        self,
        A_global: torch.Tensor,
        in_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 2,
        K: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.register_buffer('A_global', A_global)

        self.gcn1 = ChebyshevConv(in_dim, hidden_dim, K)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gcn2 = ChebyshevConv(hidden_dim, hidden_dim, K)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, H_fused: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        A_batch = self.A_global[batch_indices][:, batch_indices]
        A_norm = sym_normalize(A_batch)

        H = F.relu(self.bn1(self.gcn1(H_fused, A_norm)))
        H = self.dropout(H)
        H = F.relu(self.bn2(self.gcn2(H, A_norm)))
        H = self.dropout(H)

        return self.mlp(H)


# ══════════════════════════════════════════════════════════════════════════════
# Full Model
# ══════════════════════════════════════════════════════════════════════════════

class ASDDiagnosisModel(nn.Module):
    def __init__(
        self,
        A_global: torch.Tensor,
        pheno_dim: int = 12,
        fmri_dim: int = 2505,
        gcn_hidden: int = 256,
        gcn_out: int = 128,
        K: int = 3,
        num_heads: int = 8,
        clf_hidden: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pheno_encoder = PhenotypicEncoder(input_dim=pheno_dim)
        self.fmri_encoder = FMRIEncoder(in_dim=fmri_dim, hidden_dim=gcn_hidden, out_dim=gcn_out, K=K)
        self.fusion = GraphCrossAttentionFusion(embed_dim=gcn_out, num_heads=num_heads, dropout=0.1)
        self.classifier = GCNClassifier(
            A_global=A_global, in_dim=gcn_out, hidden_dim=clf_hidden,
            num_classes=num_classes, K=K, dropout=dropout,
        )

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[ASDDiagnosisModel] Total: {total:,} | Trainable: {trainable:,}")

    def forward(self, fmri_flat, pheno, batch_indices):
        A_batch = self.classifier.A_global[batch_indices][:, batch_indices]

        Z_pheno = self.pheno_encoder(pheno)
        Z_fmri = self.fmri_encoder(fmri_flat, A_batch)

        H_fused = self.fusion(Z_pheno, Z_fmri, A_batch)
        logits = self.classifier(H_fused, batch_indices)

        return logits, Z_pheno, Z_fmri


if __name__ == "__main__":
    torch.manual_seed(42)
    B, N, M, D = 16, 871, 12, 2505

    A_dummy = torch.rand(N, N)
    A_dummy = (A_dummy + A_dummy.T) / 2
    A_dummy.fill_diagonal_(0.0)

    model = ASDDiagnosisModel(A_global=A_dummy, pheno_dim=M, fmri_dim=D)

    fmri_flat = torch.randn(B, D)
    pheno = torch.randn(B, M)
    idx = torch.arange(B)

    logits, z_p, z_f = model(fmri_flat, pheno, idx)
    print(f"logits: {logits.shape}, Z_pheno: {z_p.shape}, Z_fmri: {z_f.shape}")
    assert logits.shape == (B, 2) and not torch.isnan(logits).any()

    cl = ContrastiveLoss()
    A_batch = A_dummy[idx][:, idx]
    loss_cl = cl(model.pheno_encoder(pheno), model.fmri_encoder(fmri_flat, A_batch))
    loss_cl.backward()
    print(f"Phase 1 CL loss: {loss_cl.item():.4f}")
    model.zero_grad()

    logits2, z_p2, z_f2 = model(fmri_flat, pheno, idx)
    loss_ce = F.cross_entropy(logits2, torch.randint(0, 2, (B,)))
    loss_cl2 = cl(z_p2, z_f2)
    loss = loss_ce + 0.5 * loss_cl2
    loss.backward()
    print(f"Phase 3 total loss: {loss.item():.4f} (CE={loss_ce.item():.4f} CL={loss_cl2.item():.4f})")

    no_grad = [n for n, p in model.named_parameters() if p.grad is None and p.requires_grad]
    assert len(no_grad) == 0, f"No gradient: {no_grad}"
    print("All gradients flow. Smoke test passed.")
