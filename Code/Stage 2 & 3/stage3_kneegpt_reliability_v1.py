import os
import json
import random
import argparse
import warnings
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

from transformers import CLIPModel, AutoImageProcessor, AutoModel


PRIMARY = "#058EEF"
SECONDARY = "#E37405"
LAVENDER = "#D6EAF8"
GRID = '#E5E7EB'
GRAY = '#6B7280'
BG = '#FFFFFF'


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def style_axes(ax):
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.25, color=GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_alpha(0.25)


def map_kl_to_binary(v, threshold: int = 2) -> int:
    return 1 if int(v) >= threshold else 0


def add_target_column(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    out = df.copy()
    out['TARGET'] = out['KL'].apply(lambda x: map_kl_to_binary(x, threshold))
    return out


def compute_col_stats_from_df(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    vals = pd.to_numeric(df[col], errors='coerce')
    mean = float(vals.mean()) if not np.isnan(vals.mean()) else 0.0
    std = float(vals.std()) if not np.isnan(vals.std()) and vals.std() > 0 else 1.0
    return mean, std


def safe_auc(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, prob_pos))
    except Exception:
        return float('nan')


def safe_auprc(y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, prob_pos))
    except Exception:
        return float('nan')


class KneeBinaryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor,
        threshold: int = 2,
        use_metadata: bool = False,
        age_group_col: str = 'AGE_GROUP_CODE',
        sex_col: str = 'SEX_MALE',
        bmi_col: str = 'BMI',
        age_group_mean: Optional[float] = None,
        age_group_std: Optional[float] = None,
        bmi_mean: Optional[float] = None,
        bmi_std: Optional[float] = None,
    ):
        self.df = add_target_column(df, threshold).copy().reset_index(drop=True)
        self.processor = processor
        self.use_metadata = use_metadata
        self.age_group_col = age_group_col
        self.sex_col = sex_col
        self.bmi_col = bmi_col
        self.age_group_mean = age_group_mean
        self.age_group_std = age_group_std
        self.bmi_mean = bmi_mean
        self.bmi_std = bmi_std

        for c in ['image_path', 'KL', 'id']:
            if c not in self.df.columns:
                raise ValueError(f'Missing column: {c}')

    def __len__(self):
        return len(self.df)

    def _encode_sex(self, val) -> float:
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ['1', 'm', 'male', 'true']:
                return 1.0
            if v in ['0', '2', 'f', 'female', 'false']:
                return 0.0
        try:
            return 1.0 if float(val) == 1 else 0.0
        except Exception:
            return 0.0

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        pixel_values = self.processor(images=img, return_tensors='pt')['pixel_values'][0]
        out = {
            'pixel_values': pixel_values,
            'label': int(row['TARGET']),
            'id': str(row['id']),
            'image_path': row['image_path'],
        }
        if self.use_metadata:
            age_group = 0.0 if pd.isna(row[self.age_group_col]) else float(row[self.age_group_col])
            bmi = 0.0 if pd.isna(row[self.bmi_col]) else float(row[self.bmi_col])
            sex = self._encode_sex(row[self.sex_col])

            if self.age_group_mean is not None and self.age_group_std is not None and self.age_group_std > 0:
                age_group = (age_group - self.age_group_mean) / self.age_group_std
            if self.bmi_mean is not None and self.bmi_std is not None and self.bmi_std > 0:
                bmi = (bmi - self.bmi_mean) / self.bmi_std
            out['meta'] = torch.tensor([age_group, sex, bmi], dtype=torch.float32)
        return out


def collate_fn(batch):
    out = {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'labels': torch.tensor([b['label'] for b in batch], dtype=torch.long),
        'ids': [b['id'] for b in batch],
        'paths': [b['image_path'] for b in batch],
    }
    if 'meta' in batch[0]:
        out['meta'] = torch.stack([b['meta'] for b in batch])
    return out


class HFVisionBackbone(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        lname = name.lower()
        if 'clip' in lname:
            self.model = CLIPModel.from_pretrained(name)
            self.feat_dim = self.model.config.projection_dim
            self.kind = 'clip'
        else:
            self.model = AutoModel.from_pretrained(name)
            self.feat_dim = getattr(self.model.config, 'hidden_size')
            self.kind = 'vit'

    def forward(self, pixel_values):
        if self.kind == 'clip':
            feats = self.model.get_image_features(pixel_values=pixel_values)
        else:
            out = self.model(pixel_values=pixel_values)
            feats = out.pooler_output if getattr(out, 'pooler_output', None) is not None else out.last_hidden_state[:, 0]
        return F.normalize(feats, dim=-1)


class KneeBinaryModel(nn.Module):
    def __init__(self, backbone_name: str, use_metadata: bool = False, dropout: float = 0.2):
        super().__init__()
        self.encoder = HFVisionBackbone(backbone_name)
        self.use_metadata = use_metadata
        if use_metadata:
            self.meta_mlp = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_in = self.encoder.feat_dim + 32
        else:
            head_in = self.encoder.feat_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, pixel_values, meta=None, return_features: bool = False):
        feats = self.encoder(pixel_values)
        feats_for_cls = torch.cat([feats, self.meta_mlp(meta)], dim=-1) if self.use_metadata else feats
        logits = self.head(feats_for_cls)
        return (logits, feats) if return_features else logits


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, probs, confs = [], [], [], []

    for batch in loader:
        x = batch['pixel_values'].to(device)
        y = batch['labels'].to(device)
        meta = batch.get('meta')
        meta = meta.to(device) if meta is not None else None

        logits = model(x, meta=meta)
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)

        ys.extend(y.cpu().numpy())
        ps.extend(pred.cpu().numpy())
        probs.extend(prob[:, 1].cpu().numpy())
        confs.extend(conf.cpu().numpy())

    y_true = np.array(ys)
    y_pred = np.array(ps)
    prob_pos = np.array(probs)
    confidence = np.array(confs)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'auc': safe_auc(y_true, prob_pos),
        'auprc': safe_auprc(y_true, prob_pos),
        'y_true': y_true,
        'y_pred': y_pred,
        'prob_pos': prob_pos,
        'confidence': confidence,
    }


def bootstrap_ci(y_true, prob_pos, preds, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    prob_pos = np.asarray(prob_pos)
    preds = np.asarray(preds)
    n = len(y_true)
    stats = {k: [] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, pp = y_true[idx], preds[idx], prob_pos[idx]
        stats['accuracy'].append(accuracy_score(yt, yp))
        stats['precision'].append(precision_score(yt, yp, zero_division=0))
        stats['recall'].append(recall_score(yt, yp, zero_division=0))
        stats['f1'].append(f1_score(yt, yp, zero_division=0))
        stats['auc'].append(safe_auc(yt, pp))
        stats['auprc'].append(safe_auprc(yt, pp))

    out = {}
    for k, v in stats.items():
        v = np.asarray(v, dtype=float)
        out[k] = {
            'mean': float(np.nanmean(v)),
            'ci_lower': float(np.nanpercentile(v, 2.5)),
            'ci_upper': float(np.nanpercentile(v, 97.5)),
        }
    return out


def compute_risk_coverage(y_true, y_pred, confidence):
    order = np.argsort(-np.asarray(confidence))
    yt = np.asarray(y_true)[order]
    yp = np.asarray(y_pred)[order]
    n = len(yt)
    coverages, risks, accuracies = [], [], []
    for k in range(1, n + 1):
        acc = float((yt[:k] == yp[:k]).mean())
        coverages.append(k / n)
        risks.append(1.0 - acc)
        accuracies.append(acc)
    return {
        'coverage': np.asarray(coverages),
        'risk': np.asarray(risks),
        'accuracy': np.asarray(accuracies),
        'aurc': float(np.trapz(risks, coverages)),
    }


def coverage_at_target_accuracy(rc_dict, target_acc: float = 0.95) -> float:
    ok = np.where(rc_dict['accuracy'] >= target_acc)[0]
    return 0.0 if len(ok) == 0 else float(rc_dict['coverage'][ok[-1]])


def compute_ece(y_true, prob_pos, n_bins: int = 10) -> float:
    conf = np.maximum(prob_pos, 1 - prob_pos)
    preds = (prob_pos >= 0.5).astype(int)
    corr = (preds == y_true).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(ece)


def mc_dropout_predict(model, loader, device, mc_runs: int = 20):
    def enable_dropout(m):
        if isinstance(m, nn.Dropout):
            m.train()

    model.eval()
    model.head.apply(enable_dropout)
    if model.use_metadata:
        model.meta_mlp.apply(enable_dropout)

    mean_prob, std_prob, entropy, ys, ps = [], [], [], [], []
    for batch in loader:
        x = batch['pixel_values'].to(device)
        y = batch['labels'].numpy()
        meta = batch.get('meta')
        meta = meta.to(device) if meta is not None else None

        runs = []
        for _ in range(mc_runs):
            logits = model(x, meta=meta)
            probs = F.softmax(logits, dim=1)[:, 1]
            runs.append(probs.detach().cpu().numpy())
        runs = np.stack(runs, axis=0)
        mp = runs.mean(axis=0)
        sp = runs.std(axis=0)
        ent = -(mp * np.log(mp + 1e-8) + (1 - mp) * np.log(1 - mp + 1e-8))
        pred = (mp >= 0.5).astype(int)

        mean_prob.append(mp)
        std_prob.append(sp)
        entropy.append(ent)
        ys.append(y)
        ps.append(pred)

    return {
        'mean_prob': np.concatenate(mean_prob),
        'std_prob': np.concatenate(std_prob),
        'entropy': np.concatenate(entropy),
        'y_true': np.concatenate(ys),
        'y_pred': np.concatenate(ps),
    }


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    for batch in loader:
        x = batch['pixel_values'].to(device)
        meta = batch.get('meta')
        meta = meta.to(device) if meta is not None else None
        _, f = model(x, meta=meta, return_features=True)
        feats.append(f.cpu().numpy())
        labels.append(batch['labels'].numpy())
    return np.concatenate(feats), np.concatenate(labels)


def plot_roc(y_true, prob_pos, path, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    auc = safe_auc(y_true, prob_pos)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(fpr, tpr, color=SECONDARY, linewidth=2.4, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], '--', color=GRAY, linewidth=1.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(frameon=True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_pr(y_true, prob_pos, path, title='Precision-Recall Curve'):
    p, r, _ = precision_recall_curve(y_true, prob_pos)
    ap = safe_auprc(y_true, prob_pos)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(r, p, color=PRIMARY, linewidth=2.4, label=f'AP = {ap:.4f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(frameon=True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_risk_coverage(rc, path, title='Risk-Coverage Curve'):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(rc['coverage'], rc['risk'], color=SECONDARY, linewidth=2.4, label=f'AURC = {rc["aurc"]:.4f}')
    ax.fill_between(rc['coverage'], rc['risk'], color=LAVENDER, alpha=0.55)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Risk')
    ax.set_title(title)
    ax.legend(frameon=True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_uncertainty_hist_single(mc, path, title='Predictive Uncertainty (MC Dropout SD)'):
    values = mc['std_prob']
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.hist(values, bins=25, color=PRIMARY, edgecolor=SECONDARY, alpha=0.72, linewidth=1.1)
    ax.set_xlabel('MC Dropout Std (SD)')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_reliability(y_true, prob_pos, path, title='Reliability Diagram'):
    conf = np.maximum(prob_pos, 1 - prob_pos)
    preds = (prob_pos >= 0.5).astype(int)
    correct = (preds == y_true).astype(int)
    frac_pos, mean_pred = calibration_curve(correct, conf, n_bins=10, strategy='uniform')
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], '--', color=GRAY, linewidth=1.5, label='Perfect calibration')
    ax.plot(mean_pred, frac_pos, marker='o', markersize=6, color=PRIMARY, linewidth=2.3, label='Model')
    ax.set_xlabel('Mean confidence')
    ax.set_ylabel('Empirical accuracy')
    ax.set_title(title)
    ax.legend(frameon=True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_embeddings(features, labels, path, method='umap', title='Embedding Visualization'):
    n = len(features)
    idx = np.arange(n)
    if n > 2000:
        idx = np.random.default_rng(42).choice(n, 2000, replace=False)
    feats = features[idx]
    labs = labels[idx]
    if method == 'umap' and HAS_UMAP:
        emb = umap.UMAP(n_components=2, random_state=42).fit_transform(feats)
        method_name = 'UMAP'
    else:
        emb = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto').fit_transform(feats)
        method_name = 'TSNE'

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    m0 = labs == 0
    m1 = labs == 1
    ax.scatter(emb[m0, 0], emb[m0, 1], s=14, alpha=0.72, color=SECONDARY, label='KL < 2')
    ax.scatter(emb[m1, 0], emb[m1, 1], s=14, alpha=0.72, color=PRIMARY, label='KL ≥ 2')
    ax.set_title(f'{title} ({method_name})')
    ax.legend(frameon=True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_metrics_tables(summary: Dict, ci: Dict, out_dir: str):
    rows = []
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']:
        rows.append({
            'metric': k,
            'value': summary[k],
            'ci95': f"[{ci[k]['ci_lower']:.4f}, {ci[k]['ci_upper']:.4f}]"
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'metrics_with_ci.csv'), index=False)


def build_test_loader(train_df, test_df, processor, args):
    age_group_mean, age_group_std = (None, None)
    bmi_mean, bmi_std = (None, None)
    if args.use_metadata:
        age_group_mean, age_group_std = compute_col_stats_from_df(train_df, args.age_group_col)
        bmi_mean, bmi_std = compute_col_stats_from_df(train_df, args.bmi_col)

    test_ds = KneeBinaryDataset(
        test_df,
        processor,
        threshold=args.binary_threshold,
        use_metadata=args.use_metadata,
        age_group_col=args.age_group_col,
        sex_col=args.sex_col,
        bmi_col=args.bmi_col,
        age_group_mean=age_group_mean,
        age_group_std=age_group_std,
        bmi_mean=bmi_mean,
        bmi_std=bmi_std,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    return test_loader


def main():
    ap = argparse.ArgumentParser(description='Stage 3: reliability and uncertainty analysis for one best Stage 2 model')
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--backbone_name', required=True)
    ap.add_argument('--model_alias', default='model')
    ap.add_argument('--binary_threshold', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--bootstrap_iters', type=int, default=1000)
    ap.add_argument('--mc_runs', type=int, default=20)
    ap.add_argument('--embedding_method', default='umap', choices=['umap', 'tsne'])
    ap.add_argument('--use_metadata', action='store_true')
    ap.add_argument('--age_group_col', default='AGE_GROUP_CODE')
    ap.add_argument('--bmi_col', default='BMI')
    ap.add_argument('--sex_col', default='SEX_MALE')
    ap.add_argument('--primary_color', default='#058EEF')
    ap.add_argument('--secondary_color', default='#E37405')
    args = ap.parse_args()


    global PRIMARY, SECONDARY
    PRIMARY = args.primary_color
    SECONDARY = args.secondary_color




    warnings.filterwarnings('ignore')
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.out_dir)

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    processor = AutoImageProcessor.from_pretrained(args.backbone_name)
    test_loader = build_test_loader(train_df, test_df, processor, args)

    model = KneeBinaryModel(
        backbone_name=args.backbone_name,
        use_metadata=args.use_metadata,
        dropout=args.dropout,
    ).to(args.device)

    state = torch.load(args.stage2_checkpoint, map_location=args.device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    print(f'[Stage3] Loaded best Stage2 model from: {args.stage2_checkpoint}')

    test = evaluate(model, test_loader, args.device)
    ci = bootstrap_ci(test['y_true'], test['prob_pos'], test['y_pred'], n_boot=args.bootstrap_iters, seed=args.seed)
    rc = compute_risk_coverage(test['y_true'], test['y_pred'], test['confidence'])
    cov95 = coverage_at_target_accuracy(rc, 0.95)
    ece = compute_ece(test['y_true'], test['prob_pos'])
    mc = mc_dropout_predict(model, test_loader, args.device, mc_runs=args.mc_runs)
    feats, labels = extract_features(model, test_loader, args.device)

    pd.DataFrame({
        'y_true': test['y_true'],
        'y_pred': test['y_pred'],
        'prob_positive': test['prob_pos'],
        'confidence': test['confidence'],
        'mc_mean_prob': mc['mean_prob'],
        'mc_std_prob': mc['std_prob'],
        'mc_entropy': mc['entropy'],
    }).to_csv(os.path.join(args.out_dir, 'stage3_test_predictions.csv'), index=False)

    prefix = f'{args.model_alias}_{"with_metadata" if args.use_metadata else "image_only"}'
    plot_roc(test['y_true'], test['prob_pos'], os.path.join(args.out_dir, 'roc_curve.png'), f'{prefix} ROC curve')
    plot_pr(test['y_true'], test['prob_pos'], os.path.join(args.out_dir, 'pr_curve.png'), f'{prefix} PR curve')
    plot_risk_coverage(rc, os.path.join(args.out_dir, 'risk_coverage_curve.png'), f'{prefix} Risk-Coverage')
    plot_uncertainty_hist_single(mc, os.path.join(args.out_dir, 'uncertainty_histogram.png'), f'{prefix} Predictive Uncertainty')
    plot_reliability(test['y_true'], test['prob_pos'], os.path.join(args.out_dir, 'reliability_diagram.png'), f'{prefix} Reliability')
    plot_embeddings(feats, labels, os.path.join(args.out_dir, 'embedding_vis.png'), method=args.embedding_method, title=prefix)

    summary = {
        'model_alias': args.model_alias,
        'backbone_name': args.backbone_name,
        'use_metadata': bool(args.use_metadata),
        'stage2_checkpoint': args.stage2_checkpoint,
        'test_size': int(len(test_df)),
        'accuracy': float(test['accuracy']),
        'precision': float(test['precision']),
        'recall': float(test['recall']),
        'f1': float(test['f1']),
        'macro_f1': float(test['macro_f1']),
        'auc': float(test['auc']),
        'auprc': float(test['auprc']),
        'ece': float(ece),
        'aurc': float(rc['aurc']),
        'coverage_at_95acc': float(cov95),
        'mc_mean_std': float(np.mean(mc['std_prob'])),
        'mc_mean_entropy': float(np.mean(mc['entropy'])),
        'output_dir': args.out_dir,
    }

    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, 'stage3_summary.csv'), index=False)
    save_metrics_tables(summary, ci, args.out_dir)

    with open(os.path.join(args.out_dir, 'stage3_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'bootstrap_ci': ci}, f, indent=2)

    print('\n[Stage3] Done.')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
