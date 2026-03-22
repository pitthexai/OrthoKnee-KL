import os
import json
import math
import random
import argparse
import warnings
import shutil
from typing import Dict, List, Optional, Tuple
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
    confusion_matrix,
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






def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def map_kl_to_binary(v, threshold: int = 2) -> int:
    return 1 if int(v) >= threshold else 0


def slugify_ratio(r: float) -> str:
    return f"{int(round(r * 100)):03d}pct"


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


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def add_target_column(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    out = df.copy()
    out['TARGET'] = out['KL'].apply(lambda x: map_kl_to_binary(x, threshold))
    return out


def stratified_subsample(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 0.999:
        return df.copy().reset_index(drop=True)
    if fraction <= 0:
        raise ValueError('fraction must be > 0')

    parts = []
    rng = np.random.default_rng(seed)
    for cls, grp in df.groupby('TARGET'):
        n = len(grp)
        k = max(1, int(round(n * fraction)))
        idx = rng.choice(grp.index.to_numpy(), size=min(k, n), replace=False)
        parts.append(df.loc[idx])
    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


def compute_col_stats_from_df(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    vals = pd.to_numeric(df[col], errors='coerce')
    mean = float(vals.mean()) if not np.isnan(vals.mean()) else 0.0
    std = float(vals.std()) if not np.isnan(vals.std()) and vals.std() > 0 else 1.0
    return mean, std


def get_age_group_mode(df: pd.DataFrame, col: str) -> str:
    vals = pd.to_numeric(df[col], errors='coerce').dropna()
    if len(vals) == 0:
        return 'categorical'
    uniq = np.sort(vals.unique())
    if len(uniq) <= 6 and np.allclose(uniq, np.round(uniq)):
        return 'categorical'
    return 'continuous'





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
        age_group_mode: str = 'categorical',
        age_group_max: Optional[float] = None,
        age_group_mean: Optional[float] = None,
        age_group_std: Optional[float] = None,
        bmi_mean: Optional[float] = None,
        bmi_std: Optional[float] = None,
    ):
        self.df = add_target_column(df, threshold).copy()
        self.processor = processor
        self.use_metadata = use_metadata
        self.age_group_col = age_group_col
        self.sex_col = sex_col
        self.bmi_col = bmi_col
        self.age_group_mode = age_group_mode
        self.age_group_max = age_group_max
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

    def _transform_age_group(self, v: float) -> float:
        if self.age_group_mode == 'categorical':
            if self.age_group_max is not None and self.age_group_max > 0:
                return float(v / self.age_group_max)
            return float(v)
        if self.age_group_mean is not None and self.age_group_std is not None and self.age_group_std > 0:
            return float((v - self.age_group_mean) / self.age_group_std)
        return float(v)

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

            age_group = self._transform_age_group(age_group)
            if self.bmi_mean is not None and self.bmi_std is not None and self.bmi_std > 0:
                bmi = float((bmi - self.bmi_mean) / self.bmi_std)

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
            out = self.model.get_image_features(pixel_values=pixel_values)

            if torch.is_tensor(out):
                feats = out
            elif hasattr(out, 'image_embeds') and out.image_embeds is not None:
                feats = out.image_embeds
            elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
                feats = out.pooler_output
            elif hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
                feats = out.last_hidden_state[:, 0]
            else:
                raise TypeError(f"Unsupported CLIP output type: {type(out)}")

        else:
            out = self.model(pixel_values=pixel_values)

            if torch.is_tensor(out):
                feats = out
            elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
                feats = out.pooler_output
            elif hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
                feats = out.last_hidden_state[:, 0]
            elif hasattr(out, 'image_embeds') and out.image_embeds is not None:
                feats = out.image_embeds
            else:
                raise TypeError(f"Unsupported ViT/DINO output type: {type(out)}")

        feats = feats.float()
        return F.normalize(feats, dim=-1)


class KneeModel(nn.Module):
    def __init__(self, backbone_name: str, use_metadata: bool = False, dropout: float = 0.3):
        super().__init__()
        self.encoder = HFVisionBackbone(backbone_name)
        self.use_metadata = use_metadata
        self.meta_dim = 32
        if use_metadata:
            self.meta_mlp = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, self.meta_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.meta_gate = nn.Sequential(
                nn.Linear(self.meta_dim, self.encoder.feat_dim),
                nn.Sigmoid(),
            )
            head_in = self.encoder.feat_dim + self.meta_dim
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
        if self.use_metadata:
            meta_feat = self.meta_mlp(meta)
            gate = self.meta_gate(meta_feat)
            gated_feats = feats * gate
            feats_for_cls = torch.cat([gated_feats, meta_feat], dim=-1)
        else:
            feats_for_cls = feats
        logits = self.head(feats_for_cls)
        return (logits, feats) if return_features else logits


def freeze_encoder(model: nn.Module):
    for p in model.encoder.parameters():
        p.requires_grad = False


def partial_unfreeze_encoder(model: KneeModel, backbone_name: str):
    freeze_encoder(model)
    lname = backbone_name.lower()

    if 'clip' in lname:
        for name, p in model.encoder.model.named_parameters():
            if (
                'vision_model.encoder.layers.11' in name
                or 'vision_model.post_layernorm' in name
                or 'visual_projection' in name
            ):
                p.requires_grad = True
    else:
        for name, p in model.encoder.model.named_parameters():
            if ('layers.11' in name) or ('layernorm' in name) or ('norm' == name.split('.')[-1]):
                p.requires_grad = True


def full_unfreeze_encoder(model: KneeModel):
    for p in model.encoder.parameters():
        p.requires_grad = True


def make_optimizer(model: KneeModel, mode: str, backbone_name: str, head_lr: float,
                   encoder_lr: float, meta_lr: float, weight_decay: float):
    params = []
    if mode == 'frozen':
        freeze_encoder(model)
    elif mode == 'partial_finetuned':
        partial_unfreeze_encoder(model, backbone_name)
    elif mode == 'full_finetuned':
        full_unfreeze_encoder(model)
    else:
        raise ValueError(f'Unknown training mode: {mode}')

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if enc_params:
        params.append({'params': enc_params, 'lr': encoder_lr, 'weight_decay': weight_decay})
    params.append({'params': model.head.parameters(), 'lr': head_lr, 'weight_decay': weight_decay})
    if model.use_metadata:
        params.append({'params': model.meta_mlp.parameters(), 'lr': meta_lr, 'weight_decay': weight_decay})
        params.append({'params': model.meta_gate.parameters(), 'lr': meta_lr, 'weight_decay': weight_decay})
    return torch.optim.AdamW(params)



def class_weights_from_df(df: pd.DataFrame, threshold: int = 2) -> torch.Tensor:
    tmp = add_target_column(df, threshold)
    counts = tmp['TARGET'].value_counts().sort_index()
    total = len(tmp)
    return torch.tensor(
        [total / (2 * counts.get(0, 1)), total / (2 * counts.get(1, 1))],
        dtype=torch.float32,
    )


def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0
    ys, ps = [], []
    for batch in loader:
        x = batch['pixel_values'].to(device)
        y = batch['labels'].to(device)
        meta = batch.get('meta')
        meta = meta.to(device) if meta is not None else None

        optimizer.zero_grad()
        logits = model(x, meta=meta)
        loss = criterion(logits, y)
        loss.backward()
        if max_grad_norm and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        ys.extend(y.detach().cpu().numpy())
        ps.extend(pred.detach().cpu().numpy())

    return total_loss / len(loader.dataset), accuracy_score(ys, ps), f1_score(ys, ps, zero_division=0)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ys, ps, probs, confs = [], [], [], []

    for batch in loader:
        x = batch['pixel_values'].to(device)
        y = batch['labels'].to(device)
        meta = batch.get('meta')
        meta = meta.to(device) if meta is not None else None

        logits = model(x, meta=meta)
        loss = criterion(logits, y)
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)

        total_loss += loss.item() * y.size(0)
        ys.extend(y.cpu().numpy())
        ps.extend(pred.cpu().numpy())
        probs.extend(prob[:, 1].cpu().numpy())
        confs.extend(conf.cpu().numpy())

    y_true = np.array(ys)
    y_pred = np.array(ps)
    prob_pos = np.array(probs)
    confidence = np.array(confs)
    return {
        'loss': total_loss / len(loader.dataset),
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


def plot_confusion_matrix(cm, labels, path, title='Confusion Matrix'):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_roc(y_true, prob_pos, path, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_true, prob_pos)
    auc = safe_auc(y_true, prob_pos)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC={auc:.4f}')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_pr(y_true, prob_pos, path, title='PR Curve'):
    p, r, _ = precision_recall_curve(y_true, prob_pos)
    ap = safe_auprc(y_true, prob_pos)
    plt.figure(figsize=(5, 4))
    plt.plot(r, p, label=f'AP={ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_risk_coverage(rc, path, title='Risk-Coverage Curve'):
    plt.figure(figsize=(6, 5))
    plt.plot(rc['coverage'], rc['risk'], label=f"AURC={rc['aurc']:.4f}")
    plt.xlabel('Coverage')
    plt.ylabel('Risk')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_uncertainty_hist(mc, path, title='MC Dropout Uncertainty'):
    c = mc['std_prob'][mc['y_true'] == mc['y_pred']]
    w = mc['std_prob'][mc['y_true'] != mc['y_pred']]
    plt.figure(figsize=(6, 4))
    plt.hist(c, bins=25, alpha=0.7, label='Correct')
    plt.hist(w, bins=25, alpha=0.7, label='Incorrect')
    plt.xlabel('Predictive Std')
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_reliability(y_true, prob_pos, path, title='Reliability Diagram'):
    conf = np.maximum(prob_pos, 1 - prob_pos)
    preds = (prob_pos >= 0.5).astype(int)
    correct = (preds == y_true).astype(int)
    frac_pos, mean_pred = calibration_curve(correct, conf, n_bins=10, strategy='uniform')
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], '--', label='Perfect')
    plt.plot(mean_pred, frac_pos, marker='o', label='Model')
    plt.xlabel('Mean confidence')
    plt.ylabel('Empirical accuracy')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_embeddings(features, labels, path, method='umap', title='Embedding Visualization'):
    n = len(features)
    idx = np.arange(n)
    if n > 2000:
        idx = np.random.default_rng(42).choice(n, 2000, replace=False)
    feats = features[idx]
    labs = labels[idx]
    if method == 'umap' and HAS_UMAP:
        emb = umap.UMAP(n_components=2, random_state=42).fit_transform(feats)
    else:
        emb = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto').fit_transform(feats)
        method = 'tsne'
    plt.figure(figsize=(6, 5))
    for cls, name in [(0, 'KL<2'), (1, 'KL>=2')]:
        m = labs == cls
        plt.scatter(emb[m, 0], emb[m, 1], s=10, alpha=0.7, label=name)
    plt.title(f'{title} ({method.upper()})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def save_metrics_tables(summary: Dict, ci: Dict, out_dir: str):
    rows = []
    for k in ['accuracy', 'precision', 'recall', 'f1', 'macro_f1', 'auc', 'auprc', 'ece', 'aurc']:
        if k in summary:
            row = {'metric': k, 'value': summary[k]}
            if k in ci:
                row['ci95'] = f"[{ci[k]['ci_lower']:.4f}, {ci[k]['ci_upper']:.4f}]"
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'metrics_with_ci.csv'), index=False)


def choose_processor(model_name: str):
    return AutoImageProcessor.from_pretrained(model_name)


def build_dataloaders(train_df, val_df, test_df, processor, args, use_metadata):
    age_group_mode = get_age_group_mode(train_df, args.age_group_col)
    age_group_max = None
    age_group_mean, age_group_std = (None, None)
    bmi_mean, bmi_std = (None, None)

    if use_metadata:
        if age_group_mode == 'categorical':
            vals = pd.to_numeric(train_df[args.age_group_col], errors='coerce').dropna()
            age_group_max = float(vals.max()) if len(vals) > 0 else 1.0
        else:
            age_group_mean, age_group_std = compute_col_stats_from_df(train_df, args.age_group_col)
        bmi_mean, bmi_std = compute_col_stats_from_df(train_df, args.bmi_col)

    train_ds = KneeBinaryDataset(
        train_df,
        processor,
        threshold=args.binary_threshold,
        use_metadata=use_metadata,
        age_group_col=args.age_group_col,
        sex_col=args.sex_col,
        bmi_col=args.bmi_col,
        age_group_mode=age_group_mode,
        age_group_max=age_group_max,
        age_group_mean=age_group_mean,
        age_group_std=age_group_std,
        bmi_mean=bmi_mean,
        bmi_std=bmi_std,
    )
    val_ds = KneeBinaryDataset(
        val_df,
        processor,
        threshold=args.binary_threshold,
        use_metadata=use_metadata,
        age_group_col=args.age_group_col,
        sex_col=args.sex_col,
        bmi_col=args.bmi_col,
        age_group_mode=age_group_mode,
        age_group_max=age_group_max,
        age_group_mean=age_group_mean,
        age_group_std=age_group_std,
        bmi_mean=bmi_mean,
        bmi_std=bmi_std,
    )
    test_ds = KneeBinaryDataset(
        test_df,
        processor,
        threshold=args.binary_threshold,
        use_metadata=use_metadata,
        age_group_col=args.age_group_col,
        sex_col=args.sex_col,
        bmi_col=args.bmi_col,
        age_group_mode=age_group_mode,
        age_group_max=age_group_max,
        age_group_mean=age_group_mean,
        age_group_std=age_group_std,
        bmi_mean=bmi_mean,
        bmi_std=bmi_std,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def train_and_evaluate(backbone_name: str, model_alias: str, train_mode: str, use_metadata: bool,
                       train_fraction: float, train_df: pd.DataFrame, val_df: pd.DataFrame,
                       test_df: pd.DataFrame, args, seed: int) -> Dict:
    fraction_tag = slugify_ratio(train_fraction)
    input_tag = 'with_metadata' if use_metadata else 'image_only'
    exp_name = f"{model_alias}_{train_mode}_{fraction_tag}_{input_tag}_seed{seed}"
    out_dir = os.path.join(args.out_root, model_alias, train_mode, fraction_tag, input_tag, f"seed_{seed}")
    ensure_dir(out_dir)

    processor = choose_processor(backbone_name)
    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df, processor, args, use_metadata)


    train_df.to_csv(os.path.join(out_dir, 'train_subset_used.csv'), index=False)
    val_df.to_csv(os.path.join(out_dir, 'val_used.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test_used.csv'), index=False)




    model = KneeModel(backbone_name=backbone_name, use_metadata=use_metadata, dropout=args.dropout).to(args.device)
    optimizer = make_optimizer(
        model,
        train_mode,
        backbone_name=backbone_name,
        head_lr=args.head_lr,
        encoder_lr=args.encoder_lr,
        meta_lr=args.meta_lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_from_df(train_df, args.binary_threshold).to(args.device),
        label_smoothing=args.label_smoothing,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=max(1, args.scheduler_patience))

    best_val = -1.0
    best_path = os.path.join(out_dir, 'best_model.pt')
    history = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion, args.device, args.max_grad_norm)
        val = evaluate(model, val_loader, criterion, args.device)
        scheduler.step(val['macro_f1'])

        history.append({
            'epoch': epoch,
            'train_loss': tr_loss,
            'train_acc': tr_acc,
            'train_f1': tr_f1,
            'val_loss': val['loss'],
            'val_acc': val['accuracy'],
            'val_precision': val['precision'],
            'val_recall': val['recall'],
            'val_f1': val['f1'],
            'val_macro_f1': val['macro_f1'],
            'val_auc': val['auc'],
            'val_auprc': val['auprc'],
        })

        print(
            f"[{exp_name}] Epoch {epoch:02d} | train_acc={tr_acc:.4f} train_f1={tr_f1:.4f} | "
            f"val_acc={val['accuracy']:.4f} val_macro_f1={val['macro_f1']:.4f} val_auc={val['auc']:.4f}"
        )

        if val['macro_f1'] > best_val:
            best_val = val['macro_f1']
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            print(f"[{exp_name}] Early stopping at epoch {epoch}")
            break

    pd.DataFrame(history).to_csv(os.path.join(out_dir, 'history.csv'), index=False)
    model.load_state_dict(torch.load(best_path, map_location=args.device))

    test = evaluate(model, test_loader, criterion, args.device)
    pd.DataFrame({
        'y_true': test['y_true'],
        'y_pred': test['y_pred'],
        'prob_positive': test['prob_pos'],
        'confidence': test['confidence'],
    }).to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)

    ci = bootstrap_ci(test['y_true'], test['prob_pos'], test['y_pred'], n_boot=args.bootstrap_iters, seed=seed)
    rc = compute_risk_coverage(test['y_true'], test['y_pred'], test['confidence'])
    cov95 = coverage_at_target_accuracy(rc, 0.95)
    ece = compute_ece(test['y_true'], test['prob_pos'])
    mc = mc_dropout_predict(model, test_loader, args.device, mc_runs=args.mc_runs)
    feats, labels = extract_features(model, test_loader, args.device)

    cm = confusion_matrix(test['y_true'], test['y_pred'])
    plot_confusion_matrix(cm, ['KL<2', 'KL>=2'], os.path.join(out_dir, 'confusion_matrix.png'), f'{exp_name} Confusion Matrix')
    plot_roc(test['y_true'], test['prob_pos'], os.path.join(out_dir, 'roc_curve.png'), f'{exp_name} ROC')
    plot_pr(test['y_true'], test['prob_pos'], os.path.join(out_dir, 'pr_curve.png'), f'{exp_name} PR')
    plot_risk_coverage(rc, os.path.join(out_dir, 'risk_coverage_curve.png'), f'{exp_name} Risk-Coverage')
    plot_uncertainty_hist(mc, os.path.join(out_dir, 'uncertainty_histogram.png'), f'{exp_name} MC Dropout Uncertainty')
    plot_reliability(test['y_true'], test['prob_pos'], os.path.join(out_dir, 'reliability_diagram.png'), f'{exp_name} Reliability')
    plot_embeddings(feats, labels, os.path.join(out_dir, 'embedding_vis.png'), method=args.embedding_method, title=exp_name)

    train_counts = add_target_column(train_df, args.binary_threshold)['TARGET'].value_counts().to_dict()
    summary = {
        'experiment_name': exp_name,
        'backbone': model_alias,
        'backbone_name': backbone_name,
        'train_mode': train_mode,
        'train_fraction': train_fraction,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'n_class0_train': int(train_counts.get(0, 0)),
        'n_class1_train': int(train_counts.get(1, 0)),
        'use_metadata': use_metadata,
        'seed': seed,
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
        'output_dir': out_dir,
    }

    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, 'summary_metrics.csv'), index=False)
    save_metrics_tables(summary, ci, out_dir)
    with open(os.path.join(out_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(summary, indent=2))
        f.write('\n\n95% Bootstrap CI\n')
        f.write(json.dumps(ci, indent=2))
    return summary


def aggregate_runs(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    group_cols = ['backbone', 'train_mode', 'train_fraction', 'use_metadata']
    metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'macro_f1', 'auc', 'auprc', 'ece', 'aurc', 'coverage_at_95acc', 'mc_mean_std', 'mc_mean_entropy']
    agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(['mean', 'std']).reset_index()
    agg.columns = ['_'.join([str(x) for x in c if x != '']).rstrip('_') for c in agg.columns.to_flat_index()]
    return agg


def pick_best_per_backbone_and_input(raw_df: pd.DataFrame, out_root: str):
    if raw_df.empty:
        return

    best_root = os.path.join(out_root, 'best_models_by_backbone_input')
    ensure_dir(best_root)

    selected_rows = []
    for (backbone, use_metadata), grp in raw_df.groupby(['backbone', 'use_metadata']):
        means = grp.groupby(['train_mode', 'train_fraction'], dropna=False)['macro_f1'].mean().reset_index()
        best_cfg = means.sort_values(['macro_f1', 'train_fraction'], ascending=[False, False]).iloc[0]
        chosen = grp[(grp['train_mode'] == best_cfg['train_mode']) & (grp['train_fraction'] == best_cfg['train_fraction'])].copy()
        chosen = chosen.sort_values(['macro_f1', 'auc', 'seed'], ascending=[False, False, True])
        best_seed_row = chosen.iloc[0].to_dict()

        input_tag = 'with_metadata' if bool(use_metadata) else 'image_only'
        dst = os.path.join(best_root, f"{backbone}_{input_tag}")
        ensure_dir(dst)

        src = best_seed_row['output_dir']
        for fname in [
            'roc_curve.png', 'pr_curve.png', 'reliability_diagram.png', 'risk_coverage_curve.png',
            'uncertainty_histogram.png', 'embedding_vis.png', 'confusion_matrix.png',
            'summary_metrics.csv', 'summary_report.txt', 'test_predictions.csv', 'history.csv'
        ]:
            sp = os.path.join(src, fname)
            if os.path.exists(sp):
                shutil.copy2(sp, os.path.join(dst, fname))

        with open(os.path.join(dst, 'selection_info.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'selection_metric': 'macro_f1_mean_then_best_seed_macro_f1',
                'backbone': backbone,
                'use_metadata': bool(use_metadata),
                'selected_train_mode': best_seed_row['train_mode'],
                'selected_train_fraction': float(best_seed_row['train_fraction']),
                'selected_seed': int(best_seed_row['seed']),
                'selected_output_dir': src,
            }, f, indent=2)

        selected_rows.append({
            'backbone': backbone,
            'use_metadata': bool(use_metadata),
            'selected_train_mode': best_seed_row['train_mode'],
            'selected_train_fraction': float(best_seed_row['train_fraction']),
            'selected_seed': int(best_seed_row['seed']),
            'macro_f1': float(best_seed_row['macro_f1']),
            'auc': float(best_seed_row['auc']),
            'auprc': float(best_seed_row['auprc']),
            'ece': float(best_seed_row['ece']),
            'aurc': float(best_seed_row['aurc']),
            'output_dir': src,
        })

    pd.DataFrame(selected_rows).to_csv(os.path.join(best_root, 'selected_best_models.csv'), index=False)



def run_suite_for_backbone(backbone_name: str, model_alias: str, args, all_rows: List[Dict]):
    train_full = pd.read_csv(args.train_csv)
    val_full = pd.read_csv(args.val_csv)
    test_full = pd.read_csv(args.test_csv)

    train_full = add_target_column(train_full, args.binary_threshold)
    val_full = add_target_column(val_full, args.binary_threshold)
    test_full = add_target_column(test_full, args.binary_threshold)

    fractions = parse_float_list(args.train_fractions)
    modes = [m.strip() for m in args.train_modes.split(',') if m.strip()]
    metadata_options = [False, True]
    seeds = parse_int_list(args.seeds)

    for frac in fractions:
        for seed in seeds:
            subset_seed = seed + int(frac * 1000)
            train_df = stratified_subsample(train_full, frac, subset_seed)
            for mode in modes:
                for use_metadata in metadata_options:
                    set_seed(seed)
                    row = train_and_evaluate(
                        backbone_name=backbone_name,
                        model_alias=model_alias,
                        train_mode=mode,
                        use_metadata=use_metadata,
                        train_fraction=frac,
                        train_df=train_df,
                        val_df=val_full,
                        test_df=test_full,
                        args=args,
                        seed=seed,
                    )
                    all_rows.append(row)



def main():
    ap = argparse.ArgumentParser(description='Knee OA classification: frozen, partial, and full fine-tuned adaptation; few-shot + metadata; performance + reliability')
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--out_root', default='outputs_kneegpt_adaptation_revised')
    ap.add_argument('--binary_threshold', type=int, default=2)

    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--early_stopping_patience', type=int, default=4)
    ap.add_argument('--scheduler_patience', type=int, default=2)
    ap.add_argument('--head_lr', type=float, default=1e-4)
    ap.add_argument('--encoder_lr', type=float, default=1e-5)
    ap.add_argument('--meta_lr', type=float, default=5e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--label_smoothing', type=float, default=0.02)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--max_grad_norm', type=float, default=1.0)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--seeds', default='13,27,42')
    ap.add_argument('--train_fractions', default='0.2,0.5,1.0')
    ap.add_argument('--train_modes', default='frozen,full_finetuned')
    ap.add_argument('--embedding_method', default='umap', choices=['umap', 'tsne'])
    ap.add_argument('--bootstrap_iters', type=int, default=1000)
    ap.add_argument('--mc_runs', type=int, default=30)
    ap.add_argument('--age_group_col', default='AGE_GROUP_CODE')
    ap.add_argument('--bmi_col', default='BMI')
    ap.add_argument('--sex_col', default='SEX_MALE')
    ap.add_argument('--clip_model_name', default='openai/clip-vit-base-patch32')
    ap.add_argument('--dino_model_name', default='facebook/dinov2-base')
    ap.add_argument('--run_clip', action='store_true')
    ap.add_argument('--run_dino', action='store_true')

    args = ap.parse_args()
    warnings.filterwarnings('ignore')
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.out_root)

    if not args.run_clip and not args.run_dino:
        args.run_clip = True
        args.run_dino = True

    all_rows: List[Dict] = []

    if args.run_clip:
        run_suite_for_backbone(args.clip_model_name, 'clip', args, all_rows)
    if args.run_dino:
        run_suite_for_backbone(args.dino_model_name, 'dino', args, all_rows)

    if all_rows:
        raw_df = pd.DataFrame(all_rows)
        raw_df.to_csv(os.path.join(args.out_root, 'all_experiments_raw.csv'), index=False)

        agg_df = aggregate_runs(all_rows)
        agg_df.to_csv(os.path.join(args.out_root, 'all_experiments_aggregated_mean_std.csv'), index=False)

        compact_cols = [
            'backbone', 'train_mode', 'train_fraction', 'use_metadata', 'seed',
            'accuracy', 'f1', 'macro_f1', 'auc', 'auprc', 'ece', 'aurc'
        ]
        raw_df[compact_cols].to_csv(os.path.join(args.out_root, 'compact_results_table.csv'), index=False)

        pick_best_per_backbone_and_input(raw_df, args.out_root)

        with open(os.path.join(args.out_root, 'run_config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=2, default=str)


if __name__ == '__main__':
    main()
