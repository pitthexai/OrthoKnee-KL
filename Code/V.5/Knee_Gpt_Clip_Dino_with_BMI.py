import os, json, random, argparse, warnings
from typing import Tuple
import numpy as np, pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def map_kl_to_binary(v, threshold=2):
    return 1 if int(v) >= threshold else 0

def compute_col_stats(train_csv: str, col: str) -> Tuple[float, float]:
    df = pd.read_csv(train_csv)
    vals = pd.to_numeric(df[col], errors='coerce')
    mean = float(vals.mean()) if not np.isnan(vals.mean()) else 0.0
    std = float(vals.std()) if not np.isnan(vals.std()) and vals.std() > 0 else 1.0
    return mean, std


def compute_age_stats(train_csv: str, age_col: str) -> Tuple[float, float]:
    df = pd.read_csv(train_csv)
    ages = pd.to_numeric(df[age_col], errors='coerce')
    mean = float(ages.mean()) if not np.isnan(ages.mean()) else 0.0
    std = float(ages.std()) if not np.isnan(ages.std()) and ages.std() > 0 else 1.0
    return mean, std


class KneeBinaryDataset(Dataset):
    def __init__(
        self,
        csv_file,
        processor,
        threshold=2,
        use_metadata=False,
        age_group_col='AGE_GROUP_CODE',
        sex_col='SEX_MALE',
        bmi_col='BMI',
        age_group_mean=None,
        age_group_std=None,
        bmi_mean=None,
        bmi_std=None
    ):
        self.df = pd.read_csv(csv_file).copy()
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
                raise ValueError(f'Missing column {c}')
        self.df['TARGET'] = self.df['KL'].apply(lambda x: map_kl_to_binary(x, threshold))

    def __len__(self):
        return len(self.df)

    def _encode_sex(self, val):
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

    def __getitem__(self, idx):
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


class ProbeModel(nn.Module):
    def __init__(self, backbone_name: str, use_metadata=False, dropout=0.2):
        super().__init__()
        self.encoder = HFVisionBackbone(backbone_name)
        self.use_metadata = use_metadata
        if use_metadata:
            self.meta_mlp = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Dropout(dropout))
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

    def forward(self, pixel_values, meta=None, return_features=False):
        feats = self.encoder(pixel_values)
        feats_for_cls = torch.cat([feats, self.meta_mlp(meta)], dim=-1) if self.use_metadata else feats
        logits = self.head(feats_for_cls)
        return (logits, feats) if return_features else logits


def freeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = False


def class_weights_from_train(csv_path, threshold=2):
    df = pd.read_csv(csv_path)
    df['TARGET'] = df['KL'].apply(lambda x: map_kl_to_binary(x, threshold))
    counts = df['TARGET'].value_counts().sort_index()
    total = len(df)
    return torch.tensor([total / (2 * counts.get(0, 1)), total / (2 * counts.get(1, 1))], dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, criterion, device):
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
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        ys.extend(y.cpu().numpy())
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
        'auc': roc_auc_score(y_true, prob_pos),
        'auprc': average_precision_score(y_true, prob_pos),
        'y_true': y_true,
        'y_pred': y_pred,
        'prob_pos': prob_pos,
        'confidence': confidence,
    }


def bootstrap_ci(y_true, prob_pos, preds, n_boot=1000, seed=42):
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
        try:
            stats['auc'].append(roc_auc_score(yt, pp))
        except Exception:
            stats['auc'].append(np.nan)
        try:
            stats['auprc'].append(average_precision_score(yt, pp))
        except Exception:
            stats['auprc'].append(np.nan)
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


def coverage_at_target_accuracy(rc_dict, target_acc=0.95):
    ok = np.where(rc_dict['accuracy'] >= target_acc)[0]
    return 0.0 if len(ok) == 0 else float(rc_dict['coverage'][ok[-1]])


def compute_ece(y_true, prob_pos, n_bins=10):
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


def mc_dropout_predict(model, loader, device, mc_runs=20):
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
    auc = roc_auc_score(y_true, prob_pos)
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
    ap = average_precision_score(y_true, prob_pos)
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


def save_metrics_tables(summary, ci, out_dir):
    rows = []
    for k in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'auprc']:
        rows.append({'metric': k, 'value': summary[k], 'ci95': f"[{ci[k]['ci_lower']:.4f}, {ci[k]['ci_upper']:.4f}]"})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'metrics_with_ci.csv'), index=False)


def choose_processor(model_name):
    return AutoImageProcessor.from_pretrained(model_name)


def train_and_evaluate(backbone_name, args, tag):
    out_dir = os.path.join(args.out_root, tag)
    ensure_dir(out_dir)
    processor = choose_processor(backbone_name)

    age_group_mean, age_group_std = (None, None)
    bmi_mean, bmi_std = (None, None)

    if args.use_metadata:
        age_group_mean, age_group_std = compute_col_stats(args.train_csv, args.age_group_col)
        bmi_mean, bmi_std = compute_col_stats(args.train_csv, args.bmi_col)

    train_ds = KneeBinaryDataset(
        args.train_csv,
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

    val_ds = KneeBinaryDataset(
        args.val_csv,
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

    test_ds = KneeBinaryDataset(
        args.test_csv,
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = ProbeModel(backbone_name, use_metadata=args.use_metadata, dropout=args.dropout).to(args.device)
    if args.freeze_encoder:
        freeze_encoder(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights_from_train(args.train_csv, args.binary_threshold).to(args.device), label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val, best_path, patience = -1.0, os.path.join(out_dir, 'best_model.pt'), 0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
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
            'lr': optimizer.param_groups[0]['lr'],
        })
        print(f"[{tag}] Epoch {epoch:02d} | train_acc={tr_acc:.4f} train_f1={tr_f1:.4f} | val_acc={val['accuracy']:.4f} val_precision={val['precision']:.4f} val_recall={val['recall']:.4f} val_f1={val['f1']:.4f} val_auc={val['auc']:.4f}")
        if val['macro_f1'] > best_val:
            best_val = val['macro_f1']
            torch.save(model.state_dict(), best_path)
            patience = 0
        else:
            patience += 1
        if patience >= args.early_stopping_patience:
            break

    pd.DataFrame(history).to_csv(os.path.join(out_dir, 'history.csv'), index=False)
    model.load_state_dict(torch.load(best_path, map_location=args.device))
    test = evaluate(model, test_loader, criterion, args.device)
    pd.DataFrame({'y_true': test['y_true'], 'y_pred': test['y_pred'], 'prob_positive': test['prob_pos'], 'confidence': test['confidence']}).to_csv(os.path.join(out_dir, 'test_predictions.csv'), index=False)

    ci = bootstrap_ci(test['y_true'], test['prob_pos'], test['y_pred'], n_boot=args.bootstrap_iters, seed=args.seed)
    rc = compute_risk_coverage(test['y_true'], test['y_pred'], test['confidence'])
    cov95 = coverage_at_target_accuracy(rc, 0.95)
    ece = compute_ece(test['y_true'], test['prob_pos'])
    mc = mc_dropout_predict(model, test_loader, args.device, mc_runs=args.mc_runs)
    feats, labels = extract_features(model, test_loader, args.device)

    cm = confusion_matrix(test['y_true'], test['y_pred'])
    plot_confusion_matrix(cm, ['KL<2', 'KL>=2'], os.path.join(out_dir, 'confusion_matrix.png'), f'{tag} Confusion Matrix')
    plot_roc(test['y_true'], test['prob_pos'], os.path.join(out_dir, 'roc_curve.png'), f'{tag} ROC')
    plot_pr(test['y_true'], test['prob_pos'], os.path.join(out_dir, 'pr_curve.png'), f'{tag} PR Curve')
    plot_risk_coverage(rc, os.path.join(out_dir, 'risk_coverage_curve.png'), f'{tag} Risk-Coverage')
    plot_uncertainty_hist(mc, os.path.join(out_dir, 'uncertainty_histogram.png'), f'{tag} MC Dropout Uncertainty')
    plot_reliability(test['y_true'], test['prob_pos'], os.path.join(out_dir, 'reliability_diagram.png'), f'{tag} Reliability')
    plot_embeddings(feats, labels, os.path.join(out_dir, 'embedding_vis.png'), method=args.embedding_method, title=tag)

    summary = {
        'model': tag,
        'backbone_name': backbone_name,
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
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, 'summary_metrics.csv'), index=False)
    save_metrics_tables(summary, ci, out_dir)
    with open(os.path.join(out_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(summary, indent=2))
        f.write('\n\n95% Bootstrap CI\n')
        f.write(json.dumps(ci, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description='KneeGPT-style CLIP/DINO probing for KL<2 vs KL>=2')
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--out_root', default='outputs_kneegpt_probe')
    ap.add_argument('--binary_threshold', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--label_smoothing', type=float, default=0.02)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--freeze_encoder', action='store_true', default=True)
    ap.add_argument('--early_stopping_patience', type=int, default=4)
    ap.add_argument('--embedding_method', default='umap', choices=['umap', 'tsne'])
    ap.add_argument('--bootstrap_iters', type=int, default=1000)
    ap.add_argument('--mc_runs', type=int, default=20)
    ap.add_argument('--use_metadata', action='store_true')
    ap.add_argument('--age_group_col', default='AGE_GROUP_CODE')
    ap.add_argument('--bmi_col', default='BMI')
    ap.add_argument('--sex_col', default='SEX_MALE')
    ap.add_argument('--clip_model_name', default='openai/clip-vit-base-patch32')
    ap.add_argument('--dino_model_name', default='facebook/dinov2-base')
    ap.add_argument('--run_clip', action='store_true', default=True)
    ap.add_argument('--run_dino', action='store_true', default=True)
    args = ap.parse_args()

    warnings.filterwarnings('ignore')
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.out_root)


    summaries = []

    settings = [
        ("image_only", False),
        ("with_metadata", True),
    ]

    if args.run_clip:
        for suffix, use_meta in settings:
            args.use_metadata = use_meta
            summaries.append(
                train_and_evaluate(args.clip_model_name, args, f"clip_probe_{suffix}")
            )

    if args.run_dino:
        for suffix, use_meta in settings:
            args.use_metadata = use_meta
            summaries.append(
                train_and_evaluate(args.dino_model_name, args, f"dino_probe_{suffix}")
            )

    if len(summaries) > 0:
        pd.DataFrame(summaries).to_csv(
            os.path.join(args.out_root, "all_models_summary.csv"),
            index=False
        )


if __name__ == '__main__':
    main()
