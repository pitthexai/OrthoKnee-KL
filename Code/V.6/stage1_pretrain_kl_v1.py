import os
import json
import random
import argparse
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import CLIPModel, AutoImageProcessor, AutoModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_col_stats_from_df(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    vals = pd.to_numeric(df[col], errors='coerce')
    mean = float(vals.mean()) if not np.isnan(vals.mean()) else 0.0
    std = float(vals.std()) if not np.isnan(vals.std()) and vals.std() > 0 else 1.0
    return mean, std


class KneeKLStage1Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor,
        use_metadata: bool = True,
        age_group_col: str = 'AGE_GROUP_CODE',
        sex_col: str = 'SEX_MALE',
        bmi_col: str = 'BMI',
        age_group_mean: Optional[float] = None,
        age_group_std: Optional[float] = None,
        bmi_mean: Optional[float] = None,
        bmi_std: Optional[float] = None,
    ):
        self.df = df.copy().reset_index(drop=True)
        self.processor = processor
        self.use_metadata = use_metadata
        self.age_group_col = age_group_col
        self.sex_col = sex_col
        self.bmi_col = bmi_col
        self.age_group_mean = age_group_mean
        self.age_group_std = age_group_std
        self.bmi_mean = bmi_mean
        self.bmi_std = bmi_std

        required = ['image_path', 'KL', 'id']
        for c in required:
            if c not in self.df.columns:
                raise ValueError(f'Missing required column: {c}')

        self.df = self.df[self.df['KL'].isin([0, 1, 2, 3, 4])].copy().reset_index(drop=True)

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
            'label': int(row['KL']),
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


class KneeStage1KLModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int = 5, use_metadata: bool = True, dropout: float = 0.2):
        super().__init__()
        self.encoder = HFVisionBackbone(backbone_name)
        self.use_metadata = use_metadata
        self.num_classes = num_classes

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
            nn.Linear(128, num_classes),
        )

    def forward(self, pixel_values, meta=None, return_features: bool = False):
        feats = self.encoder(pixel_values)
        feats_for_cls = torch.cat([feats, self.meta_mlp(meta)], dim=-1) if self.use_metadata else feats
        logits = self.head(feats_for_cls)
        return (logits, feats) if return_features else logits


class WeightedKappaLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.7, kappa_weight: float = 0.3, num_classes: int = 5, class_weights: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.kappa_weight = kappa_weight
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

        W = np.zeros((num_classes, num_classes), dtype=np.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer('W', torch.tensor(W, dtype=torch.float32))

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)
        y_onehot = F.one_hot(target, num_classes=self.num_classes).float()

        O = y_onehot.T @ probs
        O = O / O.sum().clamp(min=1e-8)

        act_hist = y_onehot.sum(dim=0)
        pred_hist = probs.sum(dim=0)
        E = torch.outer(act_hist, pred_hist)
        E = E / E.sum().clamp(min=1e-8)

        num = (self.W * O).sum()
        den = (self.W * E).sum().clamp(min=1e-8)
        qwk_loss = num / den
        return self.ce_weight * ce + self.kappa_weight * qwk_loss


def class_weights_from_df(df: pd.DataFrame, num_classes: int = 5) -> torch.Tensor:
    counts = df['KL'].value_counts().sort_index()
    total = len(df)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        weights.append(total / (num_classes * cnt))
    return torch.tensor(weights, dtype=torch.float32)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    O = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).astype(np.float64)
    if O.sum() == 0:
        return 0.0
    O = O / O.sum()

    act_hist = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    E = E / E.sum()

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    den = (W * E).sum()
    if den == 0:
        return 0.0
    return float(1.0 - ((W * O).sum() / den))


def off_by_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))


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
        ys.extend(y.detach().cpu().numpy())
        ps.extend(pred.detach().cpu().numpy())

    ys = np.asarray(ys)
    ps = np.asarray(ps)
    return {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(ys, ps),
        'macro_f1': f1_score(ys, ps, average='macro', zero_division=0),
        'qwk': quadratic_weighted_kappa(ys, ps),
        'off_by_one_acc': off_by_one_accuracy(ys, ps),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ys, ps, probs = [], [], []

    for batch in loader:
        x = batch['pixel_values'].to(device)
        y = batch['labels'].to(device)
        meta = batch.get('meta')
        meta = meta.to(device) if meta is not None else None

        logits = model(x, meta=meta)
        loss = criterion(logits, y)
        prob = F.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

        total_loss += loss.item() * y.size(0)
        ys.extend(y.cpu().numpy())
        ps.extend(pred.cpu().numpy())
        probs.extend(prob.cpu().numpy())

    y_true = np.asarray(ys)
    y_pred = np.asarray(ps)
    prob_all = np.asarray(probs)

    return {
        'loss': total_loss / len(loader.dataset),
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'qwk': quadratic_weighted_kappa(y_true, y_pred),
        'off_by_one_acc': off_by_one_accuracy(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred,
        'prob_all': prob_all,
    }


def plot_confusion_matrix(cm, labels, path, title='Confusion Matrix'):
    plt.figure(figsize=(6, 5))
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


def plot_training_curves(history_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(history_df['epoch'], history_df['train_qwk'], label='train_qwk')
    plt.plot(history_df['epoch'], history_df['val_qwk'], label='val_qwk')
    plt.plot(history_df['epoch'], history_df['train_macro_f1'], label='train_macro_f1')
    plt.plot(history_df['epoch'], history_df['val_macro_f1'], label='val_macro_f1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Stage 1 Training Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def build_dataloaders(train_df, val_df, test_df, processor, args):
    age_group_mean, age_group_std = (None, None)
    bmi_mean, bmi_std = (None, None)
    if args.use_metadata:
        age_group_mean, age_group_std = compute_col_stats_from_df(train_df, args.age_group_col)
        bmi_mean, bmi_std = compute_col_stats_from_df(train_df, args.bmi_col)

    common = dict(
        processor=processor,
        use_metadata=args.use_metadata,
        age_group_col=args.age_group_col,
        sex_col=args.sex_col,
        bmi_col=args.bmi_col,
        age_group_mean=age_group_mean,
        age_group_std=age_group_std,
        bmi_mean=bmi_mean,
        bmi_std=bmi_std,
    )

    train_ds = KneeKLStage1Dataset(train_df, **common)
    val_ds = KneeKLStage1Dataset(val_df, **common)
    test_ds = KneeKLStage1Dataset(test_df, **common)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def choose_processor(model_name: str):
    return AutoImageProcessor.from_pretrained(model_name)


def main():
    ap = argparse.ArgumentParser(description='Stage 1: KL-grade pretraining (0-4) for KneeGPT')
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--out_dir', default='outputs_stage1_kl')
    ap.add_argument('--backbone_name', default='facebook/dinov2-base')
    ap.add_argument('--model_alias', default='dino')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--head_lr', type=float, default=1e-4)
    ap.add_argument('--encoder_lr', type=float, default=1e-5)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--label_smoothing', type=float, default=0.02)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--early_stopping_patience', type=int, default=5)
    ap.add_argument('--use_metadata', action='store_true')
    ap.add_argument('--age_group_col', default='AGE_GROUP_CODE')
    ap.add_argument('--bmi_col', default='BMI')
    ap.add_argument('--sex_col', default='SEX_MALE')
    ap.add_argument('--num_classes', type=int, default=5)
    ap.add_argument('--ce_weight', type=float, default=0.7)
    ap.add_argument('--kappa_weight', type=float, default=0.3)

    args = ap.parse_args()

    warnings.filterwarnings('ignore')
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.out_dir)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    processor = choose_processor(args.backbone_name)
    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df, processor, args)

    model = KneeStage1KLModel(
        backbone_name=args.backbone_name,
        num_classes=args.num_classes,
        use_metadata=args.use_metadata,
        dropout=args.dropout,
    ).to(args.device)

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': args.weight_decay},
        {'params': model.head.parameters(), 'lr': args.head_lr, 'weight_decay': args.weight_decay},
        *([{'params': model.meta_mlp.parameters(), 'lr': args.head_lr, 'weight_decay': args.weight_decay}] if args.use_metadata else [])
    ])

    class_w = class_weights_from_df(train_df, num_classes=args.num_classes).to(args.device)
    criterion = WeightedKappaLoss(
        ce_weight=args.ce_weight,
        kappa_weight=args.kappa_weight,
        num_classes=args.num_classes,
        class_weights=class_w,
        label_smoothing=args.label_smoothing,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_qwk = -1.0
    best_ckpt_path = os.path.join(args.out_dir, 'best_stage1_model.pt')
    history: List[Dict] = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val = evaluate(model, val_loader, criterion, args.device)
        scheduler.step(val['qwk'])

        history.append({
            'epoch': epoch,
            'train_loss': tr['loss'],
            'train_accuracy': tr['accuracy'],
            'train_macro_f1': tr['macro_f1'],
            'train_qwk': tr['qwk'],
            'train_off_by_one_acc': tr['off_by_one_acc'],
            'val_loss': val['loss'],
            'val_accuracy': val['accuracy'],
            'val_macro_f1': val['macro_f1'],
            'val_qwk': val['qwk'],
            'val_off_by_one_acc': val['off_by_one_acc'],
            'encoder_lr': optimizer.param_groups[0]['lr'],
            'head_lr': optimizer.param_groups[1]['lr'],
        })

        print(
            f"[Stage1-{args.model_alias}] Epoch {epoch:02d} | "
            f"train_qwk={tr['qwk']:.4f} train_f1={tr['macro_f1']:.4f} | "
            f"val_qwk={val['qwk']:.4f} val_f1={val['macro_f1']:.4f} val_acc={val['accuracy']:.4f}"
        )

        if val['qwk'] > best_val_qwk:
            best_val_qwk = val['qwk']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'head_state_dict': model.head.state_dict(),
                'use_metadata': model.use_metadata,
                'backbone_name': args.backbone_name,
                'model_alias': args.model_alias,
                'num_classes': args.num_classes,
            }, best_ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            print(f'Early stopping triggered at epoch {epoch}.')
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.out_dir, 'history.csv'), index=False)
    plot_training_curves(history_df, os.path.join(args.out_dir, 'training_curves.png'))

    ckpt = torch.load(best_ckpt_path, map_location=args.device)
    model.load_state_dict(ckpt['model_state_dict'])

    test = evaluate(model, test_loader, criterion, args.device)
    cm = confusion_matrix(test['y_true'], test['y_pred'], labels=[0, 1, 2, 3, 4])
    plot_confusion_matrix(cm, ['KL0', 'KL1', 'KL2', 'KL3', 'KL4'], os.path.join(args.out_dir, 'confusion_matrix_test.png'), 'Stage 1 Test Confusion Matrix')

    preds_df = pd.DataFrame({
        'y_true': test['y_true'],
        'y_pred': test['y_pred'],
        'p_kl0': test['prob_all'][:, 0],
        'p_kl1': test['prob_all'][:, 1],
        'p_kl2': test['prob_all'][:, 2],
        'p_kl3': test['prob_all'][:, 3],
        'p_kl4': test['prob_all'][:, 4],
    })
    preds_df.to_csv(os.path.join(args.out_dir, 'test_predictions_stage1.csv'), index=False)

    summary = {
        'stage': 'stage1_kl_pretraining',
        'backbone_name': args.backbone_name,
        'model_alias': args.model_alias,
        'use_metadata': args.use_metadata,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'best_val_qwk': float(best_val_qwk),
        'test_accuracy': float(test['accuracy']),
        'test_macro_f1': float(test['macro_f1']),
        'test_qwk': float(test['qwk']),
        'test_off_by_one_acc': float(test['off_by_one_acc']),
        'output_dir': args.out_dir,
        'best_checkpoint': best_ckpt_path,
        'encoder_only_path': os.path.join(args.out_dir, 'stage1_encoder_only.pt'),
    }

    torch.save(model.encoder.state_dict(), summary['encoder_only_path'])

    with open(os.path.join(args.out_dir, 'summary_stage1.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, 'summary_stage1.csv'), index=False)

    print('\n===== Stage 1 complete =====')
    print(json.dumps(summary, indent=2))
    print(f"\nUse this encoder in Stage 2: {summary['encoder_only_path']}")


if __name__ == '__main__':
    main()
