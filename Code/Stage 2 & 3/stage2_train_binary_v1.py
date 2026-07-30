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
    confusion_matrix,
)
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


def stratified_subsample(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 0.999:
        return df.copy().reset_index(drop=True)
    if fraction <= 0:
        raise ValueError('fraction must be > 0')

    rng = np.random.default_rng(seed)
    parts = []
    for cls, grp in df.groupby('TARGET'):
        n = len(grp)
        k = max(1, int(round(n * fraction)))
        idx = rng.choice(grp.index.to_numpy(), size=min(k, n), replace=False)
        parts.append(df.loc[idx])
    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


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
            'kl_grade': int(row['KL']),
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
        'kl_grades': torch.tensor([b['kl_grade'] for b in batch], dtype=torch.long),
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





def freeze_encoder(model: nn.Module):
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_encoder(model: nn.Module):
    for p in model.encoder.parameters():
        p.requires_grad = True


def make_optimizer(model: KneeBinaryModel, mode: str, head_lr: float, encoder_lr: float, weight_decay: float):
    if mode == 'frozen':
        freeze_encoder(model)
        params = [
            {'params': model.head.parameters(), 'lr': head_lr, 'weight_decay': weight_decay},
        ]
        if model.use_metadata:
            params.append({'params': model.meta_mlp.parameters(), 'lr': head_lr, 'weight_decay': weight_decay})
    elif mode == 'finetuned':
        unfreeze_encoder(model)
        params = [
            {'params': model.encoder.parameters(), 'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': model.head.parameters(), 'lr': head_lr, 'weight_decay': weight_decay},
        ]
        if model.use_metadata:
            params.append({'params': model.meta_mlp.parameters(), 'lr': head_lr, 'weight_decay': weight_decay})
    else:
        raise ValueError(f'Unknown training mode: {mode}')
    return torch.optim.AdamW(params)


def class_weights_from_df(df: pd.DataFrame, threshold: int = 2) -> torch.Tensor:
    tmp = add_target_column(df, threshold)
    counts = tmp['TARGET'].value_counts().sort_index()
    total = len(tmp)
    return torch.tensor(
        [total / (2 * counts.get(0, 1)), total / (2 * counts.get(1, 1))],
        dtype=torch.float32,
    )


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
        'f1': f1_score(ys, ps, zero_division=0),
        'macro_f1': f1_score(ys, ps, average='macro', zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    ys, ps, probs, ids, kls = [], [], [], [], []

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
        probs.extend(prob[:, 1].cpu().numpy())
        ids.extend(batch['ids'])
        kls.extend(batch['kl_grades'].cpu().numpy())

    y_true = np.asarray(ys)
    y_pred = np.asarray(ps)
    prob_pos = np.asarray(probs)
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
        'ids': ids,
        'kl_grades': np.asarray(kls),
    }


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


def plot_training_curves(history_df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.plot(history_df['epoch'], history_df['train_macro_f1'], label='train_macro_f1')
    plt.plot(history_df['epoch'], history_df['val_macro_f1'], label='val_macro_f1')
    plt.plot(history_df['epoch'], history_df['train_loss'], label='train_loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Score / Loss')
    plt.title('Stage 2 Training Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def choose_processor(model_name: str):
    return AutoImageProcessor.from_pretrained(model_name)


def build_dataloaders(train_df, val_df, test_df, processor, args):
    age_group_mean, age_group_std = (None, None)
    bmi_mean, bmi_std = (None, None)
    if args.use_metadata:
        age_group_mean, age_group_std = compute_col_stats_from_df(train_df, args.age_group_col)
        bmi_mean, bmi_std = compute_col_stats_from_df(train_df, args.bmi_col)

    common = dict(
        processor=processor,
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

    train_ds = KneeBinaryDataset(train_df, **common)
    val_ds = KneeBinaryDataset(val_df, **common)
    test_ds = KneeBinaryDataset(test_df, **common)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader


def main():
    ap = argparse.ArgumentParser(description='Stage 2: Binary Knee OA training training')
    ap.add_argument('--train_csv', required=True)
    ap.add_argument('--val_csv', required=True)
    ap.add_argument('--test_csv', required=True)
    ap.add_argument('--out_dir', default='outputs_stage2_binary')
    ap.add_argument('--backbone_name', default='facebook/dinov2-base')
    ap.add_argument('--model_alias', default='dino')
    ap.add_argument('--binary_threshold', type=int, default=2)
    ap.add_argument('--train_fraction', type=float, default=1.0)
    ap.add_argument('--train_mode', default='finetuned', choices=['frozen', 'finetuned'])
    ap.add_argument('--use_metadata', action='store_true')
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
    ap.add_argument('--age_group_col', default='AGE_GROUP_CODE')
    ap.add_argument('--bmi_col', default='BMI')
    ap.add_argument('--sex_col', default='SEX_MALE')
    args = ap.parse_args()

    warnings.filterwarnings('ignore')
    set_seed(args.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ensure_dir(args.out_dir)

    train_full = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    train_full = add_target_column(train_full, args.binary_threshold)
    val_df = add_target_column(val_df, args.binary_threshold)
    test_df = add_target_column(test_df, args.binary_threshold)

    train_df = stratified_subsample(train_full, args.train_fraction, args.seed + int(args.train_fraction * 1000))

    processor = choose_processor(args.backbone_name)
    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df, processor, args)

    model = KneeBinaryModel(
        backbone_name=args.backbone_name,
        use_metadata=args.use_metadata,
        dropout=args.dropout,
    ).to(args.device)



    optimizer = make_optimizer(model, args.train_mode, args.head_lr, args.encoder_lr, args.weight_decay)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_from_df(train_df, args.binary_threshold).to(args.device),
        label_smoothing=args.label_smoothing,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val = -1.0
    best_ckpt_path = os.path.join(args.out_dir, 'best_stage2_model.pt')
    history = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val = evaluate(model, val_loader, criterion, args.device)
        scheduler.step(val['macro_f1'])

        history.append({
            'epoch': epoch,
            'train_loss': tr['loss'],
            'train_accuracy': tr['accuracy'],
            'train_f1': tr['f1'],
            'train_macro_f1': tr['macro_f1'],
            'val_loss': val['loss'],
            'val_accuracy': val['accuracy'],
            'val_precision': val['precision'],
            'val_recall': val['recall'],
            'val_f1': val['f1'],
            'val_macro_f1': val['macro_f1'],
            'val_auc': val['auc'],
            'val_auprc': val['auprc'],
            'encoder_lr': optimizer.param_groups[0]['lr'],
            'head_lr': optimizer.param_groups[-1]['lr'],
        })

        print(
            f"[Stage2-{args.model_alias}] Epoch {epoch:02d} | "
            f"train_f1={tr['f1']:.4f} train_macro_f1={tr['macro_f1']:.4f} | "
            f"val_f1={val['f1']:.4f} val_macro_f1={val['macro_f1']:.4f} val_auc={val['auc']:.4f}"
        )

        if val['macro_f1'] > best_val:
            best_val = val['macro_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'head_state_dict': model.head.state_dict(),
                'use_metadata': model.use_metadata,
                'backbone_name': args.backbone_name,
                'model_alias': args.model_alias,
                'train_mode': args.train_mode,
                'binary_threshold': args.binary_threshold,
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

    cm = confusion_matrix(test['y_true'], test['y_pred'], labels=[0, 1])
    plot_confusion_matrix(cm, ['KL<2', 'KL>=2'], os.path.join(args.out_dir, 'confusion_matrix_test.png'), 'Stage 2 Test Confusion Matrix')

    pd.DataFrame({
        'id': test['ids'],
        'kl_grade': test['kl_grades'],
        'y_true': test['y_true'],
        'y_pred': test['y_pred'],
        'prob_positive': test['prob_pos'],
    }).to_csv(os.path.join(args.out_dir, 'test_predictions_stage2.csv'), index=False)

    summary = {
        'stage': 'stage2_binary_training',
        'backbone_name': args.backbone_name,
        'model_alias': args.model_alias,
        'train_mode': args.train_mode,
        'train_fraction': args.train_fraction,
        'use_metadata': args.use_metadata,
        'binary_threshold': args.binary_threshold,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'best_val_macro_f1': float(best_val),
        'test_accuracy': float(test['accuracy']),
        'test_precision': float(test['precision']),
        'test_recall': float(test['recall']),
        'test_f1': float(test['f1']),
        'test_macro_f1': float(test['macro_f1']),
        'test_auc': float(test['auc']),
        'test_auprc': float(test['auprc']),
        'output_dir': args.out_dir,
        'best_checkpoint': best_ckpt_path,
    }

    with open(os.path.join(args.out_dir, 'summary_stage2.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, 'summary_stage2.csv'), index=False)

    print('\n===== Stage 2 complete =====')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
