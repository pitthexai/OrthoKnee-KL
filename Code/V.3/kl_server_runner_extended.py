
import os
import json
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_csv_history(history, path):
    pd.DataFrame(history).to_csv(path, index=False)


def class_weights_from_train(csv_path, num_classes=5):
    print(f"[INFO] Reading class weights from: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)

    if "KL" not in df.columns:
        raise ValueError(f"'KL' column not found in {csv_path}")

    counts = df["KL"].value_counts().sort_index()
    total = len(df)
    weights = []

    print(f"[INFO] Train size: {total}", flush=True)
    print(f"[INFO] Class counts: {counts.to_dict()}", flush=True)

    for c in range(num_classes):
        cnt = counts.get(c, 1)
        weights.append(total / (num_classes * cnt))

    w = torch.tensor(weights, dtype=torch.float32)
    print(f"[INFO] Class weights: {w.tolist()}", flush=True)
    return w


def get_model_hparams(backbone, user_lr=None, user_batch_size=None):
    defaults = {
        "resnet18": {"lr": 1e-4, "batch_size": 8},
        "resnet34": {"lr": 1e-4, "batch_size": 8},
        "efficientnet_b0": {"lr": 1e-4, "batch_size": 8},
        "convnext_tiny": {"lr": 7e-5, "batch_size": 4},
    }

    if backbone not in defaults:
        raise ValueError(f"Unsupported backbone for hparams: {backbone}")

    lr = defaults[backbone]["lr"] if user_lr is None else user_lr
    batch_size = defaults[backbone]["batch_size"] if user_batch_size is None else user_batch_size

    return lr, batch_size


def make_weighted_sampler(csv_path):
    df = pd.read_csv(csv_path)
    counts = df["KL"].value_counts().sort_index().to_dict()
    sample_weights = df["KL"].map(lambda x: 1.0 / counts[x]).values
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


class KneeKLDataset(Dataset):
    def __init__(self, csv_file, transform=None, return_meta=False):
        print(f"[INFO] Loading dataset: {csv_file}", flush=True)
        self.df = pd.read_csv(csv_file).copy()
        self.transform = transform
        self.return_meta = return_meta

        for col in ["id", "image_path", "KL"]:
            if col not in self.df.columns:
                raise ValueError(f"Missing column '{col}' in {csv_file}")

        if len(self.df) == 0:
            raise ValueError(f"Dataset is empty: {csv_file}")

        print(f"[INFO] Loaded {len(self.df)} rows from {csv_file}", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {img_path} | Error: {e}")

        try:
            y = int(row["KL"])
        except Exception as e:
            raise RuntimeError(f"Invalid KL at idx={idx}: {row['KL']} | Error: {e}")

        if self.transform is not None:
            img = self.transform(img)

        if not self.return_meta:
            return img, y

        meta = {
            "id": int(row["id"]) if "id" in row.index else idx,
            "image_path": img_path,
            "side": row["side"] if "side" in row.index else "UNK",
            "age": float(row["AGE"]) if "AGE" in row.index and not pd.isna(row["AGE"]) else np.nan,
            "sex_male": int(row["SEX_MALE"]) if "SEX_MALE" in row.index and not pd.isna(row["SEX_MALE"]) else np.nan,
        }
        return img, y, meta


class KneeKLMultiModalDataset(Dataset):
    def __init__(self, csv_file, transform=None, age_mean=None, age_std=None, return_meta=False):
        print(f"[INFO] Loading multimodal dataset: {csv_file}", flush=True)
        self.df = pd.read_csv(csv_file).copy()
        self.transform = transform
        self.return_meta = return_meta

        required = ["id", "image_path", "KL", "AGE", "SEX_MALE"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing column '{col}' in {csv_file}")

        if len(self.df) == 0:
            raise ValueError(f"Dataset is empty: {csv_file}")

        self.age_mean = float(self.df["AGE"].mean()) if age_mean is None else float(age_mean)
        std = float(self.df["AGE"].std())
        self.age_std = std if age_std is None and std > 0 else float(age_std if age_std is not None else 1.0)
        if self.age_std == 0:
            self.age_std = 1.0

        print(f"[INFO] Loaded {len(self.df)} rows from {csv_file}", flush=True)
        print(f"[INFO] age_mean={self.age_mean:.4f} age_std={self.age_std:.4f}", flush=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {img_path} | Error: {e}")

        y = int(row["KL"])
        age = float(row["AGE"])
        sex_male = float(row["SEX_MALE"])

        age_norm = (age - self.age_mean) / self.age_std
        tab = torch.tensor([age_norm, sex_male], dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        if not self.return_meta:
            return img, tab, y

        meta = {
            "id": int(row["id"]),
            "image_path": img_path,
            "side": row["side"] if "side" in row.index else "UNK",
            "age": age,
            "sex_male": int(sex_male),
        }
        return img, tab, y, meta


def get_transforms(img_size=320):
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
        transforms.ColorJitter(brightness=0.08, contrast=0.08),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def build_model(backbone, num_classes=5, pretrained=False):
    backbone = backbone.lower()
    print(f"[INFO] Building model: {backbone} | pretrained={pretrained}", flush=True)

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif backbone == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    print(f"[INFO] Model ready: {backbone}", flush=True)
    return model


def build_feature_extractor(backbone, pretrained=False):
    backbone = backbone.lower()

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()

    elif backbone == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        feat_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()

    elif backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        feat_dim = model.classifier[2].in_features
        model.classifier = nn.Identity()

    else:
        raise ValueError(f"Unsupported backbone for feature extractor: {backbone}")

    return model, feat_dim


class MultiModalKneeModel(nn.Module):
    def __init__(self, backbone, num_classes=5, pretrained=True, tabular_dim=2, tab_hidden=32):
        super().__init__()
        self.backbone_name = backbone
        self.image_encoder, img_feat_dim = build_feature_extractor(backbone, pretrained=pretrained)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, tab_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(tab_hidden, tab_hidden),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(img_feat_dim + tab_hidden, num_classes)

    def forward(self, x_img, x_tab):
        img_feat = self.image_encoder(x_img)
        if img_feat.ndim > 2:
            img_feat = torch.flatten(img_feat, 1)
        tab_feat = self.tabular_mlp(x_tab)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.classifier(fused)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx == 0:
            print(f"[DEBUG][Train][Epoch {epoch}] first batch x={tuple(x.shape)} y={tuple(y.shape)}", flush=True)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, phase="val"):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    for batch_idx, (x, y) in enumerate(loader):
        if batch_idx == 0:
            print(f"[DEBUG][{phase}] first batch x={tuple(x.shape)} y={tuple(y.shape)}", flush=True)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    return (
        loss,
        acc,
        f1_macro,
        precision_macro,
        recall_macro,
        precision_weighted,
        recall_weighted,
        y_true,
        y_pred,
    )


def train_one_epoch_multimodal(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for batch_idx, (x_img, x_tab, y) in enumerate(loader):
        if batch_idx == 0:
            print(f"[DEBUG][Train-MM][Epoch {epoch}] first batch img={tuple(x_img.shape)} tab={tuple(x_tab.shape)} y={tuple(y.shape)}", flush=True)

        x_img = x_img.to(device, non_blocking=True)
        x_tab = x_tab.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x_img, x_tab)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return loss, acc, f1


@torch.no_grad()
def evaluate_multimodal(model, loader, criterion, device, phase="val"):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    for batch_idx, (x_img, x_tab, y) in enumerate(loader):
        if batch_idx == 0:
            print(f"[DEBUG][{phase}-MM] first batch img={tuple(x_img.shape)} tab={tuple(x_tab.shape)} y={tuple(y.shape)}", flush=True)

        x_img = x_img.to(device, non_blocking=True)
        x_tab = x_tab.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x_img, x_tab)
        loss = criterion(logits, y)

        running_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    return (
        loss,
        acc,
        f1_macro,
        precision_macro,
        recall_macro,
        precision_weighted,
        recall_weighted,
        y_true,
        y_pred,
    )


def save_test_outputs(y_true, y_pred, out_dir):
    ensure_dir(out_dir)

    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv(os.path.join(out_dir, "test_confusion_matrix.csv"), index=False)

    report = classification_report(y_true, y_pred, digits=4)
    with open(os.path.join(out_dir, "test_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_f1, history, patience_counter):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_f1": best_val_f1,
        "history": history,
        "patience_counter": patience_counter,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    start_epoch = ckpt["epoch"] + 1
    best_val_f1 = ckpt["best_val_f1"]
    history = ckpt["history"]
    patience_counter = ckpt.get("patience_counter", 0)

    return model, optimizer, scheduler, start_epoch, best_val_f1, history, patience_counter


def _to_python_scalar(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
        return v.detach().cpu().tolist()
    return v


@torch.no_grad()
def collect_predictions_with_probs(model, loader, device):
    model.eval()
    rows = []

    for batch in loader:
        x, y, meta = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confs, _ = torch.max(probs, dim=1)

        probs_np = probs.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        confs_np = confs.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        batch_size = len(y_np)
        for i in range(batch_size):
            row = {
                "id": int(_to_python_scalar(meta["id"][i])),
                "image_path": str(_to_python_scalar(meta["image_path"][i])),
                "side": str(_to_python_scalar(meta["side"][i])),
                "age": float(_to_python_scalar(meta["age"][i])) if not pd.isna(_to_python_scalar(meta["age"][i])) else np.nan,
                "sex_male": int(_to_python_scalar(meta["sex_male"][i])) if not pd.isna(_to_python_scalar(meta["sex_male"][i])) else np.nan,
                "y_true": int(y_np[i]),
                "y_pred": int(preds_np[i]),
                "correct": int(y_np[i] == preds_np[i]),
                "confidence": float(confs_np[i]),
            }
            for c in range(probs_np.shape[1]):
                row[f"prob_KL{c}"] = float(probs_np[i, c])

            rows.append(row)

    return pd.DataFrame(rows)


def compute_risk_coverage(pred_df):
    df = pred_df.sort_values("confidence", ascending=False).reset_index(drop=True)

    coverages = []
    risks = []
    total = len(df)

    for k in range(1, total + 1):
        sub = df.iloc[:k]
        coverage = k / total
        risk = 1.0 - sub["correct"].mean()
        coverages.append(coverage)
        risks.append(risk)

    rc_df = pd.DataFrame({
        "coverage": coverages,
        "risk": risks,
    })

    aurc = float(np.trapz(rc_df["risk"].values, rc_df["coverage"].values))
    return rc_df, aurc


def save_risk_coverage(pred_df, out_dir, model_name):
    ensure_dir(out_dir)

    rc_df, aurc = compute_risk_coverage(pred_df)
    rc_csv = os.path.join(out_dir, "risk_coverage.csv")
    rc_png = os.path.join(out_dir, "risk_coverage.png")

    rc_df.to_csv(rc_csv, index=False)

    plt.figure(figsize=(6, 5))
    plt.plot(rc_df["coverage"], rc_df["risk"], linewidth=2, label=f"{model_name} | AURC={aurc:.4f}")
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title(f"Risk-Coverage: {model_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rc_png, dpi=200)
    plt.close()

    return aurc


def save_risk_coverage_combined(outputs_root, model_names):
    plt.figure(figsize=(7, 6))
    found_any = False

    for model_name in model_names:
        rc_csv = os.path.join(outputs_root, model_name, "risk_coverage.csv")
        if os.path.exists(rc_csv):
            rc_df = pd.read_csv(rc_csv)
            plt.plot(rc_df["coverage"], rc_df["risk"], linewidth=2, label=model_name)
            found_any = True

    if not found_any:
        print("[WARN] No risk_coverage.csv files found for combined plot.", flush=True)
        plt.close()
        return

    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title("Risk-Coverage Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(outputs_root, "risk_coverage_all_models.png")
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[INFO] Saved combined risk-coverage plot: {out_path}", flush=True)


def pick_best_model(summary_df, metric="test_macro_f1"):
    valid_df = summary_df.dropna(subset=[metric]).copy()
    if len(valid_df) == 0:
        raise ValueError("No valid rows found for best model selection.")

    best_row = valid_df.sort_values(metric, ascending=False).iloc[0].to_dict()
    return best_row


def save_best_model_age_sex_analysis(pred_df, out_dir):
    ensure_dir(out_dir)
    rows = []

    if "sex_male" in pred_df.columns and pred_df["sex_male"].notna().any():
        for sex_val in sorted(pred_df["sex_male"].dropna().unique()):
            sub = pred_df[pred_df["sex_male"] == sex_val].copy()
            if len(sub) < 5:
                continue

            sex_name = "male" if int(sex_val) == 1 else "female"
            rows.append({
                "group_type": "sex",
                "group_value": sex_name,
                "n": len(sub),
                "acc": accuracy_score(sub["y_true"], sub["y_pred"]),
                "macro_f1": f1_score(sub["y_true"], sub["y_pred"], average="macro"),
                "macro_precision": precision_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0),
                "macro_recall": recall_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0),
            })

    if "age" in pred_df.columns and pred_df["age"].notna().any():
        age_df = pred_df[pred_df["age"].notna()].copy()
        bins = [0, 59, 69, 79, 120]
        labels = ["<=59", "60-69", "70-79", "80+"]
        age_df["age_group"] = pd.cut(age_df["age"], bins=bins, labels=labels, include_lowest=True)

        for age_group in labels:
            sub = age_df[age_df["age_group"] == age_group].copy()
            if len(sub) < 5:
                continue

            rows.append({
                "group_type": "age",
                "group_value": age_group,
                "n": len(sub),
                "acc": accuracy_score(sub["y_true"], sub["y_pred"]),
                "macro_f1": f1_score(sub["y_true"], sub["y_pred"], average="macro"),
                "macro_precision": precision_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0),
                "macro_recall": recall_score(sub["y_true"], sub["y_pred"], average="macro", zero_division=0),
            })

    out_csv = os.path.join(out_dir, "best_model_age_sex_analysis.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[INFO] Saved best-model age/sex analysis to: {out_csv}", flush=True)


def get_target_layer(model, backbone):
    backbone = backbone.lower()
    if backbone in ["resnet18", "resnet34"]:
        return model.layer4[-1]
    if backbone == "efficientnet_b0":
        return model.features[-1]
    if backbone == "convnext_tiny":
        return model.features[-1]
    raise ValueError(f"Unsupported backbone for Grad-CAM++: {backbone}")


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = None
        self.backward_handle = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()
        if self.backward_handle is not None:
            self.backward_handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(output, dim=1).item())

        score = output[:, class_idx]
        score_exp = torch.exp(score).sum()
        score_exp.backward(retain_graph=True)

        grads = self.gradients
        activations = self.activations

        b, k, u, v = grads.size()
        alpha_num = grads.pow(2)
        alpha_denom = 2.0 * grads.pow(2) + torch.sum(activations * grads.pow(3), dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num / (alpha_denom + 1e-7)

        positive_gradients = F.relu(score_exp * grads)
        weights = torch.sum(alpha * positive_gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1, keepdim=False)
        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy()[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, class_idx


def denormalize_image(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.detach().cpu() * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def apply_colormap(cam):
    cmap = plt.get_cmap("jet")
    colored = cmap(cam)[..., :3]
    return colored


def overlay_heatmap_on_image(rgb_img, cam, alpha=0.45):
    heatmap = apply_colormap(cam)
    overlay = (1 - alpha) * rgb_img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay


def select_gradcam_samples(csv_path, max_samples=2):
    df = pd.read_csv(csv_path)
    selected_indices = []

    for side in ["R", "L"]:
        sub = df[df["side"] == side]
        if len(sub) > 0:
            selected_indices.append(int(sub.index[0]))

    if len(selected_indices) < max_samples:
        for idx in range(len(df)):
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= max_samples:
                break

    return selected_indices[:max_samples]


def save_gradcampp_for_all_models(args, model_names):
    print("[INFO] Generating Grad-CAM++ for all models...", flush=True)
    out_dir = os.path.join(args.out_root, "gradcampp_all_models")
    ensure_dir(out_dir)

    _, eval_tfms = get_transforms(args.img_size)
    vis_ds = KneeKLDataset(args.test_csv, transform=eval_tfms, return_meta=True)
    selected_indices = select_gradcam_samples(args.test_csv, max_samples=2)

    rows_meta = []

    for sample_i, ds_idx in enumerate(selected_indices):
        x, y, meta = vis_ds[ds_idx]
        rgb_img = denormalize_image(x)

        fig, axes = plt.subplots(1, len(model_names) + 1, figsize=(4 * (len(model_names) + 1), 4))
        if len(model_names) + 1 == 1:
            axes = [axes]

        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Raw Knee\nTrue=KL{y} | Side={meta['side']}")
        axes[0].axis("off")

        raw_out = os.path.join(out_dir, f"sample_{sample_i}_raw.png")
        plt.imsave(raw_out, rgb_img)

        rows_meta.append({
            "sample_id": sample_i,
            "dataset_index": ds_idx,
            "image_path": meta["image_path"],
            "side": meta["side"],
            "true_label": y,
            "raw_image_path": raw_out,
        })

        for col_i, backbone in enumerate(model_names, start=1):
            model = build_model(backbone, num_classes=5, pretrained=args.pretrained).to(args.device)
            ckpt_path = os.path.join(args.out_root, backbone, "best_model.pt")
            state = torch.load(ckpt_path, map_location=args.device)
            model.load_state_dict(state)
            model.eval()

            target_layer = get_target_layer(model, backbone)
            campp = GradCAMPlusPlus(model, target_layer)
            input_tensor = x.unsqueeze(0).to(args.device)

            cam, pred = campp(input_tensor)
            campp.remove_hooks()

            overlay = overlay_heatmap_on_image(rgb_img, cam, alpha=0.45)
            axes[col_i].imshow(overlay)
            axes[col_i].set_title(f"{backbone}\nPred=KL{pred}")
            axes[col_i].axis("off")

            single_path = os.path.join(out_dir, f"sample_{sample_i}_{backbone}_gradcampp.png")
            plt.imsave(single_path, overlay)

        fig.tight_layout()
        grid_path = os.path.join(out_dir, f"sample_{sample_i}_gradcampp_comparison.png")
        fig.savefig(grid_path, dpi=240, bbox_inches="tight")
        plt.close(fig)

    pd.DataFrame(rows_meta).to_csv(os.path.join(out_dir, "selected_samples.csv"), index=False)
    print(f"[INFO] Saved Grad-CAM++ comparisons to: {out_dir}", flush=True)


def run_one_model(backbone, args):
    print("\n" + "=" * 80, flush=True)
    print(f"[INFO] STARTING MODEL: {backbone}", flush=True)
    print("=" * 80, flush=True)

    lr, batch_size = get_model_hparams(backbone, user_lr=args.lr, user_batch_size=args.batch_size)

    print(f"[INFO] Hyperparameters for {backbone}: lr={lr}, batch_size={batch_size}", flush=True)
    print(f"[INFO] Shared params: epochs={args.epochs}, weight_decay={args.weight_decay}, "
          f"label_smoothing={args.label_smoothing}, patience={args.early_stopping_patience}", flush=True)

    model_dir = os.path.join(args.out_root, backbone)
    ensure_dir(model_dir)

    checkpoint_path = os.path.join(model_dir, "last_checkpoint.pt")
    best_model_path = os.path.join(model_dir, "best_model.pt")
    history_path = os.path.join(model_dir, "history.csv")
    summary_path = os.path.join(model_dir, "test_summary.txt")

    train_tfms, eval_tfms = get_transforms(args.img_size)

    train_ds = KneeKLDataset(args.train_csv, transform=train_tfms)
    val_ds = KneeKLDataset(args.val_csv, transform=eval_tfms)
    test_ds = KneeKLDataset(args.test_csv, transform=eval_tfms, return_meta=True)

    print(f"[INFO] train/val/test sizes = {len(train_ds)} / {len(val_ds)} / {len(test_ds)}", flush=True)

    first_img_path = train_ds.df.iloc[0]["image_path"]
    print(f"[INFO] first image path: {first_img_path}", flush=True)
    with Image.open(first_img_path) as im:
        print(f"[INFO] first image ok: size={im.size}, mode={im.mode}", flush=True)

    train_sampler = make_weighted_sampler(args.train_csv)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader_plain = DataLoader(
        KneeKLDataset(args.test_csv, transform=eval_tfms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader_meta = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(backbone, num_classes=5, pretrained=args.pretrained).to(args.device)

    weights = class_weights_from_train(args.train_csv, num_classes=5).to(args.device)
    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=args.label_smoothing
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_reduce_factor,
        patience=args.lr_scheduler_patience,
    )

    start_epoch = 1
    best_val_f1 = -1.0
    history = []
    patience_counter = 0

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}", flush=True)
        model, optimizer, scheduler, start_epoch, best_val_f1, history, patience_counter = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, args.device
        )
    else:
        print("[INFO] No checkpoint found. Starting fresh.", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n[INFO] {backbone} | Epoch {epoch}/{args.epochs}", flush=True)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[INFO] Current LR: {current_lr:.8f}", flush=True)

        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, args.device, epoch
        )

        va_loss, va_acc, va_f1, va_prec_macro, va_rec_macro, va_prec_weighted, va_rec_weighted, _, _ = evaluate(
            model, val_loader, criterion, args.device, phase="val"
        )

        scheduler.step(va_f1)
        new_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_f1_macro": tr_f1,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_f1_macro": va_f1,
            "val_precision_macro": va_prec_macro,
            "val_recall_macro": va_rec_macro,
            "val_precision_weighted": va_prec_weighted,
            "val_recall_weighted": va_rec_weighted,
        }
        history.append(row)

        print(
            f"[{backbone}] Epoch {epoch:02d}/{args.epochs} | "
            f"lr={current_lr:.8f} -> {new_lr:.8f} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} train_f1={tr_f1:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}",
            flush=True
        )

        improved = va_f1 > best_val_f1

        if improved:
            best_val_f1 = va_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] New best model saved: {best_model_path}", flush=True)
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Early-stop counter: {patience_counter}/{args.early_stopping_patience}", flush=True)

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_f1,
            history,
            patience_counter
        )
        save_csv_history(history, history_path)

        if patience_counter >= args.early_stopping_patience:
            print(f"[INFO] Early stopping triggered for {backbone}.", flush=True)
            break

    print(f"[INFO] Loading best model: {best_model_path}", flush=True)
    model.load_state_dict(torch.load(best_model_path, map_location=args.device))

    test_loss, test_acc, test_f1, test_prec_macro, test_rec_macro, test_prec_weighted, test_rec_weighted, y_true, y_pred = evaluate(
        model, test_loader_plain, criterion, args.device, phase="test"
    )
    save_test_outputs(y_true, y_pred, model_dir)

    pred_df = collect_predictions_with_probs(model, test_loader_meta, args.device)
    pred_df.to_csv(os.path.join(model_dir, "test_predictions_with_probs.csv"), index=False)
    aurc = save_risk_coverage(pred_df, model_dir, backbone)
    print(f"[INFO] {backbone} AURC={aurc:.6f}", flush=True)

    print("\n[INFO] Test classification report:", flush=True)
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3, 4],
        target_names=["KL0", "KL1", "KL2", "KL3", "KL4"],
        digits=4
    ), flush=True)

    print("[INFO] Test confusion matrix:", flush=True)
    print(confusion_matrix(y_true, y_pred), flush=True)

    report_path = os.path.join(model_dir, "test_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Test Classification Report\n\n")
        f.write(classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3, 4],
            target_names=["KL0", "KL1", "KL2", "KL3", "KL4"],
            digits=4
        ))
        f.write("\n\nConfusion Matrix\n")
        f.write(str(confusion_matrix(y_true, y_pred)))

    print(f"[INFO] Saved test report to: {report_path}", flush=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"backbone: {backbone}\n")
        f.write(f"test_loss: {test_loss:.6f}\n")
        f.write(f"test_acc: {test_acc:.6f}\n")
        f.write(f"test_macro_f1: {test_f1:.6f}\n")
        f.write(f"test_macro_precision: {test_prec_macro:.6f}\n")
        f.write(f"test_macro_recall: {test_rec_macro:.6f}\n")
        f.write(f"test_weighted_precision: {test_prec_weighted:.6f}\n")
        f.write(f"test_weighted_recall: {test_rec_weighted:.6f}\n")
        f.write(f"AURC: {aurc:.6f}\n")
        f.write(f"best_val_f1: {best_val_f1:.6f}\n")
        f.write(f"final_lr: {optimizer.param_groups[0]['lr']:.10f}\n")
        f.write(f"label_smoothing: {args.label_smoothing}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")

    print(f"[INFO] FINISHED MODEL: {backbone}", flush=True)
    print(
        f"[INFO] Test loss={test_loss:.4f} "
        f"acc={test_acc:.4f} "
        f"macro_f1={test_f1:.4f} "
        f"macro_precision={test_prec_macro:.4f} "
        f"macro_recall={test_rec_macro:.4f}",
        flush=True
    )

    return {
        "backbone": backbone,
        "best_val_f1": best_val_f1,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "test_macro_precision": test_prec_macro,
        "test_macro_recall": test_rec_macro,
        "test_weighted_precision": test_prec_weighted,
        "test_weighted_recall": test_rec_weighted,
        "AURC": aurc,
        "final_lr": optimizer.param_groups[0]["lr"],
        "used_batch_size": batch_size,
        "used_lr": lr,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
    }


def run_best_model_with_age_sex(best_backbone, args):
    print("\n" + "=" * 80, flush=True)
    print(f"[INFO] STARTING BEST-BACKBONE + AGE/SEX MODEL: {best_backbone}", flush=True)
    print("=" * 80, flush=True)

    lr, batch_size = get_model_hparams(best_backbone, user_lr=args.lr, user_batch_size=args.batch_size)
    out_dir = os.path.join(args.out_root, f"{best_backbone}_with_age_sex")
    ensure_dir(out_dir)

    checkpoint_path = os.path.join(out_dir, "last_checkpoint.pt")
    best_model_path = os.path.join(out_dir, "best_model.pt")
    history_path = os.path.join(out_dir, "history.csv")
    summary_path = os.path.join(out_dir, "test_summary.txt")

    train_tfms, eval_tfms = get_transforms(args.img_size)

    train_df = pd.read_csv(args.train_csv)
    age_mean = float(train_df["AGE"].mean())
    age_std = float(train_df["AGE"].std()) if float(train_df["AGE"].std()) > 0 else 1.0

    train_ds = KneeKLMultiModalDataset(args.train_csv, transform=train_tfms, age_mean=age_mean, age_std=age_std)
    val_ds = KneeKLMultiModalDataset(args.val_csv, transform=eval_tfms, age_mean=age_mean, age_std=age_std)
    test_ds = KneeKLMultiModalDataset(args.test_csv, transform=eval_tfms, age_mean=age_mean, age_std=age_std, return_meta=True)
    test_ds_plain = KneeKLMultiModalDataset(args.test_csv, transform=eval_tfms, age_mean=age_mean, age_std=age_std)

    train_sampler = make_weighted_sampler(args.train_csv)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds_plain,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader_meta = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MultiModalKneeModel(best_backbone, num_classes=5, pretrained=args.pretrained).to(args.device)
    weights = class_weights_from_train(args.train_csv, num_classes=5).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_reduce_factor,
        patience=args.lr_scheduler_patience,
    )

    start_epoch = 1
    best_val_f1 = -1.0
    history = []
    patience_counter = 0

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming multimodal checkpoint: {checkpoint_path}", flush=True)
        model, optimizer, scheduler, start_epoch, best_val_f1, history, patience_counter = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, args.device
        )
    else:
        print("[INFO] No multimodal checkpoint found. Starting fresh.", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n[INFO] {best_backbone}+age/sex | Epoch {epoch}/{args.epochs}", flush=True)
        current_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc, tr_f1 = train_one_epoch_multimodal(model, train_loader, optimizer, criterion, args.device, epoch)
        va_loss, va_acc, va_f1, va_prec_macro, va_rec_macro, va_prec_weighted, va_rec_weighted, _, _ = evaluate_multimodal(
            model, val_loader, criterion, args.device, phase="val"
        )
        scheduler.step(va_f1)
        new_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_f1_macro": tr_f1,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_f1_macro": va_f1,
            "val_precision_macro": va_prec_macro,
            "val_recall_macro": va_rec_macro,
            "val_precision_weighted": va_prec_weighted,
            "val_recall_weighted": va_rec_weighted,
        }
        history.append(row)

        print(
            f"[{best_backbone}+age/sex] Epoch {epoch:02d}/{args.epochs} | "
            f"lr={current_lr:.8f} -> {new_lr:.8f} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} train_f1={tr_f1:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}",
            flush=True
        )

        improved = va_f1 > best_val_f1
        if improved:
            best_val_f1 = va_f1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] New best multimodal model saved: {best_model_path}", flush=True)
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Early-stop counter: {patience_counter}/{args.early_stopping_patience}", flush=True)

        save_checkpoint(
            checkpoint_path, model, optimizer, scheduler, epoch, best_val_f1, history, patience_counter
        )
        save_csv_history(history, history_path)

        if patience_counter >= args.early_stopping_patience:
            print(f"[INFO] Early stopping triggered for {best_backbone}+age/sex.", flush=True)
            break

    print(f"[INFO] Loading best multimodal model: {best_model_path}", flush=True)
    model.load_state_dict(torch.load(best_model_path, map_location=args.device))

    test_loss, test_acc, test_f1, test_prec_macro, test_rec_macro, test_prec_weighted, test_rec_weighted, y_true, y_pred = evaluate_multimodal(
        model, test_loader, criterion, args.device, phase="test"
    )
    save_test_outputs(y_true, y_pred, out_dir)

    multimodal_pred_rows = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader_meta:
            x_img, x_tab, y, meta = batch
            x_img = x_img.to(args.device, non_blocking=True)
            x_tab = x_tab.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            logits = model(x_img, x_tab)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs, _ = torch.max(probs, dim=1)

            probs_np = probs.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            confs_np = confs.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            for i in range(len(y_np)):
                row = {
                    "id": int(_to_python_scalar(meta["id"][i])),
                    "image_path": str(_to_python_scalar(meta["image_path"][i])),
                    "side": str(_to_python_scalar(meta["side"][i])),
                    "age": float(_to_python_scalar(meta["age"][i])),
                    "sex_male": int(_to_python_scalar(meta["sex_male"][i])),
                    "y_true": int(y_np[i]),
                    "y_pred": int(preds_np[i]),
                    "correct": int(y_np[i] == preds_np[i]),
                    "confidence": float(confs_np[i]),
                }
                for c in range(probs_np.shape[1]):
                    row[f"prob_KL{c}"] = float(probs_np[i, c])
                multimodal_pred_rows.append(row)

    pred_df = pd.DataFrame(multimodal_pred_rows)
    pred_df.to_csv(os.path.join(out_dir, "test_predictions_with_probs.csv"), index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"backbone: {best_backbone}_with_age_sex\n")
        f.write(f"test_loss: {test_loss:.6f}\n")
        f.write(f"test_acc: {test_acc:.6f}\n")
        f.write(f"test_macro_f1: {test_f1:.6f}\n")
        f.write(f"test_macro_precision: {test_prec_macro:.6f}\n")
        f.write(f"test_macro_recall: {test_rec_macro:.6f}\n")
        f.write(f"test_weighted_precision: {test_prec_weighted:.6f}\n")
        f.write(f"test_weighted_recall: {test_rec_weighted:.6f}\n")
        f.write(f"best_val_f1: {best_val_f1:.6f}\n")
        f.write(f"age_mean: {age_mean:.6f}\n")
        f.write(f"age_std: {age_std:.6f}\n")

    save_best_model_age_sex_analysis(pred_df, out_dir)

    comparison = pd.DataFrame([{
        "model": best_backbone,
        "variant": "image_plus_age_sex",
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "test_macro_precision": test_prec_macro,
        "test_macro_recall": test_rec_macro,
    }])
    comparison.to_csv(os.path.join(out_dir, "comparison_image_plus_age_sex.csv"), index=False)

    print(f"[INFO] FINISHED BEST-BACKBONE + AGE/SEX MODEL: {best_backbone}", flush=True)
    return {
        "backbone": best_backbone,
        "variant": "image_plus_age_sex",
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "test_macro_precision": test_prec_macro,
        "test_macro_recall": test_rec_macro,
    }


def main():
    print("[INFO] Script started.", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--val_csv", type=str, default="val.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", default=True)

    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--lr_scheduler_patience", type=int, default=3)
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5)

    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18", "efficientnet_b0", "resnet34", "convnext_tiny"],
        choices=["resnet18", "efficientnet_b0", "resnet34", "convnext_tiny"],
    )
    args = parser.parse_args()

    print(f"[INFO] Args: {args}", flush=True)

    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Device: {args.device}", flush=True)
    print(f"[INFO] Models to run: {args.models}", flush=True)

    ensure_dir(args.out_root)

    all_results = []

    for backbone in args.models:
        try:
            result = run_one_model(backbone, args)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Model {backbone} failed: {e}", flush=True)

            all_results.append({
                "backbone": backbone,
                "best_val_f1": None,
                "test_loss": None,
                "test_acc": None,
                "test_macro_f1": None,
                "test_macro_precision": None,
                "test_macro_recall": None,
                "test_weighted_precision": None,
                "test_weighted_recall": None,
                "AURC": None,
                "final_lr": None,
                "used_batch_size": None,
                "used_lr": None,
                "label_smoothing": args.label_smoothing,
                "weight_decay": args.weight_decay,
                "error": str(e),
            })
            continue

    summary_df = pd.DataFrame(all_results)
    summary_csv = os.path.join(args.out_root, "summary_all_models.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n[INFO] All requested runs finished.", flush=True)
    print(f"[INFO] Summary saved to: {summary_csv}", flush=True)

    if len(summary_df) > 0 and "test_macro_f1" in summary_df.columns:
        try:
            ranked = summary_df.sort_values("test_macro_f1", ascending=False, na_position="last")
            print("[INFO] Final ranking by test_macro_f1:", flush=True)
            print(ranked[["backbone", "test_acc", "test_macro_f1", "best_val_f1", "AURC"]], flush=True)
        except Exception:
            ranked = summary_df.copy()
    else:
        ranked = summary_df.copy()

    save_risk_coverage_combined(args.out_root, args.models)

    try:
        save_gradcampp_for_all_models(args, args.models)
    except Exception as e:
        print(f"[WARN] Grad-CAM++ generation failed: {e}", flush=True)

    try:
        best_info = pick_best_model(summary_df, metric="test_macro_f1")

        best_json_path = os.path.join(args.out_root, "best_model_selected.json")
        with open(best_json_path, "w", encoding="utf-8") as f:
            json.dump(best_info, f, indent=2)

        print(f"[INFO] Best model by test_macro_f1: {best_info['backbone']}", flush=True)
        print(f"[INFO] Saved best model info to: {best_json_path}", flush=True)

        best_model_dir = os.path.join(args.out_root, best_info["backbone"])
        best_pred_path = os.path.join(best_model_dir, "test_predictions_with_probs.csv")

        if os.path.exists(best_pred_path):
            best_pred_df = pd.read_csv(best_pred_path)
            save_best_model_age_sex_analysis(best_pred_df, best_model_dir)
        else:
            print(f"[WARN] Best model prediction file not found: {best_pred_path}", flush=True)

        mm_result = run_best_model_with_age_sex(best_info["backbone"], args)
        pd.DataFrame([mm_result]).to_csv(
            os.path.join(args.out_root, "best_model_with_age_sex_summary.csv"),
            index=False
        )
    except Exception as e:
        print(f"[WARN] Best model selection / age-sex rerun failed: {e}", flush=True)


if __name__ == "__main__":
    main()
