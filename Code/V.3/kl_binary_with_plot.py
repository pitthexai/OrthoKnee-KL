import os
import random
import argparse
import copy
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score,)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_csv_history(history: List[dict], path: str):
    pd.DataFrame(history).to_csv(path, index=False)


def map_kl_to_binary(kl_value: int, threshold: int = 2) -> int:
    kl_value = int(kl_value)
    return 1 if kl_value >= threshold else 0


def get_binary_target_names(threshold: int = 2) -> List[str]:
    return [f"KL<{threshold}", f"KL>={threshold}"]


def class_weights_from_train(csv_path: str, threshold: int = 2) -> torch.Tensor:
    print(f"[INFO] Reading class weights from: {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    if "KL" not in df.columns:
        raise ValueError(f"'KL' column not found in {csv_path}")

    df["TARGET"] = df["KL"].apply(lambda x: map_kl_to_binary(x, threshold=threshold))
    counts = df["TARGET"].value_counts().sort_index()
    total = len(df)
    weights = []

    print(f"[INFO] Train size: {total}", flush=True)
    print(f"[INFO] Binary class counts: {counts.to_dict()}", flush=True)

    for c in range(2):
        cnt = counts.get(c, 1)
        weights.append(total / (2 * cnt))

    w = torch.tensor(weights, dtype=torch.float32)
    print(f"[INFO] Class weights: {w.tolist()}", flush=True)
    return w


def get_model_hparams(backbone: str, user_lr: Optional[float] = None, user_batch_size: Optional[int] = None):
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


def compute_age_stats(train_csv: str, age_col: str):
    df = pd.read_csv(train_csv)
    if age_col not in df.columns:
        raise ValueError(f"Age column '{age_col}' not found in {train_csv}")
    age_series = pd.to_numeric(df[age_col], errors="coerce")
    mean = float(age_series.mean())
    std = float(age_series.std())
    if np.isnan(mean):
        mean = 0.0
    if np.isnan(std) or std == 0:
        std = 1.0
    return mean, std


class KneeBinaryDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        transform=None,
        threshold: int = 2,
        use_metadata: bool = False,
        age_col: str = "AGE",
        sex_col: str = "SEX_MALE",
        age_mean: Optional[float] = None,
        age_std: Optional[float] = None,
    ):
        print(f"[INFO] Loading dataset: {csv_file}", flush=True)
        self.df = pd.read_csv(csv_file).copy()
        self.transform = transform
        self.threshold = threshold
        self.use_metadata = use_metadata
        self.age_col = age_col
        self.sex_col = sex_col
        self.age_mean = age_mean
        self.age_std = age_std

        for col in ["id", "image_path", "KL"]:
            if col not in self.df.columns:
                raise ValueError(f"Missing column '{col}' in {csv_file}")

        if self.use_metadata:
            for col in [self.age_col, self.sex_col]:
                if col not in self.df.columns:
                    raise ValueError(f"Missing metadata column '{col}' in {csv_file}")

        self.df["TARGET"] = self.df["KL"].apply(
            lambda x: map_kl_to_binary(x, threshold=self.threshold)
        )

        if len(self.df) == 0:
            raise ValueError(f"Dataset is empty: {csv_file}")

        print(f"[INFO] Loaded {len(self.df)} rows from {csv_file}", flush=True)

    def __len__(self):
        return len(self.df)

    def _encode_sex(self, val):
        if pd.isna(val):
            return 0.0
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ["1", "m", "male", "true"]:
                return 1.0
            if v in ["0", "2", "f", "female", "false"]:
                return 0.0
        try:
            v = float(val)
            if v == 1:
                return 1.0
            if v == 0 or v == 2:
                return 0.0
            return v
        except Exception:
            return 0.0

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {img_path} | Error: {e}")

        y = int(row["TARGET"])

        if self.transform is not None:
            img = self.transform(img)

        if not self.use_metadata:
            return img, y

        age = float(row[self.age_col]) if not pd.isna(row[self.age_col]) else 0.0
        if self.age_mean is not None and self.age_std is not None and self.age_std > 0:
            age = (age - self.age_mean) / self.age_std
        sex = self._encode_sex(row[self.sex_col])
        meta = torch.tensor([age, sex], dtype=torch.float32)
        return img, y, meta


def get_transforms(img_size: int = 320):
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


def build_model(backbone: str, pretrained: bool = False):
    backbone = backbone.lower()
    print(f"[INFO] Building binary model: {backbone} | pretrained={pretrained}", flush=True)

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

    elif backbone == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)

    elif backbone == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 2)

    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model


class ImageBackboneWithMeta(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = False, meta_dim: int = 2, meta_hidden_dim: int = 32):
        super().__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.image_encoder = model

        elif self.backbone_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.image_encoder = model

        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            feat_dim = model.classifier[1].in_features
            model.classifier[1] = nn.Identity()
            self.image_encoder = model

        elif self.backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = models.convnext_tiny(weights=weights)
            feat_dim = model.classifier[2].in_features
            model.classifier[2] = nn.Identity()
            self.image_encoder = model
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(feat_dim + meta_hidden_dim, 2)

    def forward(self, x, meta):
        img_feat = self.image_encoder(x)
        if img_feat.ndim > 2:
            img_feat = torch.flatten(img_feat, 1)
        meta_feat = self.meta_mlp(meta)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.classifier(fused)


def unpack_batch(batch, device):
    if len(batch) == 2:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        return x, y, None
    elif len(batch) == 3:
        x, y, meta = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        meta = meta.to(device, non_blocking=True)
        return x, y, meta
    raise ValueError("Unexpected batch format")


def forward_model(model, x, meta=None):
    if meta is None:
        return model(x)
    return model(x, meta)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []

    for batch_idx, batch in enumerate(loader):
        x, y, meta = unpack_batch(batch, device)
        if batch_idx == 0:
            meta_shape = None if meta is None else tuple(meta.shape)
            print(f"[DEBUG][Train][Epoch {epoch}] first batch x={tuple(x.shape)} y={tuple(y.shape)} meta={meta_shape}", flush=True)

        optimizer.zero_grad()
        logits = forward_model(model, x, meta)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        pred = torch.argmax(logits, dim=1)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    return loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, phase="val"):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    prob_pos_all, conf_all = [], []

    for batch_idx, batch in enumerate(loader):
        x, y, meta = unpack_batch(batch, device)
        if batch_idx == 0:
            meta_shape = None if meta is None else tuple(meta.shape)
            print(f"[DEBUG][{phase}] first batch x={tuple(x.shape)} y={tuple(y.shape)} meta={meta_shape}", flush=True)

        logits = forward_model(model, x, meta)
        loss = criterion(logits, y)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

        running_loss += loss.item() * y.size(0)

        y_true.extend(y.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        prob_pos_all.extend(probs[:, 1].detach().cpu().numpy().tolist())
        conf_all.extend(conf.detach().cpu().numpy().tolist())

    loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    f1_bin = f1_score(y_true, y_pred, average="binary", zero_division=0)
    precision_bin = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall_bin = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        auc = roc_auc_score(y_true, prob_pos_all)
    except Exception:
        auc = np.nan

    return {
        "loss": loss,
        "acc": acc,
        "f1_binary": f1_bin,
        "precision_binary": precision_bin,
        "recall_binary": recall_bin,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_pos": prob_pos_all,
        "confidence": conf_all,
    }


def save_test_outputs(y_true, y_pred, prob_pos, confidence, out_dir, target_names):
    ensure_dir(out_dir)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        os.path.join(out_dir, "test_confusion_matrix.csv")
    )

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, digits=4, zero_division=0
    )
    with open(os.path.join(out_dir, "test_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "prob_positive": prob_pos,
        "confidence": confidence,
    }).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)


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



def compute_risk_coverage(y_true, y_pred, confidence):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence)

    order = np.argsort(-confidence)
    y_true = y_true[order]
    y_pred = y_pred[order]

    coverages, risks, accuracies = [], [], []
    n = len(y_true)

    for k in range(1, n + 1):
        yt = y_true[:k]
        yp = y_pred[:k]
        acc = float((yt == yp).mean())
        risk = 1.0 - acc
        coverages.append(k / n)
        risks.append(risk)
        accuracies.append(acc)

    aurc = float(np.trapz(risks, coverages))
    return {
        "coverage": np.asarray(coverages),
        "risk": np.asarray(risks),
        "accuracy": np.asarray(accuracies),
        "aurc": aurc,
    }


def coverage_at_target_accuracy(risk_cov_dict, target_acc=0.95):
    coverage = risk_cov_dict["coverage"]
    accuracy = risk_cov_dict["accuracy"]
    ok = np.where(accuracy >= target_acc)[0]
    if len(ok) == 0:
        return 0.0
    return float(coverage[ok[-1]])


def save_risk_coverage_single(model_name, rc_dict, out_path):
    plt.figure(figsize=(7, 5))
    plt.plot(rc_dict["coverage"], rc_dict["risk"], label=f"{model_name} (AURC={rc_dict['aurc']:.4f})")
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title(f"Risk-Coverage: {model_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_risk_coverage_all(curves: Dict[str, dict], out_path: str):
    plt.figure(figsize=(8, 6))
    for name, rc in curves.items():
        plt.plot(rc["coverage"], rc["risk"], label=f"{name} (AURC={rc['aurc']:.4f})")
    plt.xlabel("Coverage")
    plt.ylabel("Risk (1 - accuracy)")
    plt.title("Risk-Coverage Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()



class _FeatureHook:
    def __init__(self, module):
        self.activations = None
        self.gradients = None
        self.h1 = module.register_forward_hook(self._forward_hook)
        self.h2 = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self.h1.remove()
        self.h2.remove()


def find_last_conv_module(model: nn.Module):
    last_conv = None
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("Could not find Conv2d layer for Grad-CAM.")
    return last_conv


def denormalize_image(tensor_img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor_img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor_img.device).view(3, 1, 1)
    img = tensor_img * std + mean
    return torch.clamp(img, 0.0, 1.0)


def gradcam_for_sample(model, x, target_class=None, meta=None):
    model.eval()
    target_layer = find_last_conv_module(model)
    hook = _FeatureHook(target_layer)

    logits = forward_model(model, x, meta)
    if target_class is None:
        target_class = int(torch.argmax(logits, dim=1).item())

    score = logits[:, target_class].sum()
    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    activations = hook.activations
    gradients = hook.gradients
    hook.close()

    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    grads2 = gradients ** 2
    grads3 = gradients ** 3
    denom = 2 * grads2 + (activations * grads3).sum(dim=(2, 3), keepdim=True)
    denom = torch.where(denom != 0, denom, torch.ones_like(denom))
    alpha = grads2 / (denom + 1e-7)
    positive_grad = F.relu(gradients)
    weights = (alpha * positive_grad).sum(dim=(2, 3), keepdim=True)

    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()

    if np.max(cam) > np.min(cam):
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
    else:
        cam = np.zeros_like(cam)

    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred = int(np.argmax(probs))
    return cam, pred, probs


def overlay_heatmap_on_image(img_tensor, cam):
    img = denormalize_image(img_tensor[0]).detach().cpu().permute(1, 2, 0).numpy()
    plt_img = np.clip(img, 0, 1)
    return plt_img, cam


def pick_gradcam_samples(test_csv, threshold, num_patients=2):
    df = pd.read_csv(test_csv).copy()
    df["TARGET"] = df["KL"].apply(lambda x: map_kl_to_binary(x, threshold))

    selected_rows = []
    used_ids = set()

    for target in [0, 1]:
        subset = df[df["TARGET"] == target].copy()
        for _, row in subset.iterrows():
            if row["id"] not in used_ids:
                selected_rows.append(row)
                used_ids.add(row["id"])
                break

    if len(selected_rows) < num_patients:
        for _, row in df.iterrows():
            if row["id"] not in used_ids:
                selected_rows.append(row)
                used_ids.add(row["id"])
            if len(selected_rows) >= num_patients:
                break

    return pd.DataFrame(selected_rows[:num_patients])


def save_gradcam_grid(sample_df, model_infos, args, out_path):
    if len(sample_df) == 0 or len(model_infos) == 0:
        return

    _, eval_tfms = get_transforms(args.img_size)
    n_rows = len(sample_df)
    n_cols = 1 + len(model_infos)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    target_names = get_binary_target_names(args.binary_threshold)

    for r, (_, row) in enumerate(sample_df.iterrows()):
        img_path = row["image_path"]
        image = Image.open(img_path).convert("RGB")
        x = eval_tfms(image).unsqueeze(0).to(args.device)

        y_true = map_kl_to_binary(row["KL"], args.binary_threshold)
        patient_title = f"id={row['id']} side={row.get('side', '-') } true={target_names[y_true]}"

        axes[r, 0].imshow(image)
        axes[r, 0].set_title(f"Original\n{patient_title}")
        axes[r, 0].axis("off")

        for c, info in enumerate(model_infos, start=1):
            model = info["model"]
            use_metadata = info["use_metadata"]
            meta = None

            if use_metadata:
                age = float(row[args.age_col]) if args.age_col in row and not pd.isna(row[args.age_col]) else 0.0
                sex_val = row[args.sex_col] if args.sex_col in row else 0.0
                age = (age - info["age_mean"]) / info["age_std"]
                sex = 1.0 if float(sex_val) == 1.0 else 0.0
                meta = torch.tensor([[age, sex]], dtype=torch.float32, device=args.device)

            cam, pred, probs = gradcam_for_sample(model, x, meta=meta)
            base_img, cam_map = overlay_heatmap_on_image(x, cam)

            axes[r, c].imshow(base_img)
            axes[r, c].imshow(cam_map, cmap="jet", alpha=0.4)
            subtitle = f"{info['name']}\npred={target_names[pred]} p1={probs[1]:.3f}"
            axes[r, c].set_title(subtitle)
            axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()





@torch.no_grad()
def predict_dataframe(model, df, args, use_metadata=False, age_mean=None, age_std=None):
    _, eval_tfms = get_transforms(args.img_size)
    rows = []

    model.eval()
    for _, row in df.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            continue

        x = eval_tfms(img).unsqueeze(0).to(args.device)
        meta = None
        if use_metadata:
            age = float(row[args.age_col]) if not pd.isna(row[args.age_col]) else 0.0
            age = (age - age_mean) / age_std
            sex_raw = row[args.sex_col]
            sex = 1.0 if float(sex_raw) == 1.0 else 0.0
            meta = torch.tensor([[age, sex]], dtype=torch.float32, device=args.device)

        logits = forward_model(model, x, meta)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred = int(np.argmax(probs))
        y_true = map_kl_to_binary(row["KL"], args.binary_threshold)

        row_dict = row.to_dict()
        row_dict["y_true"] = y_true
        row_dict["y_pred"] = pred
        row_dict["prob_positive"] = float(probs[1])
        row_dict["confidence"] = float(np.max(probs))
        rows.append(row_dict)

    return pd.DataFrame(rows)


def compute_metrics_from_df(df_pred: pd.DataFrame):
    y_true = df_pred["y_true"].tolist()
    y_pred = df_pred["y_pred"].tolist()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }




def run_one_model(backbone, args, use_metadata=False, out_root_override=None, resume=True):
    print("\n" + "=" * 88, flush=True)
    tag = f"{backbone} + age/sex" if use_metadata else backbone
    print(f"[INFO] STARTING MODEL: {tag}", flush=True)
    print("=" * 88, flush=True)

    lr, batch_size = get_model_hparams(backbone, user_lr=args.lr, user_batch_size=args.batch_size)

    print(f"[INFO] Hyperparameters for {tag}: lr={lr}, batch_size={batch_size}", flush=True)
    print(f"[INFO] Shared params: epochs={args.epochs}, weight_decay={args.weight_decay}, "
          f"label_smoothing={args.label_smoothing}, patience={args.early_stopping_patience}", flush=True)

    base_root = out_root_override if out_root_override is not None else args.out_root
    model_dir_name = f"{backbone}_with_age_sex" if use_metadata else backbone
    model_dir = os.path.join(base_root, model_dir_name)
    ensure_dir(model_dir)

    checkpoint_path = os.path.join(model_dir, "last_checkpoint.pt")
    best_model_path = os.path.join(model_dir, "best_model.pt")
    history_path = os.path.join(model_dir, "history.csv")
    summary_path = os.path.join(model_dir, "test_summary.txt")

    train_tfms, eval_tfms = get_transforms(args.img_size)

    age_mean, age_std = (None, None)
    if use_metadata:
        age_mean, age_std = compute_age_stats(args.train_csv, args.age_col)
        print(f"[INFO] Age normalization: mean={age_mean:.4f}, std={age_std:.4f}", flush=True)

    train_ds = KneeBinaryDataset(
        args.train_csv,
        transform=train_tfms,
        threshold=args.binary_threshold,
        use_metadata=use_metadata,
        age_col=args.age_col,
        sex_col=args.sex_col,
        age_mean=age_mean,
        age_std=age_std,
    )
    val_ds = KneeBinaryDataset(
        args.val_csv,
        transform=eval_tfms,
        threshold=args.binary_threshold,
        use_metadata=use_metadata,
        age_col=args.age_col,
        sex_col=args.sex_col,
        age_mean=age_mean,
        age_std=age_std,
    )
    test_ds = KneeBinaryDataset(
        args.test_csv,
        transform=eval_tfms,
        threshold=args.binary_threshold,
        use_metadata=use_metadata,
        age_col=args.age_col,
        sex_col=args.sex_col,
        age_mean=age_mean,
        age_std=age_std,
    )

    print(f"[INFO] train/val/test sizes = {len(train_ds)} / {len(val_ds)} / {len(test_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    if use_metadata:
        model = ImageBackboneWithMeta(backbone, pretrained=args.pretrained).to(args.device)
    else:
        model = build_model(backbone, pretrained=args.pretrained).to(args.device)

    weights = class_weights_from_train(
        args.train_csv,
        threshold=args.binary_threshold,
    ).to(args.device)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_reduce_factor, patience=args.lr_scheduler_patience
    )

    start_epoch = 1
    best_val_f1 = -1.0
    history = []
    patience_counter = 0

    if resume and os.path.exists(checkpoint_path):
        print(f"[INFO] Resuming from checkpoint: {checkpoint_path}", flush=True)
        model, optimizer, scheduler, start_epoch, best_val_f1, history, patience_counter = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler, args.device
        )
    else:
        print("[INFO] No checkpoint found. Starting fresh.", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n[INFO] {tag} | Epoch {epoch}/{args.epochs}", flush=True)

        current_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion, args.device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, args.device, phase="val")
        scheduler.step(val_metrics["f1_macro"])

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_f1_binary": tr_f1,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1_binary": val_metrics["f1_binary"],
            "val_precision_binary": val_metrics["precision_binary"],
            "val_recall_binary": val_metrics["recall_binary"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_auc": val_metrics["auc"],
        }
        history.append(row)

        print(
            f"[{tag}] Epoch {epoch:02d}/{args.epochs} | "
            f"lr={current_lr:.8f} -> {optimizer.param_groups[0]['lr']:.8f} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} train_f1_binary={tr_f1:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_f1_binary={val_metrics['f1_binary']:.4f} val_auc={val_metrics['auc']:.4f}",
            flush=True
        )

        improved = val_metrics["f1_macro"] > best_val_f1
        if improved:
            best_val_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] New best model saved: {best_model_path}", flush=True)
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Early-stop counter: {patience_counter}/{args.early_stopping_patience}", flush=True)

        save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_val_f1, history, patience_counter)
        save_csv_history(history, history_path)

        if patience_counter >= args.early_stopping_patience:
            print(f"[INFO] Early stopping triggered for {tag}.", flush=True)
            break

    print(f"[INFO] Loading best model: {best_model_path}", flush=True)
    model.load_state_dict(torch.load(best_model_path, map_location=args.device))

    test_metrics = evaluate(model, test_loader, criterion, args.device, phase="test")
    target_names = get_binary_target_names(args.binary_threshold)

    save_test_outputs(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        test_metrics["prob_pos"],
        test_metrics["confidence"],
        model_dir,
        target_names,
    )

    rc = compute_risk_coverage(test_metrics["y_true"], test_metrics["y_pred"], test_metrics["confidence"])
    coverage95 = coverage_at_target_accuracy(rc, target_acc=0.95)
    save_risk_coverage_single(tag, rc, os.path.join(model_dir, "risk_coverage_curve.png"))

    labels = [0, 1]
    full_report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=labels)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"backbone: {backbone}\n")
        f.write(f"use_metadata: {use_metadata}\n")
        f.write(f"binary_threshold: {args.binary_threshold}\n")
        f.write(f"test_loss: {test_metrics['loss']:.6f}\n")
        f.write(f"test_acc: {test_metrics['acc']:.6f}\n")
        f.write(f"test_binary_f1: {test_metrics['f1_binary']:.6f}\n")
        f.write(f"test_binary_precision: {test_metrics['precision_binary']:.6f}\n")
        f.write(f"test_binary_recall: {test_metrics['recall_binary']:.6f}\n")
        f.write(f"test_macro_f1: {test_metrics['f1_macro']:.6f}\n")
        f.write(f"test_macro_precision: {test_metrics['precision_macro']:.6f}\n")
        f.write(f"test_macro_recall: {test_metrics['recall_macro']:.6f}\n")
        f.write(f"test_auc: {test_metrics['auc']:.6f}\n")
        f.write(f"aurc: {rc['aurc']:.6f}\n")
        f.write(f"coverage_at_95acc: {coverage95:.6f}\n")
        f.write(f"best_val_f1_macro: {best_val_f1:.6f}\n")
        f.write(f"final_lr: {optimizer.param_groups[0]['lr']:.10f}\n")
        f.write(f"label_smoothing: {args.label_smoothing}\n")
        f.write(f"weight_decay: {args.weight_decay}\n\n")
        f.write("Classification Report\n\n")
        f.write(full_report)
        f.write("\nConfusion Matrix\n")
        f.write(str(cm))

    print(f"[INFO] Saved summary to: {summary_path}", flush=True)
    print(f"[INFO] FINISHED MODEL: {tag}", flush=True)

    return {
        "backbone": backbone,
        "variant": "with_age_sex" if use_metadata else "image_only",
        "best_val_f1_macro": best_val_f1,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_binary_f1": test_metrics["f1_binary"],
        "test_binary_precision": test_metrics["precision_binary"],
        "test_binary_recall": test_metrics["recall_binary"],
        "test_macro_f1": test_metrics["f1_macro"],
        "test_macro_precision": test_metrics["precision_macro"],
        "test_macro_recall": test_metrics["recall_macro"],
        "test_auc": test_metrics["auc"],
        "aurc": rc["aurc"],
        "coverage_at_95acc": coverage95,
        "final_lr": optimizer.param_groups[0]["lr"],
        "used_batch_size": batch_size,
        "used_lr": lr,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
        "use_metadata": use_metadata,
        "best_model_path": best_model_path,
        "age_mean": age_mean,
        "age_std": age_std,
        "risk_coverage_curve_path": os.path.join(model_dir, "risk_coverage_curve.png"),
    }, model, rc


def save_best_model_comparison_plot(image_row: dict, meta_row: dict, out_path: str):
    metrics = ["test_acc", "test_binary_f1", "test_binary_precision", "test_binary_recall", "test_auc"]
    labels = ["Accuracy", "F1", "Precision", "Recall", "AUC"]
    image_vals = [float(image_row[m]) for m in metrics]
    meta_vals = [float(meta_row[m]) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, image_vals, width, label="image_only")
    plt.bar(x + width / 2, meta_vals, width, label="with_age_sex")
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Best Model: Image-only vs With Age/Sex")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Binary KL classification only")
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--val_csv", type=str, default="val.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")
    parser.add_argument("--out_root", type=str, default="outputs_binary_only")
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--early_stopping_patience", type=int, default=8)
    parser.add_argument("--lr_scheduler_patience", type=int, default=3)
    parser.add_argument("--lr_reduce_factor", type=float, default=0.5)
    parser.add_argument("--binary_threshold", type=int, default=2)
    parser.add_argument("--run_best_with_metadata", action="store_true", default=True)
    parser.add_argument("--age_col", type=str, default="AGE")
    parser.add_argument("--sex_col", type=str, default="SEX_MALE")
    parser.add_argument("--make_gradcam", action="store_true", default=True)
    parser.add_argument("--gradcam_num_patients", type=int, default=2)

    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18", "efficientnet_b0", "resnet34", "convnext_tiny"],
        choices=["resnet18", "efficientnet_b0", "resnet34", "convnext_tiny"],
    )

    return parser.parse_args()

def main():
    print("[INFO] Binary-only script started.", flush=True)
    args = parse_args()
    print(f"[INFO] Args: {args}", flush=True)

    set_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_root)

    target_names = get_binary_target_names(args.binary_threshold)
    print(f"[INFO] Device: {args.device}", flush=True)
    print(f"[INFO] Binary target: {target_names[0]} vs {target_names[1]}", flush=True)
    print(f"[INFO] Models to run: {args.models}", flush=True)

    all_results = []
    risk_curves = {}
    loaded_models_for_cam = []

    for backbone in args.models:
        try:
            result, trained_model, rc = run_one_model(backbone, args, use_metadata=False, resume=True)
            all_results.append(result)
            risk_curves[backbone] = rc
            loaded_models_for_cam.append({"name": backbone, "model": copy.deepcopy(trained_model).to(args.device), "use_metadata": False})
        except Exception as e:
            print(f"[ERROR] Model {backbone} failed: {e}", flush=True)
            all_results.append({
                "backbone": backbone,
                "variant": "image_only",
                "test_acc": None,
                "test_binary_f1": None,
                "test_binary_precision": None,
                "test_binary_recall": None,
                "test_macro_f1": None,
                "test_auc": None,
                "aurc": None,
                "coverage_at_95acc": None,
                "error": str(e),
            })

    summary_df = pd.DataFrame(all_results)
    summary_csv = os.path.join(args.out_root, "summary_all_models.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[INFO] Summary saved to: {summary_csv}", flush=True)

    valid_df = summary_df[(summary_df["variant"] == "image_only") & (summary_df["test_macro_f1"].notna())].copy()
    best_backbone = None
    best_result_row = None

    if len(valid_df) > 0:
        ranked = valid_df.sort_values("test_macro_f1", ascending=False, na_position="last")
        print("[INFO] Final ranking by test_macro_f1:", flush=True)
        print(ranked[["backbone", "test_acc", "test_binary_f1", "test_macro_f1", "test_auc", "aurc"]], flush=True)
        best_backbone = ranked.iloc[0]["backbone"]
        best_result_row = ranked.iloc[0].to_dict()
        print(f"[INFO] Best image-only binary model: {best_backbone}", flush=True)

    if len(risk_curves) > 0:
        save_risk_coverage_all(risk_curves, os.path.join(args.out_root, "risk_coverage_all_models.png"))
        rc_rows = []
        for name, rc in risk_curves.items():
            rc_rows.append({
                "backbone": name,
                "aurc": rc["aurc"],
                "coverage_at_95acc": coverage_at_target_accuracy(rc, 0.95),
            })
        pd.DataFrame(rc_rows).to_csv(os.path.join(args.out_root, "risk_coverage_summary.csv"), index=False)

    metadata_result = None
    metadata_model_info = None

    if args.run_best_with_metadata and best_backbone is not None:
        print("\n" + "#" * 88, flush=True)
        print(f"[INFO] Re-running BEST binary model with age + sex: {best_backbone}", flush=True)
        print("#" * 88, flush=True)

        metadata_out_root = os.path.join(args.out_root, "best_model_with_metadata")
        ensure_dir(metadata_out_root)

        metadata_result, meta_model, meta_rc = run_one_model(
            best_backbone, args, use_metadata=True, out_root_override=metadata_out_root, resume=False
        )
        all_results.append(metadata_result)

        comparison_df = pd.DataFrame(all_results)
        comparison_csv = os.path.join(args.out_root, "summary_all_models_with_best_metadata.csv")
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"[INFO] Extended summary saved to: {comparison_csv}", flush=True)

        image_row = comparison_df[(comparison_df["backbone"] == best_backbone) & (comparison_df["variant"] == "image_only")].iloc[0].to_dict()
        meta_row = comparison_df[(comparison_df["backbone"] == best_backbone) & (comparison_df["variant"] == "with_age_sex")].iloc[0].to_dict()
        save_best_model_comparison_plot(image_row, meta_row, os.path.join(args.out_root, "best_model_image_vs_metadata.png"))

        metadata_model_info = {
            "name": f"{best_backbone}+age/sex",
            "model": copy.deepcopy(meta_model).to(args.device),
            "use_metadata": True,
            "age_mean": metadata_result["age_mean"],
            "age_std": metadata_result["age_std"],
        }

        base_info = next(x for x in loaded_models_for_cam if x["name"] == best_backbone)


    if args.make_gradcam and len(loaded_models_for_cam) > 0:
        try:
            sample_df = pick_gradcam_samples(
                args.test_csv,
                threshold=args.binary_threshold,
                num_patients=args.gradcam_num_patients
            )
            model_infos = loaded_models_for_cam.copy()
            if metadata_model_info is not None:
                model_infos.append(metadata_model_info)
            save_gradcam_grid(sample_df, model_infos, args, os.path.join(args.out_root, "gradcam_binary_models.png"))
            sample_df.to_csv(os.path.join(args.out_root, "gradcam_selected_samples.csv"), index=False)
            print("[INFO] Grad-CAM grid saved.", flush=True)
        except Exception as e:
            print(f"[WARN] Grad-CAM generation failed: {e}", flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
