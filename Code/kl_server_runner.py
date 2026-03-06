import os
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report


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



class KneeKLDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        print(f"[INFO] Loading dataset: {csv_file}", flush=True)
        self.df = pd.read_csv(csv_file).copy()
        self.transform = transform

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

        return img, y



# Transforms
def get_transforms(img_size=224):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(7),
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


# Train / Eval
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
    test_ds = KneeKLDataset(args.test_csv, transform=eval_tfms)

    print(f"[INFO] train/val/test sizes = {len(train_ds)} / {len(val_ds)} / {len(test_ds)}", flush=True)

    first_img_path = train_ds.df.iloc[0]["image_path"]
    print(f"[INFO] first image path: {first_img_path}", flush=True)
    with Image.open(first_img_path) as im:
        print(f"[INFO] first image ok: size={im.size}, mode={im.mode}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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
        model, test_loader, criterion, args.device, phase="test"
    )
    save_test_outputs(y_true, y_pred, model_dir)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"backbone: {backbone}\n")
        f.write(f"test_loss: {test_loss:.6f}\n")
        f.write(f"test_acc: {test_acc:.6f}\n")
        f.write(f"test_macro_f1: {test_f1:.6f}\n")
        f.write(f"test_macro_precision: {test_prec_macro:.6f}\n")
        f.write(f"test_macro_recall: {test_rec_macro:.6f}\n")
        f.write(f"test_weighted_precision: {test_prec_weighted:.6f}\n")
        f.write(f"test_weighted_recall: {test_rec_weighted:.6f}\n")
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
        "final_lr": optimizer.param_groups[0]["lr"],
        "used_batch_size": batch_size,
        "used_lr": lr,
        "label_smoothing": args.label_smoothing,
        "weight_decay": args.weight_decay,
    }



def main():
    print("[INFO] Script started.", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--val_csv", type=str, default="val.csv")
    parser.add_argument("--test_csv", type=str, default="test.csv")
    parser.add_argument("--out_root", type=str, default="outputs")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--lr_scheduler_patience", type=int, default=2)
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
            print(ranked[["backbone", "test_acc", "test_macro_f1", "best_val_f1"]], flush=True)
        except Exception:
            pass


if __name__ == "__main__":
    main()