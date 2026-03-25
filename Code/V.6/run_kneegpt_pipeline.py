import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

try:
    import yaml
except Exception:
    raise SystemExit("Please install pyyaml before running: pip install pyyaml")


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def bool_flag(v: bool):
    return [] if not v else ["--use_metadata"]


def slugify_ratio(r: float) -> str:
    return f"{int(round(float(r) * 100)):03d}pct"


def run_cmd(cmd, dry_run=False):
    print("\n[RUN]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x, default=float("-inf")):
    try:
        return float(x)
    except Exception:
        return default


def to_bool_str(v):
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def normalize_bool_str(v):
    return "true" if to_bool_str(v) else "false"


def safe_get(row, key, default=None):
    return row[key] if key in row else default


def compute_fraction_deltas(df, out_path):
    """
    Legacy summary: keep old behavior.
    Groups by model_alias, train_mode, use_metadata only.
    """
    results = []

    if df.empty:
        pd.DataFrame(results).to_csv(out_path, index=False)
        return

    df = df.copy()
    df["train_fraction"] = pd.to_numeric(df["train_fraction"], errors="coerce")
    df["test_auc"] = pd.to_numeric(df["test_auc"], errors="coerce")

    group_cols = ["model_alias", "train_mode", "use_metadata"]

    for keys, group in df.groupby(group_cols):
        group = group.sort_values("train_fraction")

        if 0.2 in group["train_fraction"].values and 1.0 in group["train_fraction"].values:
            row_02 = group[group["train_fraction"] == 0.2].iloc[0]
            row_10 = group[group["train_fraction"] == 1.0].iloc[0]

            delta_auc = row_10["test_auc"] - row_02["test_auc"]

            results.append({
                "model_alias": keys[0],
                "train_mode": keys[1],
                "use_metadata": keys[2],
                "auc_0.2": row_02["test_auc"],
                "auc_1.0": row_10["test_auc"],
                "delta_auc": delta_auc,
            })

    pd.DataFrame(results).to_csv(out_path, index=False)


def compute_fraction_deltas_with_stage1(df, out_path):
    """
    New additive summary: also separates init_from_stage1.
    """
    results = []

    if df.empty or "init_from_stage1" not in df.columns:
        pd.DataFrame(results).to_csv(out_path, index=False)
        return

    df = df.copy()
    df["train_fraction"] = pd.to_numeric(df["train_fraction"], errors="coerce")
    df["test_auc"] = pd.to_numeric(df["test_auc"], errors="coerce")
    df["init_from_stage1"] = df["init_from_stage1"].astype(str).str.lower()
    df["use_metadata"] = df["use_metadata"].astype(str).str.lower()

    group_cols = ["model_alias", "train_mode", "use_metadata", "init_from_stage1"]

    for keys, group in df.groupby(group_cols):
        group = group.sort_values("train_fraction")

        if 0.2 in group["train_fraction"].values and 1.0 in group["train_fraction"].values:
            row_02 = group[group["train_fraction"] == 0.2].iloc[0]
            row_10 = group[group["train_fraction"] == 1.0].iloc[0]

            delta_auc = row_10["test_auc"] - row_02["test_auc"]

            results.append({
                "model_alias": keys[0],
                "train_mode": keys[1],
                "use_metadata": keys[2],
                "init_from_stage1": keys[3],
                "auc_0.2": row_02["test_auc"],
                "auc_1.0": row_10["test_auc"],
                "delta_auc": delta_auc,
            })

    pd.DataFrame(results).to_csv(out_path, index=False)


def compute_best_models(df, metric="best_val_macro_f1"):
    """
    Legacy summary: keep old behavior.
    Best per (model_alias, use_metadata), regardless of Stage1 init.
    """
    best_rows = []

    if df.empty:
        return pd.DataFrame(best_rows)

    df = df.copy()
    if metric not in df.columns:
        raise KeyError(f"Selection metric '{metric}' not found in Stage2 summary columns.")

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    group_cols = ["model_alias", "use_metadata"]

    for keys, group in df.groupby(group_cols):
        group = group.sort_values(metric, ascending=False)
        if group.empty:
            continue
        best = group.iloc[0]

        best_rows.append({
            "model_alias": keys[0],
            "use_metadata": keys[1],
            "best_fraction": safe_get(best, "train_fraction"),
            "best_mode": safe_get(best, "train_mode"),
            "selection_metric": metric,
            "selection_value": safe_get(best, metric),
            "best_auc": safe_get(best, "test_auc"),
            "best_auprc": safe_get(best, "test_auprc"),
            "best_f1": safe_get(best, "test_f1"),
            "best_macro_f1": safe_get(best, "test_macro_f1"),
            "init_from_stage1": safe_get(best, "init_from_stage1"),
            "output_dir": safe_get(best, "output_dir"),
            "best_checkpoint": safe_get(best, "best_checkpoint"),
        })

    return pd.DataFrame(best_rows)


def compute_best_models_with_stage1(df, metric="best_val_macro_f1"):
    """
    New additive summary:
    Best per (model_alias, use_metadata, init_from_stage1).
    """
    best_rows = []

    if df.empty:
        return pd.DataFrame(best_rows)

    df = df.copy()
    if metric not in df.columns:
        raise KeyError(f"Selection metric '{metric}' not found in Stage2 summary columns.")

    if "init_from_stage1" not in df.columns:
        raise KeyError("Column 'init_from_stage1' not found in Stage2 summary columns.")

    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["init_from_stage1"] = df["init_from_stage1"].astype(str).str.lower()
    df["use_metadata"] = df["use_metadata"].astype(str).str.lower()

    group_cols = ["model_alias", "use_metadata", "init_from_stage1"]

    for keys, group in df.groupby(group_cols):
        group = group.sort_values(metric, ascending=False)
        if group.empty:
            continue
        best = group.iloc[0]

        best_rows.append({
            "model_alias": keys[0],
            "use_metadata": keys[1],
            "init_from_stage1": keys[2],
            "best_fraction": safe_get(best, "train_fraction"),
            "best_mode": safe_get(best, "train_mode"),
            "selection_metric": metric,
            "selection_value": safe_get(best, metric),
            "best_auc": safe_get(best, "test_auc"),
            "best_auprc": safe_get(best, "test_auprc"),
            "best_f1": safe_get(best, "test_f1"),
            "best_macro_f1": safe_get(best, "test_macro_f1"),
            "output_dir": safe_get(best, "output_dir"),
            "best_checkpoint": safe_get(best, "best_checkpoint"),
        })

    return pd.DataFrame(best_rows)


def compare_best_models(best_df, out_path):
    """
    Legacy comparison: keep old behavior.
    """
    results = []

    for _, row in best_df.iterrows():
        results.append({
            "model_alias": row.get("model_alias"),
            "use_metadata": row.get("use_metadata"),
            "best_fraction": row.get("best_fraction"),
            "best_mode": row.get("best_mode"),
            "selection_metric": row.get("selection_metric"),
            "selection_value": row.get("selection_value"),
            "best_auc": row.get("best_auc"),
            "best_auprc": row.get("best_auprc"),
            "best_f1": row.get("best_f1"),
            "best_macro_f1": row.get("best_macro_f1"),
            "init_from_stage1": row.get("init_from_stage1"),
        })

    pd.DataFrame(results).to_csv(out_path, index=False)


def compare_best_models_with_stage1(best_df, out_path):
    """
    New additive comparison: explicitly keeps Stage1 separation.
    """
    results = []

    for _, row in best_df.iterrows():
        results.append({
            "model_alias": row.get("model_alias"),
            "use_metadata": row.get("use_metadata"),
            "init_from_stage1": row.get("init_from_stage1"),
            "best_fraction": row.get("best_fraction"),
            "best_mode": row.get("best_mode"),
            "selection_metric": row.get("selection_metric"),
            "selection_value": row.get("selection_value"),
            "best_auc": row.get("best_auc"),
            "best_auprc": row.get("best_auprc"),
            "best_f1": row.get("best_f1"),
            "best_macro_f1": row.get("best_macro_f1"),
        })

    pd.DataFrame(results).to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Run KneeGPT pipeline from YAML")
    ap.add_argument("--config", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    py = args.python
    paths = cfg["paths"]
    scripts = cfg["scripts"]
    backbones = cfg["backbones"]
    md_opts = cfg["metadata"]["use_metadata_options"]
    outputs = cfg["outputs"]

    ensure_dir(outputs["master_dir"])

    stage1_map = {}
    stage2_rows = []

    if cfg["stage1"]["enabled"]:
        for bb in backbones:
            for use_md in md_opts:
                md_tag = "with_metadata" if use_md else "image_only"
                out_dir = os.path.join(cfg["stage1"]["out_root"], bb["alias"], md_tag)
                ensure_dir(out_dir)

                cmd = [
                    py, scripts["stage1"],
                    "--train_csv", paths["train_csv"],
                    "--val_csv", paths["val_csv"],
                    "--test_csv", paths["test_csv"],
                    "--out_dir", out_dir,
                    "--backbone_name", bb["name"],
                    "--model_alias", bb["alias"],
                    "--epochs", str(cfg["stage1"]["epochs"]),
                    "--batch_size", str(cfg["stage1"]["batch_size"]),
                    "--num_workers", str(cfg["stage1"]["num_workers"]),
                    "--head_lr", str(cfg["stage1"]["head_lr"]),
                    "--encoder_lr", str(cfg["stage1"]["encoder_lr"]),
                    "--weight_decay", str(cfg["stage1"]["weight_decay"]),
                    "--dropout", str(cfg["stage1"]["dropout"]),
                    "--label_smoothing", str(cfg["stage1"]["label_smoothing"]),
                    "--early_stopping_patience", str(cfg["stage1"]["early_stopping_patience"]),
                ] + bool_flag(use_md)

                run_cmd(cmd, dry_run=args.dry_run)
                stage1_map[(bb["alias"], use_md)] = os.path.join(
                    out_dir,
                    cfg["stage1"]["save_encoder_only_name"]
                )

    if cfg["stage2"]["enabled"]:
        stage2_init_opts = cfg["stage2"].get(
            "init_from_stage1_options",
            [cfg["stage2"].get("init_from_stage1", False)]
        )

        for bb in backbones:
            for use_md in md_opts:
                md_tag = "with_metadata" if use_md else "image_only"
                stage1_ckpt = stage1_map.get((bb["alias"], use_md), "")

                for mode in cfg["stage2"]["train_modes"]:
                    for frac in cfg["stage2"]["train_fractions"]:
                        for use_stage1 in stage2_init_opts:
                            frac_tag = slugify_ratio(frac)
                            init_tag = "with_stage1" if use_stage1 else "no_stage1"

                            # New path contains init_tag, but old outputs are otherwise preserved
                            out_dir = os.path.join(
                                cfg["stage2"]["out_root"],
                                bb["alias"],
                                mode,
                                frac_tag,
                                md_tag,
                                init_tag,
                            )
                            ensure_dir(out_dir)

                            cmd = [
                                py, scripts["stage2"],
                                "--train_csv", paths["train_csv"],
                                "--val_csv", paths["val_csv"],
                                "--test_csv", paths["test_csv"],
                                "--out_dir", out_dir,
                                "--backbone_name", bb["name"],
                                "--model_alias", bb["alias"],
                                "--train_mode", mode,
                                "--train_fraction", str(frac),
                                "--binary_threshold", str(cfg["stage2"]["binary_threshold"]),
                                "--epochs", str(cfg["stage2"]["epochs"]),
                                "--batch_size", str(cfg["stage2"]["batch_size"]),
                                "--num_workers", str(cfg["stage2"]["num_workers"]),
                                "--head_lr", str(cfg["stage2"]["head_lr"]),
                                "--encoder_lr", str(cfg["stage2"]["encoder_lr"]),
                                "--weight_decay", str(cfg["stage2"]["weight_decay"]),
                                "--dropout", str(cfg["stage2"]["dropout"]),
                                "--label_smoothing", str(cfg["stage2"]["label_smoothing"]),
                                "--early_stopping_patience", str(cfg["stage2"]["early_stopping_patience"]),
                            ] + bool_flag(use_md)

                            if use_stage1:
                                cmd += ["--init_from_stage1", "--stage1_checkpoint", stage1_ckpt]

                            run_cmd(cmd, dry_run=args.dry_run)

                            summary_csv = os.path.join(out_dir, "summary_stage2.csv")
                            if (not args.dry_run) and os.path.exists(summary_csv):
                                rows = read_csv_rows(summary_csv)
                                for r in rows:
                                    if "init_from_stage1" not in r:
                                        r["init_from_stage1"] = str(use_stage1)
                                    if "use_metadata" not in r:
                                        r["use_metadata"] = str(use_md)
                                    if "train_mode" not in r:
                                        r["train_mode"] = mode
                                    if "train_fraction" not in r:
                                        r["train_fraction"] = str(frac)
                                    if "model_alias" not in r:
                                        r["model_alias"] = bb["alias"]
                                    if "backbone_name" not in r:
                                        r["backbone_name"] = bb["name"]
                                    if "output_dir" not in r:
                                        r["output_dir"] = out_dir
                                stage2_rows.extend(rows)

    master_summary_path = os.path.join(outputs["master_dir"], outputs["master_summary_filename"])
    if stage2_rows and outputs.get("save_master_summary", True):
        fieldnames = sorted({k for r in stage2_rows for k in r.keys()})
        with open(master_summary_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(stage2_rows)
        print(f"[SAVED] {master_summary_path}")

        master_df = pd.DataFrame(stage2_rows)

        legacy_fraction_delta_path = os.path.join(outputs["master_dir"], "fraction_delta_auc.csv")
        compute_fraction_deltas(master_df, legacy_fraction_delta_path)
        print(f"[SAVED] {legacy_fraction_delta_path}")

        fraction_delta_stage1_path = os.path.join(outputs["master_dir"], "fraction_delta_auc_with_stage1.csv")
        compute_fraction_deltas_with_stage1(master_df, fraction_delta_stage1_path)
        print(f"[SAVED] {fraction_delta_stage1_path}")

        best_metric = cfg["stage2"].get("selection_metric", "best_val_macro_f1")
        print(f"[INFO] Model selection metric: {best_metric}")

        best_df = compute_best_models(master_df, metric=best_metric)
        best_models_path = os.path.join(outputs["master_dir"], "best_models_summary.csv")
        best_df.to_csv(best_models_path, index=False)
        print(f"[SAVED] {best_models_path}")

        best_models_cmp_path = os.path.join(outputs["master_dir"], "best_models_comparison.csv")
        compare_best_models(best_df, best_models_cmp_path)
        print(f"[SAVED] {best_models_cmp_path}")

        if "init_from_stage1" in master_df.columns:
            best_df_stage1 = compute_best_models_with_stage1(master_df, metric=best_metric)

            best_models_stage1_path = os.path.join(
                outputs["master_dir"],
                "best_models_summary_with_stage1.csv"
            )
            best_df_stage1.to_csv(best_models_stage1_path, index=False)
            print(f"[SAVED] {best_models_stage1_path}")

            best_models_stage1_cmp_path = os.path.join(
                outputs["master_dir"],
                "best_models_comparison_with_stage1.csv"
            )
            compare_best_models_with_stage1(best_df_stage1, best_models_stage1_cmp_path)
            print(f"[SAVED] {best_models_stage1_cmp_path}")

    if cfg["stage3"]["enabled"] and stage2_rows:
        metric = cfg["stage3"]["select_from_stage2_metric"]
        selected = []

        for use_md in md_opts:
            candidates = [
                r for r in stage2_rows
                if normalize_bool_str(r.get("use_metadata", "")) == normalize_bool_str(use_md)
            ]
            if not candidates:
                continue

            best = max(candidates, key=lambda r: to_float(r.get(metric)))
            selected.append(best)

        best_index_path = os.path.join(outputs["master_dir"], outputs["best_models_filename"])
        if outputs.get("save_best_model_index", True) and selected:
            fieldnames = sorted({k for r in selected for k in r.keys()})
            with open(best_index_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(selected)
            print(f"[SAVED] {best_index_path}")

        for row in selected:
            bb_alias = row["model_alias"]
            bb_name = row["backbone_name"]
            use_md = to_bool_str(row["use_metadata"])
            md_tag = "with_metadata" if use_md else "image_only"
            stage2_out = row["output_dir"]
            stage2_ckpt = os.path.join(stage2_out, "best_stage2_model.pt")

            out_dir = os.path.join(cfg["stage3"]["out_root"], bb_alias, md_tag)
            ensure_dir(out_dir)

            cmd = [
                py, scripts["stage3"],
                "--train_csv", paths["train_csv"],
                "--test_csv", paths["test_csv"],
                "--stage2_checkpoint", stage2_ckpt,
                "--out_dir", out_dir,
                "--backbone_name", bb_name,
                "--model_alias", bb_alias,
                "--mc_runs", str(cfg["stage3"]["mc_runs"]),
                "--bootstrap_iters", str(cfg["stage3"]["bootstrap_iters"]),
                "--embedding_method", str(cfg["stage3"]["embedding_method"]),
                "--primary_color", cfg["stage3"]["style"]["primary_hex"],
                "--secondary_color", cfg["stage3"]["style"]["secondary_hex"],
            ] + bool_flag(use_md)

            run_cmd(cmd, dry_run=args.dry_run)

    if cfg["stage3"]["enabled"] and stage2_rows:
        stage2_df = pd.DataFrame(stage2_rows).copy()

        if "init_from_stage1" in stage2_df.columns:
            metric = cfg["stage3"]["select_from_stage2_metric"]
            selected_stage1 = []

            stage2_df["use_metadata"] = stage2_df["use_metadata"].astype(str).str.lower()
            stage2_df["init_from_stage1"] = stage2_df["init_from_stage1"].astype(str).str.lower()
            stage2_df[metric] = pd.to_numeric(stage2_df[metric], errors="coerce")

            group_cols = ["use_metadata", "init_from_stage1"]

            for _, group in stage2_df.groupby(group_cols):
                group = group.sort_values(metric, ascending=False)
                if group.empty:
                    continue
                selected_stage1.append(group.iloc[0].to_dict())

            best_index_stage1_path = os.path.join(
                outputs["master_dir"],
                "best_models_stage3_index_with_stage1.csv"
            )
            if outputs.get("save_best_model_index", True) and selected_stage1:
                fieldnames = sorted({k for r in selected_stage1 for k in r.keys()})
                with open(best_index_stage1_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(selected_stage1)
                print(f"[SAVED] {best_index_stage1_path}")

            for row in selected_stage1:
                bb_alias = row["model_alias"]
                bb_name = row["backbone_name"]
                use_md = to_bool_str(row["use_metadata"])
                use_stage1 = to_bool_str(row["init_from_stage1"])

                md_tag = "with_metadata" if use_md else "image_only"
                init_tag = "with_stage1" if use_stage1 else "no_stage1"

                stage2_out = row["output_dir"]
                stage2_ckpt = os.path.join(stage2_out, "best_stage2_model.pt")

                out_dir = os.path.join(
                    cfg["stage3"]["out_root"],
                    bb_alias,
                    md_tag,
                    init_tag
                )
                ensure_dir(out_dir)

                cmd = [
                    py, scripts["stage3"],
                    "--train_csv", paths["train_csv"],
                    "--test_csv", paths["test_csv"],
                    "--stage2_checkpoint", stage2_ckpt,
                    "--out_dir", out_dir,
                    "--backbone_name", bb_name,
                    "--model_alias", bb_alias,
                    "--mc_runs", str(cfg["stage3"]["mc_runs"]),
                    "--bootstrap_iters", str(cfg["stage3"]["bootstrap_iters"]),
                    "--embedding_method", str(cfg["stage3"]["embedding_method"]),
                    "--primary_color", cfg["stage3"]["style"]["primary_hex"],
                    "--secondary_color", cfg["stage3"]["style"]["secondary_hex"],
                ] + bool_flag(use_md)

                run_cmd(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()