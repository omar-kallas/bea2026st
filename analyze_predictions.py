from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _pick_column(df: pd.DataFrame, candidates: Iterable[str], kind: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find a {kind} column. Tried {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def _safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    if len(a) < 2:
        return float("nan")
    return float(a.corr(b, method=method))


def _fmt(value: float, digits: int = 4) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.{digits}f}"


def _save_bucket_heatmap(bucket_table: pd.DataFrame, out_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save bucket heatmaps.") from exc

    plot_df = bucket_table.copy()
    if "All" in plot_df.index:
        plot_df = plot_df.drop(index="All")
    if "All" in plot_df.columns:
        plot_df = plot_df.drop(columns="All")

    counts = plot_df.values.astype(float)
    row_totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals != 0)

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    im = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels([str(c) for c in plot_df.columns])
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels([str(i) for i in plot_df.index])
    ax.set_xlabel("Predicted bucket")
    ax.set_ylabel("Reference bucket")
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val > 0.55 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-normalized proportion")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _save_language_comparison_heatmaps(
    heatmaps: list[tuple[str, float, pd.DataFrame]],
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save bucket heatmaps.") from exc

    if not heatmaps:
        return

    cleaned: list[tuple[str, float, pd.DataFrame]] = []
    for name, rmse, table in heatmaps:
        t = table.copy()
        if "All" in t.index:
            t = t.drop(index="All")
        if "All" in t.columns:
            t = t.drop(columns="All")
        cleaned.append((name, rmse, t))

    n = len(cleaned)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes_arr = np.array(axes).reshape(-1)
    im = None

    for idx, (name, rmse, table) in enumerate(cleaned):
        ax = axes_arr[idx]
        counts = table.values.astype(float)
        row_totals = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals != 0)

        im = ax.imshow(values, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([str(c) for c in table.columns])
        ax.set_yticks(range(len(table.index)))
        ax.set_yticklabels([str(i) for i in table.index])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Reference")
        ax.set_title(f"{name}\nRMSE: {rmse:.4f}", fontsize=10)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", color=color, fontsize=9)

    for idx in range(len(cleaned), len(axes_arr)):
        axes_arr[idx].axis("off")

    # Reserve right margin for a dedicated colorbar axis to avoid overlap.
    fig.subplots_adjust(right=0.88, top=0.9, wspace=0.35, hspace=0.4)
    if im is not None:
        cax = fig.add_axes([0.9, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Row-normalized proportion")
    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze lexical score predictions against reference data. "
            "Supports single-file mode and directory mode."
        )
    )
    parser.add_argument("--pred-file", type=Path, help="Prediction CSV path (single-file mode)")
    parser.add_argument("--ref-file", type=Path, help="Reference CSV path (single-file mode)")
    parser.add_argument("--pred-dir", type=Path, help="Prediction directory (directory mode)")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Reference data split directory, e.g. data/dev (directory mode)",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Split name used to infer reference file names in directory mode (default: inferred from --data-dir name)",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="Optional path to save directory-mode summary CSV",
    )
    parser.add_argument(
        "--details-dir",
        type=Path,
        default=None,
        help="Optional directory to save per-file text reports in directory mode",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="ID column name used in both files (default: auto-detect from common names)",
    )
    parser.add_argument(
        "--pred-col",
        default=None,
        help="Prediction score column in pred file (default: auto-detect)",
    )
    parser.add_argument(
        "--ref-score-col",
        default=None,
        help="Reference score column in ref file (default: auto-detect)",
    )
    parser.add_argument(
        "--pos-col",
        default=None,
        help="POS column in reference file for per-POS RMSE reporting (default: auto-detect)",
    )
    parser.add_argument(
        "--tail-quantile",
        type=float,
        default=0.25,
        help="Quantile for low/high groups on reference scores (default: 0.25)",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=1.0,
        help="Threshold to count strong under/over predictions (default: 1.0)",
    )
    parser.add_argument(
        "--accurate-thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 1.0],
        help="Absolute error thresholds for accuracy rates (default: 0.25 0.5 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many rows to print for each ranked sample table (default: 10)",
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        default=None,
        help="Optional path to save merged row-level analysis CSV",
    )
    parser.add_argument(
        "--heatmap-out",
        type=Path,
        default=None,
        help="Optional path to save the bucket heatmap image in single-file mode",
    )
    return parser.parse_args()


def _resolve_reference_file(data_dir: Path, l1: str, split: Optional[str]) -> Path:
    if split is None:
        split = data_dir.name

    preferred = data_dir / l1 / f"kvl_shared_task_{l1}_{split}.csv"
    if preferred.exists():
        return preferred

    fallback = sorted((data_dir / l1).glob(f"kvl_shared_task_{l1}_*.csv"))
    if fallback:
        return fallback[0]

    raise FileNotFoundError(
        f"Could not find reference file for L1 '{l1}' under '{data_dir}'. "
        f"Expected '{preferred}'."
    )


def _analyze_pair(
    *,
    pred_file: Path,
    ref_file: Path,
    id_col_arg: Optional[str],
    pred_col_arg: Optional[str],
    ref_score_col_arg: Optional[str],
    pos_col_arg: Optional[str],
    tail_quantile: float,
    drop_threshold: float,
    accurate_thresholds: list[float],
    top_k: int,
    details_out: Optional[Path] = None,
    heatmap_out: Optional[Path] = None,
    print_report: bool = True,
) -> dict:
    pred_df = pd.read_csv(pred_file)
    ref_df = pd.read_csv(ref_file)

    id_col = id_col_arg or _pick_column(pred_df, ["item_id", "id", "sample_id"], "ID")
    if id_col not in ref_df.columns:
        fallback_id = _pick_column(ref_df, ["item_id", "id", "sample_id"], "ID")
        if fallback_id != id_col:
            raise ValueError(
                f"ID column mismatch: pred has '{id_col}', ref has '{fallback_id}'. "
                "Pass --id-col explicitly if needed."
            )

    pred_col = pred_col_arg or _pick_column(
        pred_df, ["prediction", "pred", "score", "pred_score"], "prediction"
    )
    ref_score_col = ref_score_col_arg or _pick_column(
        ref_df,
        ["GLMM_score", "score", "label", "target", "y", "gold_score"],
        "reference score",
    )

    pred_trim = pred_df[[id_col, pred_col]].copy()
    ref_trim = ref_df[[id_col, ref_score_col]].copy()

    pred_trim = pred_trim.rename(columns={pred_col: "pred_score"})
    ref_trim = ref_trim.rename(columns={ref_score_col: "ref_score"})

    if pred_trim[id_col].duplicated().any():
        dup_count = int(pred_trim[id_col].duplicated().sum())
        print(f"Warning: found {dup_count} duplicate IDs in predictions. Keeping first occurrence.")
        pred_trim = pred_trim.drop_duplicates(subset=[id_col], keep="first")

    merged = ref_trim.merge(pred_trim, on=id_col, how="left")

    n_ref = len(ref_trim)
    n_pred = len(pred_trim)
    n_missing = int(merged["pred_score"].isna().sum())
    n_matched = n_ref - n_missing
    extra_preds = int((~pred_trim[id_col].isin(ref_trim[id_col])).sum())

    matched = merged.dropna(subset=["pred_score"]).copy()
    if matched.empty:
        raise ValueError("No overlapping IDs between reference and prediction files.")

    matched["error"] = matched["pred_score"] - matched["ref_score"]
    matched["abs_error"] = matched["error"].abs()
    matched["squared_error"] = matched["error"] ** 2

    rmse = float(np.sqrt(matched["squared_error"].mean()))
    mae = float(matched["abs_error"].mean())
    bias = float(matched["error"].mean())
    pearson = _safe_corr(matched["pred_score"], matched["ref_score"], method="pearson")
    spearman = _safe_corr(matched["pred_score"], matched["ref_score"], method="spearman")

    q = tail_quantile
    low_cut = float(matched["ref_score"].quantile(q))
    high_cut = float(matched["ref_score"].quantile(1 - q))

    low_group = matched[matched["ref_score"] <= low_cut].copy()
    high_group = matched[matched["ref_score"] >= high_cut].copy()

    if print_report:
        print("=== File Summary ===")
        print(f"prediction file: {pred_file}")
        print(f"reference file : {ref_file}")
        print(f"id column      : {id_col}")
        print(f"pred column    : {pred_col}")
        print(f"ref score col  : {ref_score_col}")
        print()

        print("=== Merge Coverage ===")
        print(f"reference rows           : {n_ref}")
        print(f"prediction rows          : {n_pred}")
        print(f"matched rows             : {n_matched}")
        print(f"missing predictions      : {n_missing}")
        print(f"extra predictions        : {extra_preds}")
        print(f"coverage (matched / ref) : {_fmt(n_matched / n_ref)}")
        print()

        print("=== Global Metrics (matched rows) ===")
        print(f"RMSE     : {_fmt(rmse)}")
        print(f"MAE      : {_fmt(mae)}")
        print(f"Bias     : {_fmt(bias)}  (positive => overprediction)")
        print(f"Pearson  : {_fmt(pearson)}")
        print(f"Spearman : {_fmt(spearman)}")
        print()

        print("=== Accuracy Rates by |error| ===")
        for thr in sorted(set(accurate_thresholds)):
            rate = float((matched["abs_error"] <= thr).mean())
            print(
                f"|error| <= {thr:<5g}: {_fmt(rate)} "
                f"({int((matched['abs_error'] <= thr).sum())}/{len(matched)})"
            )
        print()

        print("=== Tail Behavior (based on reference score quantiles) ===")
        print(f"low cutoff  ({q:.2f}) : {_fmt(low_cut)}")
        print(f"high cutoff ({1-q:.2f}) : {_fmt(high_cut)}")
        print()

        if not high_group.empty:
            pct_lower = float((high_group["pred_score"] < high_group["ref_score"]).mean())
            pct_big_drop = float((high_group["error"] <= -drop_threshold).mean())
            print("High reference-score rows:")
            print(f"count                            : {len(high_group)}")
            print(f"mean ref score                   : {_fmt(float(high_group['ref_score'].mean()))}")
            print(f"mean predicted score             : {_fmt(float(high_group['pred_score'].mean()))}")
            print(f"mean error (pred - ref)          : {_fmt(float(high_group['error'].mean()))}")
            print(f"fraction predicted lower         : {_fmt(pct_lower)}")
            print(f"fraction with error <= -{drop_threshold:g} : {_fmt(pct_big_drop)}")
            print()

        if not low_group.empty:
            pct_higher = float((low_group["pred_score"] > low_group["ref_score"]).mean())
            pct_big_rise = float((low_group["error"] >= drop_threshold).mean())
            print("Low reference-score rows:")
            print(f"count                            : {len(low_group)}")
            print(f"mean ref score                   : {_fmt(float(low_group['ref_score'].mean()))}")
            print(f"mean predicted score             : {_fmt(float(low_group['pred_score'].mean()))}")
            print(f"mean error (pred - ref)          : {_fmt(float(low_group['error'].mean()))}")
            print(f"fraction predicted higher        : {_fmt(pct_higher)}")
            print(f"fraction with error >= +{drop_threshold:g} : {_fmt(pct_big_rise)}")
            print()

    matched["ref_bucket"] = pd.cut(
        matched["ref_score"],
        bins=[-np.inf, low_cut, high_cut, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )
    matched["pred_bucket"] = pd.cut(
        matched["pred_score"],
        bins=[-np.inf, low_cut, high_cut, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    )

    bucket_table = pd.crosstab(matched["ref_bucket"], matched["pred_bucket"], margins=True)
    if heatmap_out is not None:
        try:
            _save_bucket_heatmap(
                bucket_table,
                heatmap_out,
                title=f"Bucket Heatmap: {pred_file.stem}",
            )
        except Exception as exc:
            print(f"Warning: could not save heatmap to {heatmap_out}: {exc}")
    if print_report:
        print("=== Reference-vs-Predicted Buckets (using reference cutoffs) ===")
        print(bucket_table.to_string())
        print()

    k = max(top_k, 1)
    base_cols = [id_col, "ref_score", "pred_score", "error", "abs_error"]
    extra_cols = [c for c in ref_df.columns if c not in {id_col, ref_score_col}]

    merged_for_view = merged.merge(
        ref_df[[id_col] + extra_cols], on=id_col, how="left", suffixes=("", "")
    )
    merged_for_view = merged_for_view.dropna(subset=["pred_score"]).copy()
    merged_for_view["error"] = merged_for_view["pred_score"] - merged_for_view["ref_score"]
    merged_for_view["abs_error"] = merged_for_view["error"].abs()
    show_cols = base_cols + [c for c in extra_cols[:3] if c in merged_for_view.columns]

    pos_col = None
    if pos_col_arg is not None:
        if pos_col_arg in merged_for_view.columns:
            pos_col = pos_col_arg
        elif print_report:
            print(
                f"Warning: requested --pos-col '{pos_col_arg}' not found in reference columns. "
                "Skipping RMSE-per-POS section."
            )
    else:
        for cand in ["en_target_pos", "target_pos", "pos", "POS"]:
            if cand in merged_for_view.columns:
                pos_col = cand
                break

    if print_report:
        if pos_col is not None:
            pos_metrics = (
                merged_for_view.groupby(pos_col, dropna=False)
                .agg(
                    count=("error", "size"),
                    rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
                    mae=("abs_error", "mean"),
                    bias=("error", "mean"),
                )
                .reset_index()
                .sort_values("rmse", ascending=False)
            )
            pos_metrics["rmse"] = pos_metrics["rmse"].map(_fmt)
            pos_metrics["mae"] = pos_metrics["mae"].map(_fmt)
            pos_metrics["bias"] = pos_metrics["bias"].map(_fmt)
            print("=== RMSE by POS ===")
            print(pos_metrics.to_string(index=False))
            print()

        print(f"=== Most Accurate Samples (top {k}) ===")
        print(merged_for_view.nsmallest(k, "abs_error")[show_cols].to_string(index=False))
        print()

        print(f"=== Most Over-Predicted Samples (top {k}) ===")
        print(merged_for_view.nlargest(k, "error")[show_cols].to_string(index=False))
        print()

        print(f"=== Most Under-Predicted Samples (top {k}) ===")
        print(merged_for_view.nsmallest(k, "error")[show_cols].to_string(index=False))
        print()

    if details_out is not None:
        out_df = merged_for_view.sort_values("abs_error", ascending=False)
        details_out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(details_out, index=False)
        if print_report:
            print(f"Saved row-level analysis to: {details_out}")

    return {
        "prediction_file": str(pred_file),
        "reference_file": str(ref_file),
        "l1": pred_file.parent.name,
        "matched_rows": n_matched,
        "reference_rows": n_ref,
        "prediction_rows": n_pred,
        "missing_predictions": n_missing,
        "extra_predictions": extra_preds,
        "coverage": (n_matched / n_ref),
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "pearson": pearson,
        "spearman": spearman,
        "bucket_table": bucket_table,
    }


def main() -> None:
    args = parse_args()

    if not (0 < args.tail_quantile < 0.5):
        raise ValueError("--tail-quantile must be between 0 and 0.5")

    single_mode = args.pred_file is not None or args.ref_file is not None
    dir_mode = args.pred_dir is not None or args.data_dir is not None
    if single_mode and dir_mode:
        raise ValueError("Use either single-file mode (--pred-file/--ref-file) or directory mode (--pred-dir/--data-dir), not both.")
    if not single_mode and not dir_mode:
        raise ValueError("Provide either --pred-file and --ref-file, or --pred-dir and --data-dir.")

    if single_mode:
        if args.pred_file is None or args.ref_file is None:
            raise ValueError("Single-file mode requires both --pred-file and --ref-file.")
        _analyze_pair(
            pred_file=args.pred_file,
            ref_file=args.ref_file,
            id_col_arg=args.id_col,
            pred_col_arg=args.pred_col,
            ref_score_col_arg=args.ref_score_col,
            pos_col_arg=args.pos_col,
            tail_quantile=args.tail_quantile,
            drop_threshold=args.drop_threshold,
            accurate_thresholds=args.accurate_thresholds,
            top_k=args.top_k,
            details_out=args.details_out,
            heatmap_out=args.heatmap_out,
            print_report=True,
        )
        return

    if args.pred_dir is None or args.data_dir is None:
        raise ValueError("Directory mode requires both --pred-dir and --data-dir.")

    pred_files = sorted(args.pred_dir.rglob("*.csv"))
    if not pred_files:
        raise ValueError(f"No CSV files found under: {args.pred_dir}")

    records: list[dict] = []
    lang_heatmaps: dict[str, list[tuple[str, float, pd.DataFrame]]] = {}
    failed: list[tuple[Path, str]] = []

    for pred_file in pred_files:
        l1 = pred_file.parent.name
        try:
            ref_file = _resolve_reference_file(args.data_dir, l1=l1, split=args.split)
            if args.details_dir is not None:
                args.details_dir.mkdir(parents=True, exist_ok=True)
                details_txt = args.details_dir / f"{pred_file.stem}_analysis.txt"
                heatmap_out = args.details_dir / f"{pred_file.stem}_bucket_heatmap.png"
                buf = StringIO()
                with redirect_stdout(buf):
                    rec = _analyze_pair(
                        pred_file=pred_file,
                        ref_file=ref_file,
                        id_col_arg=args.id_col,
                        pred_col_arg=args.pred_col,
                        ref_score_col_arg=args.ref_score_col,
                        pos_col_arg=args.pos_col,
                        tail_quantile=args.tail_quantile,
                        drop_threshold=args.drop_threshold,
                        accurate_thresholds=args.accurate_thresholds,
                        top_k=args.top_k,
                        details_out=None,
                        heatmap_out=heatmap_out,
                        print_report=True,
                    )
                details_txt.write_text(buf.getvalue(), encoding="utf-8")
            else:
                rec = _analyze_pair(
                    pred_file=pred_file,
                    ref_file=ref_file,
                    id_col_arg=args.id_col,
                    pred_col_arg=args.pred_col,
                    ref_score_col_arg=args.ref_score_col,
                    pos_col_arg=args.pos_col,
                    tail_quantile=args.tail_quantile,
                    drop_threshold=args.drop_threshold,
                    accurate_thresholds=args.accurate_thresholds,
                    top_k=args.top_k,
                    details_out=None,
                    heatmap_out=None,
                    print_report=False,
                )
            records.append(rec)
            lang_heatmaps.setdefault(rec["l1"], []).append(
                (pred_file.stem, float(rec["rmse"]), rec["bucket_table"])
            )
            print(
                f"[OK] {pred_file} | rmse={_fmt(rec['rmse'])} "
                f"pearson={_fmt(rec['pearson'])} coverage={_fmt(rec['coverage'])}"
            )
        except Exception as exc:
            failed.append((pred_file, str(exc)))
            print(f"[FAIL] {pred_file} | {exc}")

    if not records:
        raise RuntimeError("No prediction files were successfully analyzed in directory mode.")

    summary_records = [{k: v for k, v in rec.items() if k != "bucket_table"} for rec in records]
    summary_df = pd.DataFrame(summary_records).sort_values(["l1", "prediction_file"])
    display_cols = [
        "prediction_file",
        "reference_file",
        "l1",
        "rmse",
        "mae",
        "bias",
        "pearson",
        "spearman",
        "coverage",
        "matched_rows",
        "reference_rows",
    ]

    print("\n=== Directory Mode Summary ===")
    print(summary_df[display_cols].to_string(index=False))

    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.summary_out, index=False)
        print(f"\nSaved summary CSV to: {args.summary_out}")

    comparison_dir = args.summary_out.parent if args.summary_out is not None else Path(".")
    for l1, entries in sorted(lang_heatmaps.items()):
        try:
            comparison_path = comparison_dir / f"{l1}_bucket_heatmaps_comparison.png"
            _save_language_comparison_heatmaps(
                entries,
                comparison_path,
                title=f"{l1.upper()} Bucket Heatmaps Comparison",
            )
            print(f"Saved language comparison heatmap: {comparison_path}")
        except Exception as exc:
            print(f"Warning: could not save language comparison heatmap for {l1}: {exc}")

    if failed:
        print(f"\n{len(failed)} file(s) failed:")
        for pred_file, msg in failed:
            print(f"- {pred_file}: {msg}")


if __name__ == "__main__":
    main()
