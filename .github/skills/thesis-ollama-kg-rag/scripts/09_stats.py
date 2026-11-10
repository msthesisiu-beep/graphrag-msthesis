"""Gate 11: Statistical hypothesis testing, multiplicity control, and development-grid variant selection.

Implements the thesis statistical analysis protocol (Thesis Chapter 3.7, Tables 9 and 10):
  - Exact McNemar test for paired binary measures (Accuracy, Recall@k, Full coverage) with paired bootstrap CIs.
  - Wilcoxon signed-rank test for continuous/bounded measures (Faithfulness, Relevancy, Latency, MRR, Citations)
    with rank-biserial correlation r and paired bootstrap CIs.
  - Multiplicity control using Holm-Bonferroni step-down within pre-specified test families (Table 10).
  - Dev-grid selection (--dev-select) to evaluate all 18 C-variants and choose top variants + default for test split.

Usage:
  python 09_stats.py --split dev --dev-select     # Select optimal C variants from dev-grid
  python 09_stats.py --split test                 # Complete thesis hypothesis testing suite
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import load_config, read_json, work, write_json


# ---------------------------------------------------------------- statistical helpers
def holm_bonferroni(p_vals: list[float]) -> list[float]:
    """Applies Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_vals)
    if m <= 1:
        return list(p_vals)
    # Sort indices by raw p-value
    sorted_indices = sorted(range(m), key=lambda i: p_vals[i])
    adj = [0.0] * m
    for rank, idx in enumerate(sorted_indices):
        multiplier = m - rank
        raw_p = p_vals[idx]
        adj[idx] = min(1.0, raw_p * multiplier)
    # Enforce monotonicity
    running_max = 0.0
    for idx in sorted_indices:
        running_max = max(running_max, adj[idx])
        adj[idx] = min(1.0, running_max)
    return adj


def paired_bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 42,
                        alpha: float = 0.05, stat: str = "mean") -> tuple[float, float, float]:
    """Computes paired difference point estimate and 95% bootstrap confidence interval (10,000 resamples)."""
    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    valid = diffs[~np.isnan(diffs)]
    n = len(valid)
    if n == 0:
        return np.nan, np.nan, np.nan
    stat_fn = np.mean if stat == "mean" else np.median
    point_est = float(stat_fn(valid))
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, n, size=(n_boot, n))
    resampled_stats = stat_fn(valid[sample_indices], axis=1)
    ci_low = float(np.percentile(resampled_stats, 100 * (alpha / 2)))
    ci_high = float(np.percentile(resampled_stats, 100 * (1 - alpha / 2)))
    return point_est, ci_low, ci_high


def exact_mcnemar(y1: np.ndarray, y2: np.ndarray, seed: int = 42) -> dict:
    """Exact McNemar test for paired binary outcomes with contingency counts b, c and bootstrap CI."""
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    # Contingency counts
    a = int(np.sum((y1 == 1) & (y2 == 1)))
    b = int(np.sum((y1 == 1) & (y2 == 0)))  # condition 1 correct, condition 2 incorrect
    c = int(np.sum((y1 == 0) & (y2 == 1)))  # condition 1 incorrect, condition 2 correct
    d = int(np.sum((y1 == 0) & (y2 == 0)))
    n_discordant = b + c
    if n_discordant == 0:
        p_val = 1.0
    else:
        # Two-sided exact binomial test with p = 0.5
        p_val = stats.binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided").pvalue

    delta_pct, ci_low, ci_high = paired_bootstrap_ci(y1, y2, seed=seed, stat="mean")
    return {
        "test": "exact_mcnemar",
        "n": len(y1),
        "a": a, "b": b, "c": c, "d": d,
        "delta": round(delta_pct * 100, 2),  # percentage points
        "ci_95": [round(ci_low * 100, 2), round(ci_high * 100, 2)],
        "p_raw": float(p_val),
    }


def paired_wilcoxon(x: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    """Paired Wilcoxon signed-rank test with rank-biserial correlation r and paired bootstrap CI."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    xv, yv = x[mask], y[mask]
    n = len(xv)
    if n < 3:
        return {"test": "wilcoxon", "n": n, "delta": np.nan, "ci_95": [np.nan, np.nan], "r": np.nan, "p_raw": 1.0}

    diffs = xv - yv
    non_zero = diffs[diffs != 0]
    n_nonzero = len(non_zero)

    if n_nonzero == 0:
        p_val = 1.0
        r_biserial = 0.0
    else:
        try:
            res = stats.wilcoxon(xv, yv, alternative="two-sided", zero_method="pratt")
            p_val = float(res.pvalue)
            # Rank-biserial correlation: (W+ - W-) / total_ranks
            ranks = stats.rankdata(np.abs(diffs))
            w_plus = float(np.sum(ranks[diffs > 0]))
            w_minus = float(np.sum(ranks[diffs < 0]))
            total = w_plus + w_minus
            r_biserial = (w_plus - w_minus) / total if total > 0 else 0.0
        except Exception:
            p_val = 1.0
            r_biserial = 0.0

    delta_mean, ci_low, ci_high = paired_bootstrap_ci(xv, yv, seed=seed, stat="mean")
    med_diff, med_low, med_high = paired_bootstrap_ci(xv, yv, seed=seed, stat="median")
    return {
        "test": "wilcoxon_signed_rank",
        "n": n,
        "n_nonzero": n_nonzero,
        "delta_mean": round(delta_mean, 4),
        "delta_median": round(med_diff, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "ci_median_95": [round(med_low, 4), round(med_high, 4)],
        "rank_biserial_r": round(r_biserial, 4),
        "p_raw": p_val,
    }


# ---------------------------------------------------------------- dev-select
def run_dev_select(cfg: dict) -> None:
    path = work(cfg, "runs", "dev", "per_question_metrics.csv")
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run 07_run_configs.py --plan dev-grid and 08_evaluate.py --split dev first.")
    df = pd.read_csv(path)
    default_v = f"t{cfg['kg']['default_variant']['triples']}-h{cfg['kg']['default_variant']['hops']}-g{int(round(cfg['kg']['default_variant']['share']*100))}"

    # Extract Baseline B performance
    b_rows = df[df.config == "B"].drop_duplicates(subset=["question_id"])
    b_acc = b_rows.set_index("question_id")["correct"].to_dict()
    b_mean = b_rows["correct"].mean()

    # Compare each C variant against B
    c_df = df[df.config == "C"]
    summary = []
    variants = sorted(c_df["variant"].unique())

    for v in variants:
        v_rows = c_df[c_df.variant == v].drop_duplicates(subset=["question_id"]).set_index("question_id")
        qids = sorted(set(b_acc.keys()) & set(v_rows.index))
        b_vec = np.array([b_acc[qid] for qid in qids])
        c_vec = np.array([v_rows.loc[qid, "correct"] for qid in qids])
        mc = exact_mcnemar(c_vec, b_vec, seed=cfg["seed"])
        v_mean = float(c_vec.mean())
        summary.append({
            "variant": v,
            "accuracy": round(v_mean, 4),
            "baseline_accuracy": round(b_mean, 4),
            "delta_pp": mc["delta"],
            "ci_95": mc["ci_95"],
            "p_raw": mc["p_raw"],
            "b_correct_c_wrong": mc["c"],
            "c_correct_b_wrong": mc["b"],
            "t_total_median": round(float(v_rows["t_total"].median()), 2) if "t_total" in v_rows else np.nan,
            "faithfulness": round(float(v_rows["faithfulness_full"].mean()), 4) if "faithfulness_full" in v_rows and not v_rows["faithfulness_full"].isna().all() else np.nan,
        })

    sum_df = pd.DataFrame(summary).sort_values(by=["accuracy", "delta_pp", "p_raw"], ascending=[False, False, True])
    # Select top 3 variants plus default
    top3 = [row.variant for row in sum_df.itertuples() if row.variant != default_v][:3]
    selected = [default_v] + [v for v in top3 if v != default_v]

    out_file = work(cfg, "runs", "dev", "selected_variants.json")
    write_json(out_file, {
        "default": default_v,
        "selected_top3": top3,
        "variants": selected,
        "ranking": sum_df.to_dict(orient="records"),
    })
    print(f"\n--- Dev-Grid Selection ({len(summary)} variants evaluated) ---")
    print(sum_df[["variant", "accuracy", "delta_pp", "ci_95", "p_raw"]].to_string(index=False))
    print(f"\nSelected variants written to {out_file}: {selected}")


# ---------------------------------------------------------------- full test suite
def run_test_suite(cfg: dict, split: str) -> None:
    path = work(cfg, "runs", split, "per_question_metrics.csv")
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run 08_evaluate.py --split {split} first.")
    df = pd.read_csv(path)
    default_v = f"t{cfg['kg']['default_variant']['triples']}-h{cfg['kg']['default_variant']['hops']}-g{int(round(cfg['kg']['default_variant']['share']*100))}"
    seed = cfg["seed"]

    # Pivot to get paired series per condition
    def get_series(config: str, variant: str = "-", mode: str = "matched", col: str = "correct"):
        subset = df[(df.config == config) & (df.variant.astype(str) == variant) & (df["mode"].astype(str) == mode)]
        if subset.empty and variant == "-":
            subset = df[(df.config == config)]
        return subset.drop_duplicates(subset=["question_id"]).set_index("question_id")[col]

    all_tests = []

    def record_test(family: str, comp_name: str, metric: str, test_res: dict):
        all_tests.append({"family": family, "comparison": comp_name, "metric": metric, **test_res})

    # 1. Family 1: Confirmatory (B vs C default, matched, Accuracy)
    b_acc = get_series("B", col="correct")
    c_def_acc = get_series("C", default_v, "matched", col="correct")
    qids_conf = sorted(set(b_acc.index) & set(c_def_acc.index))
    if qids_conf:
        res = exact_mcnemar(c_def_acc.loc[qids_conf].values, b_acc.loc[qids_conf].values, seed=seed)
        record_test("1. Confirmatory", "C(default) vs B [matched]", "accuracy", res)

    # 2. Family 2: Selected Variants (B vs each selected C variant, matched, Accuracy)
    sel_path = work(cfg, "runs", "dev", "selected_variants.json")
    variants_to_test = [default_v]
    if sel_path.exists():
        variants_to_test = read_json(sel_path).get("variants", [default_v])
    for v in set(variants_to_test):
        v_acc = get_series("C", v, "matched", col="correct")
        qids_v = sorted(set(b_acc.index) & set(v_acc.index))
        if qids_v:
            res = exact_mcnemar(v_acc.loc[qids_v].values, b_acc.loc[qids_v].values, seed=seed)
            record_test("2. Variants", f"C({v}) vs B [matched]", "accuracy", res)

    # 3. Family 3: Controls (C default vs RAND, C default vs UNF, matched, Accuracy)
    for ctrl in ("RAND", "UNF"):
        ctrl_acc = get_series(ctrl, default_v, "matched", col="correct")
        qids_ctrl = sorted(set(c_def_acc.index) & set(ctrl_acc.index))
        if qids_ctrl:
            res = exact_mcnemar(c_def_acc.loc[qids_ctrl].values, ctrl_acc.loc[qids_ctrl].values, seed=seed)
            record_test("3. Controls", f"C(default) vs {ctrl} [matched]", "accuracy", res)

    # 4. Family 4: Placement (C default vs D, matched, Accuracy)
    d_acc = get_series("D", default_v, "matched", col="correct")
    qids_d = sorted(set(c_def_acc.index) & set(d_acc.index))
    if qids_d:
        res = exact_mcnemar(c_def_acc.loc[qids_d].values, d_acc.loc[qids_d].values, seed=seed)
        record_test("4. Placement", "C(default) vs D [matched]", "accuracy", res)

    # 5. Family 5: Subgroups (B vs C default within question subgroups)
    meta_df = df[["question_id", "hop_type", "question_type", "retriever_found"]].drop_duplicates().set_index("question_id")
    for group_col in ("hop_type", "question_type", "retriever_found"):
        if group_col not in meta_df:
            continue
        for group_val in sorted(meta_df[group_col].dropna().unique()):
            g_qids = meta_df[meta_df[group_col] == group_val].index
            common = sorted(set(g_qids) & set(qids_conf))
            if len(common) >= 5:
                res = exact_mcnemar(c_def_acc.loc[common].values, b_acc.loc[common].values, seed=seed)
                record_test("5. Subgroups", f"C(def) vs B [{group_col}={group_val}]", "accuracy", res)

    # 6. Family 6: Secondary Quality Measures (B vs C default, continuous)
    continuous_metrics = [
        "faithfulness_full", "faithfulness_passages", "answer_relevancy",
        "context_precision", "context_recall", "citation_validity", "citation_support", "mrr"
    ]
    for m in continuous_metrics:
        if m not in df.columns:
            continue
        b_m = get_series("B", col=m)
        c_m = get_series("C", default_v, "matched", col=m)
        common = sorted(set(b_m.dropna().index) & set(c_m.dropna().index))
        if len(common) >= 5:
            res = paired_wilcoxon(c_m.loc[common].values, b_m.loc[common].values, seed=seed)
            record_test("6. Secondary Quality", f"C(default) vs B", m, res)

    # 7. Family 7: Cost and Latency (B vs C default)
    latency_metrics = ["t_total", "t_retrieve", "t_rerank", "t_kg", "t_generate"]
    for lat in latency_metrics:
        if lat not in df.columns:
            continue
        b_lat = get_series("B", col=lat)
        c_lat = get_series("C", default_v, "matched", col=lat)
        common = sorted(set(b_lat.dropna().index) & set(c_lat.dropna().index))
        if len(common) >= 5:
            res = paired_wilcoxon(c_lat.loc[common].values, b_lat.loc[common].values, seed=seed)
            record_test("7. Cost / Latency", f"C(default) vs B", lat, res)

    # Apply Holm-Bonferroni correction within each family
    by_family = defaultdict(list)
    for t in all_tests:
        by_family[t["family"]].append(t)

    final_results = []
    for fam, tests in by_family.items():
        if fam.startswith("1."):  # Family 1 Confirmatory: no adjustment (single test)
            for t in tests:
                t["p_holm"] = t["p_raw"]
                t["significant_0.05"] = bool(t["p_holm"] < 0.05)
                final_results.append(t)
        else:
            raw_ps = [t["p_raw"] for t in tests]
            adj_ps = holm_bonferroni(raw_ps)
            for t, adj_p in zip(tests, adj_ps):
                t["p_holm"] = round(adj_p, 5)
                t["significant_0.05"] = bool(adj_p < 0.05)
                final_results.append(t)

    # Save output artifacts
    out_json = work(cfg, "runs", split, "statistical_tests.json")
    write_json(out_json, final_results)
    out_csv = work(cfg, "runs", split, "statistical_tests.csv")
    res_df = pd.DataFrame(final_results)
    res_df.to_csv(out_csv, index=False)

    print(f"\n=======================================================")
    print(f"      Thesis Statistical Test Results ({split} split)   ")
    print(f"=======================================================")
    for fam, tests in by_family.items():
        print(f"\n>>> {fam}")
        sub = pd.DataFrame([t for t in final_results if t["family"] == fam])
        cols = [c for c in ["comparison", "metric", "delta", "delta_mean", "ci_95", "rank_biserial_r", "p_raw", "p_holm", "significant_0.05"] if c in sub]
        print(sub[cols].to_string(index=False))
    print(f"\nWrote full statistical reports to:\n  - {out_json}\n  - {out_csv}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run thesis statistical analysis suite")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--dev-select", action="store_true", help="Rank and select optimal C variants on dev split")
    args = ap.parse_args()

    cfg = load_config()
    if args.dev_select:
        if args.split != "dev":
            ap.error("--dev-select requires --split dev")
        run_dev_select(cfg)
    else:
        run_test_suite(cfg, args.split)


if __name__ == "__main__":
    main()
