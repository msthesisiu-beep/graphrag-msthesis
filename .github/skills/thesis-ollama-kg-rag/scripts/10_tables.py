"""Gate 11: Camera-ready thesis tables (LaTeX & Markdown) and publication-quality tradeoff figures.

Generates:
  - Table 4: Benchmark dataset characterization and question taxonomy (single vs. multi-hop, specs).
  - Table 6: Benchmark flow and exclusion attrition stages.
  - Table 19: Main experimental results across all configurations (0, A, B, C, D, RAND, UNF).
  - Table 20: Stage-by-stage latency (retrieve, rerank, KG, generate) and token/monetary cost breakdown.
  - Table 21: KG parameter ablation grid (triples, hops, share) and subgroup breakdown.
  - Publication figures (PNG & PDF): Accuracy vs. Latency Pareto curve, Faithfulness tradeoff.

Usage:
  python 10_tables.py --split test
  python 10_tables.py --split dev
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_config, read_json, work, write_json


# ---------------------------------------------------------------- formatting helpers
def to_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Renders a standard clean GitHub Markdown table."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    data_lines = ["| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + data_lines) + "\n"


def to_latex_table(caption: str, label: str, col_align: str, headers: list[str], rows: list[list[str]]) -> str:
    """Renders a booktabs-compliant LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        f"\\begin{{tabular}}{{{col_align}}}",
        r"\toprule",
        " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


# ---------------------------------------------------------------- table generators
def generate_table4_dataset(cfg: dict, out_dir: Path) -> None:
    """Table 4: Benchmark Dataset Characterization."""
    qlog_path = work(cfg, "benchmark", "question_log.csv")
    if not qlog_path.exists():
        print(f"Skipping Table 4: {qlog_path} not found.")
        return
    df = pd.read_csv(qlog_path).fillna("")
    inc = df[df.decision == "include"]

    rows = []
    # Split distribution
    for split_name in ("development", "test"):
        sub = inc[inc.split == split_name]
        n_single = sum(sub.hop_type == "single")
        n_multi = sum(sub.hop_type == "multi")
        n_found = sum(sub.retriever_found == True)
        rows.append([
            split_name.capitalize(),
            str(len(sub)),
            f"{n_single} ({round(n_single/len(sub)*100, 1)}%)" if len(sub) else "0",
            f"{n_multi} ({round(n_multi/len(sub)*100, 1)}%)" if len(sub) else "0",
            f"{n_found} ({round(n_found/len(sub)*100, 1)}%)" if len(sub) else "0",
        ])

    headers = ["Subset", "Total Questions", "Single-Hop", "Multi-Hop", "Retriever-Found"]
    col_align = "lcccc"
    caption = "TeleQnA Benchmark Dataset Characterization and Stratified Partitions"
    label = "tab:benchmark_characterization"

    (out_dir / "table4_dataset.md").write_text(to_markdown_table(headers, rows))
    (out_dir / "table4_dataset.tex").write_text(to_latex_table(caption, label, col_align, headers, rows))
    print(f"  - Table 4 written to {out_dir}/table4_dataset.{{md,tex}}")


def generate_table6_flow(cfg: dict, out_dir: Path) -> None:
    """Table 6: Exclusion and Benchmark Flow Attrition."""
    flow_path = work(cfg, "benchmark", "flow.json")
    if not flow_path.exists():
        print(f"Skipping Table 6: {flow_path} not found.")
        return
    flow = read_json(flow_path)

    rows = [
        ["1. Initial Candidate Pool (Standards Categories)", str(flow.get("candidates", "N/A")), "100.0%"],
        ["2. Candidate Pooling (BM25 + E5 hits >= 1)", str(flow.get("after_pooling", "N/A")),
         f"{round(flow.get('after_pooling', 0) / max(1, flow.get('candidates', 1)) * 100, 1)}%"],
        ["3. Verifier Answerability Acceptance", str(flow.get("after_verifier", "N/A")),
         f"{round(flow.get('after_verifier', 0) / max(1, flow.get('candidates', 1)) * 100, 1)}%"],
        ["4. Post-Manual Gate & Deduplication", str(flow.get("after_manual", "N/A")),
         f"{round(flow.get('after_manual', 0) / max(1, flow.get('candidates', 1)) * 100, 1)}%"],
        ["5. Final Frozen Benchmark Set", str(flow.get("final_retained", "N/A")),
         f"{round(flow.get('final_retained', 0) / max(1, flow.get('candidates', 1)) * 100, 1)}%"],
        ["   - Development Subset (20%)", str(flow.get("development", "N/A")),
         f"{round(flow.get('development', 0) / max(1, flow.get('final_retained', 1)) * 100, 1)}%"],
        ["   - Held-Out Test Subset (80%)", str(flow.get("test", "N/A")),
         f"{round(flow.get('test', 0) / max(1, flow.get('final_retained', 1)) * 100, 1)}%"],
    ]

    headers = ["Pipeline Stage / Filter", "Question Count", "Retention Rate"]
    col_align = "lrr"
    caption = "TeleQnA Benchmark Flow and Exclusion Attrition"
    label = "tab:benchmark_flow"

    (out_dir / "table6_flow.md").write_text(to_markdown_table(headers, rows))
    (out_dir / "table6_flow.tex").write_text(to_latex_table(caption, label, col_align, headers, rows))
    print(f"  - Table 6 written to {out_dir}/table6_flow.{{md,tex}}")


def generate_table19_main(cfg: dict, split: str, out_dir: Path) -> None:
    """Table 19: Main Experimental Results Across All Configurations."""
    metrics_path = work(cfg, "runs", split, "per_question_metrics.csv")
    if not metrics_path.exists():
        print(f"Skipping Table 19: {metrics_path} not found.")
        return
    df = pd.read_csv(metrics_path)
    stats_path = work(cfg, "runs", split, "statistical_tests.json")
    stats_map = {}
    if stats_path.exists():
        for t in read_json(stats_path):
            stats_map[(t.get("comparison"), t.get("metric"))] = t

    order = [
        ("0", "-", "-"),
        ("A", "-", "matched"),
        ("A", "-", "overhead"),
        ("B", "-", "-"),
        ("C", "default", "matched"),
        ("C", "default", "overhead"),
        ("D", "default", "matched"),
        ("RAND", "default", "matched"),
        ("UNF", "default", "matched"),
    ]
    def_v = f"t{cfg['kg']['default_variant']['triples']}-h{cfg['kg']['default_variant']['hops']}-g{int(round(cfg['kg']['default_variant']['share']*100))}"

    rows = []
    for cfg_code, var, mode in order:
        v_label = def_v if var == "default" else var
        sub = df[(df.config == cfg_code) & (df.variant.astype(str) == v_label) & (df["mode"].astype(str) == mode)]
        if sub.empty and var == "-":
            sub = df[df.config == cfg_code]
        if sub.empty:
            continue

        n = len(sub)
        acc = round(float(sub["correct"].mean() * 100), 1)
        rec_k = f"{round(float(sub['recall_at_k'].mean() * 100), 1)}%" if "recall_at_k" in sub and not sub["recall_at_k"].isna().all() else "--"
        mrr = f"{round(float(sub['mrr'].mean()), 3)}" if "mrr" in sub and not sub["mrr"].isna().all() else "--"
        faith = f"{round(float(sub['faithfulness_full'].mean()), 3)}" if "faithfulness_full" in sub and not sub["faithfulness_full"].isna().all() else "--"
        faith_p = f"{round(float(sub['faithfulness_passages'].mean()), 3)}" if "faithfulness_passages" in sub and not sub["faithfulness_passages"].isna().all() else "--"
        relev = f"{round(float(sub['answer_relevancy'].mean()), 3)}" if "answer_relevancy" in sub and not sub["answer_relevancy"].isna().all() else "--"
        cite_val = f"{round(float(sub['citation_validity'].mean() * 100), 1)}%" if "citation_validity" in sub and not sub["citation_validity"].isna().all() else "--"
        lat = f"{round(float(sub['t_total'].median()), 2)}s" if "t_total" in sub else "--"

        cond_desc = f"{cfg_code}" + (f" ({mode})" if mode != "-" else "")
        if cfg_code in ("C", "D", "RAND", "UNF"):
            cond_desc = f"{cfg_code} [{mode}]"

        rows.append([
            cond_desc, str(n), f"{acc}%", rec_k, mrr, faith, faith_p, relev, cite_val, lat
        ])

    headers = ["Condition", "N", "Accuracy", "Recall@5", "MRR", "Faith. (Full)", "Faith. (Pass)", "Relevancy", "Cite Valid", "Latency (Med)"]
    col_align = "lccccccccc"
    caption = f"Experimental Results Across Configurations ({split.capitalize()} Partition)"
    label = f"tab:main_results_{split}"

    (out_dir / "table19_main_results.md").write_text(to_markdown_table(headers, rows))
    (out_dir / "table19_main_results.tex").write_text(to_latex_table(caption, label, col_align, headers, rows))
    print(f"  - Table 19 written to {out_dir}/table19_main_results.{{md,tex}}")


def generate_table20_cost(cfg: dict, split: str, out_dir: Path) -> None:
    """Table 20: Latency Breakdown and Token Accounting."""
    metrics_path = work(cfg, "runs", split, "per_question_metrics.csv")
    if not metrics_path.exists():
        print(f"Skipping Table 20: {metrics_path} not found.")
        return
    df = pd.read_csv(metrics_path)
    price = cfg.get("pricing_usd_per_1m_tokens", {"input": 0.15, "output": 0.60})

    rows = []
    for cond_name, grp in df.groupby("cond"):
        n = len(grp)
        t_ret = round(float(grp["t_retrieve"].median()), 2) if "t_retrieve" in grp else 0.0
        t_rr = round(float(grp["t_rerank"].median()), 2) if "t_rerank" in grp else 0.0
        t_kg = round(float(grp["t_kg"].median()), 2) if "t_kg" in grp else 0.0
        t_gen = round(float(grp["t_generate"].median()), 2) if "t_generate" in grp else 0.0
        t_tot = round(float(grp["t_total"].median()), 2) if "t_total" in grp else 0.0

        in_tok = float(grp["in_tokens"].mean()) if "in_tokens" in grp else 0.0
        out_tok = float(grp["out_tokens"].mean()) if "out_tokens" in grp else 0.0
        cost_usd = (in_tok / 1e6 * price["input"]) + (out_tok / 1e6 * price["output"])

        rows.append([
            cond_name, f"{t_ret}s", f"{t_rr}s", f"{t_kg}s", f"{t_gen}s", f"{t_tot}s",
            f"{int(round(in_tok))}", f"{int(round(out_tok))}", f"${cost_usd:.4f}"
        ])

    headers = ["Condition", "t_Retrieve", "t_Rerank", "t_KG", "t_Gen", "t_Total", "Tokens In", "Tokens Out", "Cost / Q"]
    col_align = "lcccccccc"
    caption = f"Stage-by-Stage Latency and Operational Cost Breakdown ({split.capitalize()} Partition)"
    label = f"tab:cost_breakdown_{split}"

    (out_dir / "table20_cost_latency.md").write_text(to_markdown_table(headers, rows))
    (out_dir / "table20_cost_latency.tex").write_text(to_latex_table(caption, label, col_align, headers, rows))
    print(f"  - Table 20 written to {out_dir}/table20_cost_latency.{{md,tex}}")


def generate_table21_ablation(cfg: dict, split: str, out_dir: Path) -> None:
    """Table 21: KG Parameter Ablation Grid and Subgroup Breakdown."""
    metrics_path = work(cfg, "runs", split, "per_question_metrics.csv")
    if not metrics_path.exists():
        print(f"Skipping Table 21: {metrics_path} not found.")
        return
    df = pd.read_csv(metrics_path)
    c_df = df[df.config == "C"]
    if c_df.empty:
        print("Skipping Table 21: No Configuration C rows found.")
        return

    rows = []
    # Grid by variant
    for v_name, grp in sorted(c_df.groupby("variant")):
        acc = round(float(grp["correct"].mean() * 100), 1)
        lat = round(float(grp["t_total"].median()), 2)
        n_trip = round(float(grp["n_triples"].mean()), 1) if "n_triples" in grp else "--"
        rows.append([v_name, str(len(grp)), f"{acc}%", str(n_trip), f"{lat}s"])

    headers = ["C Variant (t-h-g)", "N", "Accuracy", "Avg Triples", "Median Latency"]
    col_align = "lcccc"
    caption = f"Knowledge Graph Parameter Ablation Across Triple, Hop, and Budget Dimensions ({split.capitalize()})"
    label = f"tab:kg_ablation_{split}"

    (out_dir / "table21_ablation.md").write_text(to_markdown_table(headers, rows))
    (out_dir / "table21_ablation.tex").write_text(to_latex_table(caption, label, col_align, headers, rows))
    print(f"  - Table 21 written to {out_dir}/table21_ablation.{{md,tex}}")


def generate_figures(cfg: dict, split: str, out_dir: Path) -> None:
    """Generates Pareto tradeoff plots (PNG and PDF)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping graphical figure generation.")
        return

    metrics_path = work(cfg, "runs", split, "per_question_metrics.csv")
    if not metrics_path.exists():
        return
    df = pd.read_csv(metrics_path)

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Accuracy vs. Latency Pareto Curve
    agg = df.groupby("cond").agg({"correct": "mean", "t_total": "median"}).reset_index()
    agg["correct"] = agg["correct"] * 100

    plt.figure(figsize=(7, 4.5), dpi=300)
    for _, row in agg.iterrows():
        plt.scatter(row["t_total"], row["correct"], s=100, alpha=0.8)
        plt.annotate(row["cond"], (row["t_total"], row["correct"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.xlabel("Median Latency (seconds)", fontsize=10)
    plt.ylabel("Accuracy (%)", fontsize=10)
    plt.title(f"Accuracy vs. Latency Tradeoff ({split.capitalize()} Set)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_latency_tradeoff.png")
    plt.savefig(fig_dir / "accuracy_latency_tradeoff.pdf")
    plt.close()
    print(f"  - Figure written to {fig_dir}/accuracy_latency_tradeoff.{{png,pdf}}")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate camera-ready thesis tables and figures")
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = work(cfg, "reports", args.split)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"    Generating Camera-Ready Thesis Tables ({args.split})")
    print(f"=======================================================")

    generate_table4_dataset(cfg, out_dir)
    generate_table6_flow(cfg, out_dir)
    generate_table19_main(cfg, args.split, out_dir)
    generate_table20_cost(cfg, args.split, out_dir)
    generate_table21_ablation(cfg, args.split, out_dir)
    generate_figures(cfg, args.split, out_dir)

    print(f"\nAll thesis tables saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
