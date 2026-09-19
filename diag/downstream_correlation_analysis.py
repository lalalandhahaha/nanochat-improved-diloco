import argparse
import csv
import glob
import json
import math
import os
import random
import re
from collections import defaultdict


CONFIG_RE = re.compile(r"^(C\d_[A-Za-z0-9_]+)_seed(\d+)$")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values):
    return sum(values) / len(values)


def stdev_population(values):
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def rankdata(values):
    pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


def pearson(x, y):
    mx = mean(x)
    my = mean(y)
    sx = math.sqrt(sum((value - mx) ** 2 for value in x))
    sy = math.sqrt(sum((value - my) ** 2 for value in y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))


def bootstrap_ci(x, y, corr_fn, *, n_resamples, rng):
    n = len(x)
    samples = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        value = corr_fn([x[i] for i in idx], [y[i] for i in idx])
        if not math.isnan(value):
            samples.append(value)
    if not samples:
        return float("nan"), float("nan")
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return lo, hi


def permutation_pvalue(x, y, corr_fn, *, n_permutations, rng):
    observed = corr_fn(x, y)
    if math.isnan(observed):
        return float("nan")
    extreme = 0
    y_perm = list(y)
    for _ in range(n_permutations):
        rng.shuffle(y_perm)
        value = corr_fn(x, y_perm)
        if not math.isnan(value) and abs(value) >= abs(observed):
            extreme += 1
    return (extreme + 1) / (n_permutations + 1)


def parse_base_run(path):
    run_id = os.path.basename(os.path.dirname(path))
    match = CONFIG_RE.match(run_id)
    if match is None:
        return None
    records = read_jsonl(path)
    if not records:
        return None
    final = records[-1]
    config, seed = match.group(1), int(match.group(2))
    metadata_path = os.path.join(os.path.dirname(path), "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    user_config = metadata.get("user_config", {})
    return {
        "run_id": run_id,
        "config": config,
        "seed": seed,
        "base_line_count": len(records),
        "base_record_type": final.get("record_type"),
        "base_step": final.get("step"),
        "outer_step_idx": final.get("outer_step_idx"),
        "core_metric": final.get("core_metric"),
        "val_bpb": final.get("val_bpb"),
        "s1_global": final.get("s1_global"),
        "s2_global": final.get("s2_global"),
        "jds_raw": final.get("jds_raw"),
        "use_fisher_prox": user_config.get("use_fisher_prox"),
        "muon_proximal_lambda": user_config.get("muon_proximal_lambda"),
        "adamw_proximal_lambda": user_config.get("adamw_proximal_lambda"),
    }


def load_base_rows(base_log_root):
    rows = {}
    for path in sorted(glob.glob(os.path.join(base_log_root, "*", "metrics.jsonl"))):
        row = parse_base_run(path)
        if row is not None:
            rows[row["run_id"]] = row
    return rows


def load_sft_rows(sft_metrics_dir):
    rows = {}
    for path in sorted(glob.glob(os.path.join(sft_metrics_dir, "*.jsonl"))):
        records = read_jsonl(path)
        if not records:
            continue
        final = records[-1]
        run_id = final.get("run_id")
        if run_id:
            rows[run_id] = {
                "sft_metrics_path": path,
                "sft_record_type": final.get("record_type"),
                "sft_step": final.get("sft_step"),
                "sft_num_iterations": final.get("num_iterations"),
                "sft_train_loss": final.get("train_loss"),
                "sft_val_loss": final.get("val_loss"),
                "mmlu_acc": final.get("mmlu_acc"),
                "arc_easy_acc": final.get("arc_easy_acc"),
                "mmlu_centered": final.get("mmlu_centered"),
                "arc_easy_centered": final.get("arc_easy_centered"),
                "downstream_score": final.get("downstream_score"),
            }
    return rows


def add_z_jds(rows):
    s1_values = [row["s1_global"] for row in rows]
    s2_values = [row["s2_global"] for row in rows]
    s1_mean, s2_mean = mean(s1_values), mean(s2_values)
    s1_sd, s2_sd = stdev_population(s1_values), stdev_population(s2_values)
    for row in rows:
        row["s1_z"] = (row["s1_global"] - s1_mean) / s1_sd if s1_sd else 0.0
        row["s2_z"] = (row["s2_global"] - s2_mean) / s2_sd if s2_sd else 0.0
        row["jds_z"] = row["s1_z"] + row["s2_z"]


def join_rows(base_rows, sft_rows):
    rows = []
    for run_id in sorted(base_rows):
        if run_id not in sft_rows:
            continue
        row = dict(base_rows[run_id])
        row.update(sft_rows[run_id])
        rows.append(row)
    add_z_jds(rows)
    return rows


def write_csv(rows, path):
    fields = [
        "run_id",
        "config",
        "seed",
        "base_record_type",
        "base_step",
        "outer_step_idx",
        "sft_record_type",
        "sft_step",
        "sft_num_iterations",
        "core_metric",
        "val_bpb",
        "s1_global",
        "s2_global",
        "jds_raw",
        "s1_z",
        "s2_z",
        "jds_z",
        "sft_train_loss",
        "sft_val_loss",
        "mmlu_acc",
        "arc_easy_acc",
        "mmlu_centered",
        "arc_easy_centered",
        "downstream_score",
        "use_fisher_prox",
        "muon_proximal_lambda",
        "adamw_proximal_lambda",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def correlation_table(rows, *, n_bootstrap, n_permutations, seed):
    rng = random.Random(seed)
    targets = [
        ("downstream_score", "Fast-SFT downstream score"),
        ("sft_val_loss", "Fast-SFT validation loss"),
        ("mmlu_acc", "Fast-SFT MMLU accuracy"),
        ("arc_easy_acc", "Fast-SFT ARC-Easy accuracy"),
        ("core_metric", "Base CORE proxy"),
        ("val_bpb", "Base validation BPB"),
    ]
    signals = [
        ("s1_global", "S1"),
        ("s2_global", "S2"),
        ("jds_z", "JDS z(S1)+z(S2)"),
        ("jds_raw", "JDS raw S1+S2"),
    ]
    table = []
    for target_key, target_label in targets:
        target_rows = [row for row in rows if row.get(target_key) is not None]
        if len(target_rows) < 3:
            continue
        y = [row[target_key] for row in target_rows]
        for signal_key, signal_label in signals:
            x = [row[signal_key] for row in target_rows]
            for method, fn in (("Pearson", pearson), ("Spearman", spearman)):
                value = fn(x, y)
                ci_lo, ci_hi = bootstrap_ci(x, y, fn, n_resamples=n_bootstrap, rng=rng)
                pvalue = permutation_pvalue(x, y, fn, n_permutations=n_permutations, rng=rng)
                table.append(
                    {
                        "target": target_label,
                        "target_key": target_key,
                        "signal": signal_label,
                        "signal_key": signal_key,
                        "method": method,
                        "n": len(target_rows),
                        "r": value,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                        "p_value": pvalue,
                    }
                )
    return table


def write_correlation_csv(table, path):
    fields = ["target", "signal", "method", "n", "r", "ci_lo", "ci_hi", "p_value"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in table:
            writer.writerow({field: row[field] for field in fields})


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def write_report(rows, corr, output_dir, missing_base, missing_sft):
    by_config = defaultdict(list)
    for row in rows:
        by_config[row["config"]].append(row)

    lines = [
        "# Phase C Fast-SFT Downstream Correlation Report",
        "",
        "## Scope",
        "",
        f"- Joined runs analyzed: {len(rows)}",
        "- Base endpoint record: final diagnostics JSONL row.",
        "- Fast-SFT endpoint record: final `record_type=final_sft_eval` row.",
        "- Primary JDS definition: `jds_z = z(S1) + z(S2)` across joined runs.",
        "- Primary downstream target: mean centered MMLU and ARC-Easy after fast-SFT.",
        "",
        "## Data Quality",
        "",
    ]
    if missing_sft:
        lines.append("- Base runs missing SFT metrics: " + ", ".join(f"`{x}`" for x in missing_sft))
    if missing_base:
        lines.append("- SFT metrics missing base diagnostics: " + ", ".join(f"`{x}`" for x in missing_base))
    if not missing_base and not missing_sft:
        lines.append("- All base diagnostic runs have matching SFT metrics.")
    bad_rows = [
        row for row in rows
        if row.get("base_record_type") != "final_eval" or row.get("sft_record_type") != "final_sft_eval"
    ]
    if bad_rows:
        lines.append("- WARNING: Some joined rows do not use final endpoint records:")
        for row in bad_rows:
            lines.append(f"  - `{row['run_id']}` base={row.get('base_record_type')} sft={row.get('sft_record_type')}")

    lines.extend([
        "",
        "## Config Summary",
        "",
        "| run_id | core_metric | val_bpb | S1 | S2 | JDS_z | sft_val_loss | mmlu | arc_easy | downstream |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {fmt(row.get('core_metric'))} | {fmt(row.get('val_bpb'))} | "
            f"{fmt(row.get('s1_global'))} | {fmt(row.get('s2_global'))} | {fmt(row.get('jds_z'))} | "
            f"{fmt(row.get('sft_val_loss'))} | {fmt(row.get('mmlu_acc'))} | "
            f"{fmt(row.get('arc_easy_acc'))} | {fmt(row.get('downstream_score'))} |"
        )

    lines.extend([
        "",
        "## Correlations",
        "",
        "Permutation p-values are two-sided. With n=5, use this only as a pilot trend check.",
        "",
        "| target | signal | method | n | r/rho | 95% CI | p-value |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in corr:
        lines.append(
            f"| {row['target']} | {row['signal']} | {row['method']} | {row['n']} | "
            f"{fmt(row['r'])} | [{fmt(row['ci_lo'])}, {fmt(row['ci_hi'])}] | {fmt(row['p_value'])} |"
        )
    jds_downstream = [
        row for row in corr
        if row["target_key"] == "downstream_score" and row["signal_key"] == "jds_z"
    ]
    if jds_downstream:
        lines.extend([
            "",
            "## Quick Read",
            "",
        ])
        for row in jds_downstream:
            lines.append(f"- JDS_z vs downstream score ({row['method']}): {fmt(row['r'])}, p={fmt(row['p_value'])}.")

    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `downstream_summary.csv`",
        "- `downstream_correlations.csv`",
        "- `jds_z_vs_downstream.svg`",
    ])
    with open(os.path.join(output_dir, "downstream_correlation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def scale(values, lo, hi):
    vmin, vmax = min(values), max(values)
    if vmax == vmin:
        return [0.5 * (lo + hi)] * len(values)
    return [lo + (value - vmin) / (vmax - vmin) * (hi - lo) for value in values]


def color_for_config(config):
    colors = {
        "C1_diloco_baseline": "#4E79A7",
        "C2_a1b_weak": "#F28E2B",
        "C3_a1b_medium": "#59A14F",
        "C4_a1b_strong": "#E15759",
        "C5_a1b_medium_adamw_weak": "#B07AA1",
    }
    return colors.get(config, "#777777")


def write_svg_scatter(rows, path):
    plot_rows = [row for row in rows if row.get("downstream_score") is not None]
    if not plot_rows:
        return
    width, height = 800, 560
    ml, mr, mt, mb = 90, 40, 50, 80
    xs = [row["jds_z"] for row in plot_rows]
    ys = [row["downstream_score"] for row in plot_rows]
    x_scaled = scale(xs, ml, width - mr)
    y_scaled = scale(ys, height - mb, mt)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" font-family="Arial" font-size="20" text-anchor="middle">JDS_z vs Fast-SFT Downstream</text>',
        f'<line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#222"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-25}" font-family="Arial" font-size="14" text-anchor="middle">JDS_z = z(S1) + z(S2)</text>',
        f'<text x="24" y="{height/2}" font-family="Arial" font-size="14" text-anchor="middle" transform="rotate(-90 24 {height/2})">Fast-SFT downstream score</text>',
        f'<text x="{ml}" y="{height-mb+20}" font-family="Arial" font-size="11" text-anchor="middle">{min(xs):.4g}</text>',
        f'<text x="{width-mr}" y="{height-mb+20}" font-family="Arial" font-size="11" text-anchor="middle">{max(xs):.4g}</text>',
        f'<text x="{ml-10}" y="{height-mb+4}" font-family="Arial" font-size="11" text-anchor="end">{min(ys):.4g}</text>',
        f'<text x="{ml-10}" y="{mt+4}" font-family="Arial" font-size="11" text-anchor="end">{max(ys):.4g}</text>',
    ]
    for row, x, y in zip(plot_rows, x_scaled, y_scaled):
        parts.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color_for_config(row["config"])}" '
            f'fill-opacity="0.85"><title>{row["run_id"]}</title></circle>'
        )
    legend_y = 55
    for config in sorted({row["config"] for row in plot_rows}):
        parts.append(f'<circle cx="{width-250}" cy="{legend_y}" r="5" fill="{color_for_config(config)}"/>')
        parts.append(f'<text x="{width-238}" y="{legend_y+4}" font-family="Arial" font-size="11">{config}</text>')
        legend_y += 18
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-log-root", required=True)
    parser.add_argument("--sft-metrics-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    base_rows = load_base_rows(args.base_log_root)
    sft_rows = load_sft_rows(args.sft_metrics_dir)
    missing_sft = sorted(set(base_rows) - set(sft_rows))
    missing_base = sorted(set(sft_rows) - set(base_rows))
    rows = join_rows(base_rows, sft_rows)
    rows.sort(key=lambda row: (row["config"], row["seed"]))
    if not rows:
        raise SystemExit("No joined base/SFT rows found")

    os.makedirs(args.output_dir, exist_ok=True)
    write_csv(rows, os.path.join(args.output_dir, "downstream_summary.csv"))
    corr = correlation_table(
        rows,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        seed=args.seed,
    )
    write_correlation_csv(corr, os.path.join(args.output_dir, "downstream_correlations.csv"))
    write_report(rows, corr, args.output_dir, missing_base, missing_sft)
    write_svg_scatter(rows, os.path.join(args.output_dir, "jds_z_vs_downstream.svg"))
    print(f"wrote {len(rows)} joined downstream rows to {args.output_dir}")


if __name__ == "__main__":
    main()
