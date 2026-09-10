#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


DATASETS = ("caltech101", "ucf101", "fgvc_aircraft")
GROUPS = ("baseline", "single_view_kd", "no_kd", "v2_no_fg", "full")
SEEDS = (1, 2, 3)
ACCURACY_RE = re.compile(r"accuracy:\s*([0-9]+(?:\.[0-9]+)?)")


def harmonic_mean(base, novel):
    if base is None or novel is None:
        return None
    if base + novel == 0:
        return 0.0
    return 2 * base * novel / (base + novel)


def extract_accuracy(log_file):
    if not log_file.exists():
        return None
    matches = ACCURACY_RE.findall(log_file.read_text(errors="ignore"))
    if not matches:
        return None
    return float(matches[-1])


def mean_or_none(values):
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def fmt(value):
    if value is None:
        return "MISSING"
    return f"{value:.2f}"


def collect_rows(root):
    rows = []
    for dataset in DATASETS:
        for group in GROUPS:
            base_values = []
            novel_values = []
            missing = []

            for seed in SEEDS:
                run_dir = root / dataset / group / f"seed{seed}"
                base = extract_accuracy(run_dir / "eval_base" / "log.txt")
                novel = extract_accuracy(run_dir / "eval_new" / "log.txt")
                base_values.append(base)
                novel_values.append(novel)
                if base is None:
                    missing.append(f"seed{seed}:base")
                if novel is None:
                    missing.append(f"seed{seed}:novel")

            base_mean = mean_or_none(base_values)
            novel_mean = mean_or_none(novel_values)
            hm = harmonic_mean(base_mean, novel_mean)
            rows.append(
                {
                    "dataset": dataset,
                    "group": group,
                    "base": base_mean,
                    "novel": novel_mean,
                    "hm": hm,
                    "status": "OK" if not missing else "MISSING:" + ",".join(missing),
                }
            )
    return rows


def write_csv(rows, path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("dataset", "group", "base", "novel", "hm", "status"),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "base": fmt(row["base"]),
                    "novel": fmt(row["novel"]),
                    "hm": fmt(row["hm"]),
                }
            )


def write_markdown(rows, path):
    lines = [
        "# Table 4.3 Ablation Results",
        "",
        "| Dataset | Group | Base | Novel | HM | Status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {group} | {base} | {novel} | {hm} | {status} |".format(
                dataset=row["dataset"],
                group=row["group"],
                base=fmt(row["base"]),
                novel=fmt(row["novel"]),
                hm=fmt(row["hm"]),
                status=row["status"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="output/ablation/table_4_3")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(root)
    write_csv(rows, root / "table_4_3_results.csv")
    write_markdown(rows, root / "table_4_3_results.md")

    missing = [row for row in rows if row["status"] != "OK"]
    if missing:
        print(f"Summary written with {len(missing)} incomplete rows: {root}")
    else:
        print(f"Summary written: {root}")


if __name__ == "__main__":
    main()
