#!/usr/bin/env python3
"""汇总 results/ 下历次 benchmark，输出对比表（以 earliest baseline 为参照）。

用法:
    python3 scripts/compare_results.py            # 全部
    python3 scripts/compare_results.py baseline  # 只看目录名含 baseline 的
"""
import csv
import glob
import os
import re
import sys

RESULTS_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results"
)

# (Prefill, Decode, Conc) -> 场景名；计分只有 4K/16K 两场景
SCENARIOS = {
    (4096, 1024, 64): "4K",
    (16384, 1024, 64): "16K",
}


def load_rows():
    rows = []
    for run_dir in sorted(os.listdir(RESULTS_ROOT)):
        run_path = os.path.join(RESULTS_ROOT, run_dir)
        if not os.path.isdir(run_path):
            continue
        for csv_path in sorted(glob.glob(os.path.join(run_path, "summary_*.csv"))):
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        key = (int(row["Prefill"]), int(row["Decode"]), int(row["Conc"]))
                    except (KeyError, ValueError, TypeError):
                        continue
                    scenario = SCENARIOS.get(key, f"{key[0]//1024}K" if key[0] % 1024 == 0 else str(key[0]))
                    try:
                        rows.append({
                            "run": run_dir,
                            "scenario": scenario,
                            "total_tok_s": float(row["Total tok/s"]),
                            "output_tok_s": float(row["Output tok/s"]),
                            "ttft_ms": float(row["Mean TTFT (ms)"]),
                        })
                    except (KeyError, ValueError, TypeError):
                        continue
    return rows


def load_accuracy():
    """evalscope 精度摘要（run_baseline.sh 存的 accuracy.txt 最后一个数值）。"""
    acc = {}
    for run_dir in os.listdir(RESULTS_ROOT):
        path = os.path.join(RESULTS_ROOT, run_dir, "accuracy.txt")
        if not os.path.exists(path):
            continue
        numbers = re.findall(r"0\.\d+|1\.0+|0\b", open(path).read())
        if numbers:
            acc[run_dir] = numbers[-1]
    return acc


def main():
    filter_word = sys.argv[1] if len(sys.argv) > 1 else ""
    rows = [r for r in load_rows() if filter_word in r["run"]]
    if not rows:
        print(f"results/ 下暂无 summary CSV（过滤词: '{filter_word}'）。")
        return

    # 参照 = 最早的 baseline 目录；否则最早一次
    runs = sorted({r["run"] for r in rows})
    ref_run = next((r for r in runs if "baseline" in r.lower()), runs[0])

    acc = load_accuracy()
    print(f"参照基准: {ref_run}\n")
    header = f"{'run':<28}{'场景':<6}{'Total tok/s':>12}{'Δ%':>8}{'Out tok/s':>11}{'TTFT ms':>10}{'精度':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        ref = next(
            (b["total_tok_s"] for b in rows
             if b["run"] == ref_run and b["scenario"] == r["scenario"]),
            None,
        )
        delta = f"{(r['total_tok_s'] / ref - 1) * 100:+.1f}" if ref else "-"
        a = acc.get(r["run"], "-")
        print(
            f"{r['run']:<28}{r['scenario']:<6}"
            f"{r['total_tok_s']:>12.1f}{delta:>8}{r['output_tok_s']:>11.1f}"
            f"{r['ttft_ms']:>10.1f}{a:>8}"
        )

    print("\n红线: 精度 ≥ 0.95; 性能有效下限 = baseline−1%, TTFT 上限 = baseline+1%（信息文档 §4.5）")


if __name__ == "__main__":
    main()
