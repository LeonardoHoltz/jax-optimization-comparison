"""Build the study's XLSX report from the per-task test CSVs that tools/test.py wrote.

    pixi run python src/optim_env/tools/study_report.py --study reproduction

Writes outputs/reports/<study>/<study>.xlsx and <study>_summary.json (the numbers the
notes page shows).
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
REFS = Path(__file__).resolve().parents[1] / "configs" / "references"
# Regression scores are errors (lower is better); classification scores are accuracy.
METRICS = {"rmse": "lower", "accuracy": "higher"}

README = [
    ("What this file is", "Test-set results of the reproduction study: each task trained with the paper's model and a smaller one, by gradient descent and by every evolution strategy (ES), each with and without tuning."),
    ("Overview sheet", "One row per task. Paper = published score. Each cell is mean ± standard deviation over 5 seeds (training repeats with different random starts)."),
    ("Task sheets", "One row per model + training method + optimizer. 'untuned' uses fixed default settings; 'tuned' uses the best settings a 20-trial Optuna search found on the validation split."),
    ("rmse", "Root mean squared error, in the target's own units (California: $100k; House: $). Lower is better."),
    ("accuracy", "Fraction of test examples classified correctly. Higher is better."),
    ("gap to paper", "Our tuned mean minus the paper's number. For rmse a negative gap is better; for accuracy a positive one."),
    ("reproduced", "Yes when the gradient run on the paper's model lands within the paper's std (or 1% when the paper gives none)."),
    ("Skipped", "Combinations that cannot run, with the reason."),
]


def references() -> dict:
    refs = {k: dict(v) for k, v in json.loads((REFS / "tabm_mlp.json").read_text()).items()}
    for v in refs.values():
        v["source"] = "TabM paper (Gorishniy et al. 2024), 15 published seeds"
    refs.update(yaml.safe_load((REFS / "papers.yaml").read_text()))
    return refs


def fmt(mean, std, n):
    if pd.isna(mean):
        return ""
    return f"{mean:.4f}" if n < 2 or pd.isna(std) else f"{mean:.4f} ± {std:.4f}"


def task_table(df: pd.DataFrame, metric: str, ref: dict | None) -> pd.DataFrame:
    rows = []
    for (arch, method, algo), g in df.groupby(["arch", "method", "algorithm"]):
        row = {"model": arch, "training": "gradient" if method == "SGD" else "evolution strategy", "optimizer": algo}
        for tuning in ("untuned", "tuned"):
            s = g[g.tuning == tuning][metric]
            row[f"{tuning} mean"] = s.mean() if len(s) else None
            row[f"{tuning} std"] = s.std() if len(s) > 1 else None
            row[f"{tuning} seeds"] = len(s)
        if ref is not None and row["tuned mean"] is not None:
            row["gap to paper (tuned)"] = row["tuned mean"] - ref["mean"]
        rows.append(row)
    out = pd.DataFrame(rows)
    better_first = METRICS[metric] == "lower"
    return out.sort_values(["model", "training", "tuned mean"], ascending=[True, False, better_first])


def reproduced(df, metric, ref) -> str:
    g = df[(df.arch == "paper") & (df.method == "SGD") & (df.tuning == "untuned")][metric]
    if ref is None or g.empty:
        return ""
    tol = ref.get("std") or abs(ref["mean"]) * 0.01
    return "yes" if abs(g.mean() - ref["mean"]) <= tol else "no"


def best(df, metric, arch, method, tuning):
    g = df[(df.arch == arch) & (df.method == method) & (df.tuning == tuning)]
    if g.empty:
        return "", ""
    means = g.groupby("algorithm")[metric].agg(["mean", "std", "count"])
    pick = means["mean"].idxmin() if METRICS[metric] == "lower" else means["mean"].idxmax()
    m = means.loc[pick]
    return fmt(m["mean"], m["std"], m["count"]), pick


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", default="reproduction")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "study" / "reproduction.yaml"))
    args = parser.parse_args()

    report_dir = ROOT / "outputs" / "reports" / args.study
    study_cfg = yaml.safe_load(Path(args.config).read_text())
    refs = references()
    summary, sheets, all_runs = [], {}, []

    for task, t in study_cfg["tasks"].items():
        csv_path = report_dir / f"{task}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        variant = df["variant"].str.replace("pilot-", "", regex=False)
        df["arch"] = variant.str.split("_").str[0]
        df["tuning"] = variant.str.split("_").str[1].map({"baseline": "untuned", "hpo": "tuned"})
        df["task"] = task
        all_runs.append(df)
        ref = refs.get(t["reference"]["key"])
        metric = "rmse" if "rmse" in df and df["rmse"].notna().any() else "accuracy"

        sheets[task] = task_table(df, metric, ref)
        row = {"task": task, "metric": metric, "lower is better": METRICS[metric] == "lower",
               "paper": ref["mean"] if ref else None, "reproduced": reproduced(df, metric, ref)}
        for arch in ("paper", "simple"):
            for tuning in ("untuned", "tuned"):
                row[f"{arch} gradient {tuning}"] = best(df, metric, arch, "SGD", tuning)[0]
                val, algo = best(df, metric, arch, "NE", tuning)
                row[f"{arch} best ES {tuning}"] = f"{val} ({algo})" if algo else ""
        summary.append(row)

    if not summary:
        raise SystemExit(f"No task CSVs in {report_dir}")

    skipped_path = ROOT / "outputs" / "study" / args.study / "skipped.json"
    skipped = json.loads(skipped_path.read_text()) if skipped_path.exists() else {}
    timings_path = ROOT / "outputs" / "study" / args.study / "timings.csv"

    out = report_dir / f"{args.study}.xlsx"
    # Written aside, then renamed: two study drivers may rebuild the report at once, and a
    # reader must never see a half-written file.
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        pd.DataFrame(README, columns=["item", "meaning"]).to_excel(xw, sheet_name="Read me", index=False)
        pd.DataFrame(summary).to_excel(xw, sheet_name="Overview", index=False)
        for task, table in sheets.items():
            table.to_excel(xw, sheet_name=task[:31], index=False)
        pd.concat(all_runs).to_excel(xw, sheet_name="All runs", index=False)
        pd.DataFrame([{"key": k, **v} for k, v in refs.items()]).to_excel(xw, sheet_name="References", index=False)
        pd.DataFrame([{"combination": k, "reason": v} for k, v in skipped.items()]).to_excel(xw, sheet_name="Skipped", index=False)
        if timings_path.exists():
            pd.read_csv(timings_path).to_excel(xw, sheet_name="Run times", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 80)
    os.replace(tmp, out)

    (report_dir / f"{args.study}_summary.json").write_text(json.dumps(
        {"overview": summary, "tasks": {k: v.to_dict("records") for k, v in sheets.items()}, "skipped": skipped},
        indent=2, default=str))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
