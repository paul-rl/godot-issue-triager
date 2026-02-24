# analyze_dataset.py
from __future__ import annotations

import argparse
import os
import json
from typing import List
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json_records(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def barh_counts(counts: pd.Series, title: str, outpath: str, top_n: int = 25):
    s = counts.head(top_n).sort_values()
    plt.figure(figsize=(10, max(4, 0.35 * len(s))))
    plt.barh(s.index.astype(str), s.values)
    plt.xlabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_coverage(df: pd.DataFrame, outpath: str):
    def has_any_list(x):
        return isinstance(x, list) and len(x) > 0

    coverage = pd.Series({
        "has_topic": df["topic"].apply(has_any_list).mean(),
        "has_platform": df["platform"].apply(has_any_list).mean(),
        "has_impact": df["impact"].apply(has_any_list).mean(),
        "has_issue_type": df["issue_type"].notna().mean(),
    }) * 100

    plt.figure(figsize=(7, 4))
    plt.bar(coverage.index, coverage.values)
    plt.ylabel("Percent of issues (%)")
    plt.title("Label coverage")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_topics_per_issue(df: pd.DataFrame, outpath: str):
    topic_per_issue = df["topic"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    mx = int(topic_per_issue.max()) if len(topic_per_issue) else 0

    plt.figure(figsize=(7, 4))
    plt.hist(topic_per_issue, bins=list(range(0, mx + 2)))
    plt.xlabel("# topics on an issue")
    plt.ylabel("Count")
    plt.title("Topics per issue")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_issues_over_time(df: pd.DataFrame, outpath: str, date_col: str = "created_at"):
    if date_col not in df.columns:
        return
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty:
        return

    monthly = tmp.set_index(date_col).resample("MS").size()

    plt.figure(figsize=(10, 4))
    plt.plot(monthly.index, monthly.values)
    plt.xlabel("Month")
    plt.ylabel("# issues")
    plt.title("Issues over time (monthly)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_other_rate_over_time(df: pd.DataFrame, outpath: str, date_col: str = "created_at"):
    if date_col not in df.columns:
        return
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty:
        return

    tmp["has_other"] = tmp["topic"].apply(lambda xs: isinstance(xs, list) and ("other" in xs))
    monthly = tmp.set_index(date_col)["has_other"].resample("MS").mean() * 100

    plt.figure(figsize=(10, 4))
    plt.plot(monthly.index, monthly.values)
    plt.xlabel("Month")
    plt.ylabel("% issues with topic='other'")
    plt.title("Rate of topic='other' over time")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze exported dataset JSONs and save plots.")
    parser.add_argument("--indir", type=str, default="data/processed", help="Directory containing train/val/test.json")
    parser.add_argument("--outdir", type=str, default="", help="Directory to write plots (default: <indir>/plots)")
    parser.add_argument("--top-n", type=int, default=25, help="Top N for bar charts")
    parser.add_argument("--fill-issue-type", type=str, default="no_type", help="Fill missing issue_type for plots only")
    args = parser.parse_args()

    indir = args.indir
    outdir = args.outdir or os.path.join(indir, "plots")
    ensure_dir(outdir)

    train_path = os.path.join(indir, "train.json")
    val_path = os.path.join(indir, "val.json")
    test_path = os.path.join(indir, "test.json")

    train_df = read_json_records(train_path)
    val_df = read_json_records(val_path)
    test_df = read_json_records(test_path)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Plot-friendly issue_type
    if "issue_type" in df.columns and args.fill_issue_type:
        df["issue_type_plot"] = df["issue_type"].fillna(args.fill_issue_type)
    else:
        df["issue_type_plot"] = df.get("issue_type")

    # ---- Core distributions ----
    if "topic" in df.columns:
        topic_counts = df["topic"].explode().dropna().value_counts()
        barh_counts(topic_counts, "Top topics", os.path.join(outdir, "top_topics.png"), top_n=args.top_n)

    if "issue_type_plot" in df.columns:
        issue_counts = df["issue_type_plot"].value_counts(dropna=False)
        barh_counts(issue_counts, "Issue type distribution", os.path.join(outdir, "issue_types.png"), top_n=50)

    if "platform" in df.columns:
        platform_counts = df["platform"].explode().dropna().value_counts()
        barh_counts(platform_counts, "Platform distribution", os.path.join(outdir, "platforms.png"), top_n=50)

    if "impact" in df.columns:
        impact_counts = df["impact"].explode().dropna().value_counts()
        barh_counts(impact_counts, "Impact distribution", os.path.join(outdir, "impacts.png"), top_n=50)

    # ---- Health/coverage ----
    for col in ["topic", "platform", "impact", "issue_type"]:
        if col not in df.columns:
            print(f"[analyze] missing expected column: {col}")

    plot_coverage(df, os.path.join(outdir, "coverage.png"))
    plot_topics_per_issue(df, os.path.join(outdir, "topics_per_issue.png"))
    plot_issues_over_time(df, os.path.join(outdir, "issues_over_time.png"))
    plot_other_rate_over_time(df, os.path.join(outdir, "other_rate_over_time.png"))

    # ---- Write a quick summary ----
    summary_lines: List[str] = []
    summary_lines.append(f"splits: train={len(train_df)} val={len(val_df)} test={len(test_df)} total={len(df)}")

    # Coverage numbers
    def has_any_list(x):
        return isinstance(x, list) and len(x) > 0

    summary_lines.append(f"% has topic: {df['topic'].apply(has_any_list).mean()*100:.2f}")
    summary_lines.append(f"% has platform: {df['platform'].apply(has_any_list).mean()*100:.2f}")
    summary_lines.append(f"% has impact: {df['impact'].apply(has_any_list).mean()*100:.2f}")
    summary_lines.append(f"% has issue_type: {df['issue_type'].notna().mean()*100:.2f}")

    if "topic" in df.columns:
        other_share = df["topic"].apply(lambda xs: isinstance(xs, list) and ("other" in xs)).mean() * 100
        summary_lines.append(f"% issues with topic='other': {other_share:.2f}")

    summary_path = os.path.join(outdir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"[analyze] wrote plots + summary to: {outdir}")


if __name__ == "__main__":
    main()
