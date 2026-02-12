# collect_dataset.py
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import pandas as pd

from dataset_utils import (
    get_github_token,
    request_issues,
    labels_to_names,
    compute_top_topic_strings,
    normalize_df_drop_labels,
    parse_created_at_utc,
    add_text_fields,
    time_split_by_created_at,
    build_label_vocab,
    ensure_dir,
    write_json,
    write_json_pretty,
    df_to_records,
    DEFAULT_LABELS_TO_DROP,
)


def main():
    parser = argparse.ArgumentParser(description="Collect + preprocess Godot issues into train/val/test JSON.")
    parser.add_argument("--repo", type=str, default="godotengine/godot", help="GitHub repo as owner/name")
    parser.add_argument("--outdir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--target", type=int, default=10_000, help="Target number of issues to collect (excluding PRs)")
    parser.add_argument("--max-pages", type=int, default=500, help="Max pages to fetch")
    parser.add_argument("--min-support", type=int, default=200, help="Min support for keeping a topic:* label")
    parser.add_argument("--top-n", type=int, default=17, help="Number of top topics to keep (15-25 recommended)")
    parser.add_argument("--train-frac", type=float, default=0.70, help="Train fraction (time-based, oldest first)")
    parser.add_argument("--val-frac", type=float, default=0.10, help="Val fraction (time-based, after train)")
    parser.add_argument("--fill-issue-type", type=str, default="", help="If set, fill missing issue_type with this value on export")
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between API requests (rate-limit friendly)")
    args = parser.parse_args()

    token = get_github_token()
    ensure_dir(args.outdir)

    # 1) fetch
    raw_df = request_issues(
        repo=args.repo,
        token=token,
        target_issues=args.target,
        max_pages=args.max_pages,
        sleep_s=args.sleep,
    )

    # Keep minimal columns if present
    keep_cols = [c for c in ["id", "title", "body", "created_at", "labels"] if c in raw_df.columns]
    df = raw_df[keep_cols].copy()

    # 2) labels dict -> names (strings)
    df["labels"] = df["labels"].apply(lambda x: labels_to_names(x, labels_to_drop=set(DEFAULT_LABELS_TO_DROP)))

    # optional: drop rows with no remaining labels (keeps dataset focused)
    df = df[df["labels"].map(len) > 0].reset_index(drop=True)

    # 3) compute top topics
    top_topic_strings = compute_top_topic_strings(
        df, "labels", min_support=args.min_support, top_n=args.top_n
    )

    # 4) normalize (drops labels)
    final_df = normalize_df_drop_labels(df, top_topic_strings, labels_col="labels", default_issue_type=None)

    # 5) created_at parsing
    final_df = parse_created_at_utc(final_df, "created_at")
    final_df = final_df.dropna(subset=["created_at"]).reset_index(drop=True)

    # 6) add text fields
    final_df = add_text_fields(final_df)

    # 7) time split
    train_df, val_df, test_df = time_split_by_created_at(
        final_df, created_col="created_at", train_frac=args.train_frac, val_frac=args.val_frac
    )

    # 8) vocab + metadata
    label_vocab = build_label_vocab(final_df)

    metadata: Dict[str, Any] = {
        "repo": args.repo,
        "target_issues_requested": args.target,
        "issues_after_label_filter": int(len(final_df)),
        "split_sizes": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "top_topics": list(top_topic_strings),
        "min_support": args.min_support,
        "top_n": args.top_n,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": 1.0 - (args.train_frac + args.val_frac),
        "created_at_min": train_df["created_at"].min().isoformat() if len(train_df) else None,
        "created_at_max": test_df["created_at"].max().isoformat() if len(test_df) else None,
    }

    # For export, optionally fill missing issue_type
    export_train = train_df.copy()
    export_val = val_df.copy()
    export_test = test_df.copy()

    if args.fill_issue_type:
        export_train["issue_type"] = export_train["issue_type"].fillna(args.fill_issue_type)
        export_val["issue_type"] = export_val["issue_type"].fillna(args.fill_issue_type)
        export_test["issue_type"] = export_test["issue_type"].fillna(args.fill_issue_type)

    # 9) write files
    write_json(os.path.join(args.outdir, "train.json"), df_to_records(export_train))
    write_json(os.path.join(args.outdir, "val.json"), df_to_records(export_val))
    write_json(os.path.join(args.outdir, "test.json"), df_to_records(export_test))
    write_json_pretty(os.path.join(args.outdir, "label_vocab.json"), label_vocab)
    write_json_pretty(os.path.join(args.outdir, "metadata.json"), metadata)

    print(f"[collect] wrote outputs to: {args.outdir}")
    print(f"[collect] splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"[collect] top topics kept={len(top_topic_strings)} min_support={args.min_support} top_n={args.top_n}")


if __name__ == "__main__":
    main()
