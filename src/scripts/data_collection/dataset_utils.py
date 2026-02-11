# dataset_utils.py
from __future__ import annotations

import os
import time
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import pandas as pd


# ----------------------------
# Auth / GitHub API
# ----------------------------
def get_github_token() -> str:
    """
    Priority:
      1) config.secret.GITHUB_TOKEN (if you have it)
      2) environment variable GITHUB_TOKEN
    """
    try:
        from config.secret import GITHUB_TOKEN  # type: ignore
        if isinstance(GITHUB_TOKEN, str) and GITHUB_TOKEN.strip():
            return GITHUB_TOKEN.strip()
    except Exception:
        pass

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Missing GitHub token. Set env var GITHUB_TOKEN or provide config/secret.py with GITHUB_TOKEN."
        )
    return token


def _headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "godot-issue-triager-dataset-builder",
    }


def request_issues(
    repo: str,
    token: str,
    *,
    state: str = "all",
    per_page: int = 100,
    target_issues: int = 10_000,
    max_pages: int = 500,
    timeout_s: int = 30,
    sleep_s: float = 0.0,
) -> pd.DataFrame:
    """
    Fetch issues from the GitHub Issues endpoint. Note:
    - This endpoint can return PRs too; we filter those out while collecting.

    Returns a DataFrame of raw issue JSON objects.
    """
    base_url = f"https://api.github.com/repos/{repo}/issues"
    params = {"per_page": per_page, "state": state}

    url = base_url
    all_rows: List[Dict[str, Any]] = []
    page = 0

    while url and page < max_pages and len(all_rows) < target_issues:
        page += 1
        print(f"[collect] page={page} issues_collected={len(all_rows)} url={url}")

        # For the first request, use params. For next-page URLs, params are already embedded.
        use_params = params if url == base_url else None

        r = requests.get(url, params=use_params, headers=_headers(token), timeout=timeout_s)

        if r.status_code == 403:
            # Possible rate limit. Provide helpful debug.
            reset = r.headers.get("X-RateLimit-Reset")
            remaining = r.headers.get("X-RateLimit-Remaining")
            msg = ""
            try:
                msg = str(r.json())
            except Exception:
                msg = r.text
            raise RuntimeError(
                f"GitHub API 403 (possible rate limit). Remaining={remaining}, Reset={reset}. Response={msg}"
            )

        if not r.ok:
            try:
                err = r.json()
            except Exception:
                err = r.text
            raise RuntimeError(f"GitHub request failed {r.status_code}: {err}")

        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Expected list from /issues, got {type(data)}: {data}")

        # Keep only actual issues (exclude PRs)
        for item in data:
            if isinstance(item, dict) and ("pull_request" not in item):
                all_rows.append(item)
                if len(all_rows) >= target_issues:
                    break

        url = r.links.get("next", {}).get("url")

        if sleep_s > 0:
            time.sleep(sleep_s)

    return pd.DataFrame(all_rows)


# ----------------------------
# Label processing
# ----------------------------
DEFAULT_LABELS_TO_DROP = {
    "archived", "confirmed", "for pr meeting", "good first issue",
    "needs testing", "needs work", "spam", "salvageable", "tracker",
    # cherrypicks
    "cherrypick:3.5", "cherrypick:3.6", "cherrypick:3.x",
    "cherrypick:4.2", "cherrypick:4.3", "cherrypick:4.4",
    "cherrypick:4.5", "cherrypick:4.6",
}


def labels_to_names(labels: Any, labels_to_drop: Optional[set] = None) -> List[str]:
    """
    Accepts:
      - list[dict] from GitHub API (each dict has "name") OR
      - list[str] if already normalized

    Returns list[str] label names, filtered.
    """
    drop = labels_to_drop if labels_to_drop is not None else DEFAULT_LABELS_TO_DROP

    if not isinstance(labels, list):
        return []

    out: List[str] = []
    for x in labels:
        name: Optional[str] = None
        if isinstance(x, dict):
            name = x.get("name")
        elif isinstance(x, str):
            name = x
        if not name:
            continue
        if name in drop:
            continue
        out.append(name)
    return out


def compute_top_topic_strings(
    df: pd.DataFrame,
    labels_col: str,
    *,
    min_support: int = 200,
    top_n: int = 17,
    topic_prefix: str = "topic:",
) -> List[str]:
    """
    df[labels_col] must be list[str] per row.
    Returns list like ["topic:audio", ...]
    """
    exploded = df[labels_col].explode()
    exploded = exploded.dropna()

    # Ensure string dtype
    exploded = exploded.astype(str)

    topic_counts = exploded[exploded.str.startswith(topic_prefix)].value_counts()

    eligible = topic_counts[topic_counts >= min_support]
    return eligible.head(top_n).index.tolist()


# ----------------------------
# Normalization schema
# ----------------------------
RAW_ISSUE_TYPES = {"bug", "discussion", "enhancement", "documentation"}
ISSUE_TYPE_NORMALIZE = {
    "bug": "bug",
    "discussion": "discussion",
    "enhancement": "feature_request",
    "documentation": "docs",
}
ISSUE_TYPE_PRECEDENCE = ["bug", "feature_request", "docs", "discussion"]

PLATFORM_MAP = {
    "platform:windows": "windows",
    "platform:android": "android",
    "platform:linuxbsd": "linuxbsd",
    "platform:macos": "macos",
    "platform:web": "web",
}

IMPACTS = {"crash", "usability", "regression", "performance"}

TOPIC_PREFIX = "topic:"
TOPIC_OTHER = "other"


def normalize_df_drop_labels(
    df: pd.DataFrame,
    top_topic_strings: Sequence[str],
    *,
    labels_col: str = "labels",
    dedupe: bool = True,
    default_issue_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Produces:
      - issue_type: single string (normalized), chosen by precedence
      - platform: list[str] (prefix stripped)
      - impact: list[str]
      - topic: list[str] (top topics kept, others -> "other", prefix stripped)

    Drops labels_col.
    """
    top_topics_set = set(top_topic_strings)
    precedence_rank = {name: i for i, name in enumerate(ISSUE_TYPE_PRECEDENCE)}

    def _process(labels: Any) -> Tuple[Optional[str], List[str], List[str], List[str]]:
        if not isinstance(labels, list):
            labels = []

        issue_candidates: set = set()
        platform: List[str] = []
        impact: List[str] = []
        topic: List[str] = []

        seen_platform = set()
        seen_impact = set()
        seen_topic = set()

        needs_other = False

        for lab in labels:
            # lab may be dict if earlier steps forgot to convert
            if isinstance(lab, dict):
                lab = lab.get("name")

            if not isinstance(lab, str):
                continue

            # issue type candidate
            if lab in RAW_ISSUE_TYPES:
                issue_candidates.add(ISSUE_TYPE_NORMALIZE[lab])

            # platform
            if lab in PLATFORM_MAP:
                p = PLATFORM_MAP[lab]
                if (not dedupe) or (p not in seen_platform):
                    platform.append(p)
                    seen_platform.add(p)

            # impact
            if lab in IMPACTS:
                if (not dedupe) or (lab not in seen_impact):
                    impact.append(lab)
                    seen_impact.add(lab)

            # topic
            if lab.startswith(TOPIC_PREFIX) and len(lab) > len(TOPIC_PREFIX):
                if lab in top_topics_set:
                    t = lab[len(TOPIC_PREFIX):]
                    if (not dedupe) or (t not in seen_topic):
                        topic.append(t)
                        seen_topic.add(t)
                else:
                    needs_other = True

        if needs_other:
            if (not dedupe) or (TOPIC_OTHER not in seen_topic):
                topic.append(TOPIC_OTHER)

        if issue_candidates:
            issue_type = min(issue_candidates, key=lambda x: precedence_rank.get(x, float("inf")))
        else:
            issue_type = default_issue_type

        return issue_type, platform, impact, topic

    out = df.copy()
    extracted = out[labels_col].apply(
        lambda x: pd.Series(_process(x), index=["issue_type", "platform", "impact", "topic"])
    )
    out = pd.concat([out.drop(columns=[labels_col]), extracted], axis=1)
    return out


# ----------------------------
# Time + text fields
# ----------------------------
def parse_created_at_utc(df: pd.DataFrame, col: str = "created_at") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out


FENCED_CODE_REMOVE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE = re.compile(r"`([^`]+)`")
URLS = re.compile(r"https?://\S+")
MD_LEADERS = re.compile(r"(?m)^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|>\s+)")


def clean_markdown(
    text: Any,
    *,
    remove_code_blocks: bool = True,
    keep_urls: bool = True,
    url_placeholder: str = "URL",
) -> str:
    """
    Light cleaning:
    - optional remove fenced code blocks
    - remove inline backticks but keep contents
    - strip markdown leaders (#, bullets, quote marker) from line starts
    - optional URL replacement (if keep_urls=False)
    - collapse whitespace to single spaces
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        # handle pandas/NA/etc safely
        if pd.isna(text):
            return ""
        text = str(text)

    t = text
    if remove_code_blocks:
        t = FENCED_CODE_REMOVE.sub(" ", t)

    t = INLINE_CODE.sub(r"\1", t)
    t = MD_LEADERS.sub("", t)

    if not keep_urls:
        t = URLS.sub(f" {url_placeholder} ", t)

    t = re.sub(r"\s+", " ", t).strip()
    return t


def add_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text_raw"] = out["title"].fillna("") + "\n\n" + out["body"].fillna("")
    out["text_clean"] = out["text_raw"].apply(clean_markdown)
    return out


# ----------------------------
# Splits + exports
# ----------------------------
def time_split_by_created_at(
    df: pd.DataFrame,
    *,
    created_col: str = "created_at",
    train_frac: float = 0.70,
    val_frac: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split: oldest -> newest.
    Default: train 70%, val 10%, test 20% (matches "tune=80%, test=20%").
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1):
        raise ValueError("train_frac and val_frac must be between 0 and 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1 (so test has remainder).")

    tmp = df.dropna(subset=[created_col]).sort_values(created_col).reset_index(drop=True)
    n = len(tmp)
    train_end = int(train_frac * n)
    val_end = int((train_frac + val_frac) * n)

    train_df = tmp.iloc[:train_end].copy()
    val_df = tmp.iloc[train_end:val_end].copy()
    test_df = tmp.iloc[val_end:].copy()
    return train_df, val_df, test_df


def build_label_vocab(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "issue_type": sorted([x for x in df["issue_type"].dropna().unique().tolist()]),
        "platform": sorted({p for row in df["platform"] if isinstance(row, list) for p in row}),
        "impact": sorted({x for row in df["impact"] if isinstance(row, list) for x in row}),
        "topic": sorted({t for row in df["topic"] if isinstance(row, list) for t in row}),
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, records: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)


def write_json_pretty(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Ensure datetimes are JSON-serializable
    out = df.copy()
    if "created_at" in out.columns:
        # store ISO string
        out["created_at"] = out["created_at"].apply(lambda x: x.isoformat() if pd.notna(x) else None)
    return out.to_dict(orient="records")
