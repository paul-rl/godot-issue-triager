import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
from config.secret import GITHUB_TOKEN


# Fetch issues
def request_data(target_issues=10_000, max_pages=500):
    params = {"per_page": 100, "state": "all"}
    url = "https://api.github.com/repos/godotengine/godot/issues"

    all_rows = []
    fetched_pages = 0

    while url and fetched_pages < max_pages and len(all_rows) < target_issues:
        fetched_pages += 1
        print(f"Processing page {fetched_pages} | collected={len(all_rows)}")

        r = requests.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}"},
            timeout=30,
        )

        if not r.ok:
            try:
                err = r.json()
            except ValueError:
                err = r.text
            raise RuntimeError(f"Request failed {r.status_code}: {err}")

        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Expected list from /issues, got {type(data)}: {data}")

        # keep only actual issues (drop PRs immediately)
        for item in data:
            if "pull_request" not in item:
                all_rows.append(item)
                if len(all_rows) >= target_issues:
                    break

        url = r.links.get("next", {}).get("url")

    return pd.DataFrame(all_rows)


df = request_data(target_issues=20_000)
print("Fetched issues:", len(df))
df

# Keep only needed columns
KEEP_COLS = ["id", "title", "body", "created_at", "labels"]
df = df[KEEP_COLS].copy()
df

# 3) Convert labels: list[dict] -> list[str], drop noise labels
LABELS_TO_DROP = {
    "archived", "confirmed", "for pr meeting", "good first issue",
    "needs testing", "needs work", "spam", "salvageable", "tracker",
    "cherrypick:3.5", "cherrypick:3.6", "cherrypick:3.x",
    "cherrypick:4.2", "cherrypick:4.3", "cherrypick:4.4",
    "cherrypick:4.5", "cherrypick:4.6",
}


def labels_to_names(labels):
    """
    Accepts:
      - list[dict] from GitHub API OR
      - list[str] if already processed
    Returns:
      - list[str] label names (filtered)
    """
    if not isinstance(labels, list):
        return []
    out = []
    for x in labels:
        if isinstance(x, dict):
            name = x.get("name")
        elif isinstance(x, str):
            name = x
        else:
            continue

        if name and name not in LABELS_TO_DROP:
            out.append(name)
    return out


df["labels"] = df["labels"].apply(labels_to_names)

# 4) Compute top_topic_strings (min support + top N)
MIN_SUPPORT = 200
N = 20

labels_exploded = df["labels"].explode()
topic_counts = (
    labels_exploded[
        labels_exploded.notna()
        & labels_exploded.astype(str).str.startswith("topic:")
    ]
    .value_counts()
)

eligible = topic_counts[topic_counts >= MIN_SUPPORT]
top_topic_strings = eligible.head(N).index.tolist()

print("Top topics:", len(top_topic_strings))
print(top_topic_strings[:10])

# 5) Normalize into issue_type/platform/impact/topic and drop labels
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


def normalize_df_drop_labels(df: pd.DataFrame, labels_col="labels", default_issue_type=None):
    top_topics_set = set(top_topic_strings)
    precedence_rank = {name: i for i, name in enumerate(ISSUE_TYPE_PRECEDENCE)}

    def _process(labels):
        if not isinstance(labels, list):
            labels = []

        issue_candidates = set()
        platform, impact, topic = [], [], []

        seen_platform, seen_impact, seen_topic = set(), set(), set()
        needs_other = False

        for lab in labels:
            if not isinstance(lab, str):
                continue

            # issue type candidates
            if lab in RAW_ISSUE_TYPES:
                issue_candidates.add(ISSUE_TYPE_NORMALIZE[lab])

            # platform list (stripped)
            if lab in PLATFORM_MAP:
                p = PLATFORM_MAP[lab]
                if p not in seen_platform:
                    platform.append(p)
                    seen_platform.add(p)

            # impact list
            if lab in IMPACTS:
                if lab not in seen_impact:
                    impact.append(lab)
                    seen_impact.add(lab)

            # topic list (top only, else other)
            if lab.startswith(TOPIC_PREFIX) and len(lab) > len(TOPIC_PREFIX):
                if lab in top_topics_set:
                    t = lab[len(TOPIC_PREFIX):]
                    if t not in seen_topic:
                        topic.append(t)
                        seen_topic.add(t)
                else:
                    needs_other = True

        if needs_other and TOPIC_OTHER not in seen_topic:
            topic.append(TOPIC_OTHER)

        # choose one issue_type by precedence
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


final_df = normalize_df_drop_labels(df)
final_df.head()

# 6) Parse created_at
final_df["created_at"] = pd.to_datetime(final_df["created_at"], utc=True, errors="coerce")
final_df = final_df.dropna(subset=["created_at"]).reset_index(drop=True)
final_df

# 7) Clean markdown
FENCED_CODE_REMOVE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE = re.compile(r"`([^`]+)`")
URLS = re.compile(r"https?://\S+")
MD_LEADERS = re.compile(r"(?m)^\s{0,3}(?:#{1,6}\s+|[-*+]\s+|>\s+)")


def clean_markdown(text, remove_code_blocks=True, keep_urls=True, url_placeholder="URL"):
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    t = text
    if remove_code_blocks:
        t = FENCED_CODE_REMOVE.sub(" ", t)

    t = INLINE_CODE.sub(r"\1", t)
    t = MD_LEADERS.sub("", t)

    if not keep_urls:
        t = URLS.sub(f" {url_placeholder} ", t)

    # IMPORTANT: collapse whitespace to ONE space (not empty)
    t = re.sub(r"\s+", " ", t).strip()
    return t


final_df["text_raw"] = final_df["title"].fillna("") + "\n\n" + final_df["body"].fillna("")
final_df["text_clean"] = final_df["text_raw"].apply(clean_markdown)

# 8) Label vocab
label_vocab = {
    "issue_type": sorted(final_df["issue_type"].dropna().unique().tolist()),
    "platform": sorted({p for row in final_df["platform"] for p in row}),
    "impact": sorted({x for row in final_df["impact"] for x in row}),
    "topic": sorted({t for row in final_df["topic"] for t in row}),
}

# 9) Time-based split + export
final_df = final_df.sort_values("created_at").reset_index(drop=True)

n = len(final_df)
train_end = int(0.70 * n)
val_end = int(0.80 * n)  # tune=80%, test=20%

train_df = final_df.iloc[:train_end]
val_df = final_df.iloc[train_end:val_end]
test_df = final_df.iloc[val_end:]

train_df.to_json("train.json", orient="records")
val_df.to_json("val.json", orient="records")
test_df.to_json("test.json", orient="records")

with open("label_vocab.json", "w") as f:
    json.dump(label_vocab, f, indent=2)

print("Wrote train.json, val.json, test.json, label_vocab.json")


def barh_counts(counts: pd.Series, title: str, top_n: int = 25):
    s = counts.head(top_n).sort_values()
    plt.figure(figsize=(10, max(4, 0.35 * len(s))))
    plt.barh(s.index.astype(str), s.values)
    plt.xlabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()


topic_counts = final_df["topic"].explode().dropna().value_counts()
barh_counts(topic_counts, "Top topics (topic labels)", top_n=25)


issue_type_counts = final_df["issue_type"].fillna("no_type").value_counts(dropna=False)
barh_counts(issue_type_counts, "Issue type distribution", top_n=20)


platform_counts = final_df["platform"].explode().dropna().value_counts()
barh_counts(platform_counts, "Platform distribution", top_n=20)


impact_counts = final_df["impact"].explode().dropna().value_counts()
barh_counts(impact_counts, "Impact flag distribution", top_n=20)


def has_any_list(x):
    return isinstance(x, list) and len(x) > 0


coverage = pd.Series({
    "has_topic": final_df["topic"].apply(has_any_list).mean(),
    "has_platform": final_df["platform"].apply(has_any_list).mean(),
    "has_impact": final_df["impact"].apply(has_any_list).mean(),
    "has_issue_type": final_df["issue_type"].notna().mean(),
}) * 100

plt.figure(figsize=(7,4))
plt.bar(coverage.index, coverage.values)
plt.ylabel("Percent of issues (%)")
plt.title("Label coverage")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()


# %%
topic_per_issue = final_df["topic"].apply(lambda x: len(x) if isinstance(x, list) else 0)

plt.figure(figsize=(7,4))
plt.hist(topic_per_issue, bins=range(0, int(topic_per_issue.max()) + 2))
plt.xlabel("# topics on an issue")
plt.ylabel("Count")
plt.title("Topics per issue")
plt.tight_layout()
plt.show()


# %%
other_share = final_df["topic"].apply(lambda xs: ("other" in xs) if isinstance(xs, list) else False).mean() * 100
print(f"Percent of issues containing topic='other': {other_share:.2f}%")


# %%
tmp = final_df.dropna(subset=["created_at"]).copy()
monthly = tmp.set_index("created_at").resample("MS").size()

plt.figure(figsize=(10, 4))
plt.plot(monthly.index, monthly.values)
plt.xlabel("Month")
plt.ylabel("# issues")
plt.title("Issues over time (monthly)")
plt.tight_layout()
plt.show()
