"""
robustness.py: Robustness evaluation for Godot issue triage.

Baseline:
    Reconstructs the TF-IDF runner from saved artifacts (config.json +
    tasks.json + tuning.json), re-trains, and evaluates on all 4 perturbed
    test splits. Reports ΔμF1, ΔmacF1, ΔHamming, and prediction stability.

LLM:
    Draws a fixed random sample of ~700 issues, calls the Gemini API on each
    perturbed version (4 × 700 = ~2,800 calls), and reports ΔμF1 + stability
    vs. the clean-run predictions for that same sample. Checkpoints each
    perturbation's raw outputs so interrupted runs can resume without re-calling.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from dotenv import load_dotenv
load_dotenv(".env.local")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Perturbation functions
# ─────────────────────────────────────────────────────────────────────────────

def perturb_remove_title(row: dict) -> dict:
    """Zero out the title, forces the model to rely on body only."""
    r = row.copy()
    r["title"] = ""
    return r


def perturb_truncate_body(row: dict, keep_chars: int = 300) -> dict:
    """Keep only the first 300 characters of the body."""
    r = row.copy()
    r["body"] = (r.get("body") or "")[:keep_chars]
    return r


def perturb_strip_code(row: dict) -> dict:
    """Remove fenced code blocks (``` ... ```) and inline backtick spans."""
    r = row.copy()
    body = r.get("body") or ""
    body = re.sub(r"```[\s\S]*?```", "", body)
    body = re.sub(r"`[^`\n]+`", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    r["body"] = body
    return r


def perturb_drop_first_last(row: dict, k_tokens: int = 50) -> dict:
    """Drop the first and last k whitespace-delimited tokens from the body."""
    r = row.copy()
    tokens = (r.get("body") or "").split()
    r["body"] = " ".join(tokens[k_tokens:-k_tokens]) if len(tokens) > 2 * k_tokens else ""
    return r


def perturb_none(row: dict) -> dict:
    return row.copy()


PERTURBATIONS: dict[str, callable] = {
    "none": perturb_none,
    "remove_title":    perturb_remove_title,
    "truncate_body":   perturb_truncate_body,
    "strip_code":      perturb_strip_code,
    "drop_first_last": perturb_drop_first_last,
}

PERTURBATION_LABELS: dict[str, str] = {
    "none": "None",
    "remove_title":    "Remove Title",
    "truncate_body":   "Truncate Body (300 chars)",
    "strip_code":      "Strip Code Blocks",
    "drop_first_last": "Drop First/Last 50 Tokens",
}


def rebuild_text_clean(df: pd.DataFrame,
                       title_col: str = "title",
                       body_col: str = "body") -> pd.DataFrame:
    """
    Reconstruct text_clean from (possibly perturbed) title + body.
    Matches the simple concatenation used in data pipeline.
    """
    df = df.copy()
    title = df[title_col].fillna("").astype(str).str.strip()
    body  = df[body_col].fillna("").astype(str).str.strip()
    df["text_clean"] = (title + " " + body).str.strip()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Reconstruct ExperimentRunner from saved run artifacts
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_runner(baseline_run_dir: Path, vocab_path: Path):
    """
    Rebuild a trained + tuned ExperimentRunner from a saved run directory.
    Reads config.json, tasks.json, tuning.json; re-trains; restores thresholds.
    """
    from scripts.baseline.runner import ExperimentRunner, RunnerConfig, TaskSpec

    run_dir = Path(baseline_run_dir)

    with open(run_dir / "config.json") as f:
        config_dict = json.load(f)
    with open(run_dir / "tasks.json") as f:
        tasks_raw = json.load(f)
    with open(run_dir / "tuning.json") as f:
        tuning_info = json.load(f)
    with open(vocab_path) as f:
        vocab = json.load(f)

    saved_tasks = {t["name"]: t for t in tasks_raw}
    default_label_col = {
        "components": "topic",
        "platform":   "platform",
        "impact":     "impact",
        "issue_type": "issue_type",
    }

    tasks = []
    for t in tasks_raw:
        name = t["name"]
        tasks.append(TaskSpec(
            name=name,
            label_col=t.get("label_col", default_label_col.get(name, name)),
            classes=vocab.get(name, t.get("classes", [])),
            threshold_policy=t.get("threshold_policy", "global"),
            tune_method=t.get("tune_method", "global_micro"),
        ))

    valid_fields = {
        "train_path", "val_path", "test_path", "text_col", "id_col",
        "out_dir", "schema_path", "tfidf_params", "lr_params",
        "threshold_grid", "schema_version", "tau_components", "margin",
    }
    cfg_kwargs = {k: v for k, v in config_dict.items() if k in valid_fields}
    if cfg_kwargs.get("threshold_grid") is not None:
        cfg_kwargs["threshold_grid"] = np.array(cfg_kwargs["threshold_grid"])

    cfg = RunnerConfig(**cfg_kwargs)
    runner = ExperimentRunner(cfg, tasks)
    runner.load_splits()
    runner.train()

    # Restore tuned thresholds
    for spec in tasks:
        if spec.name not in runner.models:
            continue
        model = runner.models[spec.name]
        t = tuning_info.get(spec.name, {})
        if "global_threshold" in t:
            model.global_threshold = float(t["global_threshold"])
        elif "best_threshold" in t:
            model.global_threshold = float(t["best_threshold"])
        elif "per_label_thresholds" in t:
            model.per_label_thresholds = np.array(t["per_label_thresholds"])

    print(f"Runner reconstructed ({len(tasks)} tasks):")
    for spec in tasks:
        m = runner.models.get(spec.name)
        thr = getattr(m, "global_threshold", "n/a") if m else "missing"
        print(f"  {spec.name}: global_threshold={thr}")

    return runner


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true_lists, y_pred_lists, classes) -> dict:
    mlb = MultiLabelBinarizer(classes=classes)
    Y_true = mlb.fit_transform(y_true_lists)
    Y_pred = mlb.transform(y_pred_lists)
    return {
        "micro_f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)),
        "hamming":  float(hamming_loss(Y_true, Y_pred)),
        "coverage": float(np.mean([len(p) > 0 for p in y_pred_lists])),
    }


def prediction_stability(preds_a: list[list], preds_b: list[list]) -> float:
    """Fraction of issues where predicted label sets are identical."""
    assert len(preds_a) == len(preds_b)
    return sum(set(a) == set(b) for a, b in zip(preds_a, preds_b)) / len(preds_a)


def predict_labels_baseline(model, df: pd.DataFrame, text_col: str) -> list[list]:
    """Threshold predict_proba at model.global_threshold."""
    proba = model.predict_proba(df, text_col=text_col)
    threshold = float(getattr(model, "global_threshold", 0.5))
    classes = model.class_names
    return [
        [c for c, p in zip(classes, row) if p >= threshold]
        for row in proba
    ]


def get_gt_lists(df: pd.DataFrame, col: str) -> list[list]:
    result = []
    for v in df.get(col, pd.Series(dtype=object)):
        if isinstance(v, list):
            result.append(v)
        elif isinstance(v, str):
            result.append([v] if v else [])
        else:
            result.append([])
    return result


def get_preds_from_records(records: list[dict], field: str) -> list[list]:
    """Extract label lists from a list of triage record dicts."""
    result = []
    for rec in records:
        v = rec.get(field, [])
        if isinstance(v, list):
            result.append(v)
        elif isinstance(v, str):
            try:
                parsed = json.loads(v)
                result.append(parsed if isinstance(parsed, list) else [])
            except Exception:
                result.append([v] if v else [])
        else:
            result.append([])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. LLM inference on perturbed sample
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm_prompt(title: str, body: str, max_body_chars: int = 8000) -> str:
    body = (body or "")
    if len(body) > max_body_chars:
        body = body[:max_body_chars] + "\n\n[... truncated ...]"
    return f"## Issue Title\n{title or ''}\n\n## Issue Body\n{body}"


def _call_gemini_once(client, model_name: str, system_prompt: str,
                      user_prompt: str, temperature: float,
                      retries: int = 3) -> dict | None:
    """
    Single Gemini API call at the given temperature.
    Returns a parsed JSON dict or None on unrecoverable failure.
    Handles 429 rate limits with exponential backoff.
    """
    try:
        from google.genai import types
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    temperature=temperature,
                ),
            )
            raw = response.text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = 30 * (2 ** attempt)
                print(f"      Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(3)
            else:
                print(f"      API error after {retries} attempts: {err[:120]}")
                return None
    return None


def _extract_labels_from_sample(sample: dict, field: str) -> list:
    """
    Extract a label list from one raw LLM sample dict.
    Handles both nested {labels: {field: [...]}} and flat {field: [...]} formats.
    issue_type is stored as a list in the nested format; unwrap it here.
    """
    if sample is None:
        return []
    if "labels" in sample and isinstance(sample["labels"], dict):
        v = sample["labels"].get(field, [])
    else:
        v = sample.get(field, [])

    if field == "issue_type":
        # Normalise: may be a plain string or a list with one element
        if isinstance(v, list):
            return v[:1]   # keep as single-element list for uniform handling
        return [v] if isinstance(v, str) and v else []

    if isinstance(v, list):
        return v
    if isinstance(v, str):
        return [v] if v else []
    return []


def _aggregate_samples(samples: list[dict], field: str,
                       classes: list[str],
                       aggregation_threshold: float = 0.5) -> list[str]:
    """
    Aggregate N raw LLM samples into a final prediction.
    Uses Exact Set Consensus for multi-label fields.
    """
    n = len(samples)
    if n == 0:
        return []

    if field == "issue_type":
        # Standard majority vote for single-label field
        freqs = {}
        for s in samples:
            labels = _extract_labels_from_sample(s, field)
            val = labels[0] if labels else ""
            freqs[val] = freqs.get(val, 0) + 1
        if not freqs:
            return []
        best_cls = max(freqs, key=freqs.get)
        return [best_cls] if freqs[best_cls] / n >= aggregation_threshold else []
    else:
        # EXACT SET CONSENSUS for components, platform, impact
        exact_set_counts = {}
        # EXACT SET CONSENSUS for components, platform, impact
        exact_set_counts = {}
        for s in samples:
            raw_labels = _extract_labels_from_sample(s, field)
            # Filter out the hallucinations! Only keep official classes.
            valid_labels = sorted([c for c in raw_labels if c in classes])
            label_tuple = tuple(valid_labels)
            exact_set_counts[label_tuple] = exact_set_counts.get(label_tuple, 0) + 1

        # Return the exact set if it meets the threshold
        for label_tuple, count in exact_set_counts.items():
            if count / n >= aggregation_threshold:
                return list(label_tuple)
                
        # If no exact combination gets a majority, abstain!
        return []   

def _build_robustness_system_prompt(vocab: dict) -> str:
    """
    System prompt matching GeminiTriageModel exactly
    """
    components  = vocab.get("components", [])
    issue_types = vocab.get("issue_type", [])
    platforms   = vocab.get("platform", [])
    impacts     = vocab.get("impact", [])

    return f"""You are an expert issue triager for the Godot game engine.

Given a GitHub issue (title + body), return a JSON object with exactly this structure:

{{
  "schema_version": "1.0",
  "labels": {{
    "issue_type": ["<one of {json.dumps(issue_types)}>"],
    "components": ["<1–3 items from {json.dumps(components)}>"],
    "platform": ["<0+ items from {json.dumps(platforms)}>"],
    "impact": ["<0+ items from {json.dumps(impacts)}>"]
  }},
  "needs_human_triage": false,
  "meta": {{"source": "llm"}}
}}

Rules:
- "labels.components" is the most important field. Assign 1–3 labels.
- Set "needs_human_triage": true and "labels.components": [] only if you cannot route at all.
- "labels.platform": infer from OS mentions / stack traces. Use [] if unclear.
- "labels.impact": only if clear textual evidence (e.g. "crash", "regression"). Use [] if none.
- Return ONLY the JSON object. No explanation, no markdown."""


def run_llm_robustness(
    sample_df: pd.DataFrame,
    clean_preds: list[dict],
    vocab: dict,
    model_name: str,
    primary_task: str,
    cache_dir: Path,
    n_samples: int = 3,
    temperature: float = 0.2,
    aggregation_threshold: float = 0.5,
    requests_per_minute: int = 500,
    max_workers: int = 20,
) -> dict[str, dict]:
    """
    Run the LLM on 4 perturbed versions of sample_df with N samples per issue.
    API calls are parallelised with ThreadPoolExecutor.

    Concurrency model:
      - max_workers threads submit calls simultaneously
      - A token bucket enforces requests_per_minute across all threads
      - A threading.Lock protects the cache file and progress counter
      - Each work unit is one (issue_idx, sample_idx) pair so the thread
        pool is as fine-grained as possible

    Cache layout:
        cache_dir/<perturb_name>.jsonl
        Each line: {"_id": <issue_id>, "_samples": [{raw record}, ...]}
        Complete when len(_samples) == n_samples. Resumable.

    Returns dict of perturb_name -> metrics dict.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai not installed. Run: pip install google-genai")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Export it or add it to your .env before running."
        )

    client        = genai.Client(api_key=api_key)
    system_prompt = _build_robustness_system_prompt(vocab)
    cache_dir.mkdir(parents=True, exist_ok=True)

    classes = vocab.get(primary_task, [])
    gt_col_map = {
        "components": "topic",
        "platform":   "platform",
        "impact":     "impact",
        "issue_type": "issue_type",
    }
    gt_col           = gt_col_map.get(primary_task, primary_task)
    gt_lists         = get_gt_lists(sample_df, gt_col)
    clean_pred_lists = get_preds_from_records(clean_preds, primary_task)

    # Infer ID column once
    id_col = next(
        (c for c in ["id", "number", "issue_id"] if c in sample_df.columns), None
    )

    def get_id(i: int):
        val = sample_df.iloc[i][id_col] if id_col else i
        return val.item() if hasattr(val, "item") else val

    # ── Token bucket for rate limiting across threads ─────────────────
    # Tracks the time the next call is allowed to start.
    _bucket_lock = threading.Lock()
    _next_allowed = [time.monotonic()]   # list so closure can mutate it

    def _acquire_token():
        """Block until a rate-limit slot is available, then claim it."""
        gap = 60.0 / max(requests_per_minute, 1)
        while True:
            with _bucket_lock:
                now = time.monotonic()
                if now >= _next_allowed[0]:
                    _next_allowed[0] = now + gap
                    return
                wait = _next_allowed[0] - now
            time.sleep(wait)

    results = {}

    for perturb_name, perturb_fn in PERTURBATIONS.items():
        print(f"\n  Perturbation: {PERTURBATION_LABELS[perturb_name]}")
        cache_file = cache_dir / f"{perturb_name}.jsonl"

        # ── Load cache ────────────────────────────────────────────────
        done: dict = {}
        if cache_file.exists():
            with open(cache_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        done[entry["_id"]] = entry
            complete = sum(1 for e in done.values() if len(e["_samples"]) >= n_samples)
            print(f"    Cache: {len(done)} entries, {complete} complete ({n_samples} samples each).")

        # ── Apply perturbation ────────────────────────────────────────
        perturbed_rows = [perturb_fn(r) for r in sample_df.to_dict("records")]

        # ── Build work list: one item per (issue, sample_slot) needed ─
        # Each item: (issue_idx, issue_id, user_prompt, samples_already_done)
        work_items = []
        for i in range(len(sample_df)):
            issue_id = get_id(i)
            existing = done.get(issue_id, {}).get("_samples", [])
            needed   = n_samples - len(existing)
            if needed > 0:
                row         = perturbed_rows[i]
                user_prompt = _build_llm_prompt(row.get("title", ""), row.get("body", ""))
                for _ in range(needed):
                    work_items.append((i, issue_id, user_prompt))

        total_calls = len(work_items)
        print(f"    {sum(1 for i in range(len(sample_df)) if len(done.get(get_id(i), {}).get('_samples', [])) < n_samples)} issues need work → {total_calls} API calls.")

        if total_calls == 0:
            print("    Nothing to do (fully cached).")
        else:
            # ── Shared mutable state (protected by locks) ─────────────
            results_lock  = threading.Lock()   # protects new_samples dict + cache file
            progress_lock = threading.Lock()   # protects calls_made counter
            calls_made    = [0]                # list so closure can mutate
            # Accumulate new samples here before merging into done
            new_samples: dict = {}             # issue_id -> [sample, ...]

            def _call_one(issue_idx: int, issue_id, user_prompt: str) -> tuple:
                """Worker: rate-limit, call API, return (issue_id, sample)."""
                _acquire_token()
                sample = _call_gemini_once(
                    client, model_name, system_prompt, user_prompt, temperature
                )
                if sample is None:
                    sample = {
                        "labels": {
                            "issue_type": [], "components": [],
                            "platform": [], "impact": [],
                        },
                        "needs_human_triage": True,
                        "_api_error": True,
                    }
                return issue_id, sample

            # Open cache file for appending partial results as they arrive
            with open(cache_file, "w") as cache_f:
                # Pre-write already-complete entries so the file is valid
                # even if we crash mid-run
                for i in range(len(sample_df)):
                    issue_id = get_id(i)
                    if issue_id in done and len(done[issue_id]["_samples"]) >= n_samples:
                        cache_f.write(json.dumps(done[issue_id]) + "\n")
                cache_f.flush()

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(_call_one, i, iid, prompt): (i, iid)
                        for i, iid, prompt in work_items
                    }

                    for future in as_completed(futures):
                        issue_id, sample = future.result()

                        with results_lock:
                            if issue_id not in new_samples:
                                new_samples[issue_id] = []
                            new_samples[issue_id].append(sample)

                            # Once an issue has all its new samples, write it
                            existing  = done.get(issue_id, {}).get("_samples", [])
                            needed    = n_samples - len(existing)
                            if len(new_samples[issue_id]) >= needed:
                                entry = {
                                    "_id":     issue_id,
                                    "_samples": existing + new_samples[issue_id][:needed],
                                }
                                done[issue_id] = entry
                                cache_f.write(json.dumps(entry) + "\n")
                                cache_f.flush()

                        with progress_lock:
                            calls_made[0] += 1
                            if calls_made[0] % 100 == 0 or calls_made[0] == total_calls:
                                print(f"    [{calls_made[0]}/{total_calls}] API calls done")

            print(f"    {calls_made[0]} new API calls made. Aggregating...")

        # ── Aggregate N samples → final predictions ───────────────────
        pert_preds = []
        for i in range(len(sample_df)):
            issue_id = get_id(i)
            samples  = done.get(issue_id, {}).get("_samples", [])
            pred     = _aggregate_samples(samples, primary_task, classes, aggregation_threshold)
            pert_preds.append(pred)

        # ── Metrics ───────────────────────────────────────────────────
        pert_metrics  = compute_metrics(gt_lists, pert_preds, classes)
        clean_metrics = compute_metrics(gt_lists, clean_pred_lists, classes)
        delta_micro   = pert_metrics["micro_f1"] - clean_metrics["micro_f1"]
        delta_macro   = pert_metrics["macro_f1"] - clean_metrics["macro_f1"]
        stab          = prediction_stability(clean_pred_lists, pert_preds)

        print(f"    LLM μF1 {clean_metrics['micro_f1']:.4f} → "
              f"{pert_metrics['micro_f1']:.4f}  (Δ{delta_micro:+.4f})  "
              f"stability={stab:.4f}  pert_coverage={pert_metrics['coverage']:.4f}")

        results[perturb_name] = {
            "llm_sample_clean_micro":  round(clean_metrics["micro_f1"], 4),
            "llm_sample_pert_micro":   round(pert_metrics["micro_f1"], 4),
            "llm_delta_micro":         round(delta_micro, 4),
            "llm_delta_macro":         round(delta_macro, 4),
            "llm_stability":           round(stab, 4),
            "llm_sample_n":            len(sample_df),
            "llm_n_samples_per_issue": n_samples,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. LLM predictions loader + sample aligner
# ─────────────────────────────────────────────────────────────────────────────

def _load_llm_predictions(run_dir: Path) -> list[dict]:
    run_dir = Path(run_dir)
    candidates = (
        list(run_dir.glob("predictions*.jsonl")) +
        list(run_dir.glob("*.jsonl"))
    )
    if candidates:
        records = []
        with open(candidates[0]) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    json_candidates = list(run_dir.glob("predictions*.json"))
    if json_candidates:
        with open(json_candidates[0]) as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    raise FileNotFoundError(
        f"No predictions file found in {run_dir}. "
        "Expected predictions.jsonl or predictions*.json."
    )


def _align_llm_preds_to_sample(
    all_preds: list[dict],
    sample_df: pd.DataFrame,
    id_col: str | None,
) -> list[dict]:
    """Return the subset of all_preds that correspond to sample_df rows, in order."""
    if id_col is None or id_col not in sample_df.columns:
        # No ID column, assume first N records align to sample
        return all_preds[:len(sample_df)]

    id_to_pred = {}
    for rec in all_preds:
        rid = rec.get(id_col) or rec.get("id") or rec.get("number")
        if rid is not None:
            id_to_pred[rid] = rec

    return [id_to_pred.get(row[id_col], {}) for _, row in sample_df.iterrows()]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_robustness_eval(
    baseline_run_dir: Path,
    llm_run_dir: Path,
    vocab_path: Path,
    out_dir: Path,
    primary_task: str = "components",
    llm_sample_n: int = 700,
    llm_sample_seed: int = 42,
    llm_n_samples: int = 3,
    llm_temperature: float = 0.2,
    llm_aggregation_threshold: float = 0.5,
    llm_requests_per_minute: int = 500,
    llm_max_workers: int = 20,
    skip_llm: bool = False,
) -> pd.DataFrame:
    """
    Full robustness pipeline. Returns a DataFrame with one row per perturbation.
    """
    """
    Full robustness pipeline. Returns a DataFrame with one row per perturbation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(vocab_path) as f:
        vocab = json.load(f)

    # ── Baseline ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Reconstruct and evaluate baseline")
    print("=" * 60)
    runner = reconstruct_runner(baseline_run_dir, vocab_path)
    cfg          = runner.cfg
    test_df      = runner.test_df
    primary_spec = runner.tasks_by_name(primary_task)
    model        = runner.models[primary_task]
    gt_col       = primary_spec.label_col
    classes      = primary_spec.classes
    text_col     = cfg.text_col
    # Detect ID column from the test data itself rather than relying on
    # cfg.id_col, which is None in the baseline RunnerConfig.
    id_col = next(
        (c for c in ["id", "number", "issue_id"] if c in test_df.columns), None
    )

    print(f"\nTest set: {len(test_df)} issues")

    clean_baseline_preds = predict_labels_baseline(model, test_df, text_col)
    clean_gt             = get_gt_lists(test_df, gt_col)
    clean_baseline_m     = compute_metrics(clean_gt, clean_baseline_preds, classes)
    print(f"Baseline clean μF1={clean_baseline_m['micro_f1']:.4f}  "
          f"macF1={clean_baseline_m['macro_f1']:.4f}")

    baseline_rows = {}
    for perturb_name, perturb_fn in PERTURBATIONS.items():
        print(f"\n  Perturbation: {PERTURBATION_LABELS[perturb_name]}")
        df_pert  = pd.DataFrame([perturb_fn(r) for r in test_df.to_dict("records")])
        df_pert  = rebuild_text_clean(df_pert)
        pert_preds = predict_labels_baseline(model, df_pert, text_col)
        pert_m     = compute_metrics(clean_gt, pert_preds, classes)
        delta_micro = pert_m["micro_f1"] - clean_baseline_m["micro_f1"]
        delta_macro = pert_m["macro_f1"] - clean_baseline_m["macro_f1"]
        delta_ham   = pert_m["hamming"]  - clean_baseline_m["hamming"]
        stab        = prediction_stability(clean_baseline_preds, pert_preds)
        print(f"  μF1 {clean_baseline_m['micro_f1']:.4f} → {pert_m['micro_f1']:.4f} "
              f"(Δ{delta_micro:+.4f})  stability={stab:.4f}")
        baseline_rows[perturb_name] = {
            "baseline_clean_micro":   round(clean_baseline_m["micro_f1"], 4),
            "baseline_pert_micro":    round(pert_m["micro_f1"], 4),
            "baseline_delta_micro":   round(delta_micro, 4),
            "baseline_clean_macro":   round(clean_baseline_m["macro_f1"], 4),
            "baseline_pert_macro":    round(pert_m["macro_f1"], 4),
            "baseline_delta_macro":   round(delta_macro, 4),
            "baseline_delta_hamming": round(delta_ham, 4),
            "baseline_stability":     round(stab, 4),
        }

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_rows: dict[str, dict] = {
        name: {
            "llm_sample_clean_micro":  float("nan"),
            "llm_sample_pert_micro":   float("nan"),
            "llm_delta_micro":         float("nan"),
            "llm_delta_macro":         float("nan"),
            "llm_stability":           float("nan"),
            "llm_sample_n":            0,
            "llm_n_samples_per_issue": 0,
        }
        for name in PERTURBATIONS
    }

    if not skip_llm:
        print("\n" + "=" * 60)
        print(f"STEP 2: LLM robustness on {llm_sample_n}-issue sample")
        print("=" * 60)

        # Load full LLM predictions from clean run
        all_llm_preds = _load_llm_predictions(llm_run_dir)
        print(f"Loaded {len(all_llm_preds)} clean LLM predictions.")

        # Draw the sample, use the same indices regardless of ID column
        rng = np.random.default_rng(llm_sample_seed)
        n_sample = min(llm_sample_n, len(test_df))
        sample_idx = rng.choice(len(test_df), size=n_sample, replace=False)
        sample_idx.sort()
        sample_df = test_df.iloc[sample_idx].reset_index(drop=True)

        # Load the run_metadata.json to get model_name
        llm_run_dir = Path(llm_run_dir)
        model_name = "gemini-2.5-flash-lite"  # fallback default
        meta_path = llm_run_dir / "run_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            model_name = meta.get("model_name", model_name)
        print(f"LLM model: {model_name}")
        print(f"Sample: {n_sample} issues (seed={llm_sample_seed})")

        # Align clean predictions to the sample
        sample_clean_preds = _align_llm_preds_to_sample(all_llm_preds, sample_df, id_col)

        # Run perturbed inference
        llm_rows = run_llm_robustness(
            sample_df=sample_df,
            clean_preds=sample_clean_preds,
            vocab=vocab,
            model_name=model_name,
            primary_task=primary_task,
            cache_dir=out_dir / "llm_cache",
            n_samples=llm_n_samples,
            temperature=llm_temperature,
            aggregation_threshold=llm_aggregation_threshold,
            requests_per_minute=llm_requests_per_minute,
            max_workers=llm_max_workers,
        )

    # ── Assemble final DataFrame ──────────────────────────────────────────
    rows = []
    for perturb_name in PERTURBATIONS:
        rows.append({
            "perturbation":        perturb_name,
            "perturbation_label":  PERTURBATION_LABELS[perturb_name],
            **baseline_rows[perturb_name],
            **llm_rows[perturb_name],
        })

    results_df = pd.DataFrame(rows)
    csv_path = out_dir / "robustness_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to {csv_path}")

    _save_report_table(results_df, out_dir)
    _plot_robustness(results_df, out_dir, primary_task)
    _write_interpretation(results_df, out_dir)

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Report table
# ─────────────────────────────────────────────────────────────────────────────

def _save_report_table(df: pd.DataFrame, out_dir: Path):
    cols = [
        "perturbation_label",
        "baseline_clean_micro", "baseline_pert_micro", "baseline_delta_micro",
        "baseline_stability",
        "llm_sample_clean_micro", "llm_sample_pert_micro", "llm_delta_micro",
        "llm_stability",
    ]
    # Only include LLM cols if they have actual values
    has_llm = df["llm_delta_micro"].notna().any()
    if not has_llm:
        cols = [c for c in cols if not c.startswith("llm_")]

    table = df[cols].copy()
    rename = {
        "perturbation_label":     "Perturbation",
        "baseline_clean_micro":   "Baseline μF1 (clean)",
        "baseline_pert_micro":    "Baseline μF1 (pert.)",
        "baseline_delta_micro":   "Baseline ΔμF1",
        "baseline_stability":     "Baseline Stability",
        "llm_sample_clean_micro": "LLM μF1 (clean, sample)",
        "llm_sample_pert_micro":  "LLM μF1 (pert., sample)",
        "llm_delta_micro":        "LLM ΔμF1",
        "llm_stability":          "LLM Stability",
    }
    table = table.rename(columns={k: v for k, v in rename.items() if k in table.columns})
    path = out_dir / "robustness_table.csv"
    table.to_csv(path, index=False)
    print(f"Report table saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Plot, four panels when LLM data is present, two panels otherwise
# ─────────────────────────────────────────────────────────────────────────────

def _plot_robustness(df: pd.DataFrame, out_dir: Path, primary_task: str = "components"):
    labels = df["perturbation_label"].tolist()
    x = np.arange(len(labels))
    has_llm = df["llm_delta_micro"].notna().any()

    ncols = 4 if has_llm else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(
        f"Robustness Evaluation — {primary_task.capitalize()} Routing",
        fontsize=13, fontweight="bold",
    )

    def bar_panel(ax, values, title, ylabel, color_fn=None, ylim=None, hline=None):
        colors = [color_fn(v) for v in values] if color_fn else ["#4C72B0"] * len(values)
        bars = ax.bar(x, values, 0.55, color=colors, alpha=0.85)
        if hline is not None:
            ax.axhline(hline, color="black" if hline == 0 else "gray",
                       linewidth=0.8, linestyle="--" if hline == 0 else ":")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        for bar, val in zip(bars, values):
            if not (val != val):  # not NaN
                offset = 0.002 if val >= 0 else -0.003
                va = "bottom" if val >= 0 else "top"
                fmt = f"{val:+.3f}" if hline == 0 else f"{val:.3f}"
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + offset, fmt,
                        ha="center", va=va, fontsize=8)
        return bars

    delta_color = lambda v: "#C44E52" if v < -0.02 else "#4C72B0"

    bar_panel(axes[0],
              df["baseline_delta_micro"].tolist(),
              "Baseline ΔμF1", "Δ Micro-F1 (pert. − clean)",
              color_fn=delta_color, hline=0)

    bar_panel(axes[1],
              df["baseline_stability"].tolist(),
              "Baseline Stability", "Prediction Stability",
              color_fn=lambda v: "#55A868", ylim=(0, 1.08), hline=1.0)

    if has_llm:
        bar_panel(axes[2],
                  df["llm_delta_micro"].tolist(),
                  f"LLM ΔμF1 (n≈{int(df['llm_sample_n'].iloc[0])})",
                  "Δ Micro-F1 (pert. − clean)",
                  color_fn=delta_color, hline=0)

        bar_panel(axes[3],
                  df["llm_stability"].tolist(),
                  "LLM Stability", "Prediction Stability",
                  color_fn=lambda v: "#DD8452", ylim=(0, 1.08), hline=1.0)

    plt.tight_layout()
    path = out_dir / "robustness_plot.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Interpretation notes
# ─────────────────────────────────────────────────────────────────────────────

_INTERP = {
    "remove_title": (
        "Tests body-only routing. A large drop means the model relies heavily on the "
        "short title summary; the body alone is insufficient for confident routing."
    ),
    "truncate_body": (
        "Simulates short or incomplete reports (first 300 chars only). "
        "Degradation here reflects dependence on full context for disambiguation."
    ),
    "strip_code": (
        "Removes fenced code blocks and inline backtick spans. "
        "Relevant for gdscript/rendering issues where API names are the primary signal. "
        "High stability means natural-language framing alone is sufficient."
    ),
    "drop_first_last": (
        "Drops first and last 50 tokens, removing opening framing and trailing metadata "
        "(OS, version, reproduction steps). A large drop suggests positional anchoring "
        "that may not generalise across report styles."
    ),
}


def _write_interpretation(df: pd.DataFrame, out_dir: Path):
    has_llm = df["llm_delta_micro"].notna().any()
    lines = [
        "Robustness Interpretation Notes",
        "=" * 60,
        "Primary routing task: components (micro-F1)",
        "",
    ]
    for _, row in df.iterrows():
        name  = row["perturbation"]
        label = row["perturbation_label"]

        def severity(d):
            return (
                "negligible (<0.01)"   if abs(d) < 0.01
                else "minor (0.01–0.05)"    if abs(d) < 0.05
                else "moderate (0.05–0.10)" if abs(d) < 0.10
                else "LARGE (>0.10)"
            )

        def stab_note(s):
            return (
                "highly stable (>0.90)"         if s > 0.90
                else "mostly stable (0.75–0.90)"    if s > 0.75
                else "moderately stable (0.60–0.75)" if s > 0.60
                else "UNSTABLE (<0.60)"
            )

        b_delta = row["baseline_delta_micro"]
        b_stab  = row["baseline_stability"]

        lines += [
            f"[{label}]",
            f"  Baseline  ΔμF1={b_delta:+.4f} ({severity(b_delta)})  "
            f"stability={b_stab:.4f} ({stab_note(b_stab)})",
        ]

        if has_llm and not np.isnan(row["llm_delta_micro"]):
            l_delta = row["llm_delta_micro"]
            l_stab  = row["llm_stability"]
            n       = int(row["llm_sample_n"])
            lines.append(
                f"  LLM (n={n}) ΔμF1={l_delta:+.4f} ({severity(l_delta)})  "
                f"stability={l_stab:.4f} ({stab_note(l_stab)})"
            )

        lines += [f"  {_INTERP.get(name, '')}", ""]

    Path(out_dir / "interpretation.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Interpretation notes saved to {out_dir / 'interpretation.txt'}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Robustness evaluation for Godot issue triage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-run", required=True,
                        help="Path to baseline run dir, e.g. baseline/runs/20260226_212256")
    parser.add_argument("--llm-run",      required=True,
                        help="Path to LLM run dir, e.g. runs/llm_gemini-2.5-flash-lite_...")
    parser.add_argument("--vocab-path",
                        default="src/scripts/data_collection/data/processed/label_vocab.json")
    parser.add_argument("--out-dir",      default="final_results/robustness")
    parser.add_argument("--primary-task", default="components")
    parser.add_argument("--llm-sample-n", type=int, default=700,
                        help="Number of test issues to sample for LLM robustness (~19%% of test set)")
    parser.add_argument("--llm-sample-seed", type=int, default=42,
                        help="Random seed for sample selection (fixed = reproducible)")
    parser.add_argument("--llm-n-samples", type=int, default=10,
                        help="Samples per issue per perturbation (N=3 → ~8,400 total calls)")
    parser.add_argument("--llm-temperature", type=float, default=0.7,
                        help="Sampling temperature (must be >0 for N>1 to give different outputs)")
    parser.add_argument("--llm-agg-threshold", type=float, default=0.5,
                        help="Majority-vote threshold: predict label if freq/N >= this value")
    parser.add_argument("--llm-rpm", type=int, default=500,
                        help="Gemini requests per minute (Tier 1 Flash Lite supports 1000+)")
    parser.add_argument("--llm-max-workers", type=int, default=20,
                        help="Parallel threads for LLM API calls")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Run baseline robustness only, skip LLM re-inference")
    args = parser.parse_args()

    run_robustness_eval(
        baseline_run_dir=Path(args.baseline_run),
        llm_run_dir=Path(args.llm_run),
        vocab_path=Path(args.vocab_path),
        out_dir=Path(args.out_dir),
        primary_task=args.primary_task,
        llm_sample_n=args.llm_sample_n,
        llm_sample_seed=args.llm_sample_seed,
        llm_n_samples=args.llm_n_samples,
        llm_temperature=args.llm_temperature,
        llm_aggregation_threshold=args.llm_agg_threshold,
        llm_requests_per_minute=args.llm_rpm,
        llm_max_workers=args.llm_max_workers,
        skip_llm=args.skip_llm,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()