"""
LLM Triage Model, Gemini-based structured issue triager.

Implements the TriageModel protocol with N-sample aggregation,
schema-aware repair, confidence estimation, and abstention.
"""
from __future__ import annotations

import json
import time
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, hamming_loss

from scripts.schema.schema_validator import TriageSchemaValidator


# ═══════════════════════════════════════════════════════════════════════
# Protocol
# ═══════════════════════════════════════════════════════════════════════

@runtime_checkable
class TriageModel(Protocol):
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None: ...
    def predict(self, df: pd.DataFrame) -> list[dict]: ...
    def predict_proba(self, df: pd.DataFrame) -> dict[str, np.ndarray]: ...
    def get_run_metadata(self) -> dict: ...


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TaskConfig:
    name: str
    label_col: str
    classes: list[str]


@dataclass
class LLMRunConfig:
    # Paths
    train_path: str = ""
    val_path: str = ""
    test_path: str = ""
    vocab_path: str = ""
    schema_path: str = ""
    out_dir: str = "runs"
    env_path: str | None = None

    # Model settings
    model_name: str = "gemini-2.5-flash-lite"
    temperature: float = 0.7
    n_samples: int = 20
    max_body_chars: int = 8000

    # Column mappings
    col_id: str = "id"
    col_title: str = "title"
    col_body: str = "body"

    # Parallelization
    max_workers: int = 64

    # Aggregation threshold
    aggregation_threshold: float = 0.5

    # Confidence threshold (for abstention)
    confidence_threshold: float = 0.4

    # Repair settings
    max_repair_attempts: int = 1

    # Checkpoint
    checkpoint_dir: str = "results/checkpoints"


# ═══════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════

class GeminiTriageModel:

    def __init__(self, config: LLMRunConfig, tasks: list[TaskConfig], vocab: dict[str, list[str]]):
        self.config = config
        self.tasks = tasks
        self.vocab = vocab
        self._tasks_by_name = {t.name: t for t in tasks}
        self._label_sets = {t.name: set(t.classes) for t in tasks}

        # API client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        self.client = genai.Client(api_key=api_key)

        # Schema validator
        self.schema_validator = TriageSchemaValidator(config.schema_path)

        # System prompt
        self.system_prompt = self._build_system_prompt()

        # Caches
        self._raw_samples: dict[Any, list[dict]] = {}
        self._checkpoint_path = (
            Path(config.checkpoint_dir)
            / f"{config.model_name.replace('/', '_')}_n{config.n_samples}_samples.jsonl"
        )

        # Tracking
        self._stats_lock = threading.Lock()
        self.stats = {
            "first_pass_valid": 0,
            "repaired": 0,
            "repair_failed": 0,
            "api_errors": 0,
            "content_blocked": 0,
            "total": 0,
        }

    # ── Thread-safe stats ──────────────────────────────────────────────

    def _inc_stat(self, key: str, n: int = 1):
        with self._stats_lock:
            self.stats[key] += n

    # ── Prompt Construction ────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        config = self.config
        vocab = self.vocab
        return f"""You are an expert open-source issue triager for the Godot game engine.
        Given a GitHub issue (title + body), produce a JSON triage record that
        conforms EXACTLY to the schema below.

        ## Output Schema

        Return ONLY a valid JSON object — no markdown, no explanation.
        The object MUST have this exact structure:

        {{
          "schema_version": "1.0",
          "labels": {{
            "issue_type": [],
            "components": [],
            "platform": [],
            "impact": []
          }},
          "needs_human_triage": false,
          "meta": {{
            "source": "llm",
            "model": "{config.model_name}"
          }}
        }}

        ## Field Rules

        - "schema_version": always exactly "1.0"
        - "labels.issue_type": array with EXACTLY 0 or 1 item from {json.dumps(vocab["issue_type"])}
        - "labels.components": array with 0-3 items from {json.dumps(vocab["components"])}. This is the most important field.
        - "labels.platform": array with 0+ items from {json.dumps(vocab["platform"])}. Infer from OS mentions, stack traces. Empty [] if unclear.
        - "labels.impact": array with 0+ items from {json.dumps(vocab["impact"])}. Only include if clear textual evidence (e.g., "crash", "regression from 4.x"). Empty [] if none.
        - "needs_human_triage": true ONLY if you cannot confidently assign any component. If true, "labels.components" MUST be [].
        - "meta": always include "source": "llm" and "model": "{config.model_name}"
        - Do NOT include any fields not listed above (no "confidence", no "scores").
        - Return ONLY the JSON object."""

    def _build_user_prompt(self, title: str, body: str) -> str:
        if pd.isna(body):
            body = ""
        if pd.isna(title):
            title = ""
        if len(body) > self.config.max_body_chars:
            body = body[: self.config.max_body_chars] + "\n\n[... truncated ...]"
        return f"## Issue Title\n{title}\n\n## Issue Body\n{body}"

    # ── API Call with Repair ───────────────────────────────────────────

    def _call_api_raw(self, contents, retries: int = 2) -> str | None:
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.config.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        response_mime_type="application/json",
                        temperature=self.config.temperature,
                    ),
                )
                if response.text:
                    return response.text.strip()
                else:
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        if response.prompt_feedback.block_reason:
                            self._inc_stat("content_blocked")
                            return None
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    wait = 30 * (attempt + 1)
                    print(f"      Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                elif attempt < retries - 1:
                    time.sleep(5)
                else:
                    print(f"      API error after {retries} attempts: {error_str}")
                    self._inc_stat("api_errors")
        return None

    def _call_with_repair(self, title: str, body: str) -> dict | None:
        self._inc_stat("total")
        user_prompt = self._build_user_prompt(title, body)

        raw_text = self._call_api_raw(user_prompt)
        if raw_text is None:
            return None

        try:
            record = json.loads(raw_text)
        except json.JSONDecodeError:
            record = None

        if record is not None:
            ok, _, errors = self.schema_validator.validate_instance(record)
            if ok:
                self._inc_stat("first_pass_valid")
                record["_schema_first_pass"] = True
                return record

            for repair_num in range(self.config.max_repair_attempts):
                error_summary = json.dumps(errors, indent=2)
                repair_prompt = (
                    f"Your previous output was invalid.\n\n"
                    f"Errors:\n{error_summary}\n\n"
                    f"Original output:\n{json.dumps(record, indent=2)}\n\n"
                    f"Fix the errors and return ONLY the corrected JSON."
                )
                repair_text = self._call_api_raw(repair_prompt)
                if repair_text is None:
                    continue
                try:
                    repaired = json.loads(repair_text)
                except json.JSONDecodeError:
                    continue
                ok, _, _ = self.schema_validator.validate_instance(repaired)
                if ok:
                    self._inc_stat("repaired")
                    repaired["_schema_first_pass"] = False
                    repaired["_repair_attempt"] = repair_num + 1
                    return repaired

        self._inc_stat("repair_failed")
        if record is not None:
            clamped = self._clamp_to_vocab(record)
            clamped["_schema_first_pass"] = False
            clamped["_repair_failed"] = True
            return clamped
        return None

    def _clamp_to_vocab(self, record: dict) -> dict:
        labels = record.get("labels", {})
        clamped_labels = {}

        it = labels.get("issue_type", [])
        if not isinstance(it, list):
            it = [it] if it in self._label_sets["issue_type"] else []
        clamped_labels["issue_type"] = [v for v in it if v in self._label_sets["issue_type"]]

        for field_name in ["components", "platform", "impact"]:
            vals = labels.get(field_name, [])
            if isinstance(vals, list):
                clamped_labels[field_name] = [v for v in vals if v in self._label_sets[field_name]]
            else:
                clamped_labels[field_name] = []

        nht = record.get("needs_human_triage", False)
        if not clamped_labels["components"]:
            nht = True

        return {
            "schema_version": "1.0",
            "labels": clamped_labels,
            "needs_human_triage": bool(nht),
            "meta": {"source": "llm", "model": self.config.model_name},
        }

    # ── N-Sample Generation ────────────────────────────────────────────

    def _process_one_sample(self, issue_id, title, body, sample_idx):
        result = self._call_with_repair(title, body)
        return issue_id, sample_idx, result

    def _generate_samples(self, df: pd.DataFrame) -> dict[Any, list[dict]]:
        cfg = self.config
        n = cfg.n_samples

        existing = self._load_checkpoint()
        self._raw_samples.update(existing)

        work_items = []
        for _, row in df.iterrows():
            issue_id = row[cfg.col_id]
            have = len(self._raw_samples.get(issue_id, []))
            if have < n:
                title = str(row[cfg.col_title])
                body = str(row[cfg.col_body])
                for s_idx in range(have, n):
                    work_items.append((issue_id, title, body, s_idx))

        unique_issues = len(set(w[0] for w in work_items))
        print(f"Generating {len(work_items)} samples across {unique_issues} issues "
              f"({len(df) - unique_issues} already complete)")

        if not work_items:
            return self._raw_samples

        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        pending_results: dict[Any, list] = {}
        results_lock = threading.Lock()
        completed = 0

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = {
                executor.submit(self._process_one_sample, iid, title, body, s_idx): iid
                for iid, title, body, s_idx in work_items
            }
            for future in as_completed(futures):
                issue_id, sample_idx, result = future.result()
                with results_lock:
                    completed += 1
                    if issue_id not in pending_results:
                        pending_results[issue_id] = list(self._raw_samples.get(issue_id, []))
                    if result is not None:
                        pending_results[issue_id].append(result)
                    if completed % 500 == 0:
                        print(f"  [{completed}/{len(work_items)}] completed")

        for issue_id, samples in pending_results.items():
            self._raw_samples[issue_id] = samples
            self._save_checkpoint_line(issue_id, samples)

        print(f"\nSample generation complete.")
        print(f"Stats: {json.dumps(self.stats, indent=2)}")
        return self._raw_samples

    # ── Checkpoint IO ──────────────────────────────────────────────────

    def _load_checkpoint(self) -> dict[Any, list[dict]]:
        loaded: dict[Any, list[dict]] = {}
        if self._checkpoint_path.exists():
            with open(self._checkpoint_path) as f:
                for line in f:
                    obj = json.loads(line)
                    loaded[obj["id"]] = obj["samples"]
            print(f"Checkpoint: loaded {len(loaded)} issues from {self._checkpoint_path}")
        return loaded

    def _save_checkpoint_line(self, issue_id: Any, samples: list[dict]) -> None:
        with open(self._checkpoint_path, "a") as f:
            f.write(json.dumps({"id": issue_id, "samples": samples}) + "\n")

    # ── Aggregation ────────────────────────────────────────────────────

    def _aggregate_proba(self, samples: list[dict], task: TaskConfig) -> np.ndarray | dict:
            n = len(samples)
            if n == 0:
                if task.name == "issue_type":
                    return np.zeros(len(task.classes))
                return {}
    
            if task.name == "issue_type":
                # Standard per-label frequency for the single-label task
                freqs = np.zeros(len(task.classes))
                for sample in samples:
                    it_list = sample.get("labels", {}).get("issue_type", [])
                    val = it_list[0] if it_list else ""
                    if val in task.classes:
                        freqs[task.classes.index(val)] += 1
                return freqs / n
            else:
                # EXACT SET frequency for multi-label tasks
                exact_set_counts = {}
                for sample in samples:
                    labels = sample.get("labels", {}).get(task.name, [])
                    # Filter to valid classes and sort so ["core", "physics"] == ["physics", "core"]
                    valid_labels = sorted([c for c in labels if c in task.classes])
                    label_tuple = tuple(valid_labels)
                    exact_set_counts[label_tuple] = exact_set_counts.get(label_tuple, 0) + 1
                
                # Return normalized frequencies e.g. {('core', 'physics'): 0.6, ('gui',): 0.4}
                return {k: v / n for k, v in exact_set_counts.items()}
    
    
    def _aggregate_prediction(self, proba: np.ndarray | dict, task: TaskConfig) -> list[str] | str:
        threshold = self.config.aggregation_threshold
            
        if task.name == "issue_type":
            # Pick highest probability class for single-label
            return task.classes[int(np.argmax(proba))]
        else:
            # proba is our dict of exact set frequencies
            # Find if any exact set meets or exceeds the threshold
            for label_tuple, freq in proba.items():
                if freq >= threshold:
                    return list(label_tuple)
            
            # If no exact combination gets a majority, abstain!
            return []

    # ── Protocol Methods ───────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> None:
        print(f"fit(): zero-shot mode, no training needed.")
        print(f"  Train: {len(train_df)} issues (unused)  |  "
              f"Val: {len(val_df) if val_df is not None else 0} issues (unused)")

    def predict_proba(self, df: pd.DataFrame) -> dict[str, list]:
        samples_by_id = self._generate_samples(df)
        result = {}
        for task in self.tasks:
            # Use a list because outputs are mixed type: ndarray (single) or dict (multi)
            proba_list = []
            for i, (_, row) in enumerate(df.iterrows()):
                issue_id = row[self.config.col_id]
                samples = samples_by_id.get(issue_id, [])
                proba_list.append(self._aggregate_proba(samples, task))
            result[task.name] = proba_list
        return result

    def predict(self, df: pd.DataFrame) -> list[dict]:
        proba_dict = self.predict_proba(df)
        records = []
        for i, (_, row) in enumerate(df.iterrows()):
            record = {self.config.col_id: row[self.config.col_id]}
            
            # 1. Get predictions for all tasks
            for task in self.tasks:
                proba_row = proba_dict[task.name][i]
                record[task.name] = self._aggregate_prediction(proba_row, task)

            # 2. Calculate Exact Set Confidence for the primary routing task
            comp_proba = proba_dict["components"][i]
            comps = record.get("components", [])
            
            if comps:
                # Find the exact frequency of the winning set
                comp_tuple = tuple(comps)
                record["confidence"] = float(comp_proba.get(comp_tuple, 0.0))
            else:
                record["confidence"] = 0.0

            # 3. Abstention logic
            record["needs_human_triage"] = (
                len(comps) == 0 or record["confidence"] < self.config.confidence_threshold
            )
            records.append(record)
            
        return records

    def get_run_metadata(self) -> dict:
        return {
            "model_type": "llm",
            "model_name": self.config.model_name,
            "temperature": self.config.temperature,
            "n_samples": self.config.n_samples,
            "aggregation_threshold": self.config.aggregation_threshold,
            "confidence_threshold": self.config.confidence_threshold,
            "max_body_chars": self.config.max_body_chars,
            "max_repair_attempts": self.config.max_repair_attempts,
            "system_prompt": self.system_prompt,
            "label_vocab": {t.name: t.classes for t in self.tasks},
            "stats": self.stats,
            "timestamp": datetime.now().isoformat(),
        }

    def recompute_stats_from_samples(self):
        self.stats = {k: 0 for k in self.stats}
        for issue_id, samples in self._raw_samples.items():
            for s in samples:
                self.stats["total"] += 1
                if s.get("_schema_first_pass", False):
                    self.stats["first_pass_valid"] += 1
                elif s.get("_repair_failed", False):
                    self.stats["repair_failed"] += 1
                elif s.get("_repair_attempt"):
                    self.stats["repaired"] += 1


# ═══════════════════════════════════════════════════════════════════════
# Evaluation (model-agnostic)
# ═══════════════════════════════════════════════════════════════════════

def evaluate_predictions(
    records: list[dict],
    df_true: pd.DataFrame,
    tasks: list[TaskConfig],
    col_id: str = "id",
) -> dict[str, dict]:
    pred_by_id = {r[col_id]: r for r in records}
    results = {}
    for task in tasks:
        mlb = MultiLabelBinarizer(classes=task.classes)
        y_true_lists, y_pred_lists = [], []
        n_covered = 0
        for _, row in df_true.iterrows():
            issue_id = row[col_id]
            pred_rec = pred_by_id.get(issue_id, {})
            gt = row.get(task.label_col, [])
            if isinstance(gt, float) or gt is None:
                gt = []
            elif isinstance(gt, str):
                gt = [gt]
            y_true_lists.append(gt)
            pred = pred_rec.get(task.name, [])
            if isinstance(pred, float) or pred is None:
                pred = []
            elif isinstance(pred, str):
                pred = [pred]
            y_pred_lists.append(pred)
            if len(pred) > 0:
                n_covered += 1
        y_true = mlb.fit_transform(y_true_lists)
        y_pred = mlb.transform(y_pred_lists)
        n_total = len(df_true)
        results[task.name] = {
            "micro_f1": round(f1_score(y_true, y_pred, average="micro", zero_division=0), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "hamming": round(hamming_loss(y_true, y_pred), 4),
            "coverage": round(n_covered / n_total, 4) if n_total > 0 else 0.0,
        }
    return results


def evaluate_ranking(
    proba_dict: dict[str, list],  # <-- Updated type hint
    df_true: pd.DataFrame,
    tasks: list[TaskConfig],
) -> dict[str, dict]:
    results = {}
    for task in tasks:
        if task.name == "issue_type":
            continue
        mlb = MultiLabelBinarizer(classes=task.classes)
        y_true_lists = []
        for _, row in df_true.iterrows():
            gt = row.get(task.label_col, [])
            if isinstance(gt, float) or gt is None:
                gt = []
            elif isinstance(gt, str):
                gt = [gt]
            y_true_lists.append(gt)
        Y_true = mlb.fit_transform(y_true_lists)

        raw_proba_list = proba_dict[task.name]
        marginal_proba = np.zeros((len(df_true), len(task.classes)))

        for i, p in enumerate(raw_proba_list):

            for exact_set, freq in p.items():
                for label in exact_set:
                    if label in task.classes:
                        marginal_proba[i, task.classes.index(label)] += freq              
        proba = marginal_proba


        k = min(5, proba.shape[1])
        top_k_idx = np.argsort(proba, axis=1)[:, -k:]
        hits, n_with = 0, 0
        recall_nums = []
        for i in range(len(Y_true)):
            n_true = Y_true[i].sum()
            if n_true == 0:
                continue
            n_with += 1
            if any(Y_true[i, j] == 1 for j in top_k_idx[i]):
                hits += 1
            recalled = sum(Y_true[i, j] for j in top_k_idx[i])
            recall_nums.append(recalled / n_true)
        results[task.name] = {
            "hit_at_5": round(hits / n_with, 4) if n_with else 0.0,
            "recall_at_5": round(float(np.mean(recall_nums)), 4) if recall_nums else 0.0,
        }
    return results


def coverage_accuracy_curve(
    proba_list: list[dict], 
    df_true: pd.DataFrame, 
    task: TaskConfig
) -> dict[str, list]:
    import numpy as np
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MultiLabelBinarizer

    # Ensure we have our ground truth lists
    mlb = MultiLabelBinarizer(classes=task.classes)
    y_true_lists = []
    for _, row in df_true.iterrows():
        gt = row.get(task.label_col, [])
        if isinstance(gt, float) or gt is None:
            gt = []
        elif isinstance(gt, str):
            gt = [gt]
        y_true_lists.append(gt)

    thresholds = np.linspace(0.1, 1.0, 50)
    coverages = []
    accuracies = []

    for t in thresholds:
        preds_at_t = []
        for p_dict in proba_list:
            # p_dict is an exact set frequency dict, e.g. {('core', 'physics'): 0.8}
            passed_sets = [list(label_set) for label_set, freq in p_dict.items() if freq >= t]
            if passed_sets:
                preds_at_t.append(passed_sets[0])
            else:
                preds_at_t.append([])

        # Calculate Coverage
        cov = np.mean([len(p) > 0 for p in preds_at_t])
        coverages.append(float(cov))

        # Calculate Micro-F1 on covered samples
        covered_idx = [i for i, p in enumerate(preds_at_t) if len(p) > 0]
        if covered_idx:
            gt_cov = [y_true_lists[i] for i in covered_idx]
            pred_cov = [preds_at_t[i] for i in covered_idx]
            
            Y_true = mlb.fit_transform(gt_cov)
            Y_pred = mlb.transform(pred_cov)
            accuracies.append(float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)))
        else:
            accuracies.append(float("nan"))

    return {
        "thresholds": thresholds.tolist(),
        "coverages": coverages,
        "accuracies": accuracies
    }

# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

class LLMTriageRunner:
    """End-to-end runner: load data, run inference, evaluate, save artifacts."""

    def __init__(self, config: LLMRunConfig, tasks: list[TaskConfig]):
        self.config = config
        self.tasks = tasks

        # Load vocab
        with open(config.vocab_path) as f:
            self.vocab = json.load(f)

        # Load environment
        if config.env_path:
            load_dotenv(config.env_path)

        # State (populated by run())
        self.model: GeminiTriageModel | None = None
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        self.predictions: list[dict] | None = None
        self.proba_dict: dict[str, np.ndarray] | None = None
        self.metrics: dict[str, dict] | None = None
        self.ranking: dict[str, dict] | None = None
        self.comp_curve: list[dict] | None = None
        self.schema_summary: dict | None = None

    # ── IO ─────────────────────────────────────────────────────────────

    @staticmethod
    def _load_json_df(path: str) -> pd.DataFrame:
        with open(path, "r", encoding="utf-8") as f:
            return pd.DataFrame(json.load(f))

    def load_splits(self) -> None:
        self.train_df = self._load_json_df(self.config.train_path)
        self.val_df = self._load_json_df(self.config.val_path)
        self.test_df = self._load_json_df(self.config.test_path)
        print(f"Train: {len(self.train_df)} | Val: {len(self.val_df)} | Test: {len(self.test_df)}")

    # ── Inference ──────────────────────────────────────────────────────

    def infer(self) -> None:
        assert self.test_df is not None, "Call load_splits() first."
        self.model = GeminiTriageModel(self.config, self.tasks, self.vocab)
        assert isinstance(self.model, TriageModel)
        self.model.fit(self.train_df, self.val_df)
        self.predictions = self.model.predict(self.test_df)
        self.proba_dict = self.model.predict_proba(self.test_df)  # cached,  no extra API calls
        print(f"Generated {len(self.predictions)} predictions")

    # ── Evaluation ─────────────────────────────────────────────────────

    def evaluate(self) -> None:
        assert self.predictions is not None, "Call infer() first."
        self.metrics = evaluate_predictions(
            self.predictions, self.test_df, self.tasks, col_id=self.config.col_id,
        )
        self.ranking = evaluate_ranking(self.proba_dict, self.test_df, self.tasks)

        comp_task = self._task_by_name("components")
        self.comp_curve = coverage_accuracy_curve(
            self.proba_dict["components"], self.test_df, comp_task,
        )

        # Schema stats
        self.model.recompute_stats_from_samples()
        raw_records = [s for samples in self.model._raw_samples.values() for s in samples]
        clean_records = [{k: v for k, v in r.items() if not k.startswith("_")} for r in raw_records]
        results = self.model.schema_validator.validate_many(clean_records)

        n_total = len(raw_records)
        n_first_pass = self.model.stats["first_pass_valid"]
        n_repaired = self.model.stats["repaired"]
        n_clamp = self.model.stats["repair_failed"]
        n_post_all_valid = sum(1 for r in results if r["ok"])

        self.schema_summary = {
            "first_pass_valid": n_first_pass,
            "llm_repaired": n_repaired,
            "clamp_salvaged": n_clamp,
            "post_all_valid": n_post_all_valid,
            "total": n_total,
        }

        # Print summary
        for name, m in self.metrics.items():
            print(f"{name:>12}: micro-F1={m['micro_f1']:.4f}  macro-F1={m['macro_f1']:.4f}  "
                  f"hamming={m['hamming']:.4f}  coverage={m['coverage']:.4f}")
        for name, r in self.ranking.items():
            print(f"{name:>12}: Hit@5={r['hit_at_5']:.4f}  Recall@5={r['recall_at_5']:.4f}")

    # ── Save ───────────────────────────────────────────────────────────

    def save_artifacts(self) -> Path:
        assert self.metrics is not None, "Call evaluate() first."
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg = self.config
        run_dir = Path(cfg.out_dir) / f"llm_{cfg.model_name.replace('/', '_')}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Metadata
        metadata = self.model.get_run_metadata()
        metadata["metrics"] = self.metrics
        metadata["ranking"] = self.ranking
        metadata["schema_validation"] = self.schema_summary
        (run_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8",
        )

        # Predictions
        with open(run_dir / "predictions.jsonl", "w") as f:
            for rec in self.predictions:
                f.write(json.dumps(rec, default=str) + "\n")

        # Probabilities
        for task_name, proba in self.proba_dict.items():
            np.save(run_dir / f"proba_{task_name}.npy", proba)

        # Coverage curve
        if self.comp_curve:
            (run_dir / "coverage_curve_components.json").write_text(
                json.dumps(self.comp_curve, indent=2), encoding="utf-8",
            )

        # Eval metrics (standalone file for easy loading by comparison script)
        (run_dir / "eval.json").write_text(
            json.dumps({
                "metrics": self.metrics,
                "ranking": self.ranking,
                "schema_validation": self.schema_summary,
            }, indent=2),
            encoding="utf-8",
        )

        print(f"Run saved to {run_dir}")
        return run_dir

    # ── Main ───────────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        self.load_splits()
        self.infer()
        self.evaluate()
        run_dir = self.save_artifacts()
        return {
            "run_dir": str(run_dir),
            "metrics": self.metrics,
            "ranking": self.ranking,
            "schema_summary": self.schema_summary,
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    def _task_by_name(self, name: str) -> TaskConfig:
        for t in self.tasks:
            if t.name == name:
                return t
        raise KeyError(name)