"""
run_experiments.py:Full experiment pipeline for Godot issue triage.

Runs in order:
    1. Baseline          TF-IDF train/tune/evaluate + schema outputs
    2. LLM clean run     full test-set inference with updated prompt
    3. Robustness suite  baseline re-run + LLM sampled re-inference (N=3, 700 issues)
    4. Comparison        baseline vs LLM tables and final_results/ artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Project root resolution ───────────────────────────────────────────────────
_THIS_FILE = Path(__file__).resolve()
# Walk up until we find src/schemas (a reliable project-root marker)
PROJECT_ROOT = _THIS_FILE
for _ in range(6):
    if (PROJECT_ROOT / "src" / "schemas").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
else:
    raise RuntimeError(
        f"Could not locate project root from {_THIS_FILE}. "
        "Make sure src/schemas/ exists in project tree."
    )

# Add src/scripts to path so package imports resolve
SRC_SCRIPTS = PROJECT_ROOT / "src" / "scripts"
if str(SRC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SRC_SCRIPTS))

# ── Paths  ─────────────────────────────────────
DATA_DIR    = SRC_SCRIPTS / "data_collection" / "data" / "processed"
TRAIN_PATH  = DATA_DIR / "train.json"
VAL_PATH    = DATA_DIR / "val.json"
TEST_PATH   = DATA_DIR / "test.json"
VOCAB_PATH  = DATA_DIR / "label_vocab.json"
SCHEMA_PATH = PROJECT_ROOT / "src" / "schemas" / "triage_schema.json"
ENV_PATH    = PROJECT_ROOT / ".env.local"

# Default baseline run dir overridden at runtime if --baseline-run is passed
# or if a fresh baseline run is executed.
BASELINE_RUN    = PROJECT_ROOT / "baseline" / "runs" / "20260226_212256"
FINAL_RESULTS   = PROJECT_ROOT / "final_results"
ROBUSTNESS_DIR  = FINAL_RESULTS / "robustness"


# ─────────────────────────────────────────────────────────────────────────────
# Step 0: Baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline() -> Path:
    """
    Train, tune, and evaluate the TF-IDF baseline, then save artifacts.
    Mirrors the runner.py __main__ block exactly.
    Returns the run_dir path.
    """
    import numpy as np
    from scripts.baseline.runner import ExperimentRunner, RunnerConfig, TaskSpec

    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    tasks = [
        TaskSpec(name="issue_type", label_col="issue_type", classes=vocab["issue_type"], tune_method="global_micro"),
        TaskSpec(name="components", label_col="topic",      classes=vocab["components"], tune_method="global_micro"),
        TaskSpec(name="platform",   label_col="platform",   classes=vocab["platform"],   tune_method="global_micro"),
        TaskSpec(name="impact",     label_col="impact",     classes=vocab["impact"],     tune_method="global_micro"),
    ]

    cfg = RunnerConfig(
        train_path=str(TRAIN_PATH),
        val_path=str(VAL_PATH),
        test_path=str(TEST_PATH),
        text_col="text_clean",
        out_dir=str(PROJECT_ROOT / "baseline" / "runs"),
        schema_path=str(SCHEMA_PATH),
        threshold_grid=np.linspace(0.05, 0.95, 301),
    )

    print("=" * 60)
    print("STEP 1: Baseline (TF-IDF)")
    print(f"  train: {TRAIN_PATH}")
    print(f"  test:  {TEST_PATH}")
    print("=" * 60)

    runner = ExperimentRunner(cfg, tasks)
    res = runner.run(build_schema_on_test=True)
    run_dir = Path(res["run_dir"])
    print(f"\nBaseline run complete. Artifacts saved to: {run_dir}")

    for task_name, info in runner.eval_info.items():
        m = info.get("metrics_all", {})
        micro = m.get("micro_f1", float("nan"))
        macro = m.get("macro_f1", float("nan"))
        cov   = m.get("coverage", float("nan"))
        print(f"  {task_name:>12}: micro-F1={micro:.4f}  macro-F1={macro:.4f}  cov={cov:.4f}")

    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: LLM clean run
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_clean(n_samples: int = 20, temperature: float = 0.7) -> Path:
    """
    Instantiate LLMTriageRunner and calls .run().
    Returns the run_dir path.
    """
    from scripts.llm.llm_triage_model import LLMRunConfig, TaskConfig, LLMTriageRunner

    with open(VOCAB_PATH) as f:
        vocab = json.load(f)

    config = LLMRunConfig(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        test_path=TEST_PATH,
        vocab_path=VOCAB_PATH,
        schema_path=SCHEMA_PATH,
        env_path=ENV_PATH,
        n_samples=n_samples,
        temperature=temperature,
    )

    tasks = [
        TaskConfig(name="issue_type", label_col="issue_type", classes=vocab["issue_type"]),
        TaskConfig(name="components", label_col="topic",      classes=vocab["components"]),
        TaskConfig(name="platform",   label_col="platform",   classes=vocab["platform"]),
        TaskConfig(name="impact",     label_col="impact",     classes=vocab["impact"]),
    ]

    print("=" * 60)
    print("STEP 2: LLM clean run")
    print(f"  model:       {config.model_name}")
    print(f"  n_samples:   {n_samples}")
    print(f"  temperature: {temperature}")
    print(f"  test set:    {TEST_PATH}")
    print("=" * 60)

    result = LLMTriageRunner(config, tasks).run()
    run_dir = Path(result["run_dir"])
    print(f"\nLLM run complete. Artifacts saved to: {run_dir}")

    # Print headline metrics
    metrics = result.get("metrics", {})
    for task, m in metrics.items():
        micro = m.get("micro_f1", float("nan"))
        macro = m.get("macro_f1", float("nan"))
        cov   = m.get("coverage", float("nan"))
        print(f"  {task:>12}: micro-F1={micro:.4f}  macro-F1={macro:.4f}  cov={cov:.4f}")

    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Robustness
# ─────────────────────────────────────────────────────────────────────────────

def run_robustness(llm_run_dir: Path, baseline_run_dir: Path) -> None:
    """
    Import and call run_robustness_eval from robustness.py.
    Assumes robustness.py lives in the same directory as this file.
    """
    import importlib.util

    rob_path = _THIS_FILE.parent / "robustness.py"
    if not rob_path.exists():
        raise FileNotFoundError(
            f"robustness.py not found at {rob_path}. "
            "Make sure it is in the same directory as run_experiments.py."
        )

    spec = importlib.util.spec_from_file_location("robustness", rob_path)
    rob  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rob)

    print("\n" + "=" * 60)
    print("STEP 3: Robustness suite")
    print(f"  baseline run: {baseline_run_dir}")
    print(f"  LLM run:      {llm_run_dir}")
    print(f"  sample n:     700 issues × 4 perturbations × N=3 samples")
    print(f"  ~8,400 API calls total (resumable via cache)")
    print("=" * 60)

    rob.run_robustness_eval(
        baseline_run_dir=baseline_run_dir,
        llm_run_dir=llm_run_dir,
        vocab_path=VOCAB_PATH,
        out_dir=ROBUSTNESS_DIR,
        primary_task="components",
        llm_sample_n=700,
        llm_sample_seed=42,
        llm_n_samples=20,
        llm_temperature=0.7,
        llm_aggregation_threshold=0.5,
        llm_requests_per_minute=3500,
        skip_llm=False,
    )
    print(f"\nRobustness artifacts saved to: {ROBUSTNESS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(llm_run_dir: Path, baseline_run_dir: Path) -> None:
    from scripts.comparison_loader import RunComparison
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MultiLabelBinarizer

    print("\n" + "=" * 60)
    print("STEP 4: Baseline vs LLM comparison")
    print(f"  baseline: {baseline_run_dir}")
    print(f"  LLM:      {llm_run_dir}")
    print("=" * 60)

    FINAL_RESULTS.mkdir(parents=True, exist_ok=True)

    comp = RunComparison(
        baseline_run_dir=str(baseline_run_dir),
        llm_run_dir=str(llm_run_dir),
    )
    comp.load()
    comp.print_summary_table()
    comp.save_all(str(FINAL_RESULTS) + "/")
    print(f"\nComparison artifacts saved to: {FINAL_RESULTS}/")

    # ── LLM Sweep Logic for Coverage-Accuracy Curve ──────────────────────────
    print("\nGenerating LLM Coverage-Accuracy Curve...")
    try:
        # Load Vocab for classes
        with open(VOCAB_PATH) as f:
            vocab = json.load(f)
        classes = vocab.get("components", [])

        # Load Ground Truth from the test set
        df_test = pd.read_json(TEST_PATH, lines=True) if str(TEST_PATH).endswith('.jsonl') else pd.read_json(TEST_PATH)
        id_col = next((c for c in ["id", "number", "issue_id"] if c in df_test.columns), "id")

        gt_lists = []
        for v in df_test.get("topic", pd.Series(dtype=object)): # 'topic' is the GT col for components
            if isinstance(v, list): gt_lists.append(v)
            elif isinstance(v, str): gt_lists.append([v] if v else [])
            else: gt_lists.append([])

        # Load LLM Predictions generated from the clean run
        candidates = list(Path(llm_run_dir).glob("predictions*.jsonl")) + list(Path(llm_run_dir).glob("*.jsonl"))
        preds = []
        with open(candidates[0]) as f:
            for line in f:
                if line.strip(): preds.append(json.loads(line))

        pred_dict = {p.get(id_col) or p.get("id") or p.get("number"): p for p in preds}

        # Align predictions to the test set order
        aligned_preds = []
        aligned_confs = []
        for _, row in df_test.iterrows():
            iid = row.get(id_col)
            p = pred_dict.get(iid, {})
            comps = p.get("components", [])
            conf = p.get("confidence", 0.0) # Exact Set frequency from our updated model
            aligned_preds.append(comps if isinstance(comps, list) else [])
            aligned_confs.append(float(conf))

        # Perform the Sweep
        thresholds = np.linspace(0.1, 1.0, 50)
        coverages = []
        accuracies = []

        mlb = MultiLabelBinarizer(classes=classes)

        for t in thresholds:
            preds_at_t = []
            for comps, conf in zip(aligned_preds, aligned_confs):
                # If the exact set frequency meets the threshold, keep it. Otherwise, abstain.
                if conf >= t:
                    preds_at_t.append(comps)
                else:
                    preds_at_t.append([])

            # Calculate Coverage
            cov = np.mean([len(p) > 0 for p in preds_at_t])
            coverages.append(cov)

            # Calculate F1 only on COVERED samples (matches baseline logic)
            covered_idx = [i for i, p in enumerate(preds_at_t) if len(p) > 0]
            if covered_idx:
                gt_cov = [gt_lists[i] for i in covered_idx]
                pred_cov = [preds_at_t[i] for i in covered_idx]
                
                Y_true = mlb.fit_transform(gt_cov)
                Y_pred = mlb.transform(pred_cov)
                accuracies.append(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
            else:
                accuracies.append(np.nan)

        # Plot the Curve
        plt.figure(figsize=(8, 6))
        plt.plot(coverages, accuracies, marker='o', markersize=4, linestyle='-', color='#DD8452', linewidth=2)
        plt.xlabel("Coverage (fraction with ≥1 predicted label)")
        plt.ylabel("Micro-F1 on covered samples")
        plt.title("LLM Coverage-Accuracy Tradeoff (Exact Set Consensus)")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save Plot
        plot_path = FINAL_RESULTS / "llm_coverage_accuracy_curve.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        # Save raw data to CSV
        curve_df = pd.DataFrame({"threshold": thresholds, "coverage": coverages, "micro_f1": accuracies})
        curve_path = FINAL_RESULTS / "llm_coverage_accuracy_data.csv"
        curve_df.to_csv(curve_path, index=False)
        
        print(f"  LLM Curve plot saved to: {plot_path}")
        print(f"  LLM Curve data saved to: {curve_path}")

    except Exception as e:
        print(f"\n  [Warning] Could not generate LLM sweep curve. Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full experiment pipeline: baseline → LLM → robustness → comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline-run", default=None,
        help="Path to an existing baseline run dir. If set, skips baseline training.",
    )
    parser.add_argument(
        "--llm-run", default=None,
        help="Path to an existing LLM run dir. If set, skips LLM inference.",
    )
    parser.add_argument(
        "--skip-robustness", action="store_true",
        help="Skip robustness suite.",
    )
    parser.add_argument(
        "--skip-comparison", action="store_true",
        help="Skip comparison step.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="N samples per issue for the clean LLM run.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for the clean LLM run.",
    )
    args = parser.parse_args()

    # ── Step 0: Baseline ─────────────────────────────────────────────
    if args.baseline_run:
        baseline_run_dir = Path(args.baseline_run).resolve()
        if not baseline_run_dir.exists():
            print(f"ERROR: --baseline-run path does not exist: {baseline_run_dir}")
            sys.exit(1)
        print(f"Using existing baseline run: {baseline_run_dir}")
    else:
        baseline_run_dir = run_baseline()

    # ── Step 1: LLM clean run ─────────────────────────────────────────
    if args.llm_run:
        llm_run_dir = Path(args.llm_run).resolve()
        if not llm_run_dir.exists():
            print(f"ERROR: --llm-run path does not exist: {llm_run_dir}")
            sys.exit(1)
        print(f"\nUsing existing LLM run: {llm_run_dir}")
    else:
        llm_run_dir = run_llm_clean(
            n_samples=args.n_samples,
            temperature=args.temperature,
        )

    # ── Step 2: Robustness ────────────────────────────────────────────
    if not args.skip_robustness:
        run_robustness(llm_run_dir, baseline_run_dir)
    else:
        print("\nSkipping robustness (--skip-robustness set).")

    # ── Step 3: Comparison ────────────────────────────────────────────
    if not args.skip_comparison:
        run_comparison(llm_run_dir, baseline_run_dir)
    else:
        print("\nSkipping comparison (--skip-comparison set).")

    print("\n" + "=" * 60)
    print("All steps complete.")
    print(f"  Baseline:      {baseline_run_dir}")
    print(f"  LLM run:       {llm_run_dir}")
    print(f"  Robustness:    {ROBUSTNESS_DIR}")
    print(f"  Final results: {FINAL_RESULTS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
