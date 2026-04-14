"""
Comparison Loader — load and compare TF-IDF baseline and LLM run artifacts.

Usage:
    from comparison_loader import RunComparison

    comp = RunComparison(
        baseline_run_dir="runs/20260409_120000",
        llm_run_dir="runs/llm_gemini-2.5-flash-lite_20260409_130000",
    )
    comp.load()
    comp.print_comparison_table()
    comp.save_comparison_table("final_results/comparison.csv")
    comp.print_schema_summary()
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


class RunComparison:

    def __init__(self, baseline_run_dir: str, llm_run_dir: str):
        self.baseline_dir = Path(baseline_run_dir)
        self.llm_dir = Path(llm_run_dir)

        # Populated by load()
        self.baseline_eval: dict | None = None
        self.llm_eval: dict | None = None
        self.baseline_config: dict | None = None
        self.llm_metadata: dict | None = None
        self.baseline_curve: list[dict] | None = None
        self.llm_curve: list[dict] | None = None

    # ── Load ───────────────────────────────────────────────────────────

    def load(self) -> None:
        # Baseline
        baseline_eval_path = self.baseline_dir / "eval.json"
        if not baseline_eval_path.exists():
            raise FileNotFoundError(f"Baseline eval.json not found at {baseline_eval_path}")
        self.baseline_eval = json.loads(baseline_eval_path.read_text(encoding="utf-8"))

        baseline_config_path = self.baseline_dir / "config.json"
        if baseline_config_path.exists():
            self.baseline_config = json.loads(baseline_config_path.read_text(encoding="utf-8"))

        # LLM
        llm_eval_path = self.llm_dir / "eval.json"
        if not llm_eval_path.exists():
            raise FileNotFoundError(f"LLM eval.json not found at {llm_eval_path}")
        self.llm_eval = json.loads(llm_eval_path.read_text(encoding="utf-8"))

        llm_metadata_path = self.llm_dir / "run_metadata.json"
        if llm_metadata_path.exists():
            self.llm_metadata = json.loads(llm_metadata_path.read_text(encoding="utf-8"))

        # Coverage curves
        baseline_curve_path = self.baseline_dir / "coverage_curve_components.json"
        if not baseline_curve_path.exists():
            # Baseline stores curve inside eval.json per-task
            comp_eval = self.baseline_eval.get("components", {})
            self.baseline_curve = comp_eval.get("coverage_curve")
        else:
            self.baseline_curve = json.loads(baseline_curve_path.read_text(encoding="utf-8"))

        llm_curve_path = self.llm_dir / "coverage_curve_components.json"
        if llm_curve_path.exists():
            self.llm_curve = json.loads(llm_curve_path.read_text(encoding="utf-8"))

        print(f"Loaded baseline from {self.baseline_dir}")
        print(f"Loaded LLM from {self.llm_dir}")

    # ── Metrics extraction ─────────────────────────────────────────────

    def _get_baseline_metrics(self) -> dict[str, dict]:
        """Extract per-task metrics from baseline eval.json."""
        # Baseline runner stores metrics nested per task with metrics_all
        raw = self.baseline_eval
        metrics = {}
        for task_name, task_data in raw.items():
            if not isinstance(task_data, dict):
                continue
            if "metrics_all" in task_data:
                m = task_data["metrics_all"]
                metrics[task_name] = {
                    "micro_f1": m.get("micro_f1", 0),
                    "macro_f1": m.get("macro_f1", 0),
                    "hamming": m.get("hamming_loss", m.get("hamming", 0)),
                    "coverage": m.get("coverage", 0),
                    "hit_at_5": task_data.get("hit_at_5", None),
                    "recall_at_5": task_data.get("recall_at_5_micro", None),
                }
            elif "micro_f1" in task_data:
                # Already flat format
                metrics[task_name] = task_data
        return metrics

    def _get_llm_metrics(self) -> dict[str, dict]:
        """Extract per-task metrics from LLM eval.json."""
        raw = self.llm_eval
        metrics = raw.get("metrics", raw)
        ranking = raw.get("ranking", {})

        result = {}
        for task_name, m in metrics.items():
            if not isinstance(m, dict):
                continue
            entry = {
                "micro_f1": m.get("micro_f1", 0),
                "macro_f1": m.get("macro_f1", 0),
                "hamming": m.get("hamming", 0),
                "coverage": m.get("coverage", 0),
            }
            if task_name in ranking:
                entry["hit_at_5"] = ranking[task_name].get("hit_at_5")
                entry["recall_at_5"] = ranking[task_name].get("recall_at_5")
            result[task_name] = entry
        return result

    # ── Comparison table ───────────────────────────────────────────────

    def build_comparison_table(self) -> pd.DataFrame:
        """Build a DataFrame comparing baseline and LLM across all tasks and metrics."""
        baseline = self._get_baseline_metrics()
        llm = self._get_llm_metrics()

        all_tasks = sorted(set(list(baseline.keys()) + list(llm.keys())))
        metric_cols = ["micro_f1", "macro_f1", "hamming", "coverage", "hit_at_5", "recall_at_5"]

        rows = []
        for task in all_tasks:
            b = baseline.get(task, {})
            l = llm.get(task, {})
            for metric in metric_cols:
                bv = b.get(metric)
                lv = l.get(metric)
                if bv is None and lv is None:
                    continue
                delta = None
                if bv is not None and lv is not None:
                    delta = round(lv - bv, 4)
                rows.append({
                    "task": task,
                    "metric": metric,
                    "tfidf_baseline": round(bv, 4) if bv is not None else None,
                    "llm": round(lv, 4) if lv is not None else None,
                    "delta": delta,
                })

        return pd.DataFrame(rows)

    def build_summary_table(self) -> pd.DataFrame:
        """Build a compact per-task summary (one row per task, key metrics only)."""
        baseline = self._get_baseline_metrics()
        llm = self._get_llm_metrics()
        all_tasks = sorted(set(list(baseline.keys()) + list(llm.keys())))

        rows = []
        for task in all_tasks:
            b = baseline.get(task, {})
            l = llm.get(task, {})
            rows.append({
                "task": task,
                "baseline_micro_f1": b.get("micro_f1"),
                "llm_micro_f1": l.get("micro_f1"),
                "delta_micro_f1": round(l.get("micro_f1", 0) - b.get("micro_f1", 0), 4)
                    if b.get("micro_f1") is not None and l.get("micro_f1") is not None else None,
                "baseline_macro_f1": b.get("macro_f1"),
                "llm_macro_f1": l.get("macro_f1"),
                "baseline_coverage": b.get("coverage"),
                "llm_coverage": l.get("coverage"),
            })

        return pd.DataFrame(rows)

    # ── Schema summary ─────────────────────────────────────────────────

    def get_schema_summary(self) -> dict | None:
        """Return LLM schema validation summary if available."""
        if self.llm_eval and "schema_validation" in self.llm_eval:
            return self.llm_eval["schema_validation"]
        if self.llm_metadata and "schema_validation" in self.llm_metadata:
            return self.llm_metadata["schema_validation"]
        return None

    # ── Print helpers ──────────────────────────────────────────────────

    def print_comparison_table(self) -> None:
        df = self.build_comparison_table()
        print("\n" + "=" * 70)
        print("COMPARISON: TF-IDF Baseline vs LLM")
        print("=" * 70)
        print(df.to_string(index=False))
        print()

    def print_summary_table(self) -> None:
        df = self.build_summary_table()
        print("\n" + "=" * 70)
        print("SUMMARY: TF-IDF Baseline vs LLM")
        print("=" * 70)
        print(df.to_string(index=False))
        print()

    def print_schema_summary(self) -> None:
        schema = self.get_schema_summary()
        if schema is None:
            print("No schema validation summary available.")
            return
        total = schema.get("total", 0)
        print("\n" + "=" * 70)
        print("LLM SCHEMA VALIDATION")
        print("=" * 70)
        for key in ["first_pass_valid", "llm_repaired", "clamp_salvaged", "post_all_valid"]:
            val = schema.get(key, 0)
            pct = f"{val/total*100:.1f}%" if total > 0 else "N/A"
            print(f"  {key:>20}: {val}/{total} ({pct})")
        print()

    # ── Save ───────────────────────────────────────────────────────────

    def save_comparison_table(self, path: str) -> None:
        df = self.build_comparison_table()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Comparison table saved to {path}")

    def save_summary_table(self, path: str) -> None:
        df = self.build_summary_table()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Summary table saved to {path}")

    def save_all(self, out_dir: str = "final_results") -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.save_comparison_table(str(out / "comparison_full.csv"))
        self.save_summary_table(str(out / "comparison_summary.csv"))

        # Save schema summary
        schema = self.get_schema_summary()
        if schema:
            (out / "schema_validation.json").write_text(
                json.dumps(schema, indent=2), encoding="utf-8",
            )

        # Copy coverage curves
        if self.baseline_curve:
            (out / "coverage_curve_baseline.json").write_text(
                json.dumps(self.baseline_curve, indent=2), encoding="utf-8",
            )
        if self.llm_curve:
            (out / "coverage_curve_llm.json").write_text(
                json.dumps(self.llm_curve, indent=2), encoding="utf-8",
            )

        print(f"All comparison artifacts saved to {out}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python comparison_loader.py <baseline_run_dir> <llm_run_dir>")
        sys.exit(1)

    comp = RunComparison(sys.argv[1], sys.argv[2])
    comp.load()
    comp.print_summary_table()
    comp.print_comparison_table()
    comp.print_schema_summary()
    comp.save_all()
