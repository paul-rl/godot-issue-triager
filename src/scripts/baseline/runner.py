from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from scripts.baseline.tf_idf_baseline import TfidfOvrMultilabelBaseline
from scripts.schema.schema_validator import TriageSchemaValidator
from scripts.schema.baseline_output_builder import BaselineOutputBuilder


# ----------------------------
# Specs / Config
# ----------------------------

@dataclass(frozen=True)
class TaskSpec:
    """
    One prediction task / label-group.
    Example: name="components", label_col="components", classes=[...]
    """
    name: str
    label_col: str
    classes: list[str]
    threshold_policy: str = "global"   # "global" or "per_label"
    tune_method: str = "global_micro"  # "global_micro" | "global_macro" | "per_label_max_f1" | "per_label_min_precision"
    min_precision: float = 0.70        # used only when tune_method="per_label_min_precision"


@dataclass(frozen=True)
class RunnerConfig:
    train_path: str
    val_path: str
    test_path: str
    text_col: str = "text_clean"
    id_col: str | None = None  # optional, e.g. "number" or "issue_id"
    out_dir: str = "runs"
    schema_path: str | None = None  # optional: build schema outputs if provided

    # Model hyperparams (passed into baseline)
    tfidf_params: dict[str, Any] | None = None
    lr_params: dict[str, Any] | None = None

    # Threshold sweep grid
    threshold_grid: np.ndarray | None = None

    # Schema output behavior
    schema_version: str = "1.0"
    tau_components: float = 0.5
    margin: float = 0.0


# ----------------------------
# Runner
# ----------------------------

class ExperimentRunner:
    def __init__(self, config: RunnerConfig, tasks: list[TaskSpec]):
        self.cfg = config
        self.tasks = tasks

        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

        self.models: dict[str, TfidfOvrMultilabelBaseline] = {}
        self.tuning_info: dict[str, Any] = {}
        self.eval_info: dict[str, Any] = {}

        self.schema_validator: TriageSchemaValidator | None = None
        self.output_builder: BaselineOutputBuilder | None = None

    # ---------- IO ----------

    @staticmethod
    def _load_json_df(path: str) -> pd.DataFrame:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return pd.DataFrame(obj)

    def load_splits(self) -> None:
        self.train_df = self._load_json_df(self.cfg.train_path)
        self.val_df = self._load_json_df(self.cfg.val_path)
        self.test_df = self._load_json_df(self.cfg.test_path)

        # cheap sanity checks
        for name, df in [("train", self.train_df), ("val", self.val_df), ("test", self.test_df)]:
            if self.cfg.text_col not in df.columns:
                raise ValueError(f"{name} missing text_col='{self.cfg.text_col}'. Columns: {list(df.columns)}")

    # ---------- Train / Tune / Eval ----------

    def train(self) -> None:
        if self.train_df is None:
            raise RuntimeError("Call load_splits() first.")

        for spec in self.tasks:
            model = TfidfOvrMultilabelBaseline(
                class_names=spec.classes,
                tfidf_params=self.cfg.tfidf_params,
                lr_params=self.cfg.lr_params,
                threshold_policy=spec.threshold_policy,
            )
            model.fit(self.train_df, text_col=self.cfg.text_col, label_col=spec.label_col)
            self.models[spec.name] = model

    def tune(self) -> None:
        if self.val_df is None:
            raise RuntimeError("Call load_splits() first.")

        grid = self.cfg.threshold_grid
        if grid is None:
            grid = np.linspace(0.05, 0.95, 18)

        for spec in self.tasks:
            model = self.models[spec.name]

            if spec.tune_method == "global_micro":
                info = model.tune_global_threshold(
                    self.val_df,
                    text_col=self.cfg.text_col,
                    label_col=spec.label_col,
                    grid=grid,
                    select_by="micro",
                )
            elif spec.tune_method == "global_macro":
                info = model.tune_global_threshold(
                    self.val_df,
                    text_col=self.cfg.text_col,
                    label_col=spec.label_col,
                    grid=grid,
                    select_by="macro",
                )
            elif spec.tune_method == "per_label_max_f1":
                thr, best_f1s = model.tune_per_label_thresholds_max_f1(
                    self.val_df,
                    text_col=self.cfg.text_col,
                    label_col=spec.label_col,
                    grid=np.linspace(0.05, 0.95, 19),
                )
                info = {"per_label_thresholds": thr.tolist(), "per_label_best_f1s": best_f1s.tolist()}
            elif spec.tune_method == "per_label_min_precision":
                thr, summary = model.tune_per_label_thresholds_min_precision(
                    self.val_df,
                    text_col=self.cfg.text_col,
                    label_col=spec.label_col,
                    min_precision=spec.min_precision,
                    grid=np.linspace(0.05, 0.95, 19),
                )
                info = {"per_label_thresholds": thr.tolist(), "summary": summary}
            else:
                raise ValueError(f"Unknown tune_method: {spec.tune_method}")

            self.tuning_info[spec.name] = info

    def evaluate(self) -> None:
        if self.test_df is None:
            raise RuntimeError("Call load_splits() first.")

        for spec in self.tasks:
            model = self.models[spec.name]
            out = model.evaluate(
                self.test_df,
                text_col=self.cfg.text_col,
                label_col=spec.label_col,
                with_report=True,
            )

            # optional: ranking metrics
            proba = out["proba"]
            Y_true = out["Y_true"]

            out["hit_at_5"] = model.hit_rate_at_k(proba, Y_true, k=5)
            out["recall_at_5_micro"] = model.recall_at_k(proba, Y_true, k=5, average="micro")

            # coverage–abstention curve (threshold sweep)
            grid = self.cfg.threshold_grid
            if grid is None:
                grid = np.linspace(0.05, 0.95, 30)

            grid = np.unique(np.append(grid, float(model.global_threshold)))
            grid = np.sort(grid)

            out["coverage_curve"] = model.coverage_curve_from_proba(proba, Y_true, grid=grid)

            # trim large arrays before storing (save separately if you want)
            out_small = {
                "metrics_all": out["metrics_all"],
                "metrics_at_coverage": out["metrics_at_coverage"],
                "hit_at_5": out["hit_at_5"],
                "recall_at_5_micro": out["recall_at_5_micro"],
                "coverage": out["metrics_all"]["coverage"],
                "avg_pred_labels": out["metrics_all"]["avg_pred_labels"],
                "coverage_curve": out["coverage_curve"],
                "report": out["report"],
            }
            self.eval_info[spec.name] = out_small

    # ---------- Schema output building ----------

    @staticmethod
    def _row_score_dict(proba_row: np.ndarray, class_names: list[str]) -> dict[str, float]:
        return {class_names[i]: float(proba_row[i]) for i in range(len(class_names))}

    def _init_schema_tools(self) -> None:
        if not self.cfg.schema_path:
            return

        self.schema_validator = TriageSchemaValidator(self.cfg.schema_path)
        self.output_builder = BaselineOutputBuilder(
            schema_version=self.cfg.schema_version,
            thresholds={},  # we’ll fill below
            tau_components=self.cfg.tau_components,
            margin=self.cfg.margin,
        )

    def build_schema_outputs(self, df: pd.DataFrame) -> list[dict]:
        """
        Builds schema outputs for each row of df.
        Requires tasks named exactly: issue_type, components, platform, impact
        (or adjust mapping below).
        """
        if not self.cfg.schema_path:
            raise RuntimeError("schema_path not set in RunnerConfig.")
        if not self.output_builder:
            self._init_schema_tools()

        assert self.output_builder is not None

        required = ["issue_type", "components", "platform", "impact"]
        for k in required:
            if k not in self.models:
                raise ValueError(f"Missing task '{k}' in models. Have: {list(self.models.keys())}")

        # If you tuned global thresholds, you can wire them into the builder:
        thresholds = {}
        for spec in self.tasks:
            if spec.name in required:
                m = self.models[spec.name]
                thresholds[spec.name] = float(m.global_threshold)
        self.output_builder.thresholds = thresholds

        # Compute proba per task in batch
        probs = {}
        for name in required:
            probs[name] = self.models[name].predict_proba(df, text_col=self.cfg.text_col)

        outputs = []
        for i in range(len(df)):
            out = self.output_builder.build(
                issue_type_scores=self._row_score_dict(probs["issue_type"][i], self.tasks_by_name("issue_type").classes),
                component_scores=self._row_score_dict(probs["components"][i], self.tasks_by_name("components").classes),
                platform_scores=self._row_score_dict(probs["platform"][i], self.tasks_by_name("platform").classes),
                impact_scores=self._row_score_dict(probs["impact"][i], self.tasks_by_name("impact").classes),
                meta={
                    "source": "baseline",
                    "row_idx": i,
                    **({self.cfg.id_col: df.iloc[i][self.cfg.id_col]} if self.cfg.id_col and self.cfg.id_col in df.columns else {}),
                },
            )
            outputs.append(out)
        return outputs

    def validate_schema_outputs(
        self,
        schema_outputs: list[dict],
        *,
        max_examples: int = 25,
        max_error_types: int = 20,
    ) -> dict[str, Any]:
        """
        Validate schema outputs and return a compact summary for reporting.
        """
        if not self.schema_validator:
            raise RuntimeError("schema_validator not initialized. Call _init_schema_tools() first.")

        results = self.schema_validator.validate_many(schema_outputs)

        total = len(results)
        valid_count = sum(1 for r in results if r["ok"])
        invalid_count = total - valid_count
        valid_rate = (valid_count / total) if total else 0.0

        # Aggregate error types: (path, message) -> count
        err_counts: dict[tuple[str, str], int] = {}
        for r in results:
            if r["ok"]:
                continue
            for e in r["errors"]:
                key = (e.get("path", "<root>"), e.get("message", ""))
                err_counts[key] = err_counts.get(key, 0) + 1

        top_errors = sorted(
            [{"path": k[0], "message": k[1], "count": c} for k, c in err_counts.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:max_error_types]

        examples = []
        for r in results:
            if not r["ok"]:
                examples.append({"idx": r["idx"], "errors": r["errors"]})
                if len(examples) >= max_examples:
                    break

        return {
            "total": total,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "valid_rate": valid_rate,
            "top_errors": top_errors,
            "examples": examples,
        }
    
    def tasks_by_name(self, name: str) -> TaskSpec:
        for t in self.tasks:
            if t.name == name:
                return t
        raise KeyError(name)

    # ---------- Artifacts ----------

    def save_artifacts(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(self.cfg.out_dir) / ts
        run_dir.mkdir(parents=True, exist_ok=True)

        # Config + task specs
        (run_dir / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2, default=str), encoding="utf-8")
        (run_dir / "tasks.json").write_text(
            json.dumps([asdict(t) for t in self.tasks], indent=2, default=str), encoding="utf-8"
        )

        # Tuning + eval
        (run_dir / "tuning.json").write_text(json.dumps(self.tuning_info, indent=2), encoding="utf-8")
        (run_dir / "eval.json").write_text(json.dumps(self.eval_info, indent=2), encoding="utf-8")

        # Also write reports as txt per task for easy viewing
        reports_dir = run_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        for task_name, info in self.eval_info.items():
            rpt = info.get("report") or ""
            (reports_dir / f"{task_name}.txt").write_text(rpt, encoding="utf-8")

        return run_dir

    # ---------- Main ----------

    def run(self, *, build_schema_on_test: bool = False) -> dict[str, Any]:
        self.load_splits()
        self.train()
        self.tune()
        self.evaluate()

        run_dir = self.save_artifacts()
        result = {"run_dir": str(run_dir), "tuning": self.tuning_info, "eval": self.eval_info}

        if build_schema_on_test and self.cfg.schema_path:
            self._init_schema_tools()
            schema_outputs = self.build_schema_outputs(self.test_df)  # type: ignore[arg-type]

            out_path = Path(run_dir) / "schema_outputs_test.jsonl"
            with out_path.open("w", encoding="utf-8") as f:
                for obj in schema_outputs:
                    f.write(json.dumps(obj) + "\n")
            result["schema_outputs_test_path"] = str(out_path)

            summary = self.validate_schema_outputs(schema_outputs)
            summary_path = Path(run_dir) / "schema_validation_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            result["schema_validation_summary_path"] = str(summary_path)
            result["schema_validation"] = summary

        return result


if __name__ == "__main__":
    # Load label vocab (adjust path)
    with open("data_collection/data/processed/label_vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)

    tasks = [
        TaskSpec(name="issue_type", label_col="issue_type", classes=vocab["issue_type"], tune_method="global_micro"),
        TaskSpec(name="components", label_col="topic", classes=vocab["components"], tune_method="global_micro"),
        TaskSpec(name="platform",   label_col="platform",   classes=vocab["platform"],   tune_method="global_micro"),
        TaskSpec(name="impact",     label_col="impact",     classes=vocab["impact"],     tune_method="global_micro"),
    ]

    cfg = RunnerConfig(
        train_path="data_collection/data/processed/train.json",
        val_path="data_collection/data/processed/val.json",
        test_path="data_collection/data/processed/test.json",
        text_col="text_clean",
        out_dir="runs",
        schema_path="src/schemas/triage_schema.json",
        threshold_grid=np.linspace(0.05, 0.95, 301)
    )

    runner = ExperimentRunner(cfg, tasks)
    res = runner.run(build_schema_on_test=True)
    print("Saved run to:", res["run_dir"])
