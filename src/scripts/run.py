from scripts.llm.llm_triage_model import LLMRunConfig, TaskConfig, LLMTriageRunner
from scripts.comparison_loader import RunComparison
from pathlib import Path
import json


PROJECT_ROOT = Path("../..").resolve()  # adjust based on notebook location
DATA_DIR = Path("data_collection/data/processed")
TRAIN_PATH = Path(DATA_DIR / "train.json")
VAL_PATH = Path(DATA_DIR / "val.json")
TEST_PATH = Path(DATA_DIR / "test.json")
OUTPUT_PATH = "results/gemini_flash_predictions.jsonl"
CHECKPOINT_PATH = "results/gemini_flash_checkpoint.jsonl"
SCHEMA_PATH = Path(PROJECT_ROOT / "src/schemas/triage_schema.json")

# -- LLM run (uses cached checkpoints) --
config = LLMRunConfig(
    train_path=TRAIN_PATH,
    val_path=VAL_PATH,
    test_path=TEST_PATH,
    vocab_path=Path(DATA_DIR / "label_vocab.json"),
    schema_path=SCHEMA_PATH,
    env_path=Path(PROJECT_ROOT / ".env.local"),
    n_samples=20,
    temperature=0.7,
)

with open(config.vocab_path) as f:
    vocab = json.load(f)

tasks = [
    TaskConfig(name="issue_type", label_col="issue_type", classes=vocab["issue_type"]),
    TaskConfig(name="components", label_col="topic",      classes=vocab["components"]),
    TaskConfig(name="platform",   label_col="platform",   classes=vocab["platform"]),
    TaskConfig(name="impact",     label_col="impact",     classes=vocab["impact"]),
]

llm_result = LLMTriageRunner(config, tasks).run()

# -- Comparison --
comp = RunComparison(
    baseline_run_dir="baseline/runs/20260226_212256",
    llm_run_dir=llm_result["run_dir"],
)
comp.load()
comp.print_summary_table()
comp.save_all("final_results/")
