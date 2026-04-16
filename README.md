# Godot Issue Triager

An end-to-end automated triage pipeline for Godot Engine GitHub issues. This project evaluates and compares two models for predicting structured JSON routing records across four fields (`components`, `issue_type`, `platform`, and `impact`):
1. **TF-IDF + One-vs-Rest Logistic Regression Baseline**
2. **Zero-Shot LLM (Gemini 2.5 Flash Lite)** utilizing exact-set consensus aggregation.

The pipeline includes a time-based dataset split to simulate real-world deployment, extensive schema validation, coverage-accuracy tradeoff analysis, and a robustness suite that tests model degradation against malformed or incomplete issue reports.

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/paul-rl/godot-issue-triager.git](https://github.com/paul-rl/godot-issue-triager.git)
cd godot-issue-triager
```

**2. Install dependencies**

Ensure Python 3.9+ is installed, then install the required packages:
```bash
pip install pandas numpy scikit-learn matplotlib python-dotenv google-genai
```

**3. Configure Environment Variables**

The LLM pipeline requires a Gemini API key. Create a .env.local file in the root of the project and add your key:

```GEMINI_API_KEY=your_api_key_here```

## How to Run the Pipeline

The entire experiment suite is orchestrated by run.py. It executes sequentially: Baseline Training → LLM Inference → Robustness Evaluation → Artifact Comparison.
Run the Full Suite

To run the complete pipeline from scratch using the default configuration (N=20 samples, T=0.7), navigate to the directory containing ```run.py``` and run:

```bash
python run.py
```
Customizing the Run

You can skip steps, adjust LLM sampling parameters, or resume from existing run directories using CLI arguments:

```bash

# Skip baseline training and LLM inference, just run robustness and comparison
python run.py --baseline-run baseline/runs/20260226_212256 --llm-run runs/llm_gemini-2.5-flash-lite_...

# Run the full suite but test with different LLM sampling parameters
python run.py --n-samples 10 --temperature 0.5

# Run the pipeline but skip the expensive LLM robustness suite
python run.py --skip-robustness
```

## Generated Artifacts

Upon completion, the pipeline aggregates all metrics, plots, and tables into a final_results/ directory at the project root.

final_results/ contains the top-level performance comparisons between the Baseline and the LLM.

    comparison_summary.csv: A compact table comparing Micro-F1, Macro-F1, and Coverage across all four prediction fields.

    comparison_full.csv: A highly detailed breakdown including Hamming Loss, Hit@5, and Recall@5.

    schema_validation.json: Statistics tracking the LLM's adherence to the JSON schema (first-pass valid, repaired, clamped, and final validity rates).

    llm_coverage_accuracy_curve.png / .csv: A plot illustrating the coverage-accuracy tradeoff for the LLM based on exact-set consensus frequencies.

    coverage_curve_baseline.json: The equivalent threshold sweep data for the TF-IDF baseline.

final_results/robustness/ contains the results of the perturbation suite (Remove Title, Truncate Body, Strip Code Blocks, Drop First/Last 50 Tokens).

    robustness_table.csv: The final report table detailing ΔμF1 and prediction stability for both models under each perturbation.

    robustness_plot.png: A 4-panel visual comparison showing performance degradation and prediction stability.

    interpretation.txt: Auto-generated qualitative notes interpreting the severity of the model degradation across the various perturbations.

    llm_cache/: A directory containing .jsonl files that cache the raw LLM responses for each perturbation. If the robustness script is interrupted, it will seamlessly resume from this cache without re-triggering expensive API calls.

## Pipeline Architecture

    Baseline (ExperimentRunner): Trains OvR Logistic Regression models on TF-IDF features. Sweeps 301 threshold values on the validation set to maximize Micro-F1.

    LLM Pipeline (LLMTriageRunner): Queries Gemini 2.5 Flash Lite requiring a strict JSON schema. Uses a two-stage repair loop (LLM correction followed by hard lexical clamping) to guarantee 100% downstream format compliance.

    Robustness (run_robustness_eval): Reconstructs the baseline and draws a 700-issue random sample for the LLM to test prediction stability against synthetic data degradation.
