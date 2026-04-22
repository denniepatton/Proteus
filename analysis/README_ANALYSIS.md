# Proteus Analysis Framework — Methods & Usage

## Overview

The analysis package is organized around six reproducible top-level sections under `analysis/results/`:

- `Examples`
- `Benchmarking_Synthetic`
- `Benchmarking_Patient`
- `Validation_PDX`
- `Validation_Patient`
- `Discovery_Patient`

Outside benchmarking, the pipeline defaults to the main PLCA Proteus model (`BioDVN`). The `NoSeq` and `OnlySeq` variants are retained only for benchmarking comparisons.

## Quick Start

```bash
# Illustrative examples and RNA-seq mixture fit
python analysis/run_full_analysis.py --group Examples

# Synthetic benchmarking (PHS grid)
python analysis/run_full_analysis.py --group Benchmarking_Synthetic

# Patient benchmarking (CRPC-WCDT, BLCA-RA)
python analysis/run_full_analysis.py --group Benchmarking_Patient

# PDX validation (20 CV + 5 holdout per cohort)
python analysis/run_full_analysis.py --group Validation_PDX

# Patient validation with pooled CrossCohort output
python analysis/run_full_analysis.py --group Validation_Patient

# Discovery-only cohorts with no RNA-seq truth
python analysis/run_full_analysis.py --group Discovery_Patient

# Run the full analysis tree
python analysis/run_full_analysis.py --group all
```

## CLI Flags

| Flag | Default | Notes |
|---|---|---|
| `--group` | *(required)* | One of `Examples`, `Benchmarking_Synthetic`, `Benchmarking_Patient`, `Validation_PDX`, `Validation_Patient`, `Discovery_Patient`, `all` |
| `--overwrite` | `False` | Regenerate outputs even if target PDFs already exist |
| `--n_bootstraps` | `100` | Retained for compatibility, but inactive in the current six-section workflows |
| `--scatter_ci` | `False` | Add CI error bars to scatter plots when CI data exist |

`Discovery_Patient` always runs the configured discovery cohorts from `analysis/pipeline_config.py`; there is no per-run override path in the current CLI.

## Package Layout

| File | Role |
|---|---|
| `analysis/run_full_analysis.py` | Thin orchestrator and section routing |
| `analysis/pipeline_config.py` | Canonical section names, model registries, and path policy |
| `analysis/examples.py` | Example plots and expression-mixture fitting workflow |
| `analysis/analysis_utils.py` | Data loading, metrics, cohort utilities, and statistical helpers |
| `analysis/calibration.py` | Regime-based conformal interval calibration |
| `analysis/reliability.py` | Post-hoc gene reliability modelling |
| `analysis/plot_utils.py` | Plotting helpers used across all sections |

## Active Sections

### Examples

Produces the illustrative “NaiveExample” plots and the RNA-seq mixture fit used for method exposition. The mixture fit now uses in-workspace PLCA truth data instead of an external RNA-seq path.

### Benchmarking_Synthetic

Benchmarks Proteus, `Proteus_NoSeq`, `Proteus_OnlySeq`, XGBoost, EPIC-Seq, and a shuffled-truth `Naive` baseline on the PHS TFx/depth grid. Outputs are split into two plot views:

- `COMPLETE`: `Proteus`, `Proteus_NoSeq`, `Proteus_OnlySeq`, `XGBoost`, `EPIC_GEP`, `EPIC_NDR`, `EPIC_PFE`, `Naive`
- `SUBSET`: `Proteus`, `XGBoost`, `EPIC_GEP`, `Naive`

### Benchmarking_Patient

Benchmarks the same model family on patient cohorts with truth labels (`CRPC-WCDT`, `BLCA-RA`). Outputs are also split into `COMPLETE` and `SUBSET` plot views.

### Validation_PDX

Evaluates the main PLCA Proteus model on `PC-PDX` and `SCLC-PDX` using combined `20 CV + 5 holdout` samples per cohort. This section also carries the PLCA calibration and reliability outputs because those are anchored to the same OOF development predictions.

### Validation_Patient

Evaluates the main PLCA Proteus model on patient cohorts with truth:

- Holdout cohorts: `CRPC-WCDT`, `BLCA-RA`
- CV cohorts: `CRPC-RA`, `SCLC`

This section includes a pooled `CrossCohort` subdirectory that merges all truth-bearing patient cohorts and preserves CV-versus-holdout marker styling.

### Discovery_Patient

Runs truth-free downstream analysis on discovery cohorts (`CRPC-Pluvicto`, `CRPC-UW`, `BC-Ghana`) using the main PLCA Proteus model only. Outputs include expression profiles, curated heatmaps, and cohort-specific downstream analyses such as limma/GSEA, PAM50, ssGSEA, and survival where applicable.

## Output Structure

```text
analysis/results/
├── Examples/
│   └── NaiveExample/
├── Benchmarking_Synthetic/
│   ├── COMPLETE/
│   ├── SUBSET/
│   └── summary_benchmarking_all.tsv
├── Benchmarking_Patient/
│   ├── COMPLETE/
│   ├── SUBSET/
│   └── summary_benchmarking_all.tsv
├── Validation_PDX/
│   ├── Calibration/
│   ├── PC-PDX/
│   ├── SCLC-PDX/
│   ├── reliability_feature_weights.pdf
│   └── summary_metrics.tsv
├── Validation_Patient/
│   ├── BLCA-RA/
│   ├── CRPC-RA/
│   ├── CRPC-WCDT/
│   ├── SCLC/
│   ├── CrossCohort/
│   └── summary_metrics.tsv
└── Discovery_Patient/
    ├── BC-Ghana/
    ├── CRPC-Pluvicto/
    └── CRPC-UW/
```

## Metric Notes

### Per-gene and per-sample metrics

The pipeline reports Pearson $r$, Spearman $\rho$, RMSE, and MAE on both the gene axis and the sample axis, plus derived summaries such as top-variance gene performance and counts of genes exceeding a target correlation threshold.

### GVW metrics

GVW in the current codebase is not implemented as one flattened weighted correlation across all gene-sample pairs. Instead, it is computed as a variance-weighted average of per-gene correlations, using gene variance as the weight. That is the definition to treat as canonical for interpreting reported `GVW-r` and `GVW-ρ` outputs.

### Benchmarking uncertainty

The active benchmarking sections currently report point estimates only. The CLI still exposes `--n_bootstraps`, but the refactored benchmarking outputs do not currently compute or plot bootstrap confidence intervals.

### Calibration

`Validation_PDX` fits a regime-based conformal calibrator on PLCA OOF intervals and evaluates it on the true holdout PDX samples. Coverage curves are produced by rescaling the calibrated 95% interval widths to lower nominal levels, matching the current implementation in `analysis/calibration.py` and `analysis/run_full_analysis.py`.

### Reliability

The post-hoc reliability model is trained on fold-wise OOF features and evaluated on holdout PDX data. Static gene features come from the PLCA development truth pool; holdout truth is used only for the evaluation target.

## Inputs and Assumptions

- Ground-truth expression tables are expected in `labels/<COHORT>_Log2CPMplus1.tsv`.
- Subtype annotations are expected in `labels/<COHORT>_TFX_Subtype.tsv` when cohort-level plots need them.
- The Hallmark GMT is expected at `reference/h.all.v2026.1.Hs.symbols.gmt`.
- The shortlist GMT is expected at `reference/gsva_genesets_shortlist.gmt`.
- EPIC-Seq benchmarking uses the raw `PFE` column directly; it is not renamed to a normalized GEP-derived quantity.

## Parallelism

The heavy plotting and metric kernels use `ProcessPoolExecutor` where appropriate. The orchestrator now always uses all available CPUs automatically.
