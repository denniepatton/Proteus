# Proteus Analysis Framework — Methods, Assumptions, and Usage

## Overview

The analysis package is organized around seven top-level sections under `analysis/results/`:

- `Examples`
- `Benchmarking_Synthetic`
- `Benchmarking_Patient`
- `Calibration`
- `Validation_PDX`
- `Validation_Patient`
- `Discovery_Patient`

Outside benchmarking, the analysis workflow defaults to the main PLCA Proteus model (`BioDVN`). The `NoSeq` and `OnlySeq` variants are retained for benchmarking comparisons only.

`analysis/run_full_analysis.py` is the main analysis runner and section dispatcher. It is not a thin wrapper: the cohort-specific discovery workflows, calibration wiring, and a substantial amount of analysis orchestration currently live there.

## Quick Start

```bash
# Illustrative examples and RNA-seq mixture fit
python analysis/run_full_analysis.py --group Examples

# Synthetic benchmarking (PHS grid)
python analysis/run_full_analysis.py --group Benchmarking_Synthetic

# Patient benchmarking (CRPC-WCDT, BLCA-RA)
python analysis/run_full_analysis.py --group Benchmarking_Patient

# Standalone calibration diagnostics
python analysis/run_full_analysis.py --group Calibration

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
| `--group` | *(required)* | One of `Examples`, `Benchmarking_Synthetic`, `Benchmarking_Patient`, `Calibration`, `Validation_PDX`, `Validation_Patient`, `Discovery_Patient`, `all` |
| `--overwrite` | `False` | Regenerate outputs even if target PDFs or cached section outputs already exist |
| `--minimum_tfx` | `0.0` | Drop patient samples with tumor fraction below this threshold during data loading before downstream analyses consume them |
| `--calibrate_cis` | `False` | Replace Proteus raw 90% CIs with the CV-fitted conformal-calibrated 90% CIs for downstream analyses |
| `--dim_ambiguous` | `False` | In single-gene plots/analyses, render ambiguous points as hollow markers and exclude them from reported single-gene statistics. For Proteus MDN outputs this now uses the activity odds ratio `log2(pi_normal / (pi_delta + pi_gamma))` together with the shared lower/upper thresholds learned from pooled patient CV OOF predictions, rather than CI overlap |
| `--scatter_ci` | `False` | Pass interval surfaces into scatter analyses; current single-gene and IHC scatter plots do not render per-point CI error bars |
| `--n_workers` | all available CPUs | Worker processes to use for parallel analysis steps |

The runner defaults to all visible CPUs and prints both the total available threads and the resolved worker count. Use `--n_workers` to override that default explicitly.

`Discovery_Patient` always runs the configured discovery cohorts from `analysis/pipeline_config.py`; there is no per-run cohort override in the current CLI.

## Analysis Dependencies

The analysis workflow requires more than the core training dependencies.

- Python packages: `lifelines`, `gseapy`, `rpy2`
- R packages: `limma`, `genefu`
- Repo-local reference assets used by downstream workflows:
    - `reference/h.all.v2026.1.Hs.symbols.gmt`
    - `reference/gsva_genesets_shortlist.gmt`
    - `reference/MANE.GRCh38.v1.3.UniversalReference.tsv`
    - cohort-specific metadata and outcome tables under `reference/`

If the R bridge is unavailable, workflows that call limma through `rpy2` will fail. This currently affects discovery analyses that run limma/GSEA or R-side mixed models.

## Package Layout

| File | Role |
|---|---|
| `analysis/run_full_analysis.py` | Main analysis runner, section dispatcher, and cohort-specific orchestration |
| `analysis/pipeline_config.py` | Canonical section names, model registries, and path policy |
| `analysis/examples.py` | Example plots and expression-mixture fitting workflow |
| `analysis/analysis_utils.py` | Data loading, metrics, cohort utilities, uncertainty helpers, survival helpers, and genefu-backed PAM50 classification |
| `analysis/calibration.py` | Asymmetric regime-based conformal interval calibration |
| `analysis/reliability.py` | Post-hoc gene reliability modelling |
| `analysis/plot_utils.py` | Plotting helpers used across all sections |
| `analysis/limma_de.R` | R-side limma bridge called via `rpy2` |

## Active Sections

### Examples

Produces the illustrative `NaiveExample` plots and the RNA-seq mixture fit used for method exposition. The mixture fit uses the in-workspace PLCA truth pool.

### Benchmarking_Synthetic

Benchmarks Proteus, `Proteus_NoSeq`, `Proteus_OnlySeq`, XGBoost, EPIC-Seq, and a shuffled-truth `Naive` baseline on the PHS TFx/depth grid. Outputs are split into two plot views:

- `COMPLETE`: `Proteus`, `Proteus_NoSeq`, `Proteus_OnlySeq`, `XGBoost`, `EPIC_GEP`, `EPIC_NDR`, `EPIC_PFE`, `Naive`
- `SUBSET`: `Proteus`, `XGBoost`, `EPIC_GEP`, `Naive`

The `SUBSET` view also includes a Hallmark ssGSEA recapitulation benchmark that reports only mean Hallmark $
ho$ within each TFx/depth cell. Those outputs now include both the raw score surfaces and an uncertainty-adjusted Proteus score surface obtained by Monte Carlo aggregation across MDN draws.

### Benchmarking_Patient

Benchmarks the same model family on patient cohorts with truth labels (`CRPC-WCDT`, `BLCA-RA`). Outputs are also split into `COMPLETE` and `SUBSET` plot views.

For Proteus MDN models, this section now writes dataset-specific `ActiveConfidenceThreshold_*` and `InactiveConfidenceThreshold_*` outputs using activity odds ratio sweeps instead of `P_on` cutoffs. The non-CI `OnOffConfusion_*` outputs now compare binary calling against the uncertainty-informed Proteus variant that drops predictions inside the shared pooled-patient ambiguous AOR band.

### Calibration

Runs standalone calibration diagnostics on the PLCA OOF prediction pool and matching PDX holdout data.

- Interval calibration uses asymmetric regime-based conformal calibration of the MDN-derived 90% interval, with separate lower-tail and upper-tail conformal corrections.
- PIT diagnostics are computed from the full hurdle-gamma-normal MDN predictive CDF and written for both pooled OOF and pooled PDX holdout evaluation sets.
- OOF evaluation uses leave-one-fold-out cross-fitting so calibrated CV diagnostics do not reuse the same fold for both fitting and evaluation.
- The section also derives pooled patient-CV activity odds ratio thresholds from the CRPC-RA and SCLC OOF predictions: an upper bound where PPV first reaches 0.90 and a lower bound where NPV first reaches 0.90. Those thresholds are written to `PatientCV_ActivityOddsRatioThresholds.tsv` and accompanied by `ActiveConfidenceThreshold_*` and `InactiveConfidenceThreshold_*` sweep plots/TSVs.

### Validation_PDX

Evaluates the main PLCA Proteus model on `PC-PDX` and `SCLC-PDX` using combined `20 CV + 5 holdout` samples per cohort. This section also carries the PLCA reliability analysis because those features are anchored to the same OOF development predictions.

This section now also writes a pooled `CrossCohort` subdirectory for the two PDX cohorts, including pooled target-gene scatters and shortlist-score truth-vs-predicted scatter plots.

### Validation_Patient

Evaluates the main PLCA Proteus model on patient cohorts with truth:

- Holdout cohorts: `CRPC-WCDT`, `BLCA-RA`
- CV cohorts: `CRPC-RA`, `SCLC`

This section includes a pooled `CrossCohort` subdirectory that merges all truth-bearing patient cohorts and preserves CV-versus-holdout marker styling.

`Validation_Patient/CRPC-RA` now also writes a CRPC-RA-specific `MetastasisComposition/` subdirectory. That analysis combines the averaged patient truth labels from `labels/CRPC-RA_Log2CPMplus1.tsv`, met-level truth labels from `labels/CRPC-RA-METS_Log2CPMplus1.tsv`, Proteus OOF predictions, and the sampling metadata in `reference/CRPC-RA_METS.tsv` to quantify whether prediction error tracks metastatic sampling breadth, tissue composition, and within-patient met heterogeneity. It writes patient-level and patient-gene summary tables, closest-truth comparator tables, pathway summaries for the shortlist scores, and the corresponding CRPC-RA-only figures.

Each truth-bearing patient cohort now also writes a `PAM50/pam50_scale/` subtree with predicted PAM50 classifications, RNA-seq-truth PAM50 classifications, per-subtype truth-vs-predicted score tables, score-correlation summaries and scatters, and RNA-seq-truth-vs-predicted confusion matrices.

### Discovery_Patient

Configured discovery cohorts currently include `CRPC-Pfizer`, `CRPC-Pluvicto`, `CRPC-UW`, and `BC-Ghana`.

This section runs truth-free downstream analyses on discovery cohorts using the main PLCA Proteus model only. Outputs include expression profiles, curated heatmaps, and cohort-specific downstream analyses such as limma/GSEA, PAM50, ssGSEA, survival, and longitudinal marker analyses where applicable.

All discovery patient cohorts now also write predicted `PAM50/pam50_scale/` score and classification tables. BC-Ghana retains the metadata-subtype confusion view and its uncertainty-adjusted PAM50 outputs because there is no matched RNA-seq truth matrix for that cohort in the repo.

All discovery analyses are exploratory. Survival, subgroup comparisons, and downstream enrichment outputs should be interpreted as hypothesis-generating rather than confirmatory.

## Output Structure

```text
analysis/results/
├── Examples/
│   ├── NaiveExample/
│   └── run_manifest.json
├── Benchmarking_Synthetic/
│   ├── COMPLETE/
│   ├── SUBSET/
│   ├── run_manifest.json
│   └── summary_benchmarking_all.tsv
├── Benchmarking_Patient/
│   ├── COMPLETE/
│   ├── SUBSET/
│   ├── run_manifest.json
│   └── summary_benchmarking_all.tsv
├── Calibration/
│   └── run_manifest.json
├── Validation_PDX/
│   ├── PC-PDX/
│   ├── SCLC-PDX/
│   ├── CrossCohort/
│   ├── run_manifest.json
│   └── summary_metrics.tsv
├── Validation_Patient/
│   ├── BLCA-RA/
│   ├── CRPC-RA/
│   │   └── MetastasisComposition/
│   ├── CRPC-WCDT/
│   ├── SCLC/
│   ├── CrossCohort/
│   ├── run_manifest.json
│   └── summary_metrics.tsv
└── Discovery_Patient/
        ├── BC-Ghana/
        ├── CRPC-Pfizer/
        ├── CRPC-Pluvicto/
        ├── CRPC-UW/
        └── run_manifest.json
```

Each top-level section writes a `run_manifest.json` capturing the CLI arguments, worker settings, execution timestamp, and output root used for that run.

## Core Method Notes

### Expression scale

All analysis paths operate on continuous log2(CPM+1) expression values, including truth data when truth is available. The model does not emit raw counts, so downstream analysis is standardized on the continuous log-scale representation throughout the repo.

### Score filtering and NaNs

Gene-set scoring routines that cannot tolerate NaNs operate at the cohort matrix level and drop genes with any NaN in that cohort matrix before scoring. This is intentionally limited to score computations and related downstream analyses that require finite cohort-wide matrices; it is not a general imputation or fill-in policy.

### Per-gene and per-sample metrics

The pipeline reports Pearson $r$, Spearman $\rho$, RMSE, and MAE on both the gene axis and the sample axis, plus derived summaries such as top-variance gene performance and counts of genes exceeding a target correlation threshold.

### GVW metrics

GVW in the current codebase is not implemented as one flattened weighted correlation across all gene-sample pairs. Instead, it is computed as a variance-weighted average of per-gene correlations, using gene variance as the weight. That is the canonical definition for interpreting reported `GVW-r` and `GVW-ρ` outputs.

### Uncertainty strategy

The analysis code uses two complementary uncertainty views:

- Clinician-facing plots and simple per-observation displays use the model's 90% intervals because they are interpretable and easy to communicate.
- Grouped downstream analyses should use the full predictive uncertainty when possible rather than reducing uncertainty to interval width alone.

Current practice in this repo:

- Adjusted Hallmark ssGSEA, shortlist/CCP score workflows, and PAM50 classification propagate uncertainty by drawing Monte Carlo expression matrices from the full MDN predictive distribution and aggregating across draws. The CLI flag `--montecarlo_draws` controls the number of draws; the default is 100.
- The shortlist workflow uses the `reference/gsva_genesets_shortlist.gmt` gene sets, but the scoring algorithm in this repo is ssGSEA rather than a separate GSVA implementation.
- Adjusted limma workflows now use those same Monte Carlo expression draws rather than inverse-variance weights. The saved uncertainty-adjusted limma tables use Rubin-style pooling for `logFC`, `standard_error`, `t`, `P.Value`, and `adj.P.Val`, and report pooled 95% confidence intervals for `logFC`.
- Proteus MDN outputs now expose an activity odds ratio surface `activity_odds_ratio = log2(pi_normal / (pi_delta + pi_gamma))` for every gene-sample prediction surface. Under this convention the normal branch is the active mass and the delta spike plus gamma branch together represent off-mass.
- Calibration pools the patient CV OOF predictions from `CRPC-RA` and `SCLC` to derive a shared upper activity threshold where PPV first reaches 0.90 and a shared lower activity threshold where NPV first reaches 0.90.
- Those shared lower/upper thresholds now drive downstream ambiguity masking in patient validation scatters and IHC analyses, and they define the uncertainty-informed Proteus abstention behavior in `Benchmarking_Patient`.
- `Benchmarking_Patient` still writes dataset-specific active/inactive threshold sweep curves, but those are now descriptive AOR sweeps rather than the source of the shared operating point.

### Survival analysis

Pluvicto survival outputs are exploratory. Cox tables now include proportional-hazards diagnostic p-values, and the workflow prints compact PH summaries when those fits are run. Kaplan-Meier plots are median-split displays of selected biomarkers and should be interpreted as visual exploratory summaries rather than standalone confirmatory tests.

### Calibration

`Validation_PDX` and `Calibration` fit an asymmetric regime-based conformal calibrator on PLCA OOF intervals and evaluate it on holdout PDX samples.

Important caveat: only the calibrated 90% interval is directly produced by the conformal procedure here, because the underlying Proteus interval inputs are q0.05/q0.95. Lower nominal coverage curves are generated by symmetrically rescaling that calibrated 90% interval around the point prediction. This is useful as a width-versus-coverage diagnostic, but it is not equivalent to fitting separate conformal calibrators for each nominal level.

### Reliability

The post-hoc reliability model is trained on fold-wise OOF features and evaluated on holdout PDX data. Static gene features come from the PLCA development truth pool; holdout truth is used only for the evaluation target.

### PAM50

The analysis now standardizes on `genefu::intrinsic.cluster.predict` with the `pam50.scale` model only. Continuous PAM50 score outputs use the genefu centroid-correlation fields (`corr_*`) when available, with subtype probabilities (`proba_*`) retained in the classification tables.

For truth-bearing patient cohorts, PAM50 is computed on both RNA-seq truth and Proteus predictions, then compared through per-subtype score-correlation summaries/scatters and confusion matrices driven by the RNA-seq-derived PAM50 calls. Discovery cohorts still receive predicted PAM50 score/classification outputs even without truth; BC-Ghana additionally compares those predictions against its metadata subtype labels and keeps the uncertainty-adjusted Monte Carlo PAM50 aggregation.

## Inputs and Assumptions

- Ground-truth expression tables are expected in `labels/<COHORT>_Log2CPMplus1.tsv`.
- Subtype annotations are expected in `labels/<COHORT>_TFX_Subtype.tsv` when cohort-level plots need them.
- The Hallmark GMT is expected at `reference/h.all.v2026.1.Hs.symbols.gmt`.
- The shortlist GMT is expected at `reference/gsva_genesets_shortlist.gmt`.
- EPIC-Seq benchmarking uses the raw `PFE` column directly; it is not renamed to a normalized GEP-derived quantity.
- Discovery analyses assume the upstream prediction artifacts already exist under the directory structure configured in `analysis/pipeline_config.py`.

## Parallelism

The heavy plotting and metric kernels use `ProcessPoolExecutor` where appropriate. The runner defaults to all available CPUs and prints the resolved worker count at startup. Use `--n_workers` to override that default explicitly.
