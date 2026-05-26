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

Outside benchmarking, the analysis workflow uses the canonical BioDVN PLCA model only. The only Proteus variants consumed anywhere in this runner are `BioDVN`, `BioDVN_NoSeq`, and `BioDVN_OnlySeq`.

When an analysis section executes, prediction inputs are read directly from hard-coded relative paths under `results_training/` and `results_holdout/`. `analysis/results/` is output-only and is never used as a prediction input source.

`analysis/run_full_analysis.py` is the main analysis runner and section dispatcher. It is not a thin wrapper: the cohort-specific discovery workflows, OOF diagnostics, and a substantial amount of analysis orchestration currently live there.

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
| `--group` | *(required)* | One of `Examples`, `Benchmarking_Synthetic`, `Benchmarking_Patient`, `Calibration`, `Validation_PDX`, `Validation_Patient`, `Discovery_Patient`, `all`. `all` runs the full standard analysis stack, including `Calibration` |
| `--overwrite` | `False` | Rerun an analysis section even when its top-level output folder already exists. Without this flag, an existing section directory is left untouched and that section is skipped rather than overwritten |
| `--minimum_tfx` | `0.0` | Drop patient samples with tumor fraction below this threshold during data loading before downstream analyses consume them |
| `--dim_ambiguous` | `False` | In single-gene plots/analyses, render ambiguous points as hollow markers and exclude them from reported single-gene statistics. For Proteus MDN outputs this now uses the activity odds ratio `log2((pi_normal + 1e-6) / (1 - pi_normal + 1e-6))` together with the shared lower/upper thresholds learned from pooled patient CV OOF predictions, rather than CI overlap |
| `--scatter_ci` | `False` | Pass interval surfaces into scatter analyses; current single-gene and IHC scatter plots do not render per-point CI error bars |
| `--n_workers` | all available CPUs | Worker processes to use for parallel analysis steps |
| `--montecarlo_draws` | `100` | Number of MDN expression draws used only for 90% score intervals |

The runner defaults to all visible CPUs and prints both the total available threads and the resolved worker count. Use `--n_workers` to override that default explicitly.

`Discovery_Patient` always runs the configured discovery cohorts from `analysis/pipeline_config.py`; there is no per-run cohort override in the current CLI.

## Analysis Dependencies

The analysis workflow requires more than the core training dependencies. The
repo-level [proteus.env](/Volumes/rpatton/TritonCNN/Proteus/proteus.env) file is the canonical micromamba environment spec for this paper-reproduction stack.

- Python packages: `lifelines`, `gseapy`, `rpy2`
- R packages: `limma`, `genefu`
- Repo-local reference assets used by downstream workflows:
    - `reference/h.all.v2026.1.Hs.symbols.gmt`
    - `reference/gsva_genesets_shortlist.gmt`
    - `reference/MANE.GRCh38.v1.3.UniversalReference.tsv`
    - cohort-specific metadata and outcome tables under `reference/`

If the R bridge is unavailable, workflows that call limma through `rpy2` will fail. This currently affects discovery analyses that run limma/GSEA or R-side mixed models.

Configured prediction files, label tables, GMT files, and cohort metadata are treated as required inputs. Missing artifacts now fail loudly instead of being skipped.

## Package Layout

| File | Role |
|---|---|
| `analysis/run_full_analysis.py` | Main analysis runner, section dispatcher, and cohort-specific orchestration |
| `analysis/pipeline_config.py` | Canonical section names, model registries, and path policy |
| `analysis/examples.py` | Example plots and expression-mixture fitting workflow |
| `analysis/analysis_utils.py` | Data loading, metrics, cohort utilities, uncertainty helpers, survival helpers, and genefu-backed PAM50 classification |
| `analysis/enrichment_utils.py` | Shared GMT parsing, Entrez-to-symbol conversion, and ssGSEA helper primitives extracted from the main runner |
| `analysis/reliability.py` | Post-hoc gene reliability modelling |
| `analysis/plot_utils.py` | Plotting helpers used across all sections |
| `analysis/limma_de.R` | R-side limma bridge called via `rpy2` |

## Active Sections

### Examples

Produces the illustrative naive-baseline plots and the RNA-seq mixture fit used for method exposition directly in `analysis/results/Examples/`. The mixture fit uses the in-workspace PLCA truth pool.

### Benchmarking_Synthetic

Benchmarks Proteus, `Proteus_NoSeq`, `Proteus_OnlySeq`, XGBoost, EPIC-Seq, and a shuffled-truth `Naive` baseline on the PHS TFx/depth grid. Outputs are split into two plot views:

- `COMPLETE`: `Proteus`, `Proteus_NoSeq`, `Proteus_OnlySeq`, `XGBoost`, `EPIC_GEP`, `EPIC_NDR`, `EPIC_PFE`, `Naive`
- `SUBSET`: `Proteus`, `XGBoost`, `EPIC_GEP`, `Naive`

The `SUBSET` view also includes a Hallmark ssGSEA recapitulation benchmark that reports mean Hallmark rho within each TFx/depth cell. Synthetic benchmark plots keep the TFx/depth line summaries and no longer write benchmark-grid boxplot/histogram figures.

### Benchmarking_Patient

Benchmarks the same model family on patient cohorts with truth labels (`CRPC-WCDT`, `BLCA-RA`). Outputs are also split into `COMPLETE` and `SUBSET` plot views.

For Proteus MDN models, this section now writes dataset-specific `ActiveConfidenceThreshold_*` and `InactiveConfidenceThreshold_*` outputs using activity odds ratio sweeps instead of `P_on` cutoffs. The non-CI `OnOffConfusion_*` outputs now compare binary calling against the uncertainty-informed Proteus variant that drops predictions inside the shared pooled-patient ambiguous AOR band.

### Calibration

Runs standalone diagnostics on the PLCA OOF prediction pool. It does not fit or apply post-hoc interval calibration.

- PIT diagnostics are computed from the full hurdle-gamma-normal MDN predictive CDF for pooled OOF samples.
- OOF residual diagnostics are written for the pooled PLCA OOF set and each PLCA cohort: AOR vs signed residual, AOR distributions with shared cut-points, CI width vs signed residual, CI width distributions, and gene-level RMSE versus MANE gene meta-features.
- The section derives pooled patient-CV activity odds ratio thresholds from the CRPC-RA and SCLC OOF predictions, restricted to genes whose truth values cross the active-expression cutoff of 2.0 log2(CPM+1). The upper bound is where PPV first reaches 0.90, and the lower bound is where NPV first reaches 0.90.

### Validation_PDX

Evaluates the main PLCA Proteus model on `PC-PDX` and `SCLC-PDX` using combined `20 CV + 5 holdout` samples per cohort. This section also carries the PLCA reliability analysis because those features are anchored to the same OOF development predictions.

This section now also writes a pooled `CrossCohort` subdirectory for the two PDX cohorts, including pooled target-gene scatters and shortlist-score truth-vs-predicted scatter plots.

### Validation_Patient

Evaluates the main PLCA Proteus model on patient cohorts with truth:

- Holdout cohorts: `CRPC-WCDT`, `BLCA-RA`
- CV cohorts: `CRPC-RA`, `SCLC`

This section includes a pooled `CrossCohort` subdirectory that merges all truth-bearing patient cohorts and preserves CV-versus-holdout marker styling.

`Validation_Patient/CRPC-RA` now also writes a CRPC-RA-specific `MetastasisComposition/` subdirectory. That analysis combines the averaged patient truth labels from `labels/CRPC-RA_Log2CPMplus1.tsv`, met-level truth labels from `labels/CRPC-RA-METS_Log2CPMplus1.tsv`, Proteus OOF predictions, and the sampling metadata in `reference/CRPC-RA_METS.tsv` to quantify whether prediction error tracks metastatic sampling breadth, tissue composition, and within-patient met heterogeneity. It writes patient-level and patient-gene summary tables, closest-truth comparator tables, pathway summaries for the shortlist scores, and the corresponding CRPC-RA-only figures.

Each truth-bearing patient cohort now also writes a `PAM50/pam50_scale/` subtree with predicted PAM50 classifications, RNA-seq-truth PAM50 classifications, per-subtype truth-vs-predicted score tables, score-correlation summaries and scatters, and RNA-seq-truth-vs-predicted confusion matrices. The pooled `CrossCohort` patient validation output now also writes pooled truth-vs-predicted PAM50 subtype score comparisons across all matched patients.

### Discovery_Patient

Configured discovery cohorts currently include `CRPC-Pfizer`, `CRPC-Pluvicto_MC`, `CRPC-Pluvicto_Full`, `CRPC-UW`, and `BC-Ghana`.

This section runs truth-free downstream analyses on discovery cohorts using the main PLCA Proteus model only. Outputs include expression profiles, curated heatmaps, and cohort-specific downstream analyses such as limma/GSEA, PAM50, ssGSEA, survival, and longitudinal marker analyses where applicable. The two Pluvicto outputs both restrict to baseline (`C1`) samples: `CRPC-Pluvicto_MC` keeps the original `FH*` / not `FHL*` subset, while `CRPC-Pluvicto_Full` uses all available baseline samples.

All discovery patient cohorts write predicted `PAM50/pam50_scale/` score and classification tables. Genefu `Normal` calls and `Normal` score columns are filtered out of downstream PAM50 summaries. BC-Ghana retains the metadata-subtype confusion view and adds receptor-status AOR boxplots alongside receptor-expression boxplots.

All discovery analyses are exploratory. Survival, subgroup comparisons, and downstream enrichment outputs should be interpreted as hypothesis-generating rather than confirmatory.

## Output Structure

```text
analysis/results/
├── Examples/
│   ├── run_manifest.json
│   └── ...
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
│   ├── OOF_All/
│   ├── OOF_CRPC-RA/
│   ├── OOF_PC-PDX/
│   ├── OOF_SCLC/
│   ├── OOF_SCLC-PDX/
│   ├── PatientCV_ActivityOddsRatioThresholds.tsv
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
    ├── CRPC-Pluvicto_Full/
    ├── CRPC-Pluvicto_MC/
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

- Hallmark and shortlist point scores are computed once from the predictive mean surface. Monte Carlo expression draws are used only to write q5/q95 90% score intervals around those point scores. PAM50 classification uses the predictive mean surface only.
- The shortlist workflow uses the `reference/gsva_genesets_shortlist.gmt` gene sets, but the scoring algorithm in this repo is ssGSEA rather than a separate GSVA implementation.
- Limma workflows use expression means only as the response surface (Proteus predictive mean or RNA-seq truth when available). Predicted-expression limma fits include `TFx` as a covariate when metadata are available; truth-mode fits can run without TFx. The shared default output modes are `unweighted`, `weighted_ci90`, `weighted_aor`, and `weighted_var`, where CI-width, `abs(log2(AOR))`, and posterior variance are each mapped into bounded observation weights in `[0.1, 1.0]` via pooled 5th/95th percentile scaling.
- The shared limma input filter drops genes with any NA in the aligned fit matrix, then requires `log2(CPM+1) > 1.0` in at least 3 samples before fitting.
- Proteus MDN outputs now expose an activity odds ratio surface `activity_odds_ratio = log2((pi_normal + 1e-6) / (1 - pi_normal + 1e-6))` for every gene-sample prediction surface. Under this convention the normal branch is the active mass and the complement of that mass represents off-state weight.
- Calibration pools the patient CV OOF predictions from `CRPC-RA` and `SCLC` to derive a shared upper activity threshold where PPV first reaches 0.90 and a shared lower activity threshold where NPV first reaches 0.90.
- Those shared lower/upper thresholds now drive downstream ambiguity masking in patient validation scatters and IHC analyses, and they define the uncertainty-informed Proteus abstention behavior in `Benchmarking_Patient`.
- `Benchmarking_Patient` still writes dataset-specific active/inactive threshold sweep curves, but those are now descriptive AOR sweeps rather than the source of the shared operating point.

### Survival analysis

Pluvicto survival outputs are exploratory. Cox tables now include proportional-hazards diagnostic p-values, and the workflow prints compact PH summaries when those fits are run. Kaplan-Meier plots are median-split displays of selected biomarkers and should be interpreted as visual exploratory summaries rather than standalone confirmatory tests.

### Calibration

The `Calibration` section is diagnostics-only. It uses raw Proteus prediction means, raw 90% CIs, the MDN predictive CDF, and AOR surfaces to characterize OOF behavior; no post-hoc interval calibration is fit or applied.

### Reliability

The post-hoc reliability model is trained on fold-wise OOF features and evaluated on holdout PDX data. Static gene features come from the PLCA development truth pool; holdout truth is used only for the evaluation target.

### PAM50

The analysis now standardizes on `genefu::intrinsic.cluster.predict` with the `pam50.scale` model only. Continuous PAM50 score outputs use the genefu centroid-correlation fields (`corr_*`) when available, with subtype probabilities (`proba_*`) retained in the classification tables.

For truth-bearing patient cohorts, PAM50 is computed on both RNA-seq truth and Proteus predictions, then compared through per-subtype score-correlation summaries/scatters and confusion matrices driven by the RNA-seq-derived PAM50 calls. Discovery cohorts still receive predicted PAM50 score/classification outputs even without truth; BC-Ghana additionally compares those predictions against its metadata subtype labels. `Normal` calls are filtered out because these downstream cohorts are cancer samples.

## Inputs and Assumptions

- Ground-truth expression tables are expected in `labels/<COHORT>_Log2CPMplus1.tsv`.
- Subtype annotations are expected in `labels/<COHORT>_TFX_Subtype.tsv` when cohort-level plots need them.
- The Hallmark GMT is expected at `reference/h.all.v2026.1.Hs.symbols.gmt`.
- The shortlist GMT is expected at `reference/gsva_genesets_shortlist.gmt`.
- EPIC-Seq benchmarking uses the raw `PFE` column directly; it is not renamed to a normalized GEP-derived quantity.
- Discovery analyses assume the upstream prediction artifacts already exist under the directory structure configured in `analysis/pipeline_config.py`.

## Parallelism

The heavy plotting and metric kernels use `ProcessPoolExecutor` where appropriate. The runner defaults to all available CPUs and prints the resolved worker count at startup. Use `--n_workers` to override that default explicitly.
