# Proteus Analysis Framework — Methods & Usage

## Overview

The analysis framework evaluates Proteus predictions against ground-truth RNA expression (log₂(CPM+1) scale, range $[0.0,\; 19.93]$) across cross-validation folds, held-out PDX samples, patient biopsies, and benchmarking against alternative methods. It produces per-gene and per-sample metrics, conformal interval calibration, gene-level reliability scoring, and publication-quality PDF figures.

---

## Quick Start

```bash
# BTS cross-validation + holdout (primary analysis)
python analysis/run_full_analysis.py --group BTS-CV

# Benchmarking against EPIC-Seq / XGBoost / ablations
python analysis/run_full_analysis.py --group Benchmarking --n_bootstraps 100

# Patient holdout cohorts (CRPC-WCDT, BLCA-RA)
python analysis/run_full_analysis.py --group Patient-HO

# Fine-tune CV
python analysis/run_full_analysis.py --group PAN-FT-CV
python analysis/run_full_analysis.py --group CRPC-FT-CV

# Patient discovery (no truth — heatmaps/KDEs only)
python analysis/run_full_analysis.py --group Patient-Discovery \
    --proteus_holdout results_holdout/BioDVN/<COHORT>

# Everything
python analysis/run_full_analysis.py --group all
```

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--group` | *(required)* | `BTS-CV`, `Benchmarking`, `PAN-FT-CV`, `CRPC-FT-CV`, `Patient-HO`, `Patient-Discovery`, `all` |
| `--proteus_holdout` | — | Path to holdout directory containing `predictions_tumor.tsv` |
| `--xgb_holdout` | — | Path to XGBoost holdout directory |
| `--truth_cohort` | — | Cohort name for truth labels (e.g. `CRPC-WCDT`) |
| `--overwrite` | `False` | Regenerate plots even if PDFs already exist |
| `--n_workers` | `0` (all CPUs) | Number of parallel workers |
| `--n_bootstraps` | `100` | Number of bootstrap resamples for confidence intervals |

---

## File Organisation

| File | Role |
|---|---|
| `analysis_utils.py` | Data loading, core statistics, metric computation, constants |
| `calibration.py` | Conformalized quantile regression (CQR) interval calibration |
| `reliability.py` | Post-hoc gene-level reliability scoring via ridge regression |
| `plot_utils.py` | All PDF figure generation (parallelised where applicable) |
| `run_full_analysis.py` | CLI orchestrator for all analysis groups |

---

## Study Design

The base training set (BTS) comprises **50 patient-derived xenograft** (PDX) tumor samples:
- **PC-PDX** (prostate cancer): 20 cross-validation + 5 holdout
- **SCLC-PDX** (small-cell lung cancer): 20 cross-validation + 5 holdout

Cross-validation uses **5 pre-determined folds** specified in `data/BTS_sample-splits.tsv`. Out-of-fold (OOF) predictions from all 5 folds produce predictions for all 40 CV samples. The 10 holdout samples are predicted by the base model directly.

---

## Metrics

### Per-Gene Metrics

For each gene $g$, computed across all samples $i = 1, \ldots, n$:

| Metric | Formula |
|---|---|
| Pearson $r$ | $r_g = \frac{\sum_i (y_{ig} - \bar{y}_g)(\hat{y}_{ig} - \bar{\hat{y}}_g)}{\sqrt{\sum_i (y_{ig} - \bar{y}_g)^2 \sum_i (\hat{y}_{ig} - \bar{\hat{y}}_g)^2}}$ |
| Spearman $\rho$ | Pearson $r$ of rank-transformed $y_{ig}$ and $\hat{y}_{ig}$ |
| RMSE | $\text{RMSE}_g = \sqrt{\frac{1}{n} \sum_i (y_{ig} - \hat{y}_{ig})^2}$ |
| MAE | $\text{MAE}_g = \frac{1}{n} \sum_i |y_{ig} - \hat{y}_{ig}|$ |

Requires $n \geq 3$ finite observation pairs; otherwise NaN.

### Per-Sample (Transcriptome) Metrics

For each sample $i$, computed across all genes $g = 1, \ldots, G$:

Same four metrics (Pearson $r$, Spearman $\rho$, RMSE, MAE) with the summation axis transposed.

### Gene-Variance-Weighted Pearson $r$ (GVW-$r$)

A global correlation statistic that up-weights genes with higher biological variance, penalising models that only predict low-variance (near-zero) genes well.

Let $w_g = \text{Var}(y_g)$ be the variance of gene $g$ computed across the full 50-sample truth pool. For the flattened (gene × sample) observation vector:

$$r_{\text{GVW}} = \frac{\sum_{ig} w_g\,(y_{ig} - \bar{y}_w)(\hat{y}_{ig} - \bar{\hat{y}}_w)}{\sqrt{\sum_{ig} w_g\,(y_{ig} - \bar{y}_w)^2 \;\sum_{ig} w_g\,(\hat{y}_{ig} - \bar{\hat{y}}_w)^2}}$$

where $\bar{y}_w = \frac{\sum_{ig} w_g\, y_{ig}}{\sum_{ig} w_g}$ is the weighted mean.

### Gene-Variance-Weighted Spearman $\rho$ (GVW-$\rho$)

GVW-$r$ applied to **within-gene ranks**: for each gene, truth and predictions are ranked across samples independently, then the GVW-weighted Pearson $r$ is computed on those ranks. Within-gene ranking removes cross-gene mean shifts so the metric reflects only rank concordance per gene, weighted by biological importance.

### Variance Group Breakdown

Genes are partitioned into three variance groups based on their truth variance $\sigma^2_g$ across the full study truth pool:

| Group | Range |
|---|---|
| Low | $0 \leq \sigma^2_g < 1$ |
| Medium | $1 \leq \sigma^2_g < 3$ |
| High | $\sigma^2_g \geq 3$ |

Per-group mean/median Pearson $r$, Spearman $\rho$, and RMSE are reported.

### Bootstrap Confidence Intervals

Confidence intervals for scalar metrics use the **percentile bootstrap** method with **sample resampling**:

1. Resample $n$ samples (columns) with replacement, $B$ times (default $B = 100$).
2. Compute the metric on each bootstrap replicate.
3. Report the $\alpha/2$ and $1 - \alpha/2$ percentiles where $\alpha = 0.05$.

Metrics with CIs: GVW-$\rho$, RMSE, mean sample $\rho$, mean gene $\rho$, mean top-10% gene $\rho$.

---

## Conformalized Quantile Regression (CQR) Calibration

The model's MDN output provides per-observation prediction intervals $[L_{ig},\, U_{ig}]$. These base intervals are post-hoc calibrated using **CQR** (Romano, Patterson & Candès, NeurIPS 2019) with regime-based stratification.

### Gene Regimes

Genes are stratified into regimes based on two truth-derived properties:

1. **Variance quantile** ($q_v$): $\text{Var}_i(y_{ig})$ ranked → partitioned into $K_v = 4$ equal-frequency bins.
2. **Zero-rate quantile** ($q_z$): $\Pr(y_{ig} = 0)$ ranked → partitioned into $K_z = 4$ equal-frequency bins.

Each gene receives a regime label $(q_v, q_z)$, yielding up to $K_v \times K_z = 16$ regimes. Regimes are computed on the full 50-sample truth pool (CV + holdout).

### Non-Conformity Scores

For calibration observation $(i, g)$:

$$S_{ig} = U_{ig} - L_{ig} \quad \text{(interval width)}$$

$$A_{ig} = \max\!\bigl(L_{ig} - y_{ig},\; y_{ig} - U_{ig},\; 0\bigr) \quad \text{(raw CQR score)}$$

$$\hat{A}_{ig} = \frac{A_{ig}}{S_{ig} + \epsilon} \quad \text{where } \epsilon = 10^{-6}$$

### Per-Regime Quantile

For regime $r$ with $n_r$ valid observations and target miscoverage $\alpha$ (default 0.05):

$$q_r = \left\lceil \frac{(1 - \alpha)(n_r + 1)}{n_r} \right\rceil\text{-th quantile of } \{\hat{A}_{ig}\}_{(i,g) \in r}$$

If $n_r < 10$, the regime is assigned $\hat{q}_r = 0$ (no adjustment).

### Interval Adjustment

$$L'_{ig} = L_{ig} - \hat{q}_r \cdot S_{ig}, \qquad U'_{ig} = U_{ig} + \hat{q}_r \cdot S_{ig}$$

with safeguards:
- If $L'_{ig} > U'_{ig}$, both collapse to $(L'_{ig} + U'_{ig})/2$.
- Clamped to $[0.0,\; 19.93]$ (expression floor/ceiling).

### Calibration Evaluation

Empirical coverage at nominal level $\gamma$:

$$\hat{C}(\gamma) = \frac{1}{N} \sum_{i,g} \mathbf{1}\!\left[y_{ig} \in [L'_{ig}(\gamma),\; U'_{ig}(\gamma)]\right]$$

where intervals at level $\gamma < 0.95$ are obtained by linearly scaling the half-widths:

$$L'_{ig}(\gamma) = \hat{y}_{ig} - \frac{\gamma}{0.95}(\hat{y}_{ig} - L'_{ig}), \qquad U'_{ig}(\gamma) = \hat{y}_{ig} + \frac{\gamma}{0.95}(U'_{ig} - \hat{y}_{ig})$$

Coverage is reported at nominal levels $\{0.50, 0.60, 0.70, 0.80, 0.90, 0.95\}$.

### Pipeline

1. **Fit** on OOF (40 CV) predictions: compute gene regimes, compute $\hat{q}_r$ per regime.
2. **Apply** to holdout CIs: widen intervals by $\hat{q}_r \cdot S_{ig}$.
3. **Evaluate**: coverage curves, per-regime coverage heatmaps, error-vs-width diagnostics.

---

## Gene-Level Reliability Scoring

A post-hoc ridge regression model predicts per-gene Pearson $r$ from gene-level features, enabling users to rank genes by expected prediction quality and retain only reliable genes.

### Features (per gene)

| Feature | Symbol | Definition |
|---|---|---|
| Mean CI width | $f_1$ | $\text{mean}_i(U_{ig} - L_{ig})$ |
| Median CI width | $f_2$ | $\text{median}_i(U_{ig} - L_{ig})$ |
| Prediction dynamic range | $f_3$ | $\text{SD}_i(\hat{y}_{ig})$ across samples (ddof=1) |
| Fold disagreement | $f_4$ | Mean (across samples) of SD (across folds) of fold-specific predictions |
| Hurdle ambiguity | $f_5$ | $\text{mean}_i(p_{0,ig} \times (1 - p_{0,ig}))$ — from the MDN zero-inflation head |
| Gene zero-rate | $f_6$ | $\Pr(y_g = 0)$ in the development truth set |
| Gene variance | $f_7$ | $\text{Var}(y_g)$ in the development truth set |

### Target

The regression target is the **Fisher z-transform** of per-gene Pearson $r$ (Fisher, 1921):

$$z_g = \text{arctanh}(r_g) = \frac{1}{2}\ln\!\left(\frac{1 + r_g}{1 - r_g}\right)$$

$|r_g|$ is clamped to $< 0.9999$ to avoid $\pm\infty$.

### Ridge Regression

Standard closed-form ridge (Hoerl & Kennard, 1970) with centred features:

$$\mathbf{w} = \bigl(\mathbf{X}_c^\top \mathbf{X}_c + \alpha \mathbf{I}\bigr)^{-1} \mathbf{X}_c^\top \mathbf{z}$$

Default regularisation: $\alpha = 1.0$. Requires $\geq 10$ genes with complete features.

### Prediction

$$\hat{z}_g = \mathbf{x}_g^\top \mathbf{w} + b \qquad \Longrightarrow \qquad \hat{r}_g = \tanh(\hat{z}_g)$$

### Retention Curve

Genes are sorted by predicted reliability $\hat{r}_g$ (descending). For fractions $f \in \{0.10, 0.20, \ldots, 1.00\}$, the mean true $r$ of the top-$\lceil f \cdot G \rceil$ genes is reported. If the model is informative, retaining fewer (higher-confidence) genes yields higher mean true $r$.

### Pipeline

1. **Fit** on OOF: compute gene features from OOF predictions; target = per-gene Pearson $r$ across CV samples.
2. **Predict** on holdout: compute features from holdout predictions; predict $\hat{r}_g$.
3. **Evaluate**: Spearman $\rho(\hat{r}_g, r_g)$, RMSE, retention curve.

---

## Analysis Groups

### BTS-CV (Base Training Set Cross-Validation)

**Input**: OOF predictions from 5-fold CV on 40 PDX samples + 10 holdout PDX predictions.

**Pipeline**:
1. Load OOF predictions (full MDN output: point predictions + CIs + $p_0$ + per-fold predictions).
2. Compute gene regimes from 50-sample canonical truth pool.
3. Fit CQR calibrator on all OOF samples.
4. Fit reliability model on OOF gene features.
5. Exclude PHT (healthy-donor admixture) samples from downstream analysis.
6. Per cohort (PC-PDX, SCLC-PDX):
   - Report metrics for CV-only, HO-only, and Combined.
   - Generate per-gene metric distributions, heatmaps, sample/gene scatters.
7. Apply CQR calibration to holdout CIs; evaluate coverage.
8. Predict gene reliability on holdout; evaluate retention.

**Outputs** (`analysis/results/BTS-CV/`):
- Summary TSVs (metrics per cohort/split)
- Per-gene and per-sample metric distributions (PDF)
- Double/single heatmaps for curated gene sets
- Sample scatter plots (truth vs pred per sample)
- Gene target scatter plots (truth vs pred per gene, coloured by subtype)
- Calibration coverage curves and regime heatmaps
- Reliability scatter and retention curves
- Gene reliability scores TSV

### Benchmarking

**Input**: PHS (patient-healthy surrogate) holdout predictions across a grid of tumor fractions (TFx) and sequencing depths.

**Models compared**:
- Proteus-Base, Proteus-NoSeq, Proteus-OnlySeq
- XGBoost (optional)
- EPIC-Seq (GEP, PFE, NDR, Gini, Shannon, GC scores)
- Null (column-wise shuffled truth)

**TFx grid**: {0.50, 0.30, 0.10, 0.05, 0.03, 0.01}
**Depths**: {10x, 5x} (1x, 30x, 50x excluded)

**Metrics per (TFx, depth, model)**: sample $\rho$ (mean), GVW-$\rho$, mean/median gene $\rho$, RMSE, mean top-10% gene $\rho$, count of genes with $\rho \geq 0.70$, with bootstrap CIs.

**Outputs** (`analysis/results/Benchmarking/`):
- Grid boxplots and histograms per metric
- Line plots with CI bands
- Summary TSVs

### Patient-HO (Patient Holdout)

**Default cohorts**: CRPC-WCDT, BLCA-RA.

Predictions are evaluated against biopsy-derived truth expression. Same metrics and plots as BTS-CV but without calibration/reliability (no calibration set matched to these cohorts).

### Patient-Discovery

Holdout predictions with no truth available. Produces only heatmaps and per-sample expression KDEs.

---

## Parallelisation

Compute-intensive operations use `ProcessPoolExecutor`:

| Operation | Parallelised Over | Worker Threshold |
|---|---|---|
| `gene_level_metrics` | Genes | $> 100$ genes and `n_workers > 1` |
| `sample_level_metrics` | Samples | $> 10$ samples and `n_workers > 1` |
| `batch_sample_scatters` | Sample scatter PDFs | Always (if `n_workers > 1`) |
| `batch_gene_scatters` | Gene scatter PDFs | Always (if `n_workers > 1`) |

`--n_workers 0` (default) uses all available CPUs.

---

## Output Directory Structure

```
analysis/results/
├── BTS-CV/
│   ├── PC-PDX/            # Per-cohort plots + metrics
│   │   ├── CV/
│   │   ├── HO/
│   │   ├── Combined/
│   │   ├── calibration/
│   │   ├── reliability/
│   │   └── gene_scatters/
│   ├── SCLC-PDX/
│   └── summary_{cohort}.tsv
├── Benchmarking/
│   ├── benchmark_grid_*.pdf
│   ├── benchmark_ecdna_lines.pdf
│   └── summary_benchmarking.tsv
├── Patient-HO/
│   ├── CRPC-WCDT/
│   └── BLCA-RA/
└── Patient-Discovery/
```
