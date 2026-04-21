# Copilot instructions (Proteus)

## Non-negotiable engineering standards (read first)
- **No unapproved additions.** Do not add new functionality, new CLI flags/options, new config keys, new files, new defaults, or behavioral changes unless the user explicitly requested it and approved it. If a change was not explicitly agreed, stop and ask.
- **Follow the plan exactly.** If a plan exists (or the user has stated requirements/constraints), follow it precisely. If you believe the plan must change, ask the user to approve the plan update before making that change.
- **Approval before scope changes.** If you discover a change you *think* is needed but it was not explicitly requested, you must ask for approval before editing code. Never "sneak" refactors, options, knobs, or scaffolding into the repo.
- **Approval before new knobs.** Do not introduce new hyperparameters/experiment fields, ablation controls, CLI arguments, or default behaviors without explicit user approval.
- **No new "systems" without explicit approval.** Do not introduce mode switches, compatibility layers, versioned schemas, fallback backends, optional execution paths, or tuning frameworks unless the user explicitly asked for that system and approved the plan.
- **No patchy coding.** Do not propose or implement “work-arounds”, “quick fixes”, “try/except to ignore”, broad exception swallowing, or stylistic/programmatic shortcuts.
- **Root-cause fixes only.** Every change must address the underlying failure mode (correctness, determinism, performance, infra stability) directly.
- **Robust + reproducible by default.** Prefer deterministic behavior, explicit configuration, and clear invariants. Avoid hidden global state and side effects.
- **Fail loudly when appropriate.** If an invariant is violated, raise a clear error with actionable context rather than silently continuing.
- **No defensive suppression.** Do not silence warnings/errors unless you can prove they are benign and you document the rationale in the PR-style summary.
- **Complete solutions.** If you change an interface/behavior, update all call sites, configs/CLI flags, and any relevant docs.
- **Minimal scope, maximum rigor.** Keep diffs focused, but do not cut corners on correctness.

## Hard guardrails (tooling + workspace boundaries)
- **Workspace boundary is absolute.** Only read/write files inside this `Proteus/` workspace root. Do not access parent directories, sibling repos, external mounts, or absolute paths outside the workspace.
- **Never execute repo code.** Do not run `python`, `pytest`, training/inference scripts, shell scripts, VS Code tasks, or any repo commands.
- **No terminal usage.** Do not use the integrated terminal (including `run_in_terminal`) unless the user explicitly asks you to run a command.
- **No environment changes.** Do not create/modify virtualenvs, install packages, or run dependency tools unless the user explicitly asks.
- **Edits must be exactly requested.** If a requested change appears to require additional refactors or new knobs, stop and ask for approval before editing.

## Code quality standards
- **Catch specific exceptions only.** Every `except` must name the exact exception types that can occur in the `try` body (e.g., `except (FileNotFoundError, KeyError, ValueError)`). Never use bare `except:` or `except Exception:`.
- **No legacy compatibility layers.** The codebase targets PyTorch 2.0+ and Python 3.10+. Do not add fallback imports, version-detection branches, or backward-compat shims.
- **Comments explain *why*, not *what*.** Scientific rationale, mathematical formulas, biological motivation, and design decisions belong in comments. Do not narrate obvious code.
- **Descriptive variable names.** Prefer `train_fold_variance_raw` over `_raw_var`, `_residual_crop_offsets` over `_skip_offsets`. Single-letter names only in tight numeric loops where context is unambiguous.
- **Functions should be focused.** Orchestrating functions coordinate helpers; they should not contain 300+ lines of inline logic. Extract coherent sub-operations into named private functions.
- **Module-level `__all__` exports.** Every package `__init__.py` must declare `__all__`. No wildcard imports (exception: `analysis/__init__.py` which re-exports from submodules for notebook convenience).

## Big picture (what runs where)
- **Proteus is a PyTorch + Zarr pipeline**: Triton outputs → per-sample `*.zarr` stores → sample-splits TSV → PyTorch `DataLoader`s → `BioDVN` model train/infer.
- Main entrypoint is `Proteus.py` (CLI, ~1600 lines). Core components:

### Entrypoint & CLI
- **`Proteus.py`** (~1600 lines) — CLI entrypoint and orchestrator.
  - Key functions: `ensure_datasets_exist()`, `generate_data()`, `train_model_cv()`, `full_train_from_splits()`, `run_model()`, `_build_configs_from_spec()`, `main()`.
  - Imports tuning entry points lazily from `tuning/` when `--run_mode tune` is used.

### Data compilation & validation
- **`proteus_utils/compile_sample_data.py`** (~1270 lines) — Triton → Zarr compilation and verification.
  - `process_dataset()` orchestrates all samples → Zarr. Pre-flight label validation catches naming mismatches before processing.
  - `verify_zarr_store()`, `verify_dataset()` validate Zarr store integrity.
  - Sample processing pipeline: `_load_signal_profiles()` → `_transform_features()` → `_build_flat_arrays()` → `_write_zarr_store()`.
  - Signal channels: 8 biological signals + 1 NaN mask (9 total). Nucleotides stored separately as uint8.
  - Module-level constants: `SIGNAL_CLIP_MIN/MAX` (±10), `MAX_FEATURE_NAN_FRACTION` (0.5), `COMPRESSOR` (zstd+bitshuffle).

### Dataset & data loading
- **`proteus_utils/proteus_utils.py`** (~1700 lines) — Dataset classes and loader construction.
  - `ProteusMultiSourceBatchDataset` — primary training dataset (256-gene chunks, 32-gene batches). Source-balanced sampling via `_balanced_chunks()` with per-epoch chunk re-shuffle.
  - `ProteusInferenceDataset`, `ProteusHoldoutValDataset` — inference/validation datasets.
  - `build_loaders_from_splits()`, `build_multisource_loaders_from_splits()` — construct train/val DataLoader pairs from splits TSV.
  - `create_data_loader()`, `create_inference_loader()` — lower-level loader factories.
  - `_parse_splits_tsv()` — shared TSV parsing helper.

### Shared data helpers
- **`proteus_utils/data_helpers.py`** (~244 lines) — Data-path and sample-mapping utilities.
  - `zarr_name_for_sample()`, `zarr_path_for_sample()` — canonical Zarr path construction.
  - `resolve_pht_source()` — PHT→source assignment via regex matching.
  - `build_source_map_from_tsv()`, `compute_source_variance()` — per-source variance maps.

### Evaluation
- **`proteus_utils/evaluation.py`** (~633 lines) — Fold-level prediction accumulation and metrics.
  - `save_fold_predictions()` — accumulates batches, computes per-source variance-weighted metrics, writes prediction TSVs and scatter plots.

### Model architecture
- **`models/models.py`** (~924 lines) — `BioDVN` dual-encoder VAE with channel-mode parameterization.
  - `BioDVNConfig` frozen dataclass with `channel_mode: str` ('all', 'no_seq', 'only_seq').
  - `SignalProcessingBlock` → `ReductionBlock` per encoder head, shared latent space.
  - `GradientReversalLayer` + `TFXDiscriminator` for TFX-invariant latent learning.
  - `channel_mode_for_model()` maps model names (e.g. 'BioDVN-noseq') to channel modes.
  - `get_model_instance()` constructs a `BioDVN` from config dict.

### MDN & loss functions
- **`models/model_utils.py`** (~1165 lines) — MDN heads, CRPS loss, β-TCVAE components.
  - `MixtureDensityHead` — 3-component mixture: hurdle (point mass at 0) + gamma + normal.
  - `mixture_crps()` — analytical CRPS (Gneiting & Raftery 2007) with Gauss-Laguerre quadrature for the gamma component.
  - `K_GAMMA = 0.80` (fixed gamma shape, empirical) — used throughout CRPS calculations.
  - `InternalLossComputer` — β-TCVAE decomposition + contrastive + reconstruction losses.
  - `total_correlation()` — minibatch-weighted TC estimator (Chen et al., 2018), clamped [-100, 100].
  - `BioDVNMDNLoss` — combines external CRPS loss with internal VAE losses.

### Training loop
- **`proteus_utils/trainer.py`** (~1441 lines) — `Trainer` class.
  - `TrainingConfig` frozen dataclass.
  - `train_epoch()` — AMP training with gradient accumulation, non-finite detection/recovery, EMA updates.
  - `validate()` — evaluation with per-gene Welford GVW-r accumulators, per-cohort/source CRPS.
  - `fit()` — epoch orchestration with cosine decay (+ warmup), early stopping, EMA weight swapping for validation/checkpointing.
  - Fused AdamW optimizer (PyTorch 2.0+). Optional `torch.compile()`.

### Metrics
- **`proteus_utils/metrics.py`** (~201 lines) — Statistical functions for fold evaluation.
  - `pearson_r()`, `spearman_r()`, `gvw_r()`, `coverage()`, `compute_regression_metrics()`, `scatter_stats()`.

### Configuration
- **`config/dataset_config.py`** (~79 lines) — `DatasetConfig` dataclass, `load_dataset_config()`, `list_datasets()`.
- **`config/experiment_spec.py`** (~682 lines) — Strict frozen `ExperimentSpec`, `TuningSpec`, `ParamSpec` dataclasses. YAML validation and normalization.

### Hyperparameter tuning
- **`tuning/`** package (5 submodules, ~1700 lines total):
  - **`optuna_tuner.py`** (~790 lines) — Public entry points: `run_optuna_cv_tuning()`, `run_optuna_fold_tuning()`, `coordinated_ask()`, `coordinated_tell()`, `run_coordinated_fold()`. SQLite-backed Optuna studies.
  - **`param_resolution.py`** (~392 lines) — `suggest_from_param()`, `resolve_trial_configs()`, `FixedParamTrial`, `config_hash()`, `configs_from_trial_json()`.
  - **`promotion.py`** (~434 lines) — `PromotionManager` for cross-fold top-config promotion.
  - **`metrics.py`** (~187 lines) — `collect_fold_metrics()`, `compute_cv_objective()`.
  - **`reporting.py`** (~273 lines) — `write_study_report()`, `compute_fold_stability()`, `narrow_search_space_from_study()`.

### Analysis & interpretation (downstream)
- **`analysis/`** — Downstream analysis utilities and plotting for analysis notebooks. Entry point: `run_full_analysis.py`.
- **`interpretation/`** — Captum-based model interpretation (attribution analysis, motif visualization, kernel analysis).

## HPC execution constraint
- Assume the user is running on an HPC node (Fred Hutch, SLURM).
- **Do not execute any code from this repo.** The workspace is on a remote server with strict permissions.
  - Do not run `python`, `pytest`, training scripts, inference scripts, shell scripts, or VS Code tasks.
  - Do not use the integrated terminal for repo commands.
  - You may only read/write/edit files inside this `Proteus/` workspace using editor tools.
  - If runtime verification is needed, ask the user to run commands and paste back the output.
- Do **not** explore outside the `Proteus/` workspace root (no parent directories, no external mounts/paths).

## Data layout + config conventions
- **Raw Triton layout** is sample-folder based; dataset YAMLs define the exact relative paths.
- Dataset definitions live in `config/datasets/*.yaml` (e.g. `PC-PDX.yaml`). These provide Triton relative paths under `sources` and label filenames under `labels`.
- `config/paths.yaml` controls base paths; it supports env expansion and `${key}` references.
  - Zarr base defaults to fast scratch; override with `PROTEUS_ZARR_BASE`.
  - Raw Triton base dir can be overridden with `PROTEUS_RAW_DATA_DIR`.
- **Zarr schema** is documented at the top of `proteus_utils/compile_sample_data.py`.
  - Training expects `flat/` arrays + `gene_names`.

## Sample naming convention
- **Full TSV names are canonical.** Internal maps (`sample_to_cohort`, `sample_to_source`, prediction DataFrames) use the full `Sample` column value from the splits TSV (e.g., `CRPC-RA_00-147`).
- **Bare names for Zarr paths only.** The cohort prefix is stripped only when constructing Zarr file paths on disk (e.g., `CRPC-RA/zarr_files/00-147.zarr`). The variable for this is called `zarr_name`.
- **`sample_name_map` override.** Dataset classes accept an optional `sample_name_map` (zarr_path → display name) so that batches emitted during validation carry full TSV names for cohort/source lookup, even though the underlying Zarr file uses a bare name.

## Metric architecture
- **GVW (Gene-Variance-Weighted)**: Metrics weighted by per-gene expression variance. Variance is used (not SD) because it directly measures information content and CRPS scales with observation variance.
- **Per-source variance alignment**: Epoch-level `trainer.validate()` uses the same per-source variance maps as fold-final `save_fold_predictions()`, ensuring checkpoint selection and reported metrics are consistent.
- **Balanced metrics**: Per-cohort/source metrics are averaged with equal weight regardless of sample count, preventing large cohorts from dominating.
- **Epoch-level GVW-r**: Computed via Welford online accumulators (O(1) memory per gene per source) during the validation loop. Per-gene Pearson r is variance-weighted and averaged across sources.
- **MDN clamp bounds**: `log_gamma_beta ∈ [-6, 4]` bounds gamma mean K/β ∈ [0.015, 320] — well beyond biological expression range [0, ~20] but numerically safe.
- **K_GAMMA = 0.80**: Fixed gamma shape parameter (empirical). Hardcoded as a module-level constant in `model_utils.py`. Not a tunable hyperparameter.

## Common workflows (CLI)
- Process Triton → Zarr:
  - `python Proteus.py --process_data --use_set PC-PDX`
- Verify Zarr stores:
  - `python Proteus.py --verify_data --use_set PC-PDX`
- Train (cross-validation):
  - `python Proteus.py --run_mode train --model_name BioDVN --use_set PC-PDX`
  - Fold-only (HPC parallelization): `python Proteus.py --run_mode train --use_set PC-PDX --test_fold 1`
- Full-train (single final model):
  - `python Proteus.py --run_mode full-train --use_set PC-PDX`
- Inference:
  - `python Proteus.py --run_mode run --model_name BioDVN --use_set <dataset>`
- Tuning:
  - `python Proteus.py --run_mode tune --use_set PC-PDX`
  - Coordinated fold-parallel: `coordinated_ask` → `run_coordinated_fold` (per SLURM job) → `coordinated_tell`

## Project-specific gotchas (don’t miss these)
- **No legacy compat code.** All v4 index formats, 11-channel signal shapes, `COHORT_NAME_MAP`, backward-compat experiment spec fields, AMP fallback imports, and `fused=True` fallback have been removed. Do not re-introduce them.
- Prefer the single current Triton → Zarr pipeline.
- CV fold assignments are pre-determined in `data/BTS_sample-splits.tsv` (Cohort/Sample/Fold columns). Training reads this TSV directly via `build_loaders_from_splits()`.
- Zarr stores are typically placed on **ephemeral scratch**; they may be cleared periodically.
  - `ensure_datasets_exist()` in `Proteus.py` can auto-regenerate missing stores for datasets that have YAML configs.
- **PHT samples** are synthetic (PDX-derived). Their truth labels duplicate their parent PDX, so they are excluded from variance computation. PHT→source assignment uses regex matching via `resolve_pht_source()` in `data_helpers.py`.
- Downstream analysis expects the on-disk naming convention used by `Proteus.py`, e.g.
  - `results_training/<model>_<use_set>/CV-fold_<k>/validation-fold_<k>_predictions_tumor.tsv`
  - See `analysis/run_full_analysis.py` for how these TSVs are consumed.
- **Signal shape**: `(2000, 12)` is expected; `(12, 2000)` is accepted and transposed. Any other shape raises `ValueError`. The 12 channels split into 8 biological signals + 4 nucleotide one-hot.
- **Channel modes**: `BioDVN` supports 3 modes via `BioDVNConfig.channel_mode`:
  - `'all'` — signals (9ch) + nucleotides (4ch) + all scalars (25)
  - `'no_seq'` — signals (9ch) only + non-seq scalars (25)
  - `'only_seq'` — nucleotides (4ch) only + ref_scalars (3)
- **`config/config_loader.py` was removed.** Dataset loading uses `config/dataset_config.py`. Experiment specs use `config/experiment_spec.py` directly.

## When editing/adding features
- Keep path handling CWD-independent (see `PROJECT_ROOT = Path(__file__).resolve().parent` in `Proteus.py`).
- When adding a new dataset:
  - Add `config/datasets/<NAME>.yaml` (Triton relative paths under `sources`, label filenames under `labels`).
  - Then run `--process_data` for that dataset.
- When adding statistical metrics: add to `proteus_utils/metrics.py` and export via `proteus_utils/__init__.py` `__all__`.
- When modifying validation metrics: ensure `trainer.validate()` and `save_fold_predictions()` (in `evaluation.py`) stay aligned. Both must use the same variance weighting scheme.
- When modifying tuning: the public API is in `tuning/__init__.py` (5 entry points). Internal helpers are in `param_resolution.py`, `metrics.py`, `promotion.py`, `reporting.py`.
- When modifying model architecture: update `BioDVNConfig` fields, ensure `channel_mode_for_model()` mapping is consistent, and update `get_model_instance()` if new config fields are added.
