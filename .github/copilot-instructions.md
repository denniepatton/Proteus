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

## Big picture (what runs where)
- **Proteus is a PyTorch + Zarr pipeline**: Triton outputs → per-sample `*.zarr` stores → sample-splits TSV → PyTorch `DataLoader`s → `BioDVN` model train/infer.
- Main entrypoint is `Proteus.py` (CLI). Core components:
  - Zarr compilation + validation: `proteus_utils/compile_sample_data.py` (`process_dataset()`, `verify_zarr_store()`)
  - Datasets/loaders: `proteus_utils/proteus_utils.py` (`ProteusGeneDataset`, `build_loaders_from_splits()`, `create_inference_loader()`)
  - Model: `models/models.py` (`BioDVN`, `BioDVNConfig`)
  - Training loop: `proteus_utils/trainer.py` (`Trainer`, AMP, cosine decay, checkpointing)

## HPC execution constraint
- Assume the user is running on an HPC node.
- **Do not execute any code from this repo.** The workspace is on a remote server with strict permissions.
  - Do not run `python`, `pytest`, training scripts, inference scripts, shell scripts, or VS Code tasks.
  - Do not use the integrated terminal for repo commands.
  - You may only read/write/edit files inside this `Proteus/` workspace using editor tools.
  - If runtime verification is needed, ask the user to run commands and paste back the output.
- Do **not** explore outside the `Proteus/` workspace root (no parent directories, no external mounts/paths).

## Data layout + config conventions
- **Raw Triton layout** is sample-folder based; dataset YAMLs define the exact relative paths.
- Dataset definitions live in `config/datasets/*.yaml` (e.g. `PC-PDX.yaml`). These provide Triton relative paths + label filenames.
- `config/paths.yaml` controls base paths; it supports env expansion and `${key}` references.
  - Zarr base defaults to fast scratch; override with `PROTEUS_ZARR_BASE`.
  - Raw Triton base dir can be overridden with `PROTEUS_RAW_DATA_DIR`.
- **Zarr schema** is documented at the top of `proteus_utils/compile_sample_data.py`.
  - Training expects `flat/` arrays + `gene_names`.

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

## Project-specific gotchas (don’t miss these)
- Prefer the single current Triton → Zarr pipeline.
- CV fold assignments are pre-determined in `data/BTS_sample-splits.tsv` (Cohort/Sample/Fold columns). Training reads this TSV directly via `build_loaders_from_splits()`.
- Zarr stores are typically placed on **ephemeral scratch**; they may be cleared periodically.
  - `ensure_datasets_exist()` in `Proteus.py` can auto-regenerate missing stores for datasets that have YAML configs.
- Downstream analysis expects the on-disk naming convention used by `Proteus.py`, e.g.
  - `results_training/<model>_<use_set>/CV-fold_<k>/validation-fold_<k>_predictions_tumor.tsv`
  - See `analysis/run_analysis.py` for how these TSVs are consumed.

## When editing/adding features
- Keep path handling CWD-independent (see `PROJECT_ROOT = Path(__file__).resolve().parent` in `Proteus.py`).
- When adding a new dataset:
  - Add `config/datasets/<NAME>.yaml` (Triton relative paths under `sources`, label filenames under `labels`).
  - Then run `--process_data` for that dataset.
