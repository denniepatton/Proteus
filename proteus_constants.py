"""Shared scientific constants for the Proteus model and downstream analysis.

Top-level, dependency-free module (no torch/numpy) so it can be imported from
both the training/inference stack (``models.model_utils``) and the lightweight
analysis pipeline (``analysis.*``) without pulling in heavy dependencies or
triggering the ``models`` package import (which loads torch). Keeping these
values in one place guarantees the model and the analysis code can never drift.
"""

from __future__ import annotations

# Hard upper bound on expression values in log2(CPM + 1) space.
# Derived from the data ceiling; this is never exceeded in practice.
Y_MAX: float = 19.93

# Fixed Gamma shape parameter for the MDN mixture component, determined by
# empirical fitting of RNA-seq expression distributions. The trained BioDVN
# checkpoints assume this exact value, so training, inference, and every
# downstream uncertainty/calibration computation must use it identically.
K_GAMMA: float = 0.75

__all__ = ["Y_MAX", "K_GAMMA"]
