"""Model-package alias for the shared Proteus scientific constants.

The canonical, dependency-free definitions live in the top-level
``proteus_constants`` module so the analysis pipeline can import them without
loading the torch-heavy ``models`` package. This module simply re-exports them
for convenient ``models.model_utils`` usage.
"""

from __future__ import annotations

from proteus_constants import K_GAMMA, Y_MAX

__all__ = ["Y_MAX", "K_GAMMA"]
