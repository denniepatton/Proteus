"""Debug utilities for inspecting Proteus inputs and model behavior.

This module provides functions for debugging data loading and model issues,
particularly useful for diagnosing training performance problems.
"""

from typing import Dict, Any, Optional
import torch


def print_input_diagnostics(
    features: Dict[str, torch.Tensor],
    labels: Optional[Dict[str, torch.Tensor]] = None,
    sample_idx: int = 0,
    gene_idx: int = 0,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Print detailed diagnostics for a single gene's input tensors.
    
    This function helps diagnose data loading issues by printing tensor shapes,
    dtypes, value ranges, and statistics for all input features.
    
    Args:
        features: Dictionary of input feature tensors from DataLoader.
        labels: Optional dictionary of label tensors.
        sample_idx: Which sample in the batch to inspect (default: 0).
        gene_idx: Which gene in the chunk to inspect (default: 0).
        verbose: Whether to print to stdout.
        
    Returns:
        Dictionary of diagnostics for each tensor.
    """
    diagnostics: Dict[str, Dict[str, Any]] = {}
    
    if verbose:
        print("\n" + "=" * 80)
        print("INPUT TENSOR DIAGNOSTICS")
        print("=" * 80)
    
    # Feature tensors
    if verbose:
        print("\n--- FEATURES ---")
    
    for name, tensor in features.items():
        if not isinstance(tensor, torch.Tensor):
            if verbose:
                print(f"  {name}: (non-tensor) {type(tensor).__name__}")
            continue
        
        diag = _tensor_diagnostics(tensor, name)
        diagnostics[f"feature.{name}"] = diag
        
        if verbose:
            _print_tensor_diag(name, diag)
            
            # For the first gene, show actual values for critical tensors
            if name in ["TSS_signals", "TSS_nucleotides", "TSS_features"]:
                _print_sample_values(tensor, name, sample_idx, gene_idx)
    
    # Label tensors
    if labels is not None:
        if verbose:
            print("\n--- LABELS ---")
        
        for name, tensor in labels.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            
            diag = _tensor_diagnostics(tensor, name)
            diagnostics[f"label.{name}"] = diag
            
            if verbose:
                _print_tensor_diag(name, diag)
                
                if name in ["tumor_exp", "background_exp"]:
                    _print_sample_values(tensor, name, sample_idx, gene_idx)
    
    # Sanity checks
    if verbose:
        print("\n--- SANITY CHECKS ---")
        _run_sanity_checks(features, labels, diagnostics)
    
    if verbose:
        print("=" * 80 + "\n")
    
    return diagnostics


def _tensor_diagnostics(tensor: torch.Tensor, name: str) -> Dict[str, Any]:
    """Compute diagnostics for a single tensor."""
    with torch.no_grad():
        # Handle NaN values
        finite_mask = torch.isfinite(tensor)
        finite_tensor = tensor[finite_mask] if finite_mask.any() else tensor
        
        diag = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "n_elements": int(tensor.numel()),
            "n_nan": int((~finite_mask).sum().item()),
            "nan_fraction": float((~finite_mask).sum().item() / max(tensor.numel(), 1)),
        }
        
        if finite_tensor.numel() > 0 and finite_tensor.dtype in [torch.float32, torch.float64, torch.float16]:
            diag.update({
                "min": float(finite_tensor.min().item()),
                "max": float(finite_tensor.max().item()),
                "mean": float(finite_tensor.mean().item()),
                "std": float(finite_tensor.std().item()) if finite_tensor.numel() > 1 else 0.0,
                "has_neg": bool((finite_tensor < 0).any().item()),
                "zeros_frac": float((finite_tensor == 0).sum().item() / max(finite_tensor.numel(), 1)),
            })
        elif finite_tensor.numel() > 0:
            # Integer types
            diag.update({
                "min": int(finite_tensor.min().item()),
                "max": int(finite_tensor.max().item()),
                "unique_values": int(torch.unique(finite_tensor).numel()),
            })
    
    return diag


def _print_tensor_diag(name: str, diag: Dict[str, Any]) -> None:
    """Print formatted diagnostics for one tensor."""
    shape_str = str(diag["shape"])
    dtype_str = diag["dtype"].replace("torch.", "")
    
    # Value range
    if "mean" in diag:
        range_str = f"[{diag['min']:.4g}, {diag['max']:.4g}] mean={diag['mean']:.4g} std={diag['std']:.4g}"
    elif "unique_values" in diag:
        range_str = f"[{diag['min']}, {diag['max']}] unique={diag['unique_values']}"
    else:
        range_str = "N/A"
    
    # Warnings
    warnings = []
    if diag.get("nan_fraction", 0) > 0:
        warnings.append(f"NaN={diag['nan_fraction']*100:.1f}%")
    if diag.get("zeros_frac", 0) > 0.9:
        warnings.append(f"zeros={diag['zeros_frac']*100:.1f}%")
    if diag.get("has_neg", False) and name in ["TSS_signals", "TSS_features"]:
        warnings.append("has_negative!")
    
    warn_str = f" WARNING: {', '.join(warnings)}" if warnings else ""
    
    print(f"  {name:25s} {shape_str:20s} {dtype_str:10s} {range_str}{warn_str}")


def _print_sample_values(
    tensor: torch.Tensor, 
    name: str, 
    sample_idx: int = 0, 
    gene_idx: int = 0
) -> None:
    """Print actual values for inspection."""
    with torch.no_grad():
        if len(tensor.shape) == 1:
            val = tensor[sample_idx] if sample_idx < tensor.shape[0] else tensor[0]
            print(f"    -> gene[{gene_idx}] = {val.item():.6g}")
        elif len(tensor.shape) == 2:
            row = tensor[sample_idx] if sample_idx < tensor.shape[0] else tensor[0]
            preview = row[:min(8, len(row))].tolist()
            preview_str = ", ".join(f"{v:.4g}" for v in preview)
            if len(row) > 8:
                preview_str += f", ... ({len(row)} total)"
            print(f"    -> gene[{gene_idx}] = [{preview_str}]")
        elif len(tensor.shape) == 3:
            # For signals: (batch, length, channels)
            if sample_idx < tensor.shape[0]:
                slice_tensor = tensor[sample_idx]
            else:
                slice_tensor = tensor[0]
            # Show stats per channel at a few positions
            mid = slice_tensor.shape[0] // 2
            print(f"    -> gene[{gene_idx}] signal preview (pos 0, {mid}, -1):")
            for pos_idx, pos in enumerate([0, mid, slice_tensor.shape[0]-1]):
                vals = slice_tensor[pos, :min(4, slice_tensor.shape[1])].tolist()
                vals_str = ", ".join(f"{v:.3g}" for v in vals)
                print(f"        pos[{pos}]: [{vals_str}, ...]")
                if pos_idx >= 2:
                    break


def _run_sanity_checks(
    features: Dict[str, torch.Tensor],
    labels: Optional[Dict[str, torch.Tensor]],
    diagnostics: Dict[str, Dict[str, Any]]
) -> None:
    """Run and print sanity checks on the data."""
    issues = []
    
    # Check TSS_signals
    sig_diag = diagnostics.get("feature.TSS_signals", {})
    if sig_diag:
        if sig_diag.get("shape", (0, 0, 0))[1] != 2000:
            issues.append(f"ERROR: TSS_signals length={sig_diag['shape'][1]}, expected 2000")
        if sig_diag.get("shape", (0, 0, 0))[2] != 8:
            issues.append(f"ERROR: TSS_signals channels={sig_diag['shape'][2]}, expected 8")
        if sig_diag.get("nan_fraction", 0) > 0.1:
            issues.append(f"WARNING: TSS_signals has {sig_diag['nan_fraction']*100:.1f}% NaN values")
        if sig_diag.get("max", 0) > 1000:
            issues.append(f"WARNING: TSS_signals max={sig_diag['max']:.1f}, may need normalization")
    
    # Check TSS_nucleotides (should be one-hot)
    nt_diag = diagnostics.get("feature.TSS_nucleotides", {})
    if nt_diag:
        if nt_diag.get("min", 0) < 0 or nt_diag.get("max", 1) > 1:
            issues.append(f"ERROR: TSS_nucleotides range [{nt_diag.get('min')}, {nt_diag.get('max')}], expected [0, 1] one-hot")
    
    # Check TFX (now in labels)
    tfx_diag = diagnostics.get("label.TFX_label", {})
    if tfx_diag:
        if tfx_diag.get("min", 0) < 0 or tfx_diag.get("max", 1) > 1:
            issues.append(f"WARNING: TFX_label range [{tfx_diag.get('min'):.3f}, {tfx_diag.get('max'):.3f}], expected [0, 1]")
    
    # Check labels
    if labels is not None:
        for name in ["tumor_exp", "background_exp"]:
            lbl_diag = diagnostics.get(f"label.{name}", {})
            if lbl_diag:
                if lbl_diag.get("min", 0) < 0:
                    issues.append(f"WARNING: {name} has negative values (min={lbl_diag['min']:.3f})")
                if lbl_diag.get("max", 0) > 20:
                    issues.append(f"WARNING: {name} max={lbl_diag['max']:.2f}, expected log2(CPM+1) <= ~20")
    
    if not issues:
        print("  OK: All sanity checks passed")
    else:
        for issue in issues:
            print(f"  {issue}")
