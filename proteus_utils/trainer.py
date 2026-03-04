"""trainer.py (PyTorch)

Training loop with AMP support, learning-rate scheduling, and checkpointing.
"""

import os
import math
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# AMP (use new torch.amp API when available; keep fallback for older torch)
try:
    from torch.amp import autocast, GradScaler  # type: ignore[attr-defined]
    _USE_TORCH_AMP = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore[assignment]
    _USE_TORCH_AMP = False

# Progress bars (optional)
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    from rich.progress import (
        Progress,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TextColumn,
    )
except Exception:  # pragma: no cover
    Progress = None

# Local imports
from models import BioDVNConfig
from models.models import print_model_summary
from models.model_utils import (
    MaskedR2Metric, MetricsTracker,
    BioDVNMDNLoss
)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimizer settings
    weight_decay: float = 0.002
    
    # LR schedule settings (step-based cosine annealing with warmup)
    max_lr: float = 0.001        # Peak learning rate
    min_lr_ratio: float = 1e-4   # Minimum LR as fraction of max_lr (eta_min = min_lr_ratio * max_lr)
    warmup_start_lr: float = 0.0 # LR at the start of warmup (0 = start from 0)
    warmup_frac: float = 0.0     # Warmup as fraction of total training steps
    schedule_frac: float = 1.0   # Fraction of total training for cosine cycle (1.0 = full, 0.5 = half)
    
    # Training settings
    epochs: int = 15
    max_grad_norm: float = 1.0  # Gradient clipping threshold (prevents explosion)
    
    # Mixed precision
    use_amp: bool = True
    
    # torch.compile (PyTorch 2.0+)
    # NOTE: Disabled by default - can cause significant slowdown on first epoch
    # and may have graph-break issues with complex models. Enable only after
    # confirming model correctness.
    use_compile: bool = False  # Disabled for v1 parity testing
    compile_mode: str = 'reduce-overhead'  # 'default', 'reduce-overhead', 'max-autotune'
    
    # Early stopping
    patience: int = 5
    min_delta: float = 0.0
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_best_only: bool = True
    
    # Logging
    log_interval: int = 100  # Log every N batches

    # Progress bar
    progress_bar: bool = True

    # Progress backend: 'auto'|'tqdm'|'rich'|'none'
    # - auto: prefer Rich if installed, else tqdm
    progress_backend: str = 'auto'

    # Diagnostics
    track_batch_timing: bool = True

    # Runtime banner (prints device / memory / one batch shape)
    runtime_device_banner: bool = False
    
    # Transfer learning
    freeze_level: Optional[str] = None  # 'Medium-Heavy', 'Heavy', 'Extra-Heavy', 'Extreme'
    base_weights: Optional[str] = None


class CosineDecayScheduler:
    """Step-based cosine-annealing LR scheduler with linear warmup.

    During warmup (steps 1..warmup_steps), LR linearly interpolates from
    *warmup_start_lr* to *base_lr* (the optimizer's initial LR).  After
    warmup, LR follows a cosine curve from *base_lr* down to *eta_min*.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        eta_min: float = 1e-6,
        warmup_start_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Advance one optimizer step and update LR."""
        self.current_step += 1

        if self.warmup_steps > 0 and self.current_step <= self.warmup_steps:
            # Linear warmup from warmup_start_lr → base_lr
            frac = self.current_step / self.warmup_steps
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.warmup_start_lr + frac * (base_lr - self.warmup_start_lr)
        else:
            # Cosine decay from base_lr → eta_min
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))

    def get_last_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]





class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Metric to monitor
            epoch: Current epoch
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class Trainer:
    """
    Training loop for BioDVN with mixed precision support.
    
    Features:
    - Automatic mixed precision (AMP) for faster training
    - Cosine learning rate decay with optional warmup
    - Early stopping based on validation loss
    - Checkpointing (best model and periodic saves)
    - Metric tracking and logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        model_config: Optional[BioDVNConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.model_config = model_config or BioDVNConfig()
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Model setup
        self.model = model.to(self.device)
        
        # Print model summary (parameter counts)
        print_model_summary(self.model, model_name="BioDVN")
        
        # NOTE: channels_last disabled - it's designed for 4D NCHW tensors (Conv2d)
        # but this model primarily uses Conv1d (3D) and Linear (2D), which can cause
        # silent tensor layout corruption.
        # if self.device.type == 'cuda':
        #     self.model = self.model.to(memory_format=torch.channels_last)
        
        # Load base weights if provided
        if config.base_weights is not None:
            self._load_weights(config.base_weights)
        
        # Apply freezing if specified
        if config.freeze_level is not None:
            self._apply_freeze_level(config.freeze_level)

        # Optimizer is created lazily (after optional warmup/compile) to ensure
        # parameter sets and wrappers are finalized.
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._prepared_for_fit = False
        
        # Mixed precision
        self.use_amp = config.use_amp and self.device.type == 'cuda'
        if self.use_amp:
            if _USE_TORCH_AMP:
                # New API: torch.amp.GradScaler(device_type, ...)
                self.scaler = GradScaler('cuda')
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Loss functions
        self.external_loss_fn = BioDVNMDNLoss(
            beta_background=self.model_config.beta_background,
            beta_point_mse=getattr(self.model_config, 'beta_point_mse', 0.0),
        )
        
        # Metrics
        self.tumor_r2 = MaskedR2Metric()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='min'
        )
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Scheduler will be initialized when training starts
        self.scheduler = None

        # One-time runtime banner to confirm GPU execution during actual training.
        self._printed_runtime_device_banner = False

        # Active progress display references for cleanup on exception.
        # If train_epoch() raises mid-loop, fit() uses these to stop the
        # Rich Live display and prevent cascade "Only one live display" errors.
        self._active_rich_progress = None
        self._active_pbar = None

    def _create_optimizer(self) -> torch.optim.Optimizer:
        # Optimizer with fused AdamW if available (PyTorch 2.0+)
        try:
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.max_lr,
                weight_decay=self.config.weight_decay,
                fused=True,
            )
        except TypeError:
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.max_lr,
                weight_decay=self.config.weight_decay,
            )

    def _prepare_for_fit(self) -> None:
        if self._prepared_for_fit:
            return

        # Apply torch.compile() if enabled (PyTorch 2.0+)
        if self.config.use_compile and hasattr(torch, 'compile'):
            try:
                compile_mode = getattr(self.config, 'compile_mode', 'reduce-overhead')
                print(f'Compiling model with torch.compile(mode={compile_mode!r})...')
                self.model = torch.compile(self.model, mode=compile_mode)
                print('  Model compiled successfully.')
            except Exception as e:
                print(f'  Warning: torch.compile failed ({e}), continuing without compilation.')

        self.model.train()
        self.optimizer = self._create_optimizer()
        self._prepared_for_fit = True

    def _print_runtime_device_banner(self, features: Optional[Dict[str, torch.Tensor]] = None) -> None:
        if not bool(getattr(self.config, 'runtime_device_banner', False)):
            return
        if self._printed_runtime_device_banner:
            return
        self._printed_runtime_device_banner = True

        print('Runtime device check:')
        print(f'  device={self.device} amp={self.use_amp} torch_amp_api={_USE_TORCH_AMP}')

        if self.device.type == 'cuda' and torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                print(f'  cuda_device_index={idx} name={torch.cuda.get_device_name(idx)}')
            except Exception:
                pass
            try:
                print(f'  cudnn.benchmark={torch.backends.cudnn.benchmark}')
            except Exception:
                pass
            try:
                print(f'  matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}')
            except Exception:
                pass
        else:
            print('  WARNING: training device is not CUDA; training will run on CPU.')

        try:
            p = next(self.model.parameters())
            print(f'  model_param_device={p.device} dtype={p.dtype}')
        except Exception:
            pass

        if features is not None:
            # Show where one representative tensor landed after _move_batch_to_device.
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    print(f'  batch_tensor[{k}].device={v.device} dtype={v.dtype} shape={tuple(v.shape)}')
                    break

        if self.device.type == 'cuda' and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                print(f'  cuda_mem_allocated_GB={torch.cuda.memory_allocated()/1e9:.3f} max_allocated_GB={torch.cuda.max_memory_allocated()/1e9:.3f}')
            except Exception:
                pass
    
    def _load_weights(self, weights_path: str):
        """Load pre-trained weights."""
        print(f'Loading weights from {weights_path}')
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
    
    def _apply_freeze_level(self, level: str):
        """Apply freezing based on level."""
        print(f'Applying freeze level: {level}')
        
        if level == 'Extreme':
            self.model.freeze_encoders()
        elif level in ['Medium-Heavy', 'Heavy', 'Extra-Heavy']:
            self.model.freeze_signal_processing()
            if level in ['Extra-Heavy']:
                # Freeze first layers of encoders
                for name, param in self.model.tumor_encoder.named_parameters():
                    if any(f'layers.{i}' in name for i in range(4)):
                        param.requires_grad = False
                for name, param in self.model.background_encoder.named_parameters():
                    if any(f'layers.{i}' in name for i in range(4)):
                        param.requires_grad = False
    
    def _move_batch_to_device(
        self,
        features: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> Tuple[Dict, Dict]:
        """Move batch tensors to device."""
        features_device = {}
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                features_device[k] = v.to(self.device, non_blocking=True)
            else:
                features_device[k] = v  # Keep strings as-is
        
        labels_device = {k: v.to(self.device, non_blocking=True) for k, v in labels.items()}
        
        return features_device, labels_device
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        if self.optimizer is None:
            raise RuntimeError('Trainer is not prepared: optimizer is None. Call fit() to initialize training state.')
        metrics_tracker = MetricsTracker()
        
        start_time = time.time()
        n_batches = len(train_loader)

        # Timing diagnostics: measure time spent waiting for the next batch vs compute.
        last_iter_end = time.time()
        ema_data_time: Optional[float] = None
        ema_step_time: Optional[float] = None
        ema_iter_time: Optional[float] = None

        backend = str(getattr(self.config, 'progress_backend', 'auto')).lower()
        if backend not in ('auto', 'tqdm', 'rich', 'none'):
            backend = 'auto'

        if backend == 'auto':
            if Progress is not None:
                backend = 'rich'
            elif tqdm is not None:
                backend = 'tqdm'
            else:
                backend = 'none'

        use_pbar = bool(self.config.progress_bar and backend != 'none')

        # Fetch the first batch before creating tqdm so we can describe startup work
        # (DataLoader worker spin-up, prefetch, etc.) instead of showing a stuck 0% bar.
        data_iter = iter(train_loader)
        t_fetch0 = time.time()
        print(f'Epoch {epoch}: initializing DataLoader iterator...')
        print(f'Epoch {epoch}: waiting for first batch...')
        try:
            first_batch = next(data_iter)
        except StopIteration:
            return {'loss': float('nan'), 'epoch_time': 0.0}
        t_fetch1 = time.time()
        print(f'Epoch {epoch}: first batch ready (wait {(t_fetch1 - t_fetch0):.2f}s)')

        pbar = None
        rich_progress = None
        rich_task_id = None

        if use_pbar and backend == 'tqdm':
            pbar = tqdm(
                total=n_batches,
                desc=f'Train {epoch}',
                leave=False,
                dynamic_ncols=True,
                unit='batch',
                smoothing=0.05,
                mininterval=0.5,
            )
        elif use_pbar and backend == 'rich' and Progress is not None:
            rich_progress = Progress(
                TextColumn('{task.description}'),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
                refresh_per_second=10,
            )
            rich_progress.start()
            rich_task_id = rich_progress.add_task(f'Train {epoch}', total=n_batches)

        # Store references so fit() can clean up if train_epoch() raises.
        self._active_rich_progress = rich_progress
        self._active_pbar = pbar

        def _iter_batches():
            yield 0, first_batch
            for i, b in enumerate(data_iter, start=1):
                yield i, b

        # Reset timing origin now that we've accounted for first-batch wait.
        last_iter_end = time.time()

        for batch_idx, (features, labels) in _iter_batches():
            if self.config.track_batch_timing:
                data_time = time.time() - last_iter_end
            # Move to device
            features, labels = self._move_batch_to_device(features, labels)
            if batch_idx == 0:
                self._print_runtime_device_banner(features)
            step_start = time.time()
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            if _USE_TORCH_AMP:
                amp_ctx = autocast(device_type=self.device.type, enabled=self.use_amp)
            else:
                amp_ctx = autocast(enabled=self.use_amp)

            with amp_ctx:
                predictions, internal_loss_result = self.model(
                    features, return_internal_loss=True,
                    tfx=labels.get('TFX_label'),
                )

            # Compute losses in float32 (outside autocast).
            # The model forward benefits from float16 (conv/linear), but the
            # CRPS loss involves gammainc / erf chains that need float32.
            # The internal loss (KL, MSE reconstruction) also casts to float32
            # internally, so this placement is belt-and-suspenders.
            external_loss, external_metrics = self.external_loss_fn(
                predictions,
                labels,
                sample_weight=labels.get('sample_weight')
            )

            # Total loss
            if internal_loss_result is not None:
                internal_loss, internal_metrics = internal_loss_result
                total_loss = external_loss + internal_loss
            else:
                internal_metrics = {}
                total_loss = external_loss

            # Fail-fast on non-finite loss to avoid corrupting weights.
            # This is especially important under AMP, where a single NaN step
            # can poison subsequent epochs.
            if not torch.isfinite(total_loss):
                current_lr = (
                    self.scheduler.get_last_lr()[0]
                    if self.scheduler is not None
                    else self.config.max_lr
                )
                ext_val = float(external_loss.detach().cpu().item())
                int_val = (
                    float(internal_loss.detach().cpu().item())
                    if internal_loss_result is not None
                    else 'N/A'
                )
                # Build per-component breakdown for diagnosis.
                parts = []
                for k, v in external_metrics.items():
                    val = float(v) if not isinstance(v, torch.Tensor) else float(v.detach().cpu().item())
                    parts.append(f'{k}={val}')
                for k, v in internal_metrics.items():
                    val = float(v) if not isinstance(v, torch.Tensor) else float(v.detach().cpu().item())
                    parts.append(f'{k}={val}')
                breakdown = ', '.join(parts)
                raise RuntimeError(
                    f'Non-finite total_loss={float(total_loss.detach().cpu().item())} '
                    f'(external={ext_val}, internal={int_val}) '
                    f'at batch {batch_idx} (lr={current_lr:.3e}). '
                    f'Components: {breakdown}'
                )
            
            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            # Update LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            if self.config.track_batch_timing:
                step_time = time.time() - step_start
                alpha = 0.05
                ema_data_time = data_time if ema_data_time is None else (1 - alpha) * ema_data_time + alpha * data_time
                ema_step_time = step_time if ema_step_time is None else (1 - alpha) * ema_step_time + alpha * step_time
                iter_time = data_time + step_time
                ema_iter_time = iter_time if ema_iter_time is None else (1 - alpha) * ema_iter_time + alpha * iter_time
            
            # Track metrics - use TSS_signals for batch size
            batch_size = features['TSS_signals'].shape[0]
            metrics_tracker.update('loss', total_loss.item(), batch_size)
            for k, v in external_metrics.items():
                metrics_tracker.update(k, float(v) if not isinstance(v, torch.Tensor) else float(v.detach().cpu().item()), batch_size)
            for k, v in internal_metrics.items():
                metrics_tracker.update(k, float(v) if not isinstance(v, torch.Tensor) else float(v.detach().cpu().item()), batch_size)
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.max_lr
                avg_metrics = metrics_tracker.compute()
                avg_loss = avg_metrics.get('loss', total_loss.item())
                if pbar is not None:
                    postfix = {
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                    }
                    if self.config.track_batch_timing:
                        batches_s = (1.0 / ema_iter_time) if (ema_iter_time is not None and ema_iter_time > 0) else 0.0
                        postfix['batches_s'] = f'{batches_s:.2f}'
                        postfix['data_s'] = f'{(ema_data_time if ema_data_time is not None else 0.0):.3f}'
                        postfix['step_s'] = f'{(ema_step_time if ema_step_time is not None else 0.0):.3f}'
                    pbar.set_postfix(postfix)
                elif rich_progress is not None and rich_task_id is not None:
                    desc = f'Train {epoch}  loss={avg_loss:.4f} lr={current_lr:.2e}'
                    if self.config.track_batch_timing:
                        batches_s = (1.0 / ema_iter_time) if (ema_iter_time is not None and ema_iter_time > 0) else 0.0
                        desc += f' batches_s={batches_s:.2f} data_s={(ema_data_time if ema_data_time is not None else 0.0):.3f} step_s={(ema_step_time if ema_step_time is not None else 0.0):.3f}'
                    rich_progress.update(rich_task_id, description=desc)
                else:
                    print(f'  Batch {batch_idx}/{n_batches} | Loss: {avg_loss:.4f} | LR: {current_lr:.2e}')

            # Keep progress bar telemetry fresh (even when log_interval is large).
            if pbar is not None and self.config.track_batch_timing:
                batches_s = (1.0 / ema_iter_time) if (ema_iter_time is not None and ema_iter_time > 0) else 0.0
                avg_metrics_fresh = metrics_tracker.compute()
                avg_loss_fresh = avg_metrics_fresh.get('loss', total_loss.item())
                pbar.set_postfix({
                    'loss': f'{avg_loss_fresh:.4f}',
                    'lr': f'{(self.scheduler.get_last_lr()[0] if self.scheduler else self.config.max_lr):.2e}',
                    'batches_s': f'{batches_s:.2f}',
                    'data_s': f'{(ema_data_time if ema_data_time is not None else 0.0):.3f}',
                    'step_s': f'{(ema_step_time if ema_step_time is not None else 0.0):.3f}',
                })
            last_iter_end = time.time()

            if pbar is not None:
                pbar.update(1)
            if rich_progress is not None and rich_task_id is not None:
                rich_progress.advance(rich_task_id, 1)

        if pbar is not None:
            pbar.close()
        if rich_progress is not None:
            rich_progress.stop()
        self._active_rich_progress = None
        self._active_pbar = None
        
        epoch_time = time.time() - start_time
        metrics = metrics_tracker.compute()
        metrics['epoch_time'] = epoch_time
        
        return metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation DataLoader
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        metrics_tracker = MetricsTracker()
        self.tumor_r2.reset()

        # GVW-weighted R² accumulators
        tumor_gvw_r2 = MaskedR2Metric()
        bg_gvw_r2 = MaskedR2Metric()

        # GVW CRPS accumulators (weighted by gene variance weights).
        tumor_crps_sum = torch.tensor(0.0)
        tumor_crps_denom = torch.tensor(0.0)
        bg_crps_sum = torch.tensor(0.0)
        bg_crps_denom = torch.tensor(0.0)

        from models.model_utils import mixture_crps

        backend = str(getattr(self.config, 'progress_backend', 'auto')).lower()
        if backend not in ('auto', 'tqdm', 'rich', 'none'):
            backend = 'auto'
        if backend == 'auto':
            if Progress is not None:
                backend = 'rich'
            elif tqdm is not None:
                backend = 'tqdm'
            else:
                backend = 'none'

        use_pbar = bool(self.config.progress_bar and backend != 'none')
        iterable = val_loader
        rich_progress = None
        rich_task_id = None
        if use_pbar and backend == 'tqdm' and tqdm is not None:
            iterable = tqdm(
                iterable,
                total=len(val_loader),
                desc='Val',
                leave=False,
                dynamic_ncols=True,
                unit='batch',
                smoothing=0.05,
                mininterval=0.5,
            )
        elif use_pbar and backend == 'rich' and Progress is not None:
            rich_progress = Progress(
                TextColumn('{task.description}'),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=True,
                refresh_per_second=10,
            )
            rich_progress.start()
            rich_task_id = rich_progress.add_task('Val', total=len(val_loader))

        last_iter_end = time.time()
        ema_data_time: Optional[float] = None

        for features, labels in iterable:
            if self.config.track_batch_timing:
                data_time = time.time() - last_iter_end
            features, labels = self._move_batch_to_device(features, labels)

            if _USE_TORCH_AMP:
                amp_ctx = autocast(device_type=self.device.type, enabled=self.use_amp)
            else:
                amp_ctx = autocast(enabled=self.use_amp)

            with amp_ctx:
                predictions, _ = self.model(features, return_internal_loss=False)

            # Compute loss in float32 (outside autocast), same as training loop.
            external_loss, external_metrics = self.external_loss_fn(
                predictions,
                labels,
                sample_weight=labels.get('sample_weight')
            )

            # Resolve gene variance weights for this batch.
            w_val = None
            val_map = getattr(self, 'val_gene_weight_map', None)
            if isinstance(val_map, dict) and val_map and 'gene_name' in features:
                gnames = features['gene_name']
                if isinstance(gnames, (list, tuple)):
                    w_val = torch.tensor(
                        [float(val_map.get(str(g), 1.0)) for g in gnames],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(-1)
                else:
                    w_val = torch.tensor(
                        [[float(val_map.get(str(gnames), 1.0))]],
                        dtype=torch.float32,
                        device=self.device,
                    )
            else:
                # Fallback: use the (training-fold) gene weights attached by the loader.
                w_val = labels.get('sample_weight')

            # GVW CRPS: analytical CRPS for tumor head.
            crps_sum_b, crps_denom_b = mixture_crps(
                predictions['tumor_mdn'],
                labels['tumor_exp'],
                sample_weight=w_val,
            )
            tumor_crps_sum += crps_sum_b.cpu()
            tumor_crps_denom += crps_denom_b.cpu()

            # Background GVW weights
            w_bg = None
            bg_map = getattr(self, 'val_background_gene_weight_map', None)
            if isinstance(bg_map, dict) and bg_map and 'gene_name' in features:
                gnames = features['gene_name']
                if isinstance(gnames, (list, tuple)):
                    w_bg = torch.tensor(
                        [float(bg_map.get(str(g), 1.0)) for g in gnames],
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(-1)
                else:
                    w_bg = torch.tensor(
                        [[float(bg_map.get(str(gnames), 1.0))]],
                        dtype=torch.float32,
                        device=self.device,
                    )
            else:
                w_bg = labels.get('sample_weight')

            # GVW CRPS: analytical CRPS for background head.
            bg_crps_s, bg_crps_d = mixture_crps(
                predictions['background_mdn'],
                labels['background_exp'],
                sample_weight=w_bg,
            )
            bg_crps_sum += bg_crps_s.cpu()
            bg_crps_denom += bg_crps_d.cpu()

            # Use TSS_signals for batch size
            batch_size = features['TSS_signals'].shape[0]
            metrics_tracker.update('val_loss', external_loss.item(), batch_size)
            for k, v in external_metrics.items():
                metrics_tracker.update(f'val_{k}', float(v) if not isinstance(v, torch.Tensor) else float(v.detach().cpu().item()), batch_size)

            if use_pbar and backend == 'tqdm':
                # Show running average val loss
                current = metrics_tracker.compute().get('val_loss', external_loss.item())
                postfix = {'val_loss': f'{current:.4f}'}
                if self.config.track_batch_timing:
                    alpha = 0.05
                    ema_data_time = data_time if ema_data_time is None else (1 - alpha) * ema_data_time + alpha * data_time
                    postfix['data_s'] = f'{ema_data_time:.3f}'
                iterable.set_postfix(postfix)
            elif rich_progress is not None and rich_task_id is not None:
                current = metrics_tracker.compute().get('val_loss', external_loss.item())
                desc = f'Val  val_loss={current:.4f}'
                if self.config.track_batch_timing:
                    alpha = 0.05
                    ema_data_time = data_time if ema_data_time is None else (1 - alpha) * ema_data_time + alpha * data_time
                    desc += f' data_s={ema_data_time:.3f}'
                rich_progress.update(rich_task_id, description=desc)

            last_iter_end = time.time()

            if rich_progress is not None and rich_task_id is not None:
                rich_progress.advance(rich_task_id, 1)
            
            # Update R² metrics
            # Aggregate R² (unweighted, pooled across all genes/samples)
            self.tumor_r2.update(predictions['tumor_exp'], labels['tumor_exp'])
            # GVW R² (weighted by gene variance weights)
            tumor_gvw_r2.update(predictions['tumor_exp'], labels['tumor_exp'], sample_weight=w_val)
            bg_gvw_r2.update(predictions['background_exp'], labels['background_exp'], sample_weight=w_bg)
        
        metrics = metrics_tracker.compute()
        metrics['val_tumor_aggregate_r2'] = self.tumor_r2.compute()
        metrics['val_tumor_gvw_r2'] = tumor_gvw_r2.compute()
        metrics['val_background_gvw_r2'] = bg_gvw_r2.compute()

        if float(tumor_crps_denom.item()) > 0:
            metrics['val_tumor_gvw_crps'] = float((tumor_crps_sum / tumor_crps_denom).item())
        else:
            metrics['val_tumor_gvw_crps'] = float('nan')

        if float(bg_crps_denom.item()) > 0:
            metrics['val_background_gvw_crps'] = float((bg_crps_sum / bg_crps_denom).item())
        else:
            metrics['val_background_gvw_crps'] = float('nan')

        if rich_progress is not None:
            rich_progress.stop()
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
        epoch_end_callback=None,
        external_stop_check=None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs (overrides config)
            
        Returns:
            Dictionary of metric histories
        """
        epochs = epochs or self.config.epochs

        # One-time init: warmup/compile/optimizer.
        preflight_start = time.time()
        print('Preflight: warmup/compile/optimizer setup...')
        self._prepare_for_fit()
        print(f'Preflight complete in {(time.time() - preflight_start):.2f}s')
        
        # Initialize scheduler – derive step counts from epoch budget
        steps_per_epoch = len(train_loader)
        total_training_steps = epochs * steps_per_epoch
        total_sched_steps = int(self.config.schedule_frac * total_training_steps)
        total_sched_steps = max(total_sched_steps, steps_per_epoch)  # at least 1 epoch
        warmup_steps = int(self.config.warmup_frac * total_sched_steps)
        eta_min = self.config.min_lr_ratio * self.config.max_lr

        self.scheduler = CosineDecayScheduler(
            optimizer=self.optimizer,
            total_steps=total_sched_steps,
            warmup_steps=warmup_steps,
            eta_min=eta_min,
            warmup_start_lr=self.config.warmup_start_lr,
        )
        
        # History tracking
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_tumor_gvw_crps': [],
            'val_tumor_aggregate_r2': [],
            'val_tumor_gvw_r2': [],
            'learning_rate': [],
        }

        best_val_score = float('inf')
        
        print(f'Starting training for {epochs} epochs')
        print(f'Device: {self.device}')
        print(f'AMP: {self.use_amp}')
        print(f'LR schedule: cosine  max_lr={self.config.max_lr:.2e}  eta_min={eta_min:.2e} (ratio={self.config.min_lr_ratio:.1e})  sched_steps={total_sched_steps} ({self.config.schedule_frac:.0%} of {total_training_steps})  warmup={warmup_steps}steps ({self.config.warmup_frac:.1%})  warmup_start_lr={self.config.warmup_start_lr:.2e}')
        print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader) if val_loader else 0}')
        
        for epoch in range(1, epochs + 1):
            print(f'\nEpoch {epoch}/{epochs}')
            
            # Training
            try:
                train_metrics = self.train_epoch(train_loader, epoch)
            except Exception:
                # Clean up any active progress display to prevent cascade
                # "Only one live display may be active at once" errors in
                # subsequent trials within the same process.
                if self._active_rich_progress is not None:
                    try:
                        self._active_rich_progress.stop()
                    except Exception:
                        pass
                    self._active_rich_progress = None
                if self._active_pbar is not None:
                    try:
                        self._active_pbar.close()
                    except Exception:
                        pass
                    self._active_pbar = None
                raise
            history['train_loss'].append(train_metrics['loss'])
            history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            print(f'  Train Loss: {train_metrics["loss"]:.4f} | Time: {train_metrics["epoch_time"]:.1f}s')
            
            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_tumor_gvw_crps'].append(val_metrics.get('val_tumor_gvw_crps', float('nan')))
                history['val_tumor_aggregate_r2'].append(val_metrics.get('val_tumor_aggregate_r2', 0))
                history['val_tumor_gvw_r2'].append(val_metrics.get('val_tumor_gvw_r2', 0))

                print(
                    f'  Val Loss: {val_metrics["val_loss"]:.4f} '
                    f'| GVW-CRPS: {val_metrics.get("val_tumor_gvw_crps", float("nan")):.4f} '
                    f'| Agg R²: {val_metrics.get("val_tumor_aggregate_r2", 0):.4f} '
                    f'| GVW R²: {val_metrics.get("val_tumor_gvw_r2", 0):.4f}'
                )

                # Model selection / early stopping metric
                val_score = val_metrics.get('val_tumor_gvw_crps', val_metrics['val_loss'])

                # If validation collapses to NaN/Inf, stop immediately.
                if not isinstance(val_score, (int, float)) or not math.isfinite(float(val_score)):
                    raise RuntimeError(f'Non-finite validation score at epoch {epoch}: {val_score}')

                # Optional epoch-end hook (e.g., Optuna reporting or distributed stop)
                if epoch_end_callback is not None:
                    epoch_end_callback(epoch=epoch, val_metrics=val_metrics, val_score=val_score)

                # Optional external stop check (e.g., leader pruned this trial)
                if external_stop_check is not None and bool(external_stop_check(epoch=epoch)):
                    print('External stop requested; stopping training.')
                    break
                
                # Checkpointing
                if val_score < best_val_score:
                    best_val_score = val_score
                    self._save_checkpoint(epoch, is_best=True)
                elif not self.config.save_best_only:
                    self._save_checkpoint(epoch, is_best=False)
                
                # Early stopping
                if self.early_stopping(val_score, epoch):
                    print(f'\nEarly stopping triggered at epoch {epoch}')
                    print(f'Best epoch: {self.early_stopping.best_epoch}')
                    break
        
        print(f'\nTraining complete. Best validation score (GVW-CRPS): {best_val_score:.4f}')
        
        return history
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if self.optimizer is None:
            raise RuntimeError('Cannot save checkpoint before optimizer is initialized.')

        model_for_save = getattr(self.model, '_orig_mod', self.model)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_for_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'model_config': self.model_config,
        }
        
        if is_best:
            path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        
        torch.save(checkpoint, path)
        print(f'  Saved checkpoint: {path}')
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        model_for_load = getattr(self.model, '_orig_mod', self.model)
        model_for_load.load_state_dict(checkpoint['model_state_dict'])

        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Loaded checkpoint from {path} (epoch {checkpoint["epoch"]})')
        return checkpoint['epoch']


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None,
    model_config: Optional[BioDVNConfig] = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Convenience function for training a model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        model_config: Model configuration (for loss weights etc.)
        
    Returns:
        Trained model and training history
    """
    if config is None:
        config = TrainingConfig()
    
    trainer = Trainer(model, config, model_config)
    history = trainer.fit(train_loader, val_loader)
    
    return trainer.model, history
