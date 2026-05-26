"""
proteus_utils.py (PyTorch version)

This file contains the PyTorch Dataset class and data loading utilities
for efficient training with Zarr-based storage.

Updated for current Triton outputs:
- TSS_signals: 9 signal channels (float32)
- TSS_nucleotides: 4 NT one-hot channels (uint8)
- TSS_features: 11 numeric features (shared Triton features)
- GB_features: 11 numeric features (identical feature set)

Key features:
- Memory-mapped Zarr access for zero-copy reads
- Multi-worker DataLoader support with persistent workers
- Efficient batching with pre-shuffled indices
"""

import os
import json
import random
import re
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import zarr

from proteus_utils.data_helpers import (
    zarr_name_for_sample, zarr_path_for_sample,
    resolve_pht_source, invert_source_groups, collect_pdx_names,
    assign_source,
)


def _collate_identity(batch: List[Any]) -> Any:
    """Return the single element produced by DataLoader(batch_size=1)."""
    return batch[0]


def _seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for a worker process."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_single_item_loader(
    dataset: Union[Dataset, IterableDataset],
    *,
    num_workers: int,
    seed: int,
    collate_fn: Callable[[List[Any]], Any] = _collate_identity,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
) -> DataLoader:
    """Build a deterministic `batch_size=1` DataLoader with shared worker setup."""
    base_generator = torch.Generator()
    base_generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=_seed_dataloader_worker if num_workers > 0 else None,
        generator=base_generator,
        drop_last=False,
        pin_memory=True,
    )


def _write_temp_index_files(prefix: str, named_paths: Dict[str, List[str]]) -> Dict[str, str]:
    """Write one or more temporary JSON index files and return their paths."""
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix=prefix)
    index_paths: Dict[str, str] = {}
    for name, paths in named_paths.items():
        index_path = os.path.join(temp_dir, f'{name}.json')
        with open(index_path, 'w') as handle:
            json.dump(paths, handle)
        index_paths[name] = index_path
    return index_paths


_PHT_META_RE = re.compile(r'^PHT-Fold-\d+_(.+)-(BP[^_]+)(?:_.*)?$')
_SRL_VARIANCE_FLOOR = 0.1
_SRL_STD_FLOOR = float(np.sqrt(_SRL_VARIANCE_FLOOR))


def _parse_pht_metadata(sample_name: str) -> Tuple[bool, str, str]:
    match = _PHT_META_RE.match(str(sample_name))
    if match is None:
        return False, '', ''
    return True, match.group(1), match.group(2)


class ProteusGeneChunkIterableDataset(IterableDataset):
    """Iterable chunk loader that yields fixed-size gene microbatches.

    Goals:
      - Keep Zarr reads contiguous and chunk-friendly (read `chunk_size` genes at once).
      - Yield *training batches* of exactly `genes_per_batch` genes.
      - Shard deterministically across DataLoader workers for high throughput.

    Notes:
      - Any leftover genes at the end of a chunk smaller than `genes_per_batch`
        are skipped to maintain fixed batch size.
      - Validation can also use this loader; it will yield fixed-size batches,
        possibly skipping the tail remainder.
      - subsample_frac can be used for fast sanity checks (0.1 = ~10% of genes).
    """

    def __init__(
        self,
        index_file: str,
        chunk_size: int = 256,
        genes_per_batch: int = 32,
        seed: int = 42,
        labeled: bool = True,
        cache_stores: bool = True,
        include_metadata: bool = False,
        subsample_frac: Optional[float] = None,
        gene_weights: Optional[Dict[str, float]] = None,
        val_frac: Optional[float] = None,
        val_split: bool = False,
        cohort_map: Optional[Dict[str, str]] = None,
        balance_cohorts: Optional[List[str]] = None,
        source_map: Optional[Dict[str, str]] = None,
        target_n_per_source: Optional[int] = None,
        sample_name_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.index_file = str(index_file)
        self.chunk_size = int(chunk_size)
        self.genes_per_batch = int(genes_per_batch)
        self.seed = int(seed)
        self.labeled = bool(labeled)
        self.cache_stores = bool(cache_stores)
        self.include_metadata = bool(include_metadata)
        self.subsample_frac = subsample_frac
        self.gene_weights = gene_weights
        self.val_frac = val_frac
        self.val_split = val_split
        self.cohort_map = cohort_map       # zarr_path -> cohort name
        self.balance_cohorts = balance_cohorts  # cohorts to balance
        self.source_map = source_map       # zarr_path -> source group name
        self.target_n_per_source = target_n_per_source
        self.sample_name_map = sample_name_map  # zarr_path -> display name (TSV sample)
        self._balance_iter = 0
        self._shuffle_buffer_batches = max(
            16,
            2 * (self.chunk_size // self.genes_per_batch),
        )

        if self.genes_per_batch <= 0:
            raise ValueError('genes_per_batch must be > 0')
        if self.chunk_size < self.genes_per_batch:
            raise ValueError(f'chunk_size ({self.chunk_size}) must be >= genes_per_batch ({self.genes_per_batch})')
        if (self.chunk_size % self.genes_per_batch) != 0:
            raise ValueError(
                f'chunk_size ({self.chunk_size}) must be a multiple of genes_per_batch ({self.genes_per_batch}) '
                'to guarantee fixed-size batches.'
            )
        if subsample_frac is not None and not (0.0 < subsample_frac <= 1.0):
            raise ValueError(f'subsample_frac must be in (0.0, 1.0], got {subsample_frac}')

        with open(self.index_file, 'r') as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            raise ValueError('Fold index JSON must be a list of Zarr paths.')
        # Deduplicate while preserving first-seen order.
        seen: set[str] = set()
        self.sample_paths: List[str] = []
        for p in loaded:
            s = str(p)
            if s not in seen:
                seen.add(s)
                self.sample_paths.append(s)

        # Build chunk schedule (deterministic, cached to disk).
        self._chunks = self._build_chunk_schedule(
            self.index_file, self.sample_paths, self.chunk_size, self.seed,
        )

        # Apply subsampling if requested (for fast sanity checks).
        if self.subsample_frac is not None and self.subsample_frac < 1.0:
            n_chunks = len(self._chunks)
            n_keep = max(1, int(n_chunks * self.subsample_frac))
            rng = random.Random(self.seed + 999)
            subsample_indices = sorted(rng.sample(range(n_chunks), n_keep))
            self._chunks = [self._chunks[i] for i in subsample_indices]

        # Gene-level train/val split at chunk granularity.
        if self.val_frac is not None and self.val_frac > 0:
            n_chunks = len(self._chunks)
            n_val = max(1, int(n_chunks * self.val_frac))
            rng_split = random.Random(self.seed + 777)
            val_indices = set(rng_split.sample(range(n_chunks), n_val))
            if self.val_split:
                self._chunks = [self._chunks[i] for i in sorted(val_indices)]
            else:
                self._chunks = [c for i, c in enumerate(self._chunks) if i not in val_indices]

        self._n_batches = self._count_batches(self._chunks)
        self._balanced_n_batches = self._estimate_balanced_batches(
            self._chunks, self.sample_paths, self.genes_per_batch,
            self.source_map, self.target_n_per_source,
            self.cohort_map, self.balance_cohorts,
            self._n_batches,
        )

        # Per-worker caches (populated lazily).
        self._store_cache: Dict[str, zarr.Group] = {}
        self._flat_grp_cache: Dict[str, zarr.Group] = {}
        self._gene_names_cache: Dict[str, List[str]] = {}
        self._sample_name_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Static helpers (extracted from __init__ for clarity)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_chunk_schedule(
        index_file: str,
        sample_paths: List[str],
        chunk_size: int,
        seed: int,
    ) -> List[Tuple[str, int, int]]:
        """Build (or load cached) deterministic chunk schedule."""
        cache_path = f"{index_file}.chunksched_{chunk_size}_seed{seed}.json"
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                return [(str(p), int(s), int(e)) for (p, s, e) in json.load(f)]

        chunks: List[Tuple[str, int, int]] = []
        for path in sample_paths:
            store = zarr.open(path, mode='r')
            n_genes = int(store['flat']['TSS_signals'].shape[0])
            for start in range(0, n_genes, chunk_size):
                end = min(start + chunk_size, n_genes)
                chunks.append((str(path), int(start), int(end)))

        rng = random.Random(seed)
        rng.shuffle(chunks)
        with open(cache_path, 'w') as f:
            json.dump(chunks, f)
        return chunks

    @staticmethod
    def _estimate_balanced_batches(
        chunks: List[Tuple[str, int, int]],
        sample_paths: List[str],
        genes_per_batch: int,
        source_map: Optional[Dict[str, str]],
        target_n_per_source: Optional[int],
        cohort_map: Optional[Dict[str, str]],
        balance_cohorts: Optional[List[str]],
        fallback: int,
    ) -> int:
        """Upper-bound batch count for balanced training modes.

        With persistent_workers the main-process balance counter never
        advances, so __len__ can't predict the exact epoch size.  This
        computes a tight upper bound by assuming the worst-case sample
        selection (samples with the most batches).
        """
        from collections import defaultdict

        if source_map and target_n_per_source:
            batches_per_sample: Dict[str, int] = {}
            for path, start, end in chunks:
                batches_per_sample[path] = (
                    batches_per_sample.get(path, 0)
                    + (end - start) // genes_per_batch
                )
            source_paths: Dict[str, List[str]] = defaultdict(list)
            for path in sample_paths:
                src = source_map.get(path)
                if src is not None:
                    source_paths[src].append(path)
            total = 0
            for paths in source_paths.values():
                per_sample = sorted(
                    [batches_per_sample.get(p, 0) for p in paths],
                    reverse=True,
                )
                if len(per_sample) >= target_n_per_source:
                    total += sum(per_sample[:target_n_per_source])
                else:
                    full = target_n_per_source // len(per_sample)
                    rem = target_n_per_source % len(per_sample)
                    total += sum(per_sample) * full + sum(per_sample[:rem])
            return total

        if cohort_map and balance_cohorts:
            batches_per_sample = {}
            for path, start, end in chunks:
                batches_per_sample[path] = (
                    batches_per_sample.get(path, 0)
                    + (end - start) // genes_per_batch
                )
            cohort_paths: Dict[str, List[str]] = defaultdict(list)
            for path in sample_paths:
                coh = cohort_map.get(path)
                if coh is not None:
                    cohort_paths[coh].append(path)
            balance_set = set(balance_cohorts)
            per_cohort_counts = {
                c: len(cohort_paths[c])
                for c in balance_cohorts if c in cohort_paths
            }
            n_min = min(per_cohort_counts.values()) if per_cohort_counts else 0
            total = 0
            for coh, paths in cohort_paths.items():
                per_sample = sorted(
                    [batches_per_sample.get(p, 0) for p in paths],
                    reverse=True,
                )
                if coh in balance_set:
                    total += sum(per_sample[:n_min])
                else:
                    total += sum(per_sample)
            return total

        return fallback

    def _count_batches(self, chunks: List[Tuple[str, int, int]]) -> int:
        n_batches = 0
        for _path, start, end in chunks:
            n_batches += max(0, int((int(end) - int(start)) // self.genes_per_batch))
        return int(n_batches)

    def __len__(self) -> int:
        if self.source_map and self.target_n_per_source:
            return self._balanced_n_batches
        if self.cohort_map and self.balance_cohorts:
            return self._balanced_n_batches
        return int(self._n_batches)

    def _balanced_chunks(self, epoch: int) -> List[Tuple[str, int, int]]:
        """Return chunks filtered for balanced training.

        Two modes:

        1.  **Source-group balancing** (``source_map`` + ``target_n_per_source``):
            Groups samples by source, then draws exactly
            ``target_n_per_source`` samples per source per epoch
            (oversampling small sources with replacement,
            undersampling large sources without).

        2.  **Legacy cohort balancing** (``balance_cohorts`` only):
            Undersamples balanced cohorts to the minority count.
            Non-balanced cohorts are always included at full count.
        """
        from collections import defaultdict

        # ------- Source-group balanced sampling -------
        if self.source_map and self.target_n_per_source:
            source_paths: Dict[str, List[str]] = defaultdict(list)
            for path in self.sample_paths:
                src = self.source_map.get(path)
                if src is not None:
                    source_paths[src].append(path)

            rng = random.Random(self.seed + epoch * 997)
            selected_paths: set = set()
            for src_name, paths in source_paths.items():
                n_avail = len(paths)
                n_target = self.target_n_per_source
                if n_avail == 0:
                    continue
                if n_avail >= n_target:
                    chosen = rng.sample(paths, n_target)
                else:
                    # Oversample: full copy + random remainder
                    full_copies = n_target // n_avail
                    remainder = n_target % n_avail
                    chosen = paths * full_copies + rng.sample(paths, remainder)
                selected_paths.update(chosen)
            filtered = [c for c in self._chunks if c[0] in selected_paths]
            rng.shuffle(filtered)
            return filtered

        # ------- Legacy cohort-balanced sampling -------
        cohort_paths: Dict[str, List[str]] = defaultdict(list)
        for path in self.sample_paths:
            coh = self.cohort_map.get(path, None)
            if coh is not None:
                cohort_paths[coh].append(path)

        counts = {c: len(cohort_paths[c]) for c in self.balance_cohorts
                  if c in cohort_paths}
        if not counts:
            return self._chunks
        n_min = min(counts.values())

        rng = random.Random(self.seed + epoch * 997)
        selected_paths = set()
        for coh in self.balance_cohorts:
            paths = cohort_paths.get(coh, [])
            chosen = rng.sample(paths, min(n_min, len(paths)))
            selected_paths.update(chosen)

        for coh, paths in cohort_paths.items():
            if coh not in self.balance_cohorts:
                selected_paths.update(paths)

        filtered = [c for c in self._chunks if c[0] in selected_paths]
        rng.shuffle(filtered)
        return filtered

    def _get_store(self, zarr_path: str) -> zarr.Group:
        if self.cache_stores:
            store = self._store_cache.get(zarr_path)
            if store is None:
                store = zarr.open(zarr_path, mode='r')
                self._store_cache[zarr_path] = store
            return store
        return zarr.open(zarr_path, mode='r')

    def _get_flat_grp(self, zarr_path: str) -> zarr.Group:
        """Return the 'flat' data group, caching the reference.

        The flat group is resolved once per worker per store and then
        reused for all subsequent epochs.  This avoids repeated
        filesystem metadata probes (os.path.isfile on the scratch FS)
        that can transiently fail under heavy concurrent I/O.
        """
        if self.cache_stores:
            grp = self._flat_grp_cache.get(zarr_path)
            if grp is not None:
                return grp
        store = self._get_store(zarr_path)
        grp = store['flat']
        if self.cache_stores:
            self._flat_grp_cache[zarr_path] = grp
        return grp

    def _get_sample_name(self, store: zarr.Group, zarr_path: str) -> str:
        cached = self._sample_name_cache.get(zarr_path)
        if cached is not None:
            return cached
        if isinstance(self.sample_name_map, dict) and zarr_path in self.sample_name_map:
            name = self.sample_name_map[zarr_path]
        else:
            name = store.attrs.get('sample_name')
            if name is None:
                name = Path(zarr_path).stem.replace('.zarr', '')
        name = str(name)
        self._sample_name_cache[zarr_path] = name
        return name

    def _get_gene_names(self, store: zarr.Group, zarr_path: str) -> List[str]:
        cached = self._gene_names_cache.get(zarr_path)
        if cached is not None:
            return cached
        names = [str(g) for g in list(store['gene_names'][:])]
        self._gene_names_cache[zarr_path] = names
        return names

    def _build_chunk_microbatches(
        self,
        zarr_path: str,
        start: int,
        end: int,
        rng: random.Random,
    ) -> List[Tuple[Dict[str, Any], Dict[str, torch.Tensor]]]:
        """Load one contiguous chunk and return shuffled fixed-size microbatches.

        Zarr reads stay chunk-friendly because the full 256-gene window is read
        contiguously.  We then shuffle the 32-gene microbatches from that chunk
        before they enter the worker-local mix buffer so training does not see
        8 back-to-back batches from the same sample unless the buffer is fully
        drained.
        """
        store = self._get_store(zarr_path)
        data_grp = self._get_flat_grp(zarr_path)

        tss_signals = torch.from_numpy(data_grp['TSS_signals'][start:end])
        tss_nt = torch.from_numpy(data_grp['TSS_nucleotides'][start:end].astype(np.float32))
        tss_feat = torch.from_numpy(data_grp['TSS_features'][start:end])
        gb_feat = torch.from_numpy(data_grp['GB_features'][start:end])
        ref_scalars = torch.from_numpy(data_grp['ref_scalars'][start:end])
        tfx = torch.from_numpy(data_grp['TFX_label'][start:end]).to(torch.float32).reshape(-1)

        if self.gene_weights is not None:
            gene_names_chunk = self._get_gene_names(store, zarr_path)
            gene_weight = torch.tensor(
                [self.gene_weights.get(gene_names_chunk[i], 1.0) for i in range(start, end)],
                dtype=torch.float32,
            )
        elif 'gene_weight' in data_grp:
            gene_weight = torch.from_numpy(data_grp['gene_weight'][start:end]).to(torch.float32).reshape(-1)
        else:
            gene_weight = torch.ones(int(end - start), dtype=torch.float32)

        if self.labeled:
            tumor_exp = torch.from_numpy(data_grp['tumor_exp'][start:end]).to(torch.float32).reshape(-1)
            background_exp = torch.from_numpy(data_grp['background_exp'][start:end]).to(torch.float32).reshape(-1)

        if self.include_metadata:
            sample_name = self._get_sample_name(store, zarr_path)
            gene_names = self._get_gene_names(store, zarr_path)

        chunk_len = int(end - start)
        batch_offsets = [
            off for off in range(0, chunk_len, self.genes_per_batch)
            if off + self.genes_per_batch <= chunk_len
        ]
        rng.shuffle(batch_offsets)

        microbatches: List[Tuple[Dict[str, Any], Dict[str, torch.Tensor]]] = []
        for off in batch_offsets:
            mb_end = off + self.genes_per_batch
            sl = slice(int(off), int(mb_end))
            features: Dict[str, Any] = {
                'TSS_signals': tss_signals[sl],
                'TSS_nucleotides': tss_nt[sl],
                'TSS_features': tss_feat[sl],
                'GB_features': gb_feat[sl],
                'ref_scalars': ref_scalars[sl],
            }

            if self.include_metadata:
                features['sample_name'] = [sample_name] * self.genes_per_batch
                features['gene_name'] = [str(g) for g in gene_names[int(start + off):int(start + mb_end)]]

            if self.labeled:
                labels: Dict[str, torch.Tensor] = {
                    'tumor_exp': tumor_exp[sl].unsqueeze(-1),
                    'background_exp': background_exp[sl].unsqueeze(-1),
                    'sample_weight': gene_weight[sl].unsqueeze(-1),
                    'TFX_label': tfx[sl].unsqueeze(-1),
                }
            else:
                labels = {
                    'sample_weight': gene_weight[sl].unsqueeze(-1),
                    'TFX_label': tfx[sl].unsqueeze(-1),
                }

            microbatches.append((features, labels))

        return microbatches

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)

        # Per-epoch deterministic chunk reshuffling.  This preserves exact
        # once-per-epoch example exposure while preventing the model from
        # seeing the same sample-local chunk order every epoch.
        self._balance_iter += 1
        epoch_seed = self.seed + self._balance_iter * 997
        active_chunks = list(self._chunks)
        if self.source_map and self.target_n_per_source:
            active_chunks = self._balanced_chunks(self._balance_iter)
        elif self.cohort_map and self.balance_cohorts:
            active_chunks = self._balanced_chunks(self._balance_iter)
        else:
            random.Random(epoch_seed).shuffle(active_chunks)

        worker_rng = random.Random(epoch_seed + 104729 * (worker_id + 1))
        microbatch_buffer: List[Tuple[Dict[str, Any], Dict[str, torch.Tensor]]] = []

        # Deterministic sharding of chunks across workers.
        for chunk_idx in range(worker_id, len(active_chunks), num_workers):
            zarr_path, start, end = active_chunks[chunk_idx]
            microbatch_buffer.extend(
                self._build_chunk_microbatches(
                    zarr_path,
                    int(start),
                    int(end),
                    worker_rng,
                )
            )
            while len(microbatch_buffer) >= self._shuffle_buffer_batches:
                pop_idx = worker_rng.randrange(len(microbatch_buffer))
                yield microbatch_buffer.pop(pop_idx)

        while microbatch_buffer:
            pop_idx = worker_rng.randrange(len(microbatch_buffer))
            yield microbatch_buffer.pop(pop_idx)


class ProteusInferenceDataset(Dataset):
    """
    Simplified dataset for inference on new samples.
    Loads all genes from a single Zarr store efficiently.
    
    Supports both a gene-level hierarchy + flat/ group layout and a flat-only layout.
    
    Args:
        zarr_path: Path to sample Zarr store
    """
    
    def __init__(self, zarr_path: str):
        self.zarr_path = zarr_path
        self.store = zarr.open(zarr_path, mode='r')
        self.n_genes = self.store.attrs['n_genes']
        self.sample_name = self.store.attrs['sample_name']
        self.gene_names = list(self.store['gene_names'][:])
        
        # Determine data source (flat/ group if available, else root)
        data_grp = self.store['flat'] if 'flat' in self.store else self.store
        
        # Pre-load all data into memory for inference (typically single samples)
        self._tss_signals = data_grp['TSS_signals'][:]
        self._tss_nucleotides = data_grp['TSS_nucleotides'][:]
        self._tss_features = data_grp['TSS_features'][:]
        self._gb_features = data_grp['GB_features'][:]
        self._ref_scalars = data_grp['ref_scalars'][:]
    
    def __len__(self) -> int:
        return self.n_genes
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get features for a single gene (no labels)."""
        return {
            'TSS_signals': torch.from_numpy(self._tss_signals[idx]),
            'TSS_nucleotides': torch.from_numpy(self._tss_nucleotides[idx].astype(np.float32)),
            'TSS_features': torch.from_numpy(self._tss_features[idx]),
            'GB_features': torch.from_numpy(self._gb_features[idx]),
            'ref_scalars': torch.from_numpy(self._ref_scalars[idx]),
        }


class ProteusMultiSourceBatchDataset(IterableDataset):
    """PHT-aware source-balanced gene-conditioned batch loader.

    Each training batch samples a small set of genes and, for each gene,
    draws multiple samples across source groups.  Gene choice is variance
    weighted, PHT draws are grouped by parent tumor or HD donor when possible,
    and metadata are emitted so CCC/PHT losses can operate on the actual
    training mini-panel.

    Loss weights per gene are compressed: ``sqrt(variance) / median_stdev``,
    clamped to ``[gene_weight_min, gene_weight_max]``.

    All training samples are preloaded into RAM at init time to eliminate
    Zarr chunk-decompression overhead on the hot path.  TSS_signals are
    stored as float16 and TSS_nucleotides as uint8 to fit within typical
    HPC node memory budgets (~114 GB for 180 samples × 18K genes).
    DataLoader workers share the preloaded arrays via fork copy-on-write.

    An optional ``val_mask`` (2-D bool array, shape [n_samples, n_union_genes])
    can be provided for full-training holdout mode: marked pairs are skipped
    during training iteration (drawn samples that hit a held-out pair are
    replaced by the next available sample from the same source).
    """

    def __init__(
        self,
        sample_paths_by_source: Dict[str, List[str]],
        gene_variance: Dict[str, float],
        n_eff_per_source: int = 24,
        genes_per_batch: int = 2,
        batch_size: int = 64,
        steps_per_epoch: int = 4000,
        gene_weight_min: float = 0.5,
        gene_weight_max: float = 2.0,
        seed: int = 42,
        labeled: bool = True,
        sample_name_map: Optional[Dict[str, str]] = None,
        include_metadata: bool = False,
        subsample_frac: Optional[float] = None,
        val_mask: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.sample_paths_by_source = dict(sample_paths_by_source)
        self.source_names = sorted(self.sample_paths_by_source.keys())
        self.n_sources = len(self.source_names)
        self.n_eff_per_source = n_eff_per_source
        self.genes_per_batch = genes_per_batch
        self.batch_size = batch_size
        self.steps_per_epoch = int(steps_per_epoch)
        self.gene_weight_min = gene_weight_min
        self.gene_weight_max = gene_weight_max
        self.seed = seed
        self.labeled = labeled
        self.sample_name_map = sample_name_map
        self.include_metadata = include_metadata
        self._epoch_counter = 0
        self.val_mask = val_mask  # shape [n_samples, n_union_genes] or None

        # Temporary narrower multi-source panel default: 2 genes × 32 rows/gene
        # = 64 rows total, which preserves the prior batch size while halving
        # the number of genes represented in each stochastic batch.

        # Build gene universe: union of all sources' gene sets.
        all_paths = [p for paths in self.sample_paths_by_source.values()
                     for p in paths]
        if not all_paths:
            raise ValueError("No sample paths provided to "
                             "ProteusMultiSourceBatchDataset")
        if self.genes_per_batch <= 0:
            raise ValueError('genes_per_batch must be > 0')
        if self.batch_size <= 0:
            raise ValueError('batch_size must be > 0')
        if self.batch_size % self.genes_per_batch != 0:
            raise ValueError('batch_size must be divisible by genes_per_batch')
        if self.steps_per_epoch <= 0:
            raise ValueError('steps_per_epoch must be > 0')

        # Discover per-source gene sets.  Samples within a source can have
        # slightly different gene panels (genes with >50% NaN features are
        # dropped per-sample at Zarr compilation time), so we union across
        # all samples in each source.
        source_gene_sets: Dict[str, set] = {}
        for src in self.source_names:
            genes: set = set()
            for p in self.sample_paths_by_source[src]:
                store = zarr.open(p, mode='r')
                genes.update(str(g) for g in store['gene_names'][:])
            source_gene_sets[src] = genes

        union_genes = sorted(set.union(*source_gene_sets.values()))
        self._union_genes: List[str] = union_genes
        n_genes = len(union_genes)
        gene_to_idx = {g: i for i, g in enumerate(union_genes)}

        # Per-source gene availability mask.
        self._source_gene_mask: Dict[str, np.ndarray] = {}
        for src in self.source_names:
            mask = np.zeros(n_genes, dtype=np.bool_)
            for g in source_gene_sets[src]:
                mask[gene_to_idx[g]] = True
            self._source_gene_mask[src] = mask

        # Ordered list of all unique sample paths (for val_mask indexing).
        self._all_sample_paths: List[str] = sorted(set(all_paths))
        self._sample_path_to_idx: Dict[str, int] = {
            p: i for i, p in enumerate(self._all_sample_paths)
        }

        # Preload all samples into RAM, reindexed to union gene order.
        # Genes missing from a source get zeros (never accessed at runtime
        # because source_gene_mask prevents drawing them).
        self._preloaded: Dict[str, Dict[str, np.ndarray]] = {}
        self._preload_all_samples(all_paths, source_gene_sets, gene_to_idx)
        self._build_sampling_groups()

        # Per-gene loss weights: sqrt(variance) / median(stdev), clamped.
        variance_arr = np.array(
            [gene_variance.get(g, 0.0) for g in union_genes],
            dtype=np.float64,
        )
        variance_arr = np.maximum(variance_arr, 0.0)
        stdevs = np.sqrt(variance_arr)
        positive_stdevs = stdevs[stdevs > 0]
        median_std = float(np.median(positive_stdevs)) if len(positive_stdevs) > 0 else 1.0
        if median_std <= 0:
            median_std = 1.0
        raw_weights = stdevs / median_std
        self._gene_loss_weights: np.ndarray = np.clip(
            raw_weights, gene_weight_min, gene_weight_max,
        ).astype(np.float32)

        positive_variances = variance_arr[variance_arr > 0]
        variance_floor = (
            0.05 * float(np.median(positive_variances))
            if positive_variances.size > 0 else 1.0
        )
        sampling_weights = variance_arr + variance_floor
        self._gene_sampling_probs = (
            sampling_weights / sampling_weights.sum()
        ).astype(np.float64)

        items_per_gene = self.batch_size // self.genes_per_batch
        self._n_batches = self.steps_per_epoch

        if subsample_frac is not None and 0.0 < subsample_frac < 1.0:
            self._n_batches = max(1, int(self._n_batches * subsample_frac))

        print(f"  [MultiSource] {n_genes} union genes, "
              f"{self.n_sources} sources, "
              f"batch={self.batch_size} ({self.genes_per_batch}g × "
              f"{items_per_gene}sps), "
              f"{self._n_batches} batches/epoch")

    def __len__(self) -> int:
        return self._n_batches

    # -- RAM preloading --

    def _preload_all_samples(
        self,
        all_paths: List[str],
        source_gene_sets: Dict[str, set],
        gene_to_idx: Dict[str, int],
    ) -> None:
        """Preload all training samples' arrays into RAM, reindexed to the
        union gene order.

        Genes that a source does not have are filled with zeros in the
        preloaded arrays — they are never accessed at runtime because
        ``_source_gene_mask`` prevents drawing those (source, gene) pairs.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        n_union = len(self._union_genes)

        # Invert: zarr_path → source name (for determining which genes exist).
        path_to_source: Dict[str, str] = {}
        for src, paths in self.sample_paths_by_source.items():
            for p in paths:
                path_to_source[p] = src

        def _load_one(zarr_path: str) -> tuple:
            store = zarr.open(zarr_path, mode='r')
            gene_names = [str(g) for g in store['gene_names'][:]]
            n_local = len(gene_names)
            flat = store['flat']

            # Build reindex map: for each local gene, where does it go
            # in the union array.
            local_to_union = np.array(
                [gene_to_idx[g] for g in gene_names], dtype=np.intp,
            )

            # Read full local arrays, then scatter into union-sized arrays.
            local_signals = flat['TSS_signals'][:].astype(np.float16)
            local_nt = flat['TSS_nucleotides'][:]  # uint8
            local_tss = flat['TSS_features'][:].astype(np.float32)
            local_gb = flat['GB_features'][:].astype(np.float32)
            local_ref = flat['ref_scalars'][:].astype(np.float32)
            local_tumor = flat['tumor_exp'][:].astype(np.float32)
            local_bg = flat['background_exp'][:].astype(np.float32)
            local_tfx = flat['TFX_label'][:].astype(np.float32)

            # Allocate union-sized arrays (zeros for missing genes).
            sig_shape = (n_union,) + local_signals.shape[1:]
            signals = np.zeros(sig_shape, dtype=np.float16)
            nt_shape = (n_union,) + local_nt.shape[1:]
            nucleotides = np.zeros(nt_shape, dtype=local_nt.dtype)
            tss_shape = (n_union,) + local_tss.shape[1:]
            tss_features = np.zeros(tss_shape, dtype=np.float32)
            gb_shape = (n_union,) + local_gb.shape[1:]
            gb_features = np.zeros(gb_shape, dtype=np.float32)
            ref_shape = (n_union,) + local_ref.shape[1:]
            ref_scalars = np.zeros(ref_shape, dtype=np.float32)
            tumor_exp = np.zeros(n_union, dtype=np.float32)
            bg_exp = np.zeros(n_union, dtype=np.float32)
            tfx_label = np.zeros(n_union, dtype=np.float32)

            signals[local_to_union] = local_signals
            nucleotides[local_to_union] = local_nt
            tss_features[local_to_union] = local_tss
            gb_features[local_to_union] = local_gb
            ref_scalars[local_to_union] = local_ref
            tumor_exp[local_to_union] = local_tumor
            bg_exp[local_to_union] = local_bg
            tfx_label[local_to_union] = local_tfx

            # Resolve sample display name.
            if isinstance(self.sample_name_map, dict) and zarr_path in self.sample_name_map:
                sample_display = self.sample_name_map[zarr_path]
            else:
                name = store.attrs.get('sample_name')
                if name is None:
                    name = Path(zarr_path).stem.replace('.zarr', '')
                sample_display = str(name)

            available_mask = np.zeros(n_union, dtype=np.bool_)
            available_mask[local_to_union] = True

            return zarr_path, {
                'TSS_signals': signals,
                'TSS_nucleotides': nucleotides,
                'TSS_features': tss_features,
                'GB_features': gb_features,
                'ref_scalars': ref_scalars,
                'tumor_exp': tumor_exp,
                'background_exp': bg_exp,
                'TFX_label': tfx_label,
                'available_mask': available_mask,
            }, sample_display

        unique_paths = sorted(set(all_paths))
        n_samples = len(unique_paths)

        # Estimate memory (conservative upper bound).
        per_sample_mb = (
            n_union * 2000 * 9 * 2     # TSS_signals float16
            + n_union * 2000 * 4 * 1   # TSS_nucleotides uint8
            + n_union * (11 + 11 + 3 + 1 + 1 + 1) * 4  # small float32
        ) / (1024 ** 2)
        total_gb = per_sample_mb * n_samples / 1024
        print(f"  [MultiSource] Preloading {n_samples} samples "
              f"({n_union} union genes) into RAM — "
              f"estimated {total_gb:.1f} GB...")

        self._sample_display_names: Dict[str, str] = {}
        self._sample_source_names: Dict[str, str] = {}
        self._sample_is_pht: Dict[str, bool] = {}
        self._sample_pht_parent: Dict[str, str] = {}
        self._sample_pht_donor: Dict[str, str] = {}

        max_threads = min(4, n_samples)
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(_load_one, p): p for p in unique_paths}
            done = 0
            for future in as_completed(futures):
                zarr_path, arrays, display_name = future.result()
                self._preloaded[zarr_path] = arrays
                self._sample_display_names[zarr_path] = display_name
                self._sample_source_names[zarr_path] = path_to_source[zarr_path]
                is_pht, parent_id, donor_id = _parse_pht_metadata(display_name)
                self._sample_is_pht[zarr_path] = is_pht
                self._sample_pht_parent[zarr_path] = parent_id
                self._sample_pht_donor[zarr_path] = donor_id
                done += 1
                if done % 20 == 0 or done == n_samples:
                    print(f"    ... loaded {done}/{n_samples} samples")

        print(f"  [MultiSource] Preload complete ({total_gb:.1f} GB in RAM)")

    def _get_sample_name(self, zarr_path: str) -> str:
        return self._sample_display_names.get(
            zarr_path,
            Path(zarr_path).stem.replace('.zarr', ''),
        )

    def _build_sampling_groups(self) -> None:
        self._source_real_paths: Dict[str, List[str]] = {src: [] for src in self.source_names}
        self._source_pht_paths: Dict[str, List[str]] = {src: [] for src in self.source_names}
        self._source_pht_by_parent: Dict[str, Dict[str, List[str]]] = {src: {} for src in self.source_names}
        self._source_pht_by_donor: Dict[str, Dict[str, List[str]]] = {src: {} for src in self.source_names}

        for src in self.source_names:
            for path in self.sample_paths_by_source[src]:
                if self._sample_is_pht.get(path, False):
                    self._source_pht_paths[src].append(path)
                    parent = self._sample_pht_parent.get(path, '')
                    donor = self._sample_pht_donor.get(path, '')
                    if parent:
                        self._source_pht_by_parent[src].setdefault(parent, []).append(path)
                    if donor:
                        self._source_pht_by_donor[src].setdefault(donor, []).append(path)
                else:
                    self._source_real_paths[src].append(path)

        for src in self.source_names:
            print(
                f"  [MultiSource] source '{src}': "
                f"{len(self._source_real_paths[src])} real + "
                f"{len(self._source_pht_paths[src])} PHT samples"
            )

    def _path_has_gene(self, zarr_path: str, gene_idx: int) -> bool:
        return bool(self._preloaded[zarr_path]['available_mask'][gene_idx])

    def _passes_val_mask(self, zarr_path: str, gene_idx: int) -> bool:
        if self.val_mask is None:
            return True
        sample_idx = self._sample_path_to_idx[zarr_path]
        return not bool(self.val_mask[sample_idx, gene_idx])

    def _valid_paths_for_gene(self, paths: List[str], gene_idx: int) -> List[str]:
        return [
            path for path in paths
            if self._path_has_gene(path, gene_idx) and self._passes_val_mask(path, gene_idx)
        ]

    def _sample_unique_paths(
        self,
        candidates: List[str],
        n_draws: int,
        rng: random.Random,
    ) -> List[str]:
        if not candidates or n_draws <= 0:
            return []
        if len(candidates) <= n_draws:
            drawn = list(candidates)
            rng.shuffle(drawn)
            return drawn
        return rng.sample(candidates, n_draws)

    def _synthetic_origin_key(self, zarr_path: str) -> str:
        if self._sample_is_pht.get(zarr_path, False):
            parent_id = self._sample_pht_parent.get(zarr_path, '')
            if parent_id:
                return f'origin:{parent_id}'
        return f"origin:{Path(zarr_path).stem.replace('.zarr', '')}"

    def _draw_synthetic_paths(self, src: str, gene_idx: int, n_draws: int, rng: random.Random) -> List[str]:
        if n_draws <= 0:
            return []
        combined_candidates = self._valid_paths_for_gene(self.sample_paths_by_source[src], gene_idx)
        if not combined_candidates:
            return []

        paths_by_origin: Dict[str, List[str]] = {}
        for path in combined_candidates:
            origin_key = self._synthetic_origin_key(path)
            paths_by_origin.setdefault(origin_key, []).append(path)

        chosen_origins = self._sample_unique_paths(list(paths_by_origin.keys()), n_draws, rng)
        drawn: List[str] = []
        for origin_key in chosen_origins:
            origin_candidates = paths_by_origin[origin_key]
            drawn.append(rng.choice(origin_candidates))
        rng.shuffle(drawn)
        return drawn

    def _sample_gene_indices(
        self,
        gene_population: List[int],
        gene_sampling_probs: List[float],
        rng: random.Random,
    ) -> List[int]:
        if not gene_population or self.genes_per_batch <= 0:
            return []
        if len(gene_population) <= self.genes_per_batch:
            drawn = list(gene_population)
            rng.shuffle(drawn)
            return drawn

        # Original implementation kept for reference:
        # return rng.choices(
        #     gene_population,
        #     weights=gene_sampling_probs,
        #     k=self.genes_per_batch,
        # )
        # Sample genes without replacement so a batch never contains the same
        # gene twice, which would otherwise reopen same-sample/origin repeats
        # inside a single CCC group.
        np_rng = np.random.default_rng(rng.randrange(2**32))
        drawn = np_rng.choice(
            np.asarray(gene_population, dtype=np.int64),
            size=self.genes_per_batch,
            replace=False,
            p=np.asarray(gene_sampling_probs, dtype=np.float64),
        )
        return [int(idx) for idx in drawn.tolist()]

    def _draw_real_paths(self, src: str, gene_idx: int, n_draws: int, rng: random.Random) -> List[str]:
        candidates = self._valid_paths_for_gene(self._source_real_paths[src], gene_idx)
        # Original implementation kept for reference:
        # if not candidates or n_draws <= 0:
        #     return []
        # return [rng.choice(candidates) for _ in range(n_draws)]
        # Draw real samples without replacement so a gene group never repeats
        # the same biological sample within one batch.
        return self._sample_unique_paths(candidates, n_draws, rng)

    def _draw_pht_paths(self, src: str, gene_idx: int, n_draws: int, rng: random.Random) -> List[str]:
        # Original implementation kept for reference:
        # if n_draws <= 0:
        #     return []
        # grouped = self._source_pht_by_parent[src] if rng.random() < 0.5 else self._source_pht_by_donor[src]
        # valid_groups = []
        # for group_paths in grouped.values():
        #     valid = self._valid_paths_for_gene(group_paths, gene_idx)
        #     if len(valid) >= 2:
        #         valid_groups.append(valid)
        # if valid_groups:
        #     chosen_group = rng.choice(valid_groups)
        #     if len(chosen_group) >= n_draws:
        #         return rng.sample(chosen_group, n_draws)
        #     return chosen_group + [rng.choice(chosen_group) for _ in range(n_draws - len(chosen_group))]
        # candidates = self._valid_paths_for_gene(self._source_pht_paths[src], gene_idx)
        # if not candidates:
        #     return []
        # return [rng.choice(candidates) for _ in range(n_draws)]
        # PHT-only draws now also respect without-replacement semantics.
        candidates = self._valid_paths_for_gene(self._source_pht_paths[src], gene_idx)
        return self._sample_unique_paths(candidates, n_draws, rng)

    def _draw_source_paths(self, src: str, gene_idx: int, n_draws: int, rng: random.Random) -> List[str]:
        if n_draws <= 0:
            return []
        pht_available = bool(self._valid_paths_for_gene(self._source_pht_paths[src], gene_idx))
        # Original implementation kept for reference:
        # real_available = bool(self._valid_paths_for_gene(self._source_real_paths[src], gene_idx))
        # if pht_available and real_available:
        #     n_pht = max(1, n_draws // 4)
        #     n_real = n_draws - n_pht
        # elif pht_available:
        #     n_pht = n_draws
        #     n_real = 0
        # else:
        #     n_pht = 0
        #     n_real = n_draws
        # drawn = self._draw_real_paths(src, gene_idx, n_real, rng)
        # drawn.extend(self._draw_pht_paths(src, gene_idx, n_pht, rng))
        # while len(drawn) < n_draws:
        #     filler = self._draw_real_paths(src, gene_idx, 1, rng) or self._draw_pht_paths(src, gene_idx, 1, rng)
        #     if not filler:
        #         break
        #     drawn.extend(filler)
        # rng.shuffle(drawn)
        # return drawn[:n_draws]
        # Mixed synthetic sources now draw from the combined PDX/PHT pool with
        # no explicit non-PHT preference.  We sample distinct model origins
        # without replacement, then choose one concrete path per origin.
        if pht_available:
            return self._draw_synthetic_paths(src, gene_idx, n_draws, rng)
        return self._draw_real_paths(src, gene_idx, n_draws, rng)

    # -- Core iteration --

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)

        self._epoch_counter += 1
        epoch_seed = self.seed + self._epoch_counter * 997 + worker_id * 31
        rng = random.Random(epoch_seed)

        n_genes = len(self._union_genes)
        gene_population = list(range(n_genes))
        gene_sampling_probs = self._gene_sampling_probs.tolist()

        # Divide batches across workers.
        total_batches = self._n_batches
        worker_batches = total_batches // num_workers
        if worker_id < (total_batches % num_workers):
            worker_batches += 1

        items_per_gene = self.batch_size // self.genes_per_batch

        for _batch_idx in range(worker_batches):
            batch_gene_indices = self._sample_gene_indices(
                gene_population,
                gene_sampling_probs,
                rng,
            )

            feat_signals = []
            feat_nt = []
            feat_tss = []
            feat_gb = []
            feat_ref = []
            lab_tumor = []
            lab_bg = []
            lab_weight = []
            lab_tfx = []
            meta_samples: List[str] = []
            meta_genes: List[str] = []
            meta_sources: List[str] = []
            meta_parent_ids: List[str] = []
            meta_donor_ids: List[str] = []
            meta_is_pht: List[float] = []

            for g_idx in batch_gene_indices:
                gene_name = self._union_genes[g_idx]
                gene_weight = float(self._gene_loss_weights[g_idx])

                # Determine which sources have this gene.
                available_sources = [
                    src for src in self.source_names
                    if self._source_gene_mask[src][g_idx]
                    and self._valid_paths_for_gene(self.sample_paths_by_source[src], g_idx)
                ]
                if not available_sources:
                    continue  # gene not in any source (shouldn't happen)

                # Distribute items_per_gene slots roughly equally.
                n_avail_src = len(available_sources)
                base_per_src = items_per_gene // n_avail_src
                remainder = items_per_gene % n_avail_src
                # Rotate remainder assignment to avoid always favoring
                # the same sources.
                src_counts = {}
                for si, src in enumerate(available_sources):
                    src_counts[src] = base_per_src + (1 if si < remainder else 0)

                for src, n_draws in src_counts.items():
                    for zarr_path in self._draw_source_paths(src, g_idx, n_draws, rng):
                        sd = self._preloaded[zarr_path]
                        is_pht = self._sample_is_pht.get(zarr_path, False)

                        feat_signals.append(
                            torch.from_numpy(
                                sd['TSS_signals'][g_idx].astype(np.float32)))
                        feat_nt.append(
                            torch.from_numpy(
                                sd['TSS_nucleotides'][g_idx]
                                .astype(np.float32)))
                        feat_tss.append(
                            torch.from_numpy(sd['TSS_features'][g_idx]))
                        feat_gb.append(
                            torch.from_numpy(sd['GB_features'][g_idx]))
                        feat_ref.append(
                            torch.from_numpy(sd['ref_scalars'][g_idx]))
                        lab_tfx.append(float(sd['TFX_label'][g_idx]))
                        lab_weight.append(gene_weight)

                        if self.labeled:
                            lab_tumor.append(float(sd['tumor_exp'][g_idx]))
                            lab_bg.append(
                                float(sd['background_exp'][g_idx]))

                        if self.include_metadata:
                            meta_samples.append(
                                self._get_sample_name(zarr_path))
                            meta_genes.append(gene_name)
                            meta_sources.append(src)
                            meta_parent_ids.append(self._sample_pht_parent.get(zarr_path, ''))
                            meta_donor_ids.append(self._sample_pht_donor.get(zarr_path, ''))
                            meta_is_pht.append(1.0 if is_pht else 0.0)

            # If val_mask filtering removed all items, skip this batch.
            if not feat_signals:
                continue

            # Stack into batch tensors.
            features: Dict[str, Any] = {
                'TSS_signals': torch.stack(feat_signals),
                'TSS_nucleotides': torch.stack(feat_nt),
                'TSS_features': torch.stack(feat_tss),
                'GB_features': torch.stack(feat_gb),
                'ref_scalars': torch.stack(feat_ref),
            }
            if self.include_metadata:
                features['sample_name'] = meta_samples
                features['gene_name'] = meta_genes
                features['source_name'] = meta_sources
                features['pht_parent_id'] = meta_parent_ids
                features['pht_donor_id'] = meta_donor_ids
                features['is_pht'] = torch.tensor(meta_is_pht, dtype=torch.float32).unsqueeze(-1)

            w_t = torch.tensor(lab_weight, dtype=torch.float32)
            t_t = torch.tensor(lab_tfx, dtype=torch.float32)

            if self.labeled:
                labels: Dict[str, torch.Tensor] = {
                    'tumor_exp': torch.tensor(
                        lab_tumor, dtype=torch.float32).unsqueeze(-1),
                    'background_exp': torch.tensor(
                        lab_bg, dtype=torch.float32).unsqueeze(-1),
                    'sample_weight': w_t.unsqueeze(-1),
                    'TFX_label': t_t.unsqueeze(-1),
                }
            else:
                labels = {
                    'sample_weight': w_t.unsqueeze(-1),
                    'TFX_label': t_t.unsqueeze(-1),
                }

            yield features, labels


class ProteusHoldoutValDataset(IterableDataset):
    """Validation dataset for full-training holdout mode.

    Iterates over all held-out (sample, gene) pairs from a preloaded
    ``ProteusMultiSourceBatchDataset``, yielding fixed-size batches of 32
    with sample/gene metadata for per-cohort/source metrics.

    Shares the preloaded arrays from the training dataset (no extra RAM).
    """

    def __init__(
        self,
        train_dataset: 'ProteusMultiSourceBatchDataset',
        val_mask: np.ndarray,
        batch_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()
        self._train_ds = train_dataset
        self.val_mask = val_mask
        self.batch_size = batch_size
        self.seed = seed

        # Pre-compute all (sample_idx, gene_idx) val pairs.
        sample_indices, gene_indices = np.where(val_mask)
        self._val_pairs = list(zip(sample_indices.tolist(), gene_indices.tolist()))
        self._n_batches = max(1, len(self._val_pairs) // self.batch_size)

        print(f"  [HoldoutVal] {len(self._val_pairs)} held-out pairs, "
              f"{self._n_batches} batches of {self.batch_size}")

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)

        ds = self._train_ds
        all_paths = ds._all_sample_paths
        union_genes = ds._union_genes
        preloaded = ds._preloaded

        # Deterministic shuffle of val pairs.
        rng = random.Random(self.seed + 555)
        pairs = list(self._val_pairs)
        rng.shuffle(pairs)

        # Shard across workers.
        worker_pairs = pairs[worker_id::num_workers]

        # Yield fixed-size batches.
        buf_signals = []
        buf_nt = []
        buf_tss = []
        buf_gb = []
        buf_ref = []
        buf_tumor = []
        buf_bg = []
        buf_weight = []
        buf_tfx = []
        buf_samples: List[str] = []
        buf_genes: List[str] = []
        buf_sources: List[str] = []
        buf_parents: List[str] = []
        buf_donors: List[str] = []
        buf_is_pht: List[float] = []

        for s_idx, g_idx in worker_pairs:
            zarr_path = all_paths[s_idx]
            sd = preloaded[zarr_path]
            gene_name = union_genes[g_idx]
            gene_weight = float(ds._gene_loss_weights[g_idx])

            buf_signals.append(torch.from_numpy(
                sd['TSS_signals'][g_idx].astype(np.float32)))
            buf_nt.append(torch.from_numpy(
                sd['TSS_nucleotides'][g_idx].astype(np.float32)))
            buf_tss.append(torch.from_numpy(sd['TSS_features'][g_idx]))
            buf_gb.append(torch.from_numpy(sd['GB_features'][g_idx]))
            buf_ref.append(torch.from_numpy(sd['ref_scalars'][g_idx]))
            buf_tfx.append(float(sd['TFX_label'][g_idx]))
            buf_weight.append(gene_weight)
            is_pht = ds._sample_is_pht.get(zarr_path, False)
            buf_tumor.append(float(sd['tumor_exp'][g_idx]))
            buf_bg.append(float(sd['background_exp'][g_idx]))
            buf_samples.append(ds._get_sample_name(zarr_path))
            buf_genes.append(gene_name)
            buf_sources.append(ds._sample_source_names.get(zarr_path, ''))
            buf_parents.append(ds._sample_pht_parent.get(zarr_path, ''))
            buf_donors.append(ds._sample_pht_donor.get(zarr_path, ''))
            buf_is_pht.append(1.0 if is_pht else 0.0)

            if len(buf_signals) == self.batch_size:
                features: Dict[str, Any] = {
                    'TSS_signals': torch.stack(buf_signals),
                    'TSS_nucleotides': torch.stack(buf_nt),
                    'TSS_features': torch.stack(buf_tss),
                    'GB_features': torch.stack(buf_gb),
                    'ref_scalars': torch.stack(buf_ref),
                    'sample_name': list(buf_samples),
                    'gene_name': list(buf_genes),
                    'source_name': list(buf_sources),
                    'pht_parent_id': list(buf_parents),
                    'pht_donor_id': list(buf_donors),
                    'is_pht': torch.tensor(buf_is_pht, dtype=torch.float32).unsqueeze(-1),
                }
                labels: Dict[str, torch.Tensor] = {
                    'tumor_exp': torch.tensor(
                        buf_tumor, dtype=torch.float32).unsqueeze(-1),
                    'background_exp': torch.tensor(
                        buf_bg, dtype=torch.float32).unsqueeze(-1),
                    'sample_weight': torch.tensor(
                        buf_weight, dtype=torch.float32).unsqueeze(-1),
                    'TFX_label': torch.tensor(
                        buf_tfx, dtype=torch.float32).unsqueeze(-1),
                }
                yield features, labels

                buf_signals.clear()
                buf_nt.clear()
                buf_tss.clear()
                buf_gb.clear()
                buf_ref.clear()
                buf_tumor.clear()
                buf_bg.clear()
                buf_weight.clear()
                buf_tfx.clear()
                buf_samples.clear()
                buf_genes.clear()
                buf_sources.clear()
                buf_parents.clear()
                buf_donors.clear()
                buf_is_pht.clear()


def collate_inference(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for inference (no labels)."""
    return {
        'TSS_signals': torch.stack([b['TSS_signals'] for b in batch]),
        'TSS_nucleotides': torch.stack([b['TSS_nucleotides'] for b in batch]),
        'TSS_features': torch.stack([b['TSS_features'] for b in batch]),
        'GB_features': torch.stack([b['GB_features'] for b in batch]),
        'ref_scalars': torch.stack([b['ref_scalars'] for b in batch]),
    }


# ==============================================================================
# Splits TSV Parsing
# ==============================================================================

def _parse_splits_tsv(
    training_set_tsv: str,
    zarr_base: str,
    val_fold: int,
    train_only_cohorts: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, str]]:
    """Parse a Cohort/Sample/Fold TSV and partition into train/val Zarr paths.

    Args:
        training_set_tsv: Path to TSV with Cohort/Sample/Fold columns.
        zarr_base: Base directory for Zarr stores.
        val_fold: Which fold to hold out for validation.
        train_only_cohorts: Cohorts always routed to training.

    Returns:
        (train_paths, val_paths, zarr_path_to_cohort, zarr_path_to_sample_name)

    Raises:
        FileNotFoundError: If TSV or any Zarr store is missing.
        ValueError: If TSV format is invalid or partitions are empty.
    """
    import csv

    if not os.path.exists(training_set_tsv):
        raise FileNotFoundError(f"Training set TSV not found: {training_set_tsv}")

    train_paths: List[str] = []
    val_paths: List[str] = []
    missing_samples: List[str] = []
    zarr_path_to_cohort: Dict[str, str] = {}
    zarr_path_to_sample_name: Dict[str, str] = {}
    _train_only = set(train_only_cohorts or [])

    with open(training_set_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        required = {'Cohort', 'Sample', 'Fold'}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"TSV must have columns: {sorted(required)}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            cohort = row['Cohort'].strip()
            sample = row['Sample'].strip()
            fold = int(row['Fold'])

            zpath = zarr_path_for_sample(sample, cohort, zarr_base)
            if not os.path.isdir(zpath):
                bare = zarr_name_for_sample(sample, cohort)
                missing_samples.append(
                    f"  {cohort}/zarr_files/{bare}.zarr  "
                    f"(Fold={fold}, TSV sample={sample})")
                continue

            zarr_path_to_cohort[zpath] = cohort
            zarr_path_to_sample_name[zpath] = sample

            if cohort in _train_only:
                train_paths.append(zpath)
            elif fold == val_fold:
                val_paths.append(zpath)
            else:
                train_paths.append(zpath)

    if missing_samples:
        raise FileNotFoundError(
            f"Missing {len(missing_samples)} Zarr store(s) under {zarr_base}:\n"
            + "\n".join(missing_samples))
    if not train_paths:
        raise ValueError(f"No training samples found (all in fold {val_fold}?)")
    if not val_paths:
        raise ValueError(f"No validation samples found for fold {val_fold}")

    return train_paths, val_paths, zarr_path_to_cohort, zarr_path_to_sample_name


def _print_multisource_split_audit(
    training_set_tsv: str,
    val_fold: int,
    train_paths: List[str],
    val_paths: List[str],
    zarr_path_to_sample_name: Dict[str, str],
) -> None:
    import csv
    from collections import Counter

    fold_counts: Counter[int] = Counter()
    sample_to_fold: Dict[str, int] = {}

    with open(training_set_tsv, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            sample = row['Sample'].strip()
            fold = int(row['Fold'])
            fold_counts[fold] += 1
            sample_to_fold[sample] = fold

    train_samples = {
        zarr_path_to_sample_name[zpath]
        for zpath in train_paths
        if zpath in zarr_path_to_sample_name
    }
    val_samples = {
        zarr_path_to_sample_name[zpath]
        for zpath in val_paths
        if zpath in zarr_path_to_sample_name
    }
    train_fold_counts = Counter(
        sample_to_fold[sample]
        for sample in train_samples
        if sample in sample_to_fold
    )
    val_fold_counts = Counter(
        sample_to_fold[sample]
        for sample in val_samples
        if sample in sample_to_fold
    )

    def _fmt_counts(counts: Counter[int]) -> str:
        return ', '.join(
            f'{fold}:{counts[fold]}' for fold in sorted(counts)
        )

    print(
        f"  [SplitAudit] held-out fold {val_fold}: "
        f"train={len(train_paths)} val={len(val_paths)}"
    )
    print(f"  [SplitAudit] TSV fold counts: {_fmt_counts(fold_counts)}")
    if 2 in fold_counts:
        print(
            "  [SplitAudit] Fold 2 membership in this run: "
            f"train={train_fold_counts.get(2, 0)} "
            f"val={val_fold_counts.get(2, 0)}"
        )


def create_data_loader(
    index_file: str,
    num_workers: int = 4,
    labeled: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    seed: int = 42,
    include_metadata: bool = False,
    subsample_frac: Optional[float] = None,
    gene_weights: Optional[Dict[str, float]] = None,
    val_frac: Optional[float] = None,
    val_split: bool = False,
    cohort_map: Optional[Dict[str, str]] = None,
    balance_cohorts: Optional[List[str]] = None,
    source_map: Optional[Dict[str, str]] = None,
    target_n_per_source: Optional[int] = None,
    sample_name_map: Optional[Dict[str, str]] = None,
) -> Tuple[DataLoader, int]:
    """Create a chunk-based DataLoader for Proteus training/validation.

    Uses fixed invariants: chunk_size=256, genes_per_batch=32.

    Args:
        index_file: Path to fold index JSON (list of Zarr paths).
        num_workers: Number of data loading workers.
        labeled: Whether dataset has labels.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Number of batches to prefetch per worker.
        seed: Random seed for reproducibility.
        include_metadata: Include sample/gene metadata in each batch.
        subsample_frac: Optional fraction (0.0-1.0) of genes for fast sanity checks.
        gene_weights: Per-gene variance weights for loss scaling.
        val_frac: Fraction of chunks to reserve for validation.
        val_split: If True, yield only val chunks; if False, yield only train chunks.
        cohort_map: zarr_path → cohort name (for cohort balancing).
        balance_cohorts: Cohort names to balance by undersampling.
        source_map: zarr_path → source group name (for source balancing).
        target_n_per_source: Target samples per source per epoch.
        sample_name_map: zarr_path → display name (TSV sample name).

    Returns:
        (DataLoader, n_batches)
    """
    dataset = ProteusGeneChunkIterableDataset(
        index_file=index_file,
        seed=seed,
        labeled=labeled,
        cache_stores=True,
        include_metadata=include_metadata,
        subsample_frac=subsample_frac,
        gene_weights=gene_weights,
        val_frac=val_frac,
        val_split=val_split,
        cohort_map=cohort_map,
        balance_cohorts=balance_cohorts,
        source_map=source_map,
        target_n_per_source=target_n_per_source,
        sample_name_map=sample_name_map,
    )

    loader = create_single_item_loader(
        dataset,
        num_workers=num_workers,
        seed=seed,
        collate_fn=_collate_identity,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return loader, len(dataset)


def create_inference_loader(
    zarr_path: str,
    num_workers: int = 2,
) -> Tuple[DataLoader, List[str]]:
    """
    Create a DataLoader for inference on a single sample.
    
    Args:
        zarr_path: Path to sample Zarr store
        num_workers: Number of workers
        
    Returns:
        DataLoader and list of gene names (for output ordering)
    """
    dataset = ProteusInferenceDataset(zarr_path)
    
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,  # Keep gene order for output
        num_workers=num_workers,
        collate_fn=collate_inference,
        pin_memory=True,
    )
    
    return loader, dataset.gene_names


def build_loaders_from_splits(
    training_set_tsv: str,
    zarr_base: str,
    val_fold: int,
    num_workers: int = 4,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    seed: int = 42,
    subsample_frac: Optional[float] = None,
    gene_weights: Optional[Dict[str, float]] = None,
    train_only_cohorts: Optional[List[str]] = None,
    balance_cohorts: Optional[List[str]] = None,
    source_groups: Optional[Dict[str, List[str]]] = None,
    target_n_per_source: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Build train/val DataLoaders from a deterministic splits TSV file.

    The TSV must have columns: Cohort, Sample, Fold (tab-separated, with header).
    Zarr paths are constructed as ``{zarr_base}/{cohort}/zarr_files/{sample}.zarr``.
    Training uses all folds except ``val_fold``.

    Args:
        training_set_tsv: Path to TSV with Cohort/Sample/Fold columns.
        zarr_base: Base directory containing cohort subdirs with sample.zarr stores.
        val_fold: Which fold to hold out for validation.
        num_workers: DataLoader workers.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Batches to prefetch per worker.
        seed: Random seed for reproducibility.
        subsample_frac: Optional fraction (0.0-1.0) of genes to use per epoch.
        gene_weights: Per-gene variance weights for loss scaling.
        train_only_cohorts: Cohorts that always go to training (never
                            validation), regardless of fold assignment.
        balance_cohorts: Cohorts to balance by per-epoch undersampling.
        source_groups: Source grouping for validation/evaluation metrics.
                   When ``batch_mode='chunk'`` this no longer changes
                   training sample exposure.
        target_n_per_source: Legacy source-balance control for the dedicated
                   ``multi_source`` loader path only.

    Returns:
        (train_loader, val_loader, train_steps, val_steps)

    Raises:
        FileNotFoundError: If training_set_tsv or any Zarr store doesn't exist.
        ValueError: If TSV format is invalid.
    """
    train_paths, val_paths, zarr_path_to_cohort, zarr_path_to_sample_name = (
        _parse_splits_tsv(training_set_tsv, zarr_base, val_fold, train_only_cohorts)
    )

    # Shuffle training paths deterministically
    rng = np.random.RandomState(seed)
    rng.shuffle(train_paths)

    # Build source assignments for validation / reporting.
    zarr_path_to_source: Optional[Dict[str, str]] = None
    if source_groups:
        _cohort_to_source = invert_source_groups(source_groups)
        _pdx_names = collect_pdx_names(training_set_tsv, _cohort_to_source)

        zarr_path_to_source = {}
        for zpath, coh in zarr_path_to_cohort.items():
            if zpath not in train_paths:
                continue
            basename = os.path.basename(zpath).replace('.zarr', '')
            source = assign_source(coh, basename, _cohort_to_source,
                                   _pdx_names)
            if source is not None:
                zarr_path_to_source[zpath] = source

        from collections import Counter
        src_counts = Counter(zarr_path_to_source.values())
        print(f"  [Source groups] {dict(src_counts)}, "
              f"target_n_per_source={target_n_per_source}")

    index_paths = _write_temp_index_files(
        'proteus_splits_',
        {
            f'train_excl_fold{val_fold}': train_paths,
            f'val_fold{val_fold}': val_paths,
        },
    )
    train_index = index_paths[f'train_excl_fold{val_fold}']
    val_index = index_paths[f'val_fold{val_fold}']

    # Log subsampling info if active
    if subsample_frac is not None and subsample_frac < 1.0:
        print(f"  [Subsample] Using ~{subsample_frac*100:.0f}% of genes for fast sanity check")

    train_loader, train_steps = create_data_loader(
        index_file=train_index,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        subsample_frac=subsample_frac,
        gene_weights=gene_weights,
        cohort_map=zarr_path_to_cohort if balance_cohorts else None,
        balance_cohorts=balance_cohorts,
    )

    val_loader, val_steps = create_data_loader(
        index_file=val_index,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        include_metadata=True,
        subsample_frac=subsample_frac,
        gene_weights=gene_weights,
        sample_name_map=zarr_path_to_sample_name,
    )

    return train_loader, val_loader, train_steps, val_steps


def build_multisource_loaders_from_splits(
    training_set_tsv: str,
    zarr_base: str,
    val_fold: int,
    source_groups: Dict[str, List[str]],
    gene_variance: Dict[str, float],
    n_eff_per_source: int = 24,
    genes_per_batch: int = 2,
    batch_size: int = 64,
    steps_per_epoch: int = 4000,
    gene_weight_min: float = 0.5,
    gene_weight_max: float = 2.0,
    num_workers: int = 4,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    seed: int = 42,
    train_only_cohorts: Optional[List[str]] = None,
    gene_weights: Optional[Dict[str, float]] = None,
    subsample_frac: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """Build multi-source train loader + chunk-based val loader from splits TSV.

    Training uses :class:`ProteusMultiSourceBatchDataset` (gene-conditioned
    batches drawn from all source groups simultaneously).  Validation uses the
    standard chunk-based exhaustive loader so every gene × sample is evaluated.

    Args:
        training_set_tsv: Path to TSV with Cohort/Sample/Fold columns.
        zarr_base: Base directory for Zarr stores.
        val_fold: Fold number held out for validation.
        source_groups: Mapping of source name → list of cohort names.
        gene_variance: Per-gene raw variance (for loss weights).
        n_eff_per_source: Retained for caller compatibility; fixed stochastic
            epochs are controlled by ``steps_per_epoch``.
        genes_per_batch: Unique genes drawn per batch (default 2).
        batch_size: Total items per batch (default 64).
        steps_per_epoch: Fixed stochastic batches per epoch (default 4000).
        gene_weight_min: Minimum compressed loss weight (default 0.5).
        gene_weight_max: Maximum compressed loss weight (default 2.0).
        num_workers: DataLoader workers.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Batches to prefetch per worker.
        seed: Random seed.
        train_only_cohorts: Cohorts always routed to training (e.g. ``['PHT']``).
        gene_weights: Clipped variance weights for the validation loader.
        subsample_frac: Optional epoch-length fraction (for fast sanity checks).

    Returns:
        (train_loader, val_loader, train_steps, val_steps)
    """
    train_paths, val_paths, zarr_path_to_cohort, zarr_path_to_sample_name = (
        _parse_splits_tsv(training_set_tsv, zarr_base, val_fold, train_only_cohorts)
    )
    _print_multisource_split_audit(
        training_set_tsv,
        val_fold,
        train_paths,
        val_paths,
        zarr_path_to_sample_name,
    )

    # -- Assign training paths to source groups (including PHT → parent PDX) --
    _cohort_to_source = invert_source_groups(source_groups)
    _pdx_names = collect_pdx_names(training_set_tsv, _cohort_to_source)

    sample_paths_by_source: Dict[str, List[str]] = {
        src: [] for src in source_groups
    }
    for zpath in train_paths:
        coh = zarr_path_to_cohort[zpath]
        basename = os.path.basename(zpath).replace('.zarr', '')
        source = assign_source(coh, basename, _cohort_to_source, _pdx_names)
        if source is not None:
            sample_paths_by_source[source].append(zpath)

    # Report source assignments.
    for src_name in sorted(sample_paths_by_source):
        print(f"  [MultiSource] source '{src_name}': "
              f"{len(sample_paths_by_source[src_name])} training samples")

    # -- Training loader (multi-source) --
    train_dataset = ProteusMultiSourceBatchDataset(
        sample_paths_by_source=sample_paths_by_source,
        gene_variance=gene_variance,
        n_eff_per_source=n_eff_per_source,
        genes_per_batch=genes_per_batch,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        gene_weight_min=gene_weight_min,
        gene_weight_max=gene_weight_max,
        seed=seed,
        labeled=True,
        sample_name_map=zarr_path_to_sample_name,
        include_metadata=True,
        subsample_frac=subsample_frac,
    )

    train_loader = create_single_item_loader(
        train_dataset,
        num_workers=num_workers,
        seed=seed,
        collate_fn=_collate_identity,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    train_steps = len(train_dataset)

    # -- Validation loader (chunk-based, exhaustive) --
    val_index = _write_temp_index_files(
        'proteus_msplits_',
        {f'val_fold{val_fold}': val_paths},
    )[f'val_fold{val_fold}']

    val_loader, val_steps = create_data_loader(
        index_file=val_index,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=seed,
        include_metadata=True,
        subsample_frac=subsample_frac,
        gene_weights=gene_weights,
        sample_name_map=zarr_path_to_sample_name,
    )

    return train_loader, val_loader, train_steps, val_steps


def extract_gene_depths(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Extract mean depth from TSS_features (index 10 = mean-region-depth).
    
    Note: mean-region-depth is log1p-transformed only (NOT z-scored) during
    Zarr preprocessing to preserve absolute depth information for scaling.
    
    Feature order (11 features):
    0: fl-mean, 1: fl-stdev, 2: fl-skew, 3: fl-kurtosis, 
    4: fl-subnucleosomal-ratio, 5: fl-entropy, 6: fl-gini-simpson,
    7: pn-compaction-score, 8: pn-mean-spacing, 9: pn-mean-amplitude,
    10: mean-region-depth (log1p only)
    """
    return features['TSS_features'][:, 10]


# ==============================================================================
# Gene Weight Computation (for MDN loss)
# ==============================================================================

def compute_gene_weights_from_zarr(
    zarr_paths: List[str],
    label_key: str = 'flat/tumor_exp',
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> Dict[str, float]:
    """Compute per-gene variance weights from Zarr stores (for loss weighting).

    For each gene, the weight is the **variance** of the label across the given
    samples, clipped to [clip_min, clip_max].  Genes with zero variance (e.g.
    always NaN or constant) receive weight = clip_min.

    Variance (rather than SD) is used so that the weight scales quadratically
    with expression spread.  This gives bimodal on/off genes (~25× weight)
    much stronger emphasis than near-constant genes (~1× weight), which better
    reflects their biological information content.

    Args:
        zarr_paths: List of paths to sample ``*.zarr`` stores.
        label_key: Zarr array path for the label vector (default ``'flat/tumor_exp'``).
        clip_min: Minimum gene weight (default 0.1).
        clip_max: Maximum gene weight (default 10.0).

    Returns:
        Dictionary mapping gene_name → clipped variance weight (float).
    """
    _, clipped = _compute_gene_variances(zarr_paths, label_key, clip_min, clip_max)
    return clipped


def compute_gene_variances_unclipped(
    zarr_paths: List[str],
    label_key: str = 'flat/tumor_exp',
) -> Dict[str, float]:
    """Compute per-gene **unclipped** variance from Zarr stores (for evaluation).

    Identical accumulation to :func:`compute_gene_weights_from_zarr` but
    returns the raw variance without any clipping, suitable for GVW-R²
    evaluation where clipping would distort the metric.

    Args:
        zarr_paths: List of paths to sample ``*.zarr`` stores.
        label_key: Zarr array path for the label vector (default ``'flat/tumor_exp'``).

    Returns:
        Dictionary mapping gene_name → raw variance (float, ≥ 0).
    """
    raw, _ = _compute_gene_variances(zarr_paths, label_key)
    return raw


def compute_gene_mean_std_from_zarr(
    zarr_paths: List[str],
    label_key: str = 'flat/tumor_exp',
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-gene mean and std from Zarr stores.

    Uses the same accumulation path as the variance helpers so SRL lookup
    tables are defined over the identical training-only truth pool used for
    the existing gene-variance weights. The SRL std is floored to the same
    minimum-variance convention used elsewhere so near-constant genes cannot
    explode standardized residual magnitudes.
    """
    import zarr

    gene_sum: Dict[str, float] = {}
    gene_sum_sq: Dict[str, float] = {}
    gene_count: Dict[str, int] = {}

    for zpath in zarr_paths:
        store = zarr.open(zpath, mode='r')
        gene_names = list(store['gene_names'][:])
        labels = store[label_key][:]

        for i, gname in enumerate(gene_names):
            g = str(gname)
            val = float(labels[i])
            if not np.isfinite(val):
                continue
            gene_sum[g] = gene_sum.get(g, 0.0) + val
            gene_sum_sq[g] = gene_sum_sq.get(g, 0.0) + val * val
            gene_count[g] = gene_count.get(g, 0) + 1

    gene_mean: Dict[str, float] = {}
    gene_std: Dict[str, float] = {}
    for g in gene_sum:
        n = gene_count[g]
        mean = gene_sum[g] / n
        var = max(0.0, gene_sum_sq[g] / n - mean * mean)
        gene_mean[g] = float(mean)
        gene_std[g] = float(max(np.sqrt(var), _SRL_STD_FLOOR))

    return gene_mean, gene_std


def _compute_gene_variances(
    zarr_paths: List[str],
    label_key: str = 'flat/tumor_exp',
    clip_min: float = 0.1,
    clip_max: float = 10.0,
) -> tuple:
    """Shared implementation: accumulate per-gene variance.

    Returns:
        (raw_dict, clipped_dict) — both map gene_name → float.
        raw_dict has unclipped variance; clipped_dict has variance clipped to
        [clip_min, clip_max].
    """
    import zarr

    # Accumulate per-gene sum and sum-of-squares for online variance computation.
    gene_sum: Dict[str, float] = {}
    gene_sum_sq: Dict[str, float] = {}
    gene_count: Dict[str, int] = {}

    for zpath in zarr_paths:
        store = zarr.open(zpath, mode='r')
        gene_names = list(store['gene_names'][:])
        labels = store[label_key][:]  # (n_genes,)

        for i, gname in enumerate(gene_names):
            g = str(gname)
            val = float(labels[i])
            if not np.isfinite(val):
                continue
            gene_sum[g] = gene_sum.get(g, 0.0) + val
            gene_sum_sq[g] = gene_sum_sq.get(g, 0.0) + val * val
            gene_count[g] = gene_count.get(g, 0) + 1

    # Compute variance per gene
    raw_weights: Dict[str, float] = {}
    clipped_weights: Dict[str, float] = {}
    for g in gene_sum:
        n = gene_count[g]
        if n < 2:
            raw_weights[g] = 0.0
            clipped_weights[g] = clip_min
            continue
        mean = gene_sum[g] / n
        var = max(0.0, gene_sum_sq[g] / n - mean * mean)
        raw_weights[g] = float(var)
        clipped_weights[g] = float(np.clip(var, clip_min, clip_max))

    return raw_weights, clipped_weights

