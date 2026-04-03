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
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import zarr

import bisect


def _normalize_fold_index_to_sample_paths(loaded: Any) -> List[str]:
    """Normalize a fold index JSON payload to a unique list of sample Zarr paths.

    Expects a flat list of absolute Zarr paths:
        ["/abs/path/sample1.zarr", "/abs/path/sample2.zarr", ...]

    Returns paths in first-seen order with duplicates removed.
    """
    if not isinstance(loaded, list):
        raise ValueError('Fold index JSON must be a list.')

    paths = [str(p) for p in loaded]

    seen: set[str] = set()
    uniq: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


class ProteusGeneDataset(Dataset):
    """Gene-level Dataset backed by flat per-sample Zarr arrays.

    This keeps the model's training semantics intact (each item = one gene),
    while still avoiding the on-disk per-gene hierarchy.

    Input index format:
      - ["/path/to/sampleA.zarr", "/path/to/sampleB.zarr", ...]
        -> dataset length is total genes across all samples.
    """

    def __init__(
        self,
        index_file: Optional[str] = None,
        zarr_paths: Optional[List[str]] = None,
        labeled: bool = True,
        cache_stores: bool = True,
        include_metadata: bool = False,
        gene_weights: Optional[Dict[str, float]] = None,
    ):
        self.labeled = labeled
        self.cache_stores = cache_stores
        self.include_metadata = bool(include_metadata)
        self.gene_weights = gene_weights
        self._store_cache: Dict[str, zarr.Group] = {}
        self._gene_names_cache: Dict[str, List[str]] = {}
        self._sample_name_cache: Dict[str, str] = {}

        if index_file is not None:
            with open(index_file, 'r') as f:
                loaded = json.load(f)
        elif zarr_paths is not None:
            loaded = list(zarr_paths)
        else:
            raise ValueError("Either index_file or zarr_paths must be provided")

        self._sample_paths: Optional[List[str]] = None
        self._cum_gene_offsets: Optional[List[int]] = None
        self._n_genes_per_sample: Optional[List[int]] = None

        self._sample_paths = [str(p) for p in loaded]
        if cache_stores:
            for path in set(self._sample_paths):
                self._store_cache[path] = zarr.open(path, mode='r')

        n_genes_per_sample: List[int] = []
        for path in self._sample_paths:
            store = self._get_store(path)
            data_grp = store['flat']
            n_genes_per_sample.append(int(data_grp['TSS_signals'].shape[0]))
        self._n_genes_per_sample = n_genes_per_sample

        cum = [0]
        running = 0
        for n in n_genes_per_sample:
            running += int(n)
            cum.append(running)
        self._cum_gene_offsets = cum

    def __len__(self) -> int:
        if self._cum_gene_offsets is None:
            raise RuntimeError('Dataset not initialized: _cum_gene_offsets is None')
        return self._cum_gene_offsets[-1]

    def _get_store(self, zarr_path: str) -> zarr.Group:
        if self.cache_stores and zarr_path in self._store_cache:
            return self._store_cache[zarr_path]
        return zarr.open(zarr_path, mode='r')

    def _get_sample_name(self, store: zarr.Group, zarr_path: str) -> str:
        cached = self._sample_name_cache.get(zarr_path)
        if cached is not None:
            return cached
        # Prefer explicit display name (full TSV sample name with cohort
        # prefix) so that internal maps and output TSVs carry unambiguous
        # identifiers like "CRPC-RA_00-010" instead of bare "00-010".
        name_map = getattr(self, 'sample_name_map', None)
        if isinstance(name_map, dict) and zarr_path in name_map:
            name = name_map[zarr_path]
        else:
            name = store.attrs.get('sample_name')
            if name is None:
                name = Path(zarr_path).stem.replace('.zarr', '')
        name = str(name)
        self._sample_name_cache[zarr_path] = name
        return name

    def _get_gene_name(self, store: zarr.Group, zarr_path: str, gene_idx: int) -> str:
        names = self._gene_names_cache.get(zarr_path)
        if names is None:
            names = [str(g) for g in list(store['gene_names'][:])]
            self._gene_names_cache[zarr_path] = names
        return str(names[int(gene_idx)])

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self._sample_paths is None or self._cum_gene_offsets is None:
            raise RuntimeError('Dataset not initialized: _sample_paths or _cum_gene_offsets is None')
        # Find which sample this gene index belongs to
        sample_i = bisect.bisect_right(self._cum_gene_offsets, idx) - 1
        zarr_path = self._sample_paths[sample_i]
        gene_idx = idx - self._cum_gene_offsets[sample_i]
        store = self._get_store(zarr_path)
        data_grp = store['flat']

        features = {
            'TSS_signals': torch.from_numpy(data_grp['TSS_signals'][gene_idx]),
            'TSS_nucleotides': torch.from_numpy(data_grp['TSS_nucleotides'][gene_idx].astype(np.float32)),
            'TSS_features': torch.from_numpy(data_grp['TSS_features'][gene_idx]),
            'GB_features': torch.from_numpy(data_grp['GB_features'][gene_idx]),
            'ref_scalars': torch.from_numpy(data_grp['ref_scalars'][gene_idx]),
        }

        tfx = torch.tensor(float(data_grp['TFX_label'][gene_idx]), dtype=torch.float32)

        if self.include_metadata:
            features['sample_name'] = self._get_sample_name(store, zarr_path)
            features['gene_name'] = self._get_gene_name(store, zarr_path, int(gene_idx))

        if self.gene_weights is not None:
            gname = self._get_gene_name(store, zarr_path, int(gene_idx))
            gene_weight = torch.tensor(self.gene_weights.get(gname, 1.0), dtype=torch.float32)
        elif 'gene_weight' in data_grp:
            gene_weight = torch.tensor(float(data_grp['gene_weight'][gene_idx]), dtype=torch.float32)
        else:
            gene_weight = torch.tensor(1.0, dtype=torch.float32)

        if self.labeled:
            tumor_exp = torch.tensor(float(data_grp['tumor_exp'][gene_idx]), dtype=torch.float32)
            background_exp = torch.tensor(float(data_grp['background_exp'][gene_idx]), dtype=torch.float32)
            labels: Dict[str, torch.Tensor] = {
                'tumor_exp': tumor_exp,
                'background_exp': background_exp,
                'sample_weight': gene_weight,
                'TFX_label': tfx,
            }
        else:
            labels = {'sample_weight': gene_weight, 'TFX_label': tfx}

        return features, labels


def _collate_identity(batch: List[Any]) -> Any:
    """Return the single element produced by DataLoader(batch_size=1)."""
    return batch[0]


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
        self.sample_paths = _normalize_fold_index_to_sample_paths(loaded)

        # Precompute (and persist) a deterministic chunk schedule.
        cache_path = f"{self.index_file}.chunksched_{self.chunk_size}_seed{self.seed}.json"
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self._chunks: List[Tuple[str, int, int]] = [
                    (str(p), int(s), int(e)) for (p, s, e) in json.load(f)
                ]
        else:
            chunks: List[Tuple[str, int, int]] = []
            for path in self.sample_paths:
                store = zarr.open(path, mode='r')
                n_genes = int(store['flat']['TSS_signals'].shape[0])
                for start in range(0, n_genes, self.chunk_size):
                    end = min(start + self.chunk_size, n_genes)
                    chunks.append((str(path), int(start), int(end)))

            rng = random.Random(self.seed)
            rng.shuffle(chunks)
            self._chunks = chunks
            with open(cache_path, 'w') as f:
                json.dump(self._chunks, f)

        # Apply subsampling if requested (for fast sanity checks).
        # This deterministically selects a fraction of chunks.
        if self.subsample_frac is not None and self.subsample_frac < 1.0:
            n_chunks = len(self._chunks)
            n_keep = max(1, int(n_chunks * self.subsample_frac))
            rng = random.Random(self.seed + 999)  # Different seed for subsample selection
            subsample_indices = sorted(rng.sample(range(n_chunks), n_keep))
            self._chunks = [self._chunks[i] for i in subsample_indices]

        # Gene-level train/val split at chunk granularity.
        # Deterministic: same seed + val_frac → same partition.
        if self.val_frac is not None and self.val_frac > 0:
            n_chunks = len(self._chunks)
            n_val = max(1, int(n_chunks * self.val_frac))
            rng_split = random.Random(self.seed + 777)
            val_indices = set(rng_split.sample(range(n_chunks), n_val))
            if self.val_split:
                self._chunks = [self._chunks[i] for i in sorted(val_indices)]
            else:
                self._chunks = [c for i, c in enumerate(self._chunks) if i not in val_indices]

        # Compute how many fixed-size microbatches we will yield.
        self._n_batches = self._count_batches(self._chunks)

        # For balanced training, pre-compute a guaranteed upper-bound
        # batch count.  With persistent_workers the main-process
        # _balance_iter never advances, so __len__ can't predict
        # exactly which epoch the workers will use.  Instead we take
        # the maximum over all possible random subsets by selecting the
        # n_min samples with the highest per-sample batch counts from
        # each balanced cohort.  The bound is tight (within 1-2
        # batches of the actual count) and prevents the PyTorch
        # "more samples fetched than reported" warning.
        if self.source_map and self.target_n_per_source:
            from collections import defaultdict
            batches_per_sample: Dict[str, int] = {}
            for path, start, end in self._chunks:
                batches_per_sample[path] = (
                    batches_per_sample.get(path, 0)
                    + (end - start) // self.genes_per_batch
                )
            source_paths_map: Dict[str, List[str]] = defaultdict(list)
            for path in self.sample_paths:
                src = self.source_map.get(path)
                if src is not None:
                    source_paths_map[src].append(path)
            max_balanced = 0
            for src_name, paths in source_paths_map.items():
                per_sample = sorted(
                    [batches_per_sample.get(p, 0) for p in paths],
                    reverse=True,
                )
                n_target = self.target_n_per_source
                # Worst case: we pick the samples with highest batch counts
                if len(per_sample) >= n_target:
                    max_balanced += sum(per_sample[:n_target])
                else:
                    # Oversampling: each sample can appear multiple times
                    full_copies = n_target // len(per_sample)
                    remainder = n_target % len(per_sample)
                    max_balanced += sum(per_sample) * full_copies + sum(per_sample[:remainder])
            self._balanced_n_batches = max_balanced
        elif self.cohort_map and self.balance_cohorts:
            from collections import defaultdict
            batches_per_sample: Dict[str, int] = {}
            for path, start, end in self._chunks:
                batches_per_sample[path] = (
                    batches_per_sample.get(path, 0)
                    + (end - start) // self.genes_per_batch
                )
            cohort_paths_map: Dict[str, List[str]] = defaultdict(list)
            for path in self.sample_paths:
                coh = self.cohort_map.get(path)
                if coh is not None:
                    cohort_paths_map[coh].append(path)
            balance_set = set(self.balance_cohorts)
            per_cohort_counts = {
                c: len(cohort_paths_map[c])
                for c in self.balance_cohorts if c in cohort_paths_map
            }
            n_min = min(per_cohort_counts.values()) if per_cohort_counts else 0
            max_balanced = 0
            for coh, paths in cohort_paths_map.items():
                per_sample = sorted(
                    [batches_per_sample.get(p, 0) for p in paths],
                    reverse=True,
                )
                if coh in balance_set:
                    max_balanced += sum(per_sample[:n_min])
                else:
                    max_balanced += sum(per_sample)
            self._balanced_n_batches = max_balanced
        else:
            self._balanced_n_batches = self._n_batches

        # These caches live per-process (per DataLoader worker).
        self._store_cache: Dict[str, zarr.Group] = {}
        self._flat_grp_cache: Dict[str, zarr.Group] = {}
        self._gene_names_cache: Dict[str, List[str]] = {}
        self._sample_name_cache: Dict[str, str] = {}

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

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = int(worker_info.id)
            num_workers = int(worker_info.num_workers)

        # Per-epoch balanced filtering (source-group or cohort-level).
        # _balance_iter increments on every __iter__() call so each
        # worker deterministically varies the selected samples.
        active_chunks = self._chunks
        if self.source_map and self.target_n_per_source:
            self._balance_iter += 1
            active_chunks = self._balanced_chunks(self._balance_iter)
        elif self.cohort_map and self.balance_cohorts:
            self._balance_iter += 1
            active_chunks = self._balanced_chunks(self._balance_iter)

        # Deterministic sharding of chunks across workers.
        for chunk_idx in range(worker_id, len(active_chunks), num_workers):
            zarr_path, start, end = active_chunks[chunk_idx]
            store = self._get_store(zarr_path)
            data_grp = self._get_flat_grp(zarr_path)

            # Load one contiguous chunk into memory.
            tss_signals = torch.from_numpy(data_grp['TSS_signals'][start:end])
            tss_nt = torch.from_numpy(data_grp['TSS_nucleotides'][start:end].astype(np.float32))
            tss_feat = torch.from_numpy(data_grp['TSS_features'][start:end])
            gb_feat = torch.from_numpy(data_grp['GB_features'][start:end])
            ref_scalars = torch.from_numpy(data_grp['ref_scalars'][start:end])
            # Force shape (n_genes,) then we will unsqueeze(-1) for (n_genes,1)
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

            # Yield fixed-size microbatches.
            chunk_len = int(end - start)
            for off in range(0, chunk_len, self.genes_per_batch):
                mb_end = off + self.genes_per_batch
                if mb_end > chunk_len:
                    break  # keep fixed size

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

                yield features, labels


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
    """Multi-source gene-conditioned batch loader.

    Each batch contains ``genes_per_batch × n_sources × samples_per_source``
    items, where different sources contribute observations of the same genes.
    Default geometry: 2 genes × 4 sources × 4 samples/source = 32 items.

    Gene sampling uses a mixture distribution:
        P(gene_i) = frac × (var_i / Σvar) + (1 − frac) × (1/n)
    where *frac* controls the variance-priority weight (default 0.70).

    Source balancing builds an exposure list of ``n_eff_per_source`` sample
    slots per source per epoch.  Small sources are oversampled via
    repeat-plus-shuffle to reach ``n_eff_per_source``.  The list is cycled
    and reshuffled when exhausted so that every active sample sees a
    statistically representative gene set.

    Loss weights per gene are compressed: ``sqrt(variance) / median_stdev``,
    clamped to ``[gene_weight_min, gene_weight_max]``.

    All training samples are preloaded into RAM at init time to eliminate
    Zarr chunk-decompression overhead on the hot path.  TSS_signals are
    stored as float16 and TSS_nucleotides as uint8 to fit within typical
    HPC node memory budgets (~114 GB for 144 samples × 18K genes).
    DataLoader workers share the preloaded arrays via fork copy-on-write.
    """

    def __init__(
        self,
        sample_paths_by_source: Dict[str, List[str]],
        gene_variance: Dict[str, float],
        n_eff_per_source: int = 24,
        genes_per_batch: int = 2,
        samples_per_source: int = 4,
        variance_sample_frac: float = 0.70,
        gene_weight_min: float = 0.5,
        gene_weight_max: float = 2.0,
        seed: int = 42,
        labeled: bool = True,
        sample_name_map: Optional[Dict[str, str]] = None,
        include_metadata: bool = False,
        subsample_frac: Optional[float] = None,
    ):
        super().__init__()
        self.sample_paths_by_source = dict(sample_paths_by_source)
        self.source_names = sorted(self.sample_paths_by_source.keys())
        self.n_sources = len(self.source_names)
        self.n_eff_per_source = n_eff_per_source
        self.genes_per_batch = genes_per_batch
        self.samples_per_source = samples_per_source
        self.variance_sample_frac = variance_sample_frac
        self.gene_weight_min = gene_weight_min
        self.gene_weight_max = gene_weight_max
        self.seed = seed
        self.labeled = labeled
        self.sample_name_map = sample_name_map
        self.include_metadata = include_metadata
        self.batch_size = genes_per_batch * self.n_sources * samples_per_source
        self._epoch_counter = 0

        # Build gene universe: intersection of all samples' gene sets.
        all_paths = [p for paths in self.sample_paths_by_source.values()
                     for p in paths]
        if not all_paths:
            raise ValueError("No sample paths provided to "
                             "ProteusMultiSourceBatchDataset")

        gene_sets: List[set] = []
        for zpath in all_paths:
            store = zarr.open(zpath, mode='r')
            names = [str(g) for g in store['gene_names'][:]]
            gene_sets.append(set(names))
        common = set.intersection(*gene_sets)
        if not common:
            raise ValueError("No common genes found across all samples")
        self._common_genes: List[str] = sorted(common)
        n_genes = len(self._common_genes)

        # Preload all samples into RAM (eliminates Zarr chunk decompression
        # from the hot path).  Must happen before workers are forked so they
        # share the arrays via OS copy-on-write pages.
        self._preloaded: Dict[str, Dict[str, np.ndarray]] = {}
        self._preload_all_samples(all_paths)

        # Gene sampling distribution (mixture of variance-priority + uniform).
        variance_arr = np.array(
            [gene_variance.get(g, 0.0) for g in self._common_genes],
            dtype=np.float64,
        )
        variance_arr = np.maximum(variance_arr, 0.0)
        sum_var = variance_arr.sum()
        if sum_var > 0:
            var_probs = variance_arr / sum_var
        else:
            var_probs = np.ones(n_genes, dtype=np.float64) / n_genes
        uniform_probs = np.ones(n_genes, dtype=np.float64) / n_genes
        self._gene_probs = (
            self.variance_sample_frac * var_probs
            + (1.0 - self.variance_sample_frac) * uniform_probs
        )
        self._gene_probs /= self._gene_probs.sum()

        # Per-gene loss weights: sqrt(variance) / median(stdev), clamped.
        stdevs = np.sqrt(variance_arr)
        positive_stdevs = stdevs[stdevs > 0]
        median_std = float(np.median(positive_stdevs)) if len(positive_stdevs) > 0 else 1.0
        if median_std <= 0:
            median_std = 1.0
        raw_weights = stdevs / median_std
        self._gene_loss_weights: np.ndarray = np.clip(
            raw_weights, gene_weight_min, gene_weight_max,
        ).astype(np.float32)

        # Epoch length: each of n_eff samples per source sees ~all genes once.
        slots_per_source_per_batch = self.genes_per_batch * self.samples_per_source
        total_source_slots = self.n_eff_per_source * n_genes
        self._n_batches = max(1, total_source_slots // slots_per_source_per_batch)

        if subsample_frac is not None and 0.0 < subsample_frac < 1.0:
            self._n_batches = max(1, int(self._n_batches * subsample_frac))

        print(f"  [MultiSource] {n_genes} common genes, "
              f"{self.n_sources} sources × {self.n_eff_per_source} eff/source, "
              f"batch={self.batch_size} ({self.genes_per_batch}g × "
              f"{self.n_sources}s × {self.samples_per_source}sps), "
              f"{self._n_batches} batches/epoch")

    def __len__(self) -> int:
        return self._n_batches

    # -- RAM preloading --

    def _preload_all_samples(self, all_paths: List[str]) -> None:
        """Preload all training samples' common-gene arrays into RAM.

        Each sample's arrays are reindexed to common-gene canonical order
        so that ``__iter__`` can use a direct integer index into the
        preloaded arrays without per-sample gene name → index lookup.

        Compact dtypes minimise memory footprint:
        - TSS_signals: float16 (~648 MB / sample for 18K genes)
        - TSS_nucleotides: uint8 (~144 MB / sample)
        - All other arrays: float32 (negligible)

        Decompression is parallelised across threads (zstd releases the
        GIL) to keep init time short (~30-60 s for 144 samples).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        common_gene_set = set(self._common_genes)
        n_common = len(self._common_genes)
        common_gene_order = {g: i for i, g in enumerate(self._common_genes)}

        def _load_one(zarr_path: str) -> tuple:
            store = zarr.open(zarr_path, mode='r')
            gene_names = [str(g) for g in store['gene_names'][:]]

            # Build index array: position i in common_genes → local index.
            local_indices = np.empty(n_common, dtype=np.intp)
            gene_to_local = {g: i for i, g in enumerate(gene_names)}
            for g, common_pos in common_gene_order.items():
                local_indices[common_pos] = gene_to_local[g]

            flat = store['flat']

            # Read full arrays and reindex to common gene order.
            signals = flat['TSS_signals'][:][local_indices].astype(np.float16)
            nucleotides = flat['TSS_nucleotides'][:][local_indices]  # uint8
            tss_features = flat['TSS_features'][:][local_indices].astype(np.float32)
            gb_features = flat['GB_features'][:][local_indices].astype(np.float32)
            ref_scalars = flat['ref_scalars'][:][local_indices].astype(np.float32)
            tumor_exp = flat['tumor_exp'][:][local_indices].astype(np.float32)
            bg_exp = flat['background_exp'][:][local_indices].astype(np.float32)
            tfx_label = flat['TFX_label'][:][local_indices].astype(np.float32)

            # Resolve sample display name at load time (avoids keeping
            # Zarr stores open after init).
            if isinstance(self.sample_name_map, dict) and zarr_path in self.sample_name_map:
                sample_display = self.sample_name_map[zarr_path]
            else:
                name = store.attrs.get('sample_name')
                if name is None:
                    name = Path(zarr_path).stem.replace('.zarr', '')
                sample_display = str(name)

            return zarr_path, {
                'TSS_signals': signals,
                'TSS_nucleotides': nucleotides,
                'TSS_features': tss_features,
                'GB_features': gb_features,
                'ref_scalars': ref_scalars,
                'tumor_exp': tumor_exp,
                'background_exp': bg_exp,
                'TFX_label': tfx_label,
            }, sample_display

        unique_paths = sorted(set(all_paths))
        n_samples = len(unique_paths)

        # Estimate memory requirement.
        per_sample_mb = (
            n_common * 2000 * 9 * 2     # TSS_signals float16
            + n_common * 2000 * 4 * 1   # TSS_nucleotides uint8
            + n_common * (11 + 11 + 3 + 1 + 1 + 1) * 4  # small float32
        ) / (1024 ** 2)
        total_gb = per_sample_mb * n_samples / 1024
        print(f"  [MultiSource] Preloading {n_samples} samples "
              f"({n_common} common genes) into RAM — "
              f"estimated {total_gb:.1f} GB...")

        self._sample_display_names: Dict[str, str] = {}

        # Use 4 threads: limits transient peak memory (~5 GB) while still
        # saturating zstd decompression throughput.
        max_threads = min(4, n_samples)
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(_load_one, p): p for p in unique_paths}
            done = 0
            for future in as_completed(futures):
                zarr_path, arrays, display_name = future.result()
                self._preloaded[zarr_path] = arrays
                self._sample_display_names[zarr_path] = display_name
                done += 1
                if done % 20 == 0 or done == n_samples:
                    print(f"    ... loaded {done}/{n_samples} samples")

        print(f"  [MultiSource] Preload complete ({total_gb:.1f} GB in RAM)")

    def _get_sample_name(self, zarr_path: str) -> str:
        return self._sample_display_names.get(
            zarr_path,
            Path(zarr_path).stem.replace('.zarr', ''),
        )

    # -- Exposure list management --

    def _build_exposure_lists(
        self, rng: random.Random,
    ) -> Dict[str, List[str]]:
        exposure: Dict[str, List[str]] = {}
        for src in self.source_names:
            paths = self.sample_paths_by_source[src]
            n_avail = len(paths)
            n_eff = self.n_eff_per_source
            if n_avail >= n_eff:
                chosen = rng.sample(paths, n_eff)
            else:
                full_copies = n_eff // n_avail
                remainder = n_eff % n_avail
                chosen = paths * full_copies + rng.sample(paths, remainder)
            rng.shuffle(chosen)
            exposure[src] = chosen
        return exposure

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
        np_rng = np.random.RandomState(epoch_seed % (2**31))

        # Build initial exposure lists and position cursors.
        exposure = self._build_exposure_lists(rng)
        positions: Dict[str, int] = {src: 0 for src in self.source_names}

        # Divide batches across workers.
        total_batches = self._n_batches
        worker_batches = total_batches // num_workers
        if worker_id < (total_batches % num_workers):
            worker_batches += 1

        n_genes = len(self._common_genes)
        sps = self.samples_per_source

        for _batch_idx in range(worker_batches):
            # Sample genes from mixture distribution (without replacement
            # within a single batch to maximise gene diversity).
            gene_indices = np_rng.choice(
                n_genes, size=self.genes_per_batch,
                replace=False, p=self._gene_probs,
            )

            # Collect items: genes_per_batch × n_sources × sps.
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

            for g_idx in gene_indices:
                gene_name = self._common_genes[g_idx]
                gene_weight = float(self._gene_loss_weights[g_idx])

                for src in self.source_names:
                    for _ in range(sps):
                        pos = positions[src]
                        if pos >= len(exposure[src]):
                            rng.shuffle(exposure[src])
                            pos = 0
                        zarr_path = exposure[src][pos]
                        positions[src] = pos + 1

                        sd = self._preloaded[zarr_path]

                        # Preloaded arrays are in common-gene order,
                        # so g_idx indexes directly (no name lookup).
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


def collate_proteus_genes(batch: List[Tuple[Dict, Dict]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Collate gene-level items (each item is one gene)."""
    features_list, labels_list = zip(*batch)

    batched_features = {
        'TSS_signals': torch.stack([f['TSS_signals'] for f in features_list]),
        'TSS_nucleotides': torch.stack([f['TSS_nucleotides'] for f in features_list]),
        'TSS_features': torch.stack([f['TSS_features'] for f in features_list]),
        'GB_features': torch.stack([f['GB_features'] for f in features_list]),
        'ref_scalars': torch.stack([f['ref_scalars'] for f in features_list]),
    }

    if features_list and 'sample_name' in features_list[0]:
        batched_features['sample_name'] = [str(f['sample_name']) for f in features_list]
    if features_list and 'gene_name' in features_list[0]:
        batched_features['gene_name'] = [str(f['gene_name']) for f in features_list]

    if labels_list and 'tumor_exp' in labels_list[0]:
        batched_labels = {
            'tumor_exp': torch.stack([l['tumor_exp'] for l in labels_list]).unsqueeze(-1),
            'background_exp': torch.stack([l['background_exp'] for l in labels_list]).unsqueeze(-1),
            'sample_weight': torch.stack([l.get('sample_weight', torch.tensor(1.0)) for l in labels_list]).unsqueeze(-1),
            'TFX_label': torch.stack([l['TFX_label'] for l in labels_list]).unsqueeze(-1),
        }
    else:
        batched_labels = {
            'sample_weight': torch.stack([l.get('sample_weight', torch.tensor(1.0)) for l in labels_list]).unsqueeze(-1),
            'TFX_label': torch.stack([l['TFX_label'] for l in labels_list]).unsqueeze(-1),
        }

    return batched_features, batched_labels


def collate_inference(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for inference (no labels)."""
    return {
        'TSS_signals': torch.stack([b['TSS_signals'] for b in batch]),
        'TSS_nucleotides': torch.stack([b['TSS_nucleotides'] for b in batch]),
        'TSS_features': torch.stack([b['TSS_features'] for b in batch]),
        'GB_features': torch.stack([b['GB_features'] for b in batch]),
        'ref_scalars': torch.stack([b['ref_scalars'] for b in batch]),
    }


def create_data_loader(
    index_file: Optional[str] = None,
    zarr_paths: Optional[List[str]] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    labeled: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    loader_mode: str = 'gene',
    chunk_size: int = 256,
    genes_per_batch: int = 32,
    shuffle_genes_in_batch: bool = True,
    seed: int = 42,
    include_metadata: bool = False,
    drop_last: bool = False,
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
    """
    Create a DataLoader for Proteus training/validation.
    
    Args:
        index_file: Path to fold index JSON file
        zarr_paths: List of Zarr store paths (alternative)
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        labeled: Whether dataset has labels
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        subsample_frac: Optional fraction (0.0-1.0) of genes for fast sanity checks.
        
    Returns:
        DataLoader and number of batches
    """
    mode = str(loader_mode).lower()

    # Invariant: a "batch" is always exactly 32 genes.
    if int(genes_per_batch) != 32:
        raise ValueError(f"genes_per_batch must be 32 (invariant). Got {genes_per_batch}.")
    if int(chunk_size) != 256:
        raise ValueError(f"chunk_size must be 256 (invariant). Got {chunk_size}.")

    base_seed = int(seed)
    base_generator = torch.Generator()
    base_generator.manual_seed(base_seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    if mode == 'chunk':
        if index_file is None:
            raise ValueError('Chunk loader requires index_file (v5 sample-path list).')

        dataset = ProteusGeneChunkIterableDataset(
            index_file=index_file,
            chunk_size=chunk_size,
            genes_per_batch=genes_per_batch,
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

        # IterableDataset already yields fixed-size training batches.
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_identity,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=base_generator,
            drop_last=False,
            pin_memory=True,
        )
    else:
        # Default: per-gene items
        dataset = ProteusGeneDataset(
            index_file=index_file,
            zarr_paths=zarr_paths,
            labeled=labeled,
            cache_stores=True,
            include_metadata=include_metadata,
            gene_weights=gene_weights,
        )

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_proteus_genes,
            persistent_workers=persistent_workers and num_workers > 0,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=base_generator,
            drop_last=bool(drop_last),
            pin_memory=True,
        )

    n_batches = len(dataset) if mode == 'chunk' else len(loader)
    
    return loader, n_batches


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
        batch_size=32,
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
    loader_mode: str = 'chunk',
    chunk_size: int = 256,
    genes_per_batch: int = 32,
    shuffle_genes_in_batch: bool = True,
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
    This function reads the TSV, constructs Zarr paths as:
        {zarr_base}/{cohort}/{sample}.zarr
    and builds loaders where training = all folds except val_fold.

    Args:
        training_set_tsv: Path to TSV with Cohort/Sample/Fold columns.
        zarr_base: Base directory containing cohort subdirs with sample.zarr stores.
        val_fold: Which fold to hold out for validation.
        num_workers: DataLoader workers.
        loader_mode: 'chunk' (recommended) or 'gene'.
        chunk_size: Genes per chunk (256 invariant).
        genes_per_batch: Genes per batch (32 invariant).
        shuffle_genes_in_batch: Shuffle genes within batch.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Batches to prefetch per worker.
        seed: Random seed for reproducibility.
        subsample_frac: Optional fraction (0.0-1.0) of genes to use per epoch.
                        Useful for fast sanity checks. None = use all genes.
        train_only_cohorts: Cohorts that always go to training (never
                            validation), regardless of fold assignment.
                            E.g. ``['PHT']`` for PDX admixtures.
        balance_cohorts: Cohorts to balance by per-epoch undersampling
                         of the majority cohort(s) to the minority count.
                         E.g. ``['CRPC-RA', 'SCLC']``.  Cohorts not
                         listed are always included at full count.
        source_groups: Source-level batch balancing.  Maps source name to
                       list of cohort names.  PHT samples are automatically
                       assigned to the source whose PDX cohort they derive
                       from (parsed from the embedded PDX name in the
                       sample name: ``PHT-Fold-X_<PDX_NAME>-BP...``).
                       Mutually exclusive with ``balance_cohorts``.
        target_n_per_source: Target samples per source group per epoch.
                             Only used when ``source_groups`` is set.

    Returns:
        (train_loader, val_loader, train_steps, val_steps)

    Raises:
        FileNotFoundError: If training_set_tsv or any Zarr store doesn't exist.
        ValueError: If TSV format is invalid.
    """
    import csv

    if not os.path.exists(training_set_tsv):
        raise FileNotFoundError(f"Training set TSV not found: {training_set_tsv}")

    train_paths: List[str] = []
    val_paths: List[str] = []
    missing_samples: List[str] = []
    zarr_path_to_cohort: Dict[str, str] = {}       # for cohort balancing
    zarr_path_to_sample_name: Dict[str, str] = {}  # zarr_path -> full TSV sample name
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

            # Zarr files on disk use bare names (without cohort prefix),
            # but we retain the full TSV name for internal maps / outputs.
            zarr_name = sample
            if sample.startswith(f"{cohort}_"):
                zarr_name = sample[len(cohort) + 1:]

            zarr_path = os.path.join(zarr_base, cohort, 'zarr_files', f"{zarr_name}.zarr")
            if not os.path.isdir(zarr_path):
                missing_samples.append(f"  {cohort}/zarr_files/{zarr_name}.zarr  (Fold={fold}, TSV sample={sample})")
                continue

            zarr_path_to_cohort[zarr_path] = cohort
            zarr_path_to_sample_name[zarr_path] = sample

            if cohort in _train_only:
                # Always route to training (e.g. PDX admixtures).
                train_paths.append(zarr_path)
            elif fold == val_fold:
                val_paths.append(zarr_path)
            else:
                train_paths.append(zarr_path)

    if missing_samples:
        raise FileNotFoundError(
            f"Missing {len(missing_samples)} Zarr store(s) under {zarr_base}:\n"
            + "\n".join(missing_samples)
        )

    if not train_paths:
        raise ValueError(f"No training samples found (all in fold {val_fold}?)")
    if not val_paths:
        raise ValueError(f"No validation samples found for fold {val_fold}")

    # Shuffle training paths deterministically
    rng = np.random.RandomState(seed)
    rng.shuffle(train_paths)

    # Build source_map: zarr_path -> source group name.
    # For PHT samples, parse the embedded PDX name from the sample name
    # (format: PHT-Fold-X_<PDX_NAME>-BP...) and match against known
    # PDX sample names to determine which source group they belong to.
    zarr_path_to_source: Optional[Dict[str, str]] = None
    if source_groups:
        import re
        # Invert source_groups: cohort -> source name
        cohort_to_source: Dict[str, str] = {}
        for src_name, cohorts in source_groups.items():
            for coh in cohorts:
                cohort_to_source[coh] = src_name

        # Collect known PDX sample names per cohort (for matching PHT names).
        pdx_sample_to_cohort: Dict[str, str] = {}
        with open(training_set_tsv, 'r', newline='') as f2:
            reader2 = csv.DictReader(f2, delimiter='\t')
            for row in reader2:
                coh = row['Cohort'].strip()
                samp = row['Sample'].strip()
                if samp.startswith(f"{coh}_"):
                    samp = samp[len(coh) + 1:]
                # Only index non-PHT cohorts that appear in source groups
                if coh in cohort_to_source:
                    pdx_sample_to_cohort[samp] = coh

        # Build the source map for all training paths.
        zarr_path_to_source = {}
        pht_re = re.compile(r'^PHT-Fold-\d+_(.+)-BP\d+')
        for zpath, coh in zarr_path_to_cohort.items():
            if zpath not in train_paths:
                continue
            if coh in cohort_to_source:
                zarr_path_to_source[zpath] = cohort_to_source[coh]
            elif coh == 'PHT':
                # Extract PDX name from the zarr filename.
                basename = os.path.basename(zpath).replace('.zarr', '')
                m = pht_re.match(basename)
                if m:
                    pdx_name = m.group(1)
                    pdx_coh = pdx_sample_to_cohort.get(pdx_name)
                    if pdx_coh is not None:
                        zarr_path_to_source[zpath] = cohort_to_source[pdx_coh]
                    else:
                        raise ValueError(
                            f"PHT sample '{basename}' has PDX name '{pdx_name}' "
                            f"that does not match any known PDX sample. "
                            f"Known: {sorted(pdx_sample_to_cohort.keys())}"
                        )
                else:
                    raise ValueError(
                        f"Could not parse PDX name from PHT sample: '{basename}'. "
                        f"Expected format: PHT-Fold-X_<PDX_NAME>-BP..."
                    )

        # Report source group assignments.
        from collections import Counter
        src_counts = Counter(zarr_path_to_source.values())
        print(f"  [Source groups] {dict(src_counts)}, "
              f"target_n_per_source={target_n_per_source}")

    # Write temp index files (required by chunk loader)
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix='proteus_splits_')
    train_index = os.path.join(temp_dir, f'train_excl_fold{val_fold}.json')
    val_index = os.path.join(temp_dir, f'val_fold{val_fold}.json')

    with open(train_index, 'w') as f:
        json.dump(train_paths, f)
    with open(val_index, 'w') as f:
        json.dump(val_paths, f)

    # Log subsampling info if active
    if subsample_frac is not None and subsample_frac < 1.0:
        print(f"  [Subsample] Using ~{subsample_frac*100:.0f}% of genes for fast sanity check")

    train_loader, train_steps = create_data_loader(
        index_file=train_index,
        num_workers=num_workers,
        shuffle=(loader_mode.lower() == 'gene'),
        labeled=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        loader_mode=loader_mode,
        chunk_size=chunk_size,
        genes_per_batch=genes_per_batch,
        shuffle_genes_in_batch=shuffle_genes_in_batch,
        seed=seed,
        drop_last=(loader_mode.lower() == 'gene'),
        subsample_frac=subsample_frac,
        gene_weights=gene_weights,
        cohort_map=zarr_path_to_cohort if balance_cohorts else None,
        balance_cohorts=balance_cohorts,
        source_map=zarr_path_to_source,
        target_n_per_source=target_n_per_source,
    )

    val_loader, val_steps = create_data_loader(
        index_file=val_index,
        num_workers=num_workers,
        shuffle=False,
        labeled=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        loader_mode=loader_mode,
        chunk_size=chunk_size,
        genes_per_batch=genes_per_batch,
        shuffle_genes_in_batch=False,
        seed=seed,
        include_metadata=True,
        drop_last=False,
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
    samples_per_source: int = 4,
    variance_sample_frac: float = 0.70,
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
        gene_variance: Per-gene raw variance (for mixture sampler + loss weights).
        n_eff_per_source: Effective samples per source per epoch.
        genes_per_batch: Unique genes drawn per batch (default 2).
        samples_per_source: Samples drawn per source per gene (default 4).
        variance_sample_frac: Weight for variance-priority sampling (default 0.70).
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
    import csv
    import re

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

            zarr_name = sample
            if sample.startswith(f"{cohort}_"):
                zarr_name = sample[len(cohort) + 1:]

            zarr_path = os.path.join(
                zarr_base, cohort, 'zarr_files', f"{zarr_name}.zarr")
            if not os.path.isdir(zarr_path):
                missing_samples.append(
                    f"  {cohort}/zarr_files/{zarr_name}.zarr  "
                    f"(Fold={fold}, TSV sample={sample})")
                continue

            zarr_path_to_cohort[zarr_path] = cohort
            zarr_path_to_sample_name[zarr_path] = sample

            if cohort in _train_only:
                train_paths.append(zarr_path)
            elif fold == val_fold:
                val_paths.append(zarr_path)
            else:
                train_paths.append(zarr_path)

    if missing_samples:
        raise FileNotFoundError(
            f"Missing {len(missing_samples)} Zarr store(s) under {zarr_base}:\n"
            + "\n".join(missing_samples))
    if not train_paths:
        raise ValueError(f"No training samples found (all in fold {val_fold}?)")
    if not val_paths:
        raise ValueError(f"No validation samples found for fold {val_fold}")

    # -- Assign training paths to source groups (including PHT → parent PDX) --
    cohort_to_source: Dict[str, str] = {}
    for src_name, cohorts in source_groups.items():
        for coh in cohorts:
            cohort_to_source[coh] = src_name

    # Collect known PDX sample names per cohort for PHT matching.
    pdx_sample_to_cohort: Dict[str, str] = {}
    with open(training_set_tsv, 'r', newline='') as f2:
        reader2 = csv.DictReader(f2, delimiter='\t')
        for row in reader2:
            coh = row['Cohort'].strip()
            samp = row['Sample'].strip()
            if samp.startswith(f"{coh}_"):
                samp = samp[len(coh) + 1:]
            if coh in cohort_to_source:
                pdx_sample_to_cohort[samp] = coh

    pht_re = re.compile(r'^PHT-Fold-\d+_(.+)-BP\d+')
    sample_paths_by_source: Dict[str, List[str]] = {
        src: [] for src in source_groups
    }

    for zpath in train_paths:
        coh = zarr_path_to_cohort[zpath]
        if coh in cohort_to_source:
            sample_paths_by_source[cohort_to_source[coh]].append(zpath)
        elif coh == 'PHT':
            basename = os.path.basename(zpath).replace('.zarr', '')
            m = pht_re.match(basename)
            if m:
                pdx_name = m.group(1)
                pdx_coh = pdx_sample_to_cohort.get(pdx_name)
                if pdx_coh is not None:
                    sample_paths_by_source[cohort_to_source[pdx_coh]].append(zpath)
                else:
                    raise ValueError(
                        f"PHT sample '{basename}' has PDX name '{pdx_name}' "
                        f"that does not match any known PDX sample. "
                        f"Known: {sorted(pdx_sample_to_cohort.keys())}")
            else:
                raise ValueError(
                    f"Could not parse PDX name from PHT sample: '{basename}'. "
                    f"Expected format: PHT-Fold-X_<PDX_NAME>-BP...")

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
        samples_per_source=samples_per_source,
        variance_sample_frac=variance_sample_frac,
        gene_weight_min=gene_weight_min,
        gene_weight_max=gene_weight_max,
        seed=seed,
        labeled=True,
        sample_name_map=zarr_path_to_sample_name,
        include_metadata=False,
        subsample_frac=subsample_frac,
    )

    base_generator = torch.Generator()
    base_generator.manual_seed(seed)

    def _seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_identity,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=base_generator,
        drop_last=False,
        pin_memory=True,
    )
    train_steps = len(train_dataset)

    # -- Validation loader (chunk-based, exhaustive) --
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix='proteus_msplits_')
    val_index = os.path.join(temp_dir, f'val_fold{val_fold}.json')
    with open(val_index, 'w') as f:
        json.dump(val_paths, f)

    val_loader, val_steps = create_data_loader(
        index_file=val_index,
        num_workers=num_workers,
        shuffle=False,
        labeled=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        loader_mode='chunk',
        chunk_size=256,
        genes_per_batch=32,
        shuffle_genes_in_batch=False,
        seed=seed,
        include_metadata=True,
        drop_last=False,
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

