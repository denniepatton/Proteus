"""
proteus_utils.py (PyTorch version)

This file contains the PyTorch Dataset class and data loading utilities
for efficient training with Zarr-based storage.

Updated for current Triton outputs:
- TSS_signals: 8 signal channels (float32)
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

    Supports:
    - sample-path list: ["/abs/path/sample1.zarr", ...]
      - legacy explicit gene indices: [["/abs/path/sample1.zarr", 123], ...]

    Returns paths in first-seen order with duplicates removed.
    """
    if not isinstance(loaded, list):
        raise ValueError('Fold index JSON must be a list.')

    # Legacy explicit indices: list of 2-tuples/lists.
    if loaded and isinstance(loaded[0], (list, tuple)) and len(loaded[0]) == 2:
        paths = [str(p) for (p, _gene) in loaded]
    else:
        paths = [str(p) for p in loaded]

    seen: set[str] = set()
    uniq: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


class ProteusSampleDataset(Dataset):
    """Sample-level Dataset for Proteus.

    Each item is one complete sample (all genes kept together). During training,
    a collate function concatenates genes across a batch of samples and shuffles
    genes in-memory.

        Expected Zarr layout:
      sample.zarr/
        ├── gene_names
        └── flat/ (arrays with leading dim = n_genes)
    """

    def __init__(
        self,
        index_file: Optional[str] = None,
        zarr_paths: Optional[List[str]] = None,
        labeled: bool = True,
        cache_stores: bool = True,
    ):
        self.labeled = labeled
        self.cache_stores = cache_stores
        self._store_cache: Dict[str, zarr.Group] = {}

        if index_file is not None:
            with open(index_file, 'r') as f:
                self.zarr_paths = json.load(f)
        elif zarr_paths is not None:
            self.zarr_paths = list(zarr_paths)
        else:
            raise ValueError("Either index_file or zarr_paths must be provided")

        if cache_stores:
            for path in set(self.zarr_paths):
                self._store_cache[path] = zarr.open(path, mode='r')

    def __len__(self) -> int:
        return len(self.zarr_paths)

    def _get_store(self, zarr_path: str) -> zarr.Group:
        if self.cache_stores and zarr_path in self._store_cache:
            return self._store_cache[zarr_path]
        return zarr.open(zarr_path, mode='r')

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        zarr_path = self.zarr_paths[idx]
        store = self._get_store(zarr_path)
        data_grp = store['flat'] if 'flat' in store else store

        # Features (all genes together)
        features = {
            'TSS_signals': torch.from_numpy(data_grp['TSS_signals'][:]),
            'TSS_nucleotides': torch.from_numpy(data_grp['TSS_nucleotides'][:].astype(np.float32)),
            'TSS_features': torch.from_numpy(data_grp['TSS_features'][:]),
            'GB_features': torch.from_numpy(data_grp['GB_features'][:]),
            'ref_scalars': torch.from_numpy(data_grp['ref_scalars'][:]),
        }

        tfx = torch.from_numpy(data_grp['TFX_label'][:])

        if 'gene_weight' in data_grp:
            gene_weight = torch.from_numpy(data_grp['gene_weight'][:])
        else:
            gene_weight = torch.ones(tfx.shape[0], dtype=torch.float32)

        labels: Dict[str, torch.Tensor]
        if self.labeled:
            tumor_exp = torch.from_numpy(data_grp['tumor_exp'][:])
            background_exp = torch.from_numpy(data_grp['background_exp'][:])
            labels = {
                'tumor_exp': tumor_exp,
                'background_exp': background_exp,
                'sample_weight': gene_weight,
                'TFX_label': tfx,
            }
        else:
            labels = {'sample_weight': gene_weight, 'TFX_label': tfx}

        return features, labels


class ProteusGeneDataset(Dataset):
    """Gene-level Dataset backed by flat per-sample Zarr arrays.

    This keeps the model's training semantics intact (each item = one gene),
    while still avoiding the on-disk per-gene hierarchy.

    Input index formats supported:
      - New (v5 fold indices): ["/path/to/sampleA.zarr", "/path/to/sampleB.zarr", ...]
        -> dataset length is total genes across all samples.
      - Legacy (v4.x fold indices): [["/path/to/sample.zarr", "GENE"], ...]
        -> dataset length is the explicit list length.
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
        self._gene_name_to_index: Dict[str, Dict[str, int]] = {}
        self._gene_names_cache: Dict[str, List[str]] = {}
        self._sample_name_cache: Dict[str, str] = {}

        if index_file is not None:
            with open(index_file, 'r') as f:
                loaded = json.load(f)
        elif zarr_paths is not None:
            loaded = list(zarr_paths)
        else:
            raise ValueError("Either index_file or zarr_paths must be provided")

        self._explicit_pairs: Optional[List[Tuple[str, Union[str, int]]]] = None
        self._sample_paths: Optional[List[str]] = None
        self._cum_gene_offsets: Optional[List[int]] = None
        self._n_genes_per_sample: Optional[List[int]] = None

        # Determine index format
        if isinstance(loaded, list) and loaded and isinstance(loaded[0], (list, tuple)) and len(loaded[0]) == 2:
            # Legacy explicit pairs: [[zarr_path, gene_id], ...]
            self._explicit_pairs = [(p[0], p[1]) for p in loaded]
            if cache_stores:
                for path in set(p[0] for p in self._explicit_pairs):
                    self._store_cache[path] = zarr.open(path, mode='r')
        else:
            # Sample-path list
            self._sample_paths = [str(p) for p in loaded]
            if cache_stores:
                for path in set(self._sample_paths):
                    self._store_cache[path] = zarr.open(path, mode='r')

            n_genes_per_sample: List[int] = []
            for path in self._sample_paths:
                store = self._get_store(path)
                data_grp = store['flat'] if 'flat' in store else store
                n_genes_per_sample.append(int(data_grp['TSS_signals'].shape[0]))
            self._n_genes_per_sample = n_genes_per_sample

            cum = [0]
            running = 0
            for n in n_genes_per_sample:
                running += int(n)
                cum.append(running)
            self._cum_gene_offsets = cum

    def __len__(self) -> int:
        if self._explicit_pairs is not None:
            return len(self._explicit_pairs)
        assert self._cum_gene_offsets is not None
        return self._cum_gene_offsets[-1]

    def _get_store(self, zarr_path: str) -> zarr.Group:
        if self.cache_stores and zarr_path in self._store_cache:
            return self._store_cache[zarr_path]
        return zarr.open(zarr_path, mode='r')

    def _resolve_gene_index(self, store: zarr.Group, zarr_path: str, gene_id: Union[str, int]) -> int:
        if isinstance(gene_id, int):
            return gene_id
        # gene_id as string
        mapping = self._gene_name_to_index.get(zarr_path)
        if mapping is None:
            gene_names = [str(g) for g in list(store['gene_names'][:])]
            self._gene_names_cache[zarr_path] = gene_names
            mapping = {g: i for i, g in enumerate(gene_names)}
            self._gene_name_to_index[zarr_path] = mapping
        return mapping[gene_id]

    def _get_sample_name(self, store: zarr.Group, zarr_path: str) -> str:
        cached = self._sample_name_cache.get(zarr_path)
        if cached is not None:
            return cached
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
        if self._explicit_pairs is not None:
            zarr_path, gene_id = self._explicit_pairs[idx]
            store = self._get_store(zarr_path)
            data_grp = store['flat'] if 'flat' in store else store
            gene_idx = self._resolve_gene_index(store, zarr_path, gene_id)
        else:
            assert self._sample_paths is not None
            assert self._cum_gene_offsets is not None
            # Find which sample this gene index belongs to
            sample_i = bisect.bisect_right(self._cum_gene_offsets, idx) - 1
            zarr_path = self._sample_paths[sample_i]
            gene_idx = idx - self._cum_gene_offsets[sample_i]
            store = self._get_store(zarr_path)
            data_grp = store['flat'] if 'flat' in store else store

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


class ProteusGeneChunkDataset(Dataset):
    """Chunked gene Dataset backed by flat per-sample Zarr arrays.

    Each item is a *contiguous slice* of genes from one sample, e.g. [start:end).
    This is much friendlier to Zarr chunking/compression than single-gene random reads.

    Intended usage:
      - DataLoader shuffle=False
      - A deterministic, pre-shuffled chunk index is generated once and saved next to
        the fold index file (so training order is reproducible across runs).
      - Collate concatenates chunks and (optionally) shuffles genes in-memory.

    Notes:
      - Supports v5 fold indices (sample-path list). Legacy explicit (path,gene)
        indices are not supported in chunk mode.
    """

    def __init__(
        self,
        index_file: str,
        chunk_size: int = 256,
        seed: int = 42,
        labeled: bool = True,
        cache_stores: bool = True,
        include_metadata: bool = False,
    ):
        self.index_file = index_file
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.labeled = labeled
        self.cache_stores = cache_stores
        self.include_metadata = bool(include_metadata)
        self._store_cache: Dict[str, zarr.Group] = {}
        self._gene_names_cache: Dict[str, List[str]] = {}
        self._sample_name_cache: Dict[str, str] = {}

        with open(index_file, 'r') as f:
            loaded = json.load(f)

        # Accept both v5 sample-path lists and legacy explicit (path,gene) lists.
        self.sample_paths = _normalize_fold_index_to_sample_paths(loaded)

        if cache_stores:
            for path in set(self.sample_paths):
                self._store_cache[path] = zarr.open(path, mode='r')

        # Build or load a deterministic chunk index.
        cache_path = f"{index_file}.chunks_{self.chunk_size}_seed{self.seed}.json"
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self._chunks: List[Tuple[str, int, int]] = [
                    (str(p), int(s), int(e)) for (p, s, e) in json.load(f)
                ]
        else:
            chunks: List[Tuple[str, int, int]] = []
            for path in self.sample_paths:
                store = self._get_store(path)
                data_grp = store['flat'] if 'flat' in store else store
                n_genes = int(data_grp['TSS_signals'].shape[0])
                for start in range(0, n_genes, self.chunk_size):
                    end = min(start + self.chunk_size, n_genes)
                    chunks.append((path, start, end))

            rng = random.Random(self.seed)
            rng.shuffle(chunks)
            self._chunks = chunks

            # Persist for reproducibility + faster startup next time.
            with open(cache_path, 'w') as f:
                json.dump(self._chunks, f)

    def __len__(self) -> int:
        return len(self._chunks)

    def _get_store(self, zarr_path: str) -> zarr.Group:
        if self.cache_stores and zarr_path in self._store_cache:
            return self._store_cache[zarr_path]
        return zarr.open(zarr_path, mode='r')

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        zarr_path, start, end = self._chunks[idx]
        store = self._get_store(zarr_path)
        data_grp = store['flat'] if 'flat' in store else store

        # Slice contiguous genes [start:end)
        features = {
            'TSS_signals': torch.from_numpy(data_grp['TSS_signals'][start:end]),
            'TSS_nucleotides': torch.from_numpy(data_grp['TSS_nucleotides'][start:end].astype(np.float32)),
            'TSS_features': torch.from_numpy(data_grp['TSS_features'][start:end]),
            'GB_features': torch.from_numpy(data_grp['GB_features'][start:end]),
            'ref_scalars': torch.from_numpy(data_grp['ref_scalars'][start:end]),
        }

        tfx = torch.from_numpy(data_grp['TFX_label'][start:end]).to(torch.float32)

        if self.include_metadata:
            sample_name = store.attrs.get('sample_name')
            if sample_name is None:
                sample_name = Path(zarr_path).stem.replace('.zarr', '')
            sample_name = str(sample_name)
            gene_names = self._gene_names_cache.get(zarr_path)
            if gene_names is None:
                gene_names = [str(g) for g in list(store['gene_names'][:])]
                self._gene_names_cache[zarr_path] = gene_names
            features['sample_name'] = [sample_name] * int(end - start)
            features['gene_name'] = [str(g) for g in gene_names[int(start):int(end)]]

        if 'gene_weight' in data_grp:
            gene_weight = torch.from_numpy(data_grp['gene_weight'][start:end]).to(torch.float32)
        else:
            gene_weight = torch.ones(end - start, dtype=torch.float32)

        if self.labeled:
            tumor_exp = torch.from_numpy(data_grp['tumor_exp'][start:end]).to(torch.float32)
            background_exp = torch.from_numpy(data_grp['background_exp'][start:end]).to(torch.float32)
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
                data_grp = store['flat'] if 'flat' in store else store
                n_genes = int(data_grp['TSS_signals'].shape[0])
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

        # Compute how many fixed-size microbatches we will yield.
        self._n_batches = 0
        for _path, start, end in self._chunks:
            self._n_batches += max(0, int((int(end) - int(start)) // self.genes_per_batch))

        # These caches live per-process (per DataLoader worker).
        self._store_cache: Dict[str, zarr.Group] = {}
        self._gene_names_cache: Dict[str, List[str]] = {}

    def __len__(self) -> int:
        return int(self._n_batches)

    def _get_store(self, zarr_path: str) -> zarr.Group:
        if self.cache_stores:
            store = self._store_cache.get(zarr_path)
            if store is None:
                store = zarr.open(zarr_path, mode='r')
                self._store_cache[zarr_path] = store
            return store
        return zarr.open(zarr_path, mode='r')

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

        # Deterministic sharding of chunks across workers.
        for chunk_idx in range(worker_id, len(self._chunks), num_workers):
            zarr_path, start, end = self._chunks[chunk_idx]
            store = self._get_store(zarr_path)
            data_grp = store['flat'] if 'flat' in store else store

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
                sample_name = store.attrs.get('sample_name')
                if sample_name is None:
                    sample_name = Path(zarr_path).stem.replace('.zarr', '')
                sample_name = str(sample_name)
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


def collate_proteus_samples(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    shuffle_genes: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Collate a batch of complete samples into a gene-level batch.

    - Concatenates genes across samples (dim=0).
    - Optionally shuffles genes across the combined batch.
    """
    features_list, labels_list = zip(*batch)

    # Concatenate along gene axis
    batched_features = {
        'TSS_signals': torch.cat([f['TSS_signals'] for f in features_list], dim=0),
        'TSS_nucleotides': torch.cat([f['TSS_nucleotides'] for f in features_list], dim=0),
        'TSS_features': torch.cat([f['TSS_features'] for f in features_list], dim=0),
        'GB_features': torch.cat([f['GB_features'] for f in features_list], dim=0),
        'ref_scalars': torch.cat([f['ref_scalars'] for f in features_list], dim=0),
    }

    # Optional metadata passthrough (lists of strings)
    if features_list and 'sample_name' in features_list[0]:
        batched_features['sample_name'] = sum([list(f['sample_name']) if isinstance(f['sample_name'], (list, tuple)) else [f['sample_name']] for f in features_list], [])
    if features_list and 'gene_name' in features_list[0]:
        batched_features['gene_name'] = sum([list(f['gene_name']) if isinstance(f['gene_name'], (list, tuple)) else [f['gene_name']] for f in features_list], [])

    # Labels
    if labels_list and 'tumor_exp' in labels_list[0]:
        batched_labels = {
            'tumor_exp': torch.cat([l['tumor_exp'] for l in labels_list], dim=0).unsqueeze(-1),
            'background_exp': torch.cat([l['background_exp'] for l in labels_list], dim=0).unsqueeze(-1),
            'sample_weight': torch.cat([l.get('sample_weight', torch.ones_like(l['tumor_exp'])) for l in labels_list], dim=0).unsqueeze(-1),
            'TFX_label': torch.cat([l['TFX_label'] for l in labels_list], dim=0).unsqueeze(-1),
        }
    else:
        # Unlabeled path (inference-style training not typical, but keep consistent)
        batched_labels = {
            'sample_weight': torch.cat([l.get('sample_weight', torch.ones(l['TFX_label'].shape[0], dtype=torch.float32)) for l in labels_list], dim=0).unsqueeze(-1),
            'TFX_label': torch.cat([l['TFX_label'] for l in labels_list], dim=0).unsqueeze(-1),
        }

    if shuffle_genes:
        n = batched_features['TSS_signals'].shape[0]
        perm = torch.randperm(n, generator=generator)
        for k, v in batched_features.items():
            if isinstance(v, torch.Tensor):
                batched_features[k] = v[perm]
            elif isinstance(v, list):
                order = perm.tolist()
                batched_features[k] = [v[i] for i in order]
        for k, v in batched_labels.items():
            batched_labels[k] = v[perm]

    return batched_features, batched_labels


class CollateProteusChunks:
    """Pickle-friendly collate for chunked gene loading.

    Concatenates chunk items along the gene axis, optionally shuffles genes
    in-memory, then (optionally) truncates to a fixed genes-per-step size.
    """

    def __init__(
        self,
        shuffle_genes: bool = True,
        seed: Optional[int] = None,
    ):
        self.shuffle_genes = bool(shuffle_genes)
        self.seed = int(seed) if seed is not None else None
        self._rng: Optional[torch.Generator] = None

    def __call__(
        self,
        batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if self._rng is None:
            # In worker processes, torch.initial_seed() is deterministically derived
            # from the DataLoader's base generator seed.
            worker_info = torch.utils.data.get_worker_info()
            self._rng = torch.Generator()
            if worker_info is None and self.seed is not None:
                self._rng.manual_seed(self.seed)
            else:
                self._rng.manual_seed(torch.initial_seed())

        features, labels = collate_proteus_samples(
            batch,
            shuffle_genes=self.shuffle_genes,
            generator=self._rng,
        )
        return features, labels


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


# Backwards-compatible defaults (model trains per-gene)
ProteusDataset = ProteusGeneDataset
collate_proteus = collate_proteus_genes


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
    )
    
    return loader, dataset.gene_names


def get_fold_indices(
    use_set: str,
    data_dir: str = 'data'
) -> Dict[int, str]:
    """
    Get paths to fold index files for a dataset.
    
    Args:
        use_set: Dataset name
        data_dir: Base data directory
        
    Returns:
        Dictionary mapping fold numbers to index file paths
    """
    index_dir = os.path.join(data_dir, use_set, 'fold_indices')
    
    if not os.path.exists(index_dir):
        raise FileNotFoundError(f"Fold indices not found at {index_dir}. Run create_fold_index first.")
    
    fold_files = {}
    for f in os.listdir(index_dir):
        if f.startswith('fold_') and f.endswith('_index.json'):
            fold_num = int(f.split('_')[1])
            fold_files[fold_num] = os.path.join(index_dir, f)
    
    return fold_files


def build_train_val_loaders(
    use_set: str,
    val_fold: int,
    num_workers: int = 4,
    data_dir: str = 'data',
    loader_mode: str = 'gene',
    chunk_size: int = 256,
    genes_per_batch: int = 32,
    shuffle_genes_in_batch: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Build training and validation DataLoaders for cross-validation.
    
    Args:
        use_set: Dataset name
        val_fold: Fold number to use for validation
        num_workers: Number of workers
        data_dir: Base data directory
        
    Returns:
        train_loader, val_loader, train_steps, val_steps
    """
    fold_files = get_fold_indices(use_set, data_dir)
    
    # Combine non-validation folds for training
    train_indices: List[str] = []
    for fold_num, index_file in fold_files.items():
        if fold_num != val_fold:
            with open(index_file, 'r') as f:
                train_indices.extend(_normalize_fold_index_to_sample_paths(json.load(f)))

    # Deduplicate (folds may overlap in legacy formats)
    train_indices = list(dict.fromkeys(train_indices))
    
    # Shuffle training indices deterministically (sample-path list order)
    rng = np.random.RandomState(seed)
    rng.shuffle(train_indices)
    
    # Create temporary index file for training
    train_index_file = os.path.join(data_dir, use_set, 'fold_indices', f'train_excl_{val_fold}_temp.json')
    with open(train_index_file, 'w') as f:
        json.dump(train_indices, f)
    
    train_loader, train_steps = create_data_loader(
        index_file=train_index_file,
        num_workers=num_workers,
        shuffle=(str(loader_mode).lower() == 'gene'),
        labeled=True
        ,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        loader_mode=loader_mode,
        chunk_size=chunk_size,
        genes_per_batch=genes_per_batch,
        shuffle_genes_in_batch=shuffle_genes_in_batch,
        seed=seed,
        drop_last=(str(loader_mode).lower() == 'gene'),
    )

    # Normalize validation fold index to a v5-style sample-path list (chunk loader requirement).
    with open(fold_files[val_fold], 'r') as f:
        val_indices = _normalize_fold_index_to_sample_paths(json.load(f))
    val_index_file = os.path.join(data_dir, use_set, 'fold_indices', f'val_fold_{val_fold}_temp.json')
    with open(val_index_file, 'w') as f:
        json.dump(val_indices, f)
    
    val_loader, val_steps = create_data_loader(
        index_file=val_index_file,
        num_workers=num_workers,
        shuffle=False,
        labeled=True
        ,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        loader_mode=loader_mode,
        chunk_size=chunk_size,
        genes_per_batch=genes_per_batch,
        shuffle_genes_in_batch=False,
        seed=seed,
        include_metadata=True,
        drop_last=False,
    )
    
    return train_loader, val_loader, train_steps, val_steps


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

            # Strip cohort prefix from sample name if present (e.g., "CRPC-RA_00-147" -> "00-147")
            sample_name = sample
            if sample.startswith(f"{cohort}_"):
                sample_name = sample[len(cohort) + 1:]

            zarr_path = os.path.join(zarr_base, cohort, 'zarr_files', f"{sample_name}.zarr")
            if not os.path.isdir(zarr_path):
                missing_samples.append(f"  {cohort}/zarr_files/{sample_name}.zarr  (Fold={fold}, TSV sample={sample})")
                continue

            if fold == val_fold:
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

