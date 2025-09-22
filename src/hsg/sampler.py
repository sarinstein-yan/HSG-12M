import numpy as np
import heapq
import torch
from typing import Iterator, List, Sequence, Optional


def rebalance_batch(
    sample_sizes: np.ndarray, 
    batch_size: int, 
    max_num_per_batch: int
) -> Optional[np.ndarray]:
    """Rebalance the batch by permuting the sample sizes.

    Parameters
    ----------
    sample_sizes : np.ndarray
        1D array of sample sizes (e.g., num_nodes or num_edges)
    batch_size : int
        Number of samples per batch
    max_sum_per_batch : int
        Maximum sum of sample sizes per batch. Recommended to set to 80% of the
        maximum that your hardware can handle.

    Returns
    -------
    Optional[np.ndarray]
        1D array of indices representing the new order of samples, 
        or None if rebalancing is not possible
    """    
    if not isinstance(sample_sizes, np.ndarray) or sample_sizes.ndim != 1:
        raise TypeError("arr must be 1D NumPy array")
    if batch_size <= 0 or max_num_per_batch <= 0:
        raise ValueError("batch_size and max_sum_per_batch must be positive")
    if sample_sizes.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(sample_sizes > max_num_per_batch):
        return None

    n = sample_sizes.size
    m = (n + batch_size - 1) // batch_size  # num batches
    # batch capacities: last batch may be partial
    caps = np.full(m, batch_size, dtype=np.int32)
    r = n % batch_size
    if r != 0:
        caps[-1] = r

    # Sort once in C: indices descending by value
    order = np.argsort(sample_sizes, kind="stable")[::-1]
    values = sample_sizes[order]

    # Heap of (current_sum, batch_idx)
    batch_sum = np.zeros(m, dtype=np.int64)
    batch_rem = caps.copy()
    heap = [(0, i) for i in range(m)]
    heapq.heapify(heap)

    # Preallocate output placement
    # Compute fixed starting offsets for each batch:
    # e.g., batch 0 starts at 0, batch 1 at caps[0], etc.
    start = np.zeros(m, dtype=np.int64)
    np.cumsum(caps[:-1], out=start[1:])
    fill = np.zeros(m, dtype=np.int64)
    out = np.empty(n, dtype=np.int64)

    rejected = []  # reuse list to avoid reallocs

    for k in range(n):
        idx = order[k]
        x = int(values[k])

        placed = False
        # Try batches, starting from smallest sum
        while heap:
            s, b = heap[0]  # peek
            if batch_rem[b] == 0:
                heapq.heappop(heap)          # permanently discard full batch
                continue
            if s + x > max_num_per_batch:
                # Temporarily move it aside while we try others
                heapq.heappop(heap)
                rejected.append((s, b))
                continue

            # Place item
            heapq.heappop(heap)
            pos = start[b] + fill[b]
            out[pos] = idx
            fill[b] += 1
            batch_rem[b] -= 1
            new_sum = s + x
            batch_sum[b] = new_sum
            # If capacity remains, push back; else never push again
            if batch_rem[b] > 0:
                heapq.heappush(heap, (int(new_sum), b))
            placed = True
            break

        # Restore rejected
        if rejected:
            for item in rejected:
                heapq.heappush(heap, item)
            rejected.clear()

        if not placed:
            return None

    return out


class StaticBatchSampler(torch.utils.data.Sampler[List[int]]):
    r"""Precomputes fixed batches so that the sum of per-sample "sizes"
    (num_nodes or num_edges) in every batch is <= max_num.
    """
    def __init__(
        self,
        dataset,
        sizes: Sequence[int],
        max_num: int,
        *,
        skip_too_big: bool = True,
        drop_last: bool = False,
        pack_strategy: str = "sequential",
        shuffle_batches_each_epoch: bool = False,
        seed: int = 0,
        dist_shard: bool = True,
        ensure_equal_batch_counts: bool = True,
    ):
        if max_num <= 0:
            raise ValueError(f"`max_num` must be > 0 (got {max_num}).")
        if len(sizes) != len(dataset):
            # Handle case where dataset might be empty (e.g. placeholder used)
            if len(dataset) == 0:
                sizes = []
            else:
                raise ValueError("`sizes` must align with the dataset length.")

        self.dataset = dataset
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.max_num = int(max_num)
        self.skip_too_big = bool(skip_too_big)
        self.drop_last = bool(drop_last)
        self.pack_strategy = pack_strategy
        self.shuffle_batches_each_epoch = bool(shuffle_batches_each_epoch)
        self._rng = torch.Generator()
        self._base_seed = int(seed)
        self._rng.manual_seed(self._base_seed)
        self.dist_shard = bool(dist_shard)
        self.ensure_equal_batch_counts = bool(ensure_equal_batch_counts)

        # Precompute fixed batch membership (list[list[int]])
        self._batches: List[List[int]] = self._prepack()

    def _world_info(self) -> tuple[int, int]:
        # (rank, world_size)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _prepack(self) -> List[List[int]]:
        N = len(self.dataset)
        order = list(range(N))
        if self.pack_strategy == "sorted_desc":
            order.sort(key=lambda i: int(self.sizes[i]), reverse=True)
        elif self.pack_strategy != "sequential":
            raise ValueError(f"Unknown pack_strategy: {self.pack_strategy}")

        batches: List[List[int]] = []
        cur: List[int] = []
        cur_total = 0

        for i in order:
            s = int(self.sizes[i])

            if s > self.max_num:
                if self.skip_too_big:
                    # just skip this sample altogether
                    continue
                else:
                    # finalize current first
                    if cur:
                        batches.append(cur)
                        cur, cur_total = [], 0
                    # emit an oversized singleton (may OOM; not recommended)
                    batches.append([i])
                    continue

            if cur_total + s > self.max_num:
                # finalize current batch and start new
                if cur:
                    batches.append(cur)
                cur, cur_total = [i], s
            else:
                cur.append(i)
                cur_total += s

        if cur and not self.drop_last:
            batches.append(cur)

        # Convert from packing order back to dataset-relative indices.
        # (They're already dataset-relative; nothing more to do.)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        idxs = list(range(len(self._batches)))
        if self.shuffle_batches_each_epoch and len(self._batches) > 1:
            idxs = torch.randperm(len(self._batches), generator=self._rng).tolist()

        rank, world_size = self._world_info()
        if self.dist_shard and self.ensure_equal_batch_counts and world_size > 1:
            # Pad to a multiple of world_size so all ranks see the same #steps.
            if len(idxs) > 0:
                pad = (-len(idxs)) % world_size
                if pad:
                    idxs.extend([idxs[-1]] * pad)  # repeat last batch id
            # if len(idxs) == 0: keep empty

        # Manual DDP Sharding
        for local_j, j in enumerate(idxs):
            if self.dist_shard:
                # Use the rank/world_size obtained earlier
                if world_size > 1 and (local_j % world_size) != rank:
                    continue
            yield self._batches[j]

    def __len__(self) -> int:
        n = len(self._batches)
        _, world_size = self._world_info()
        if self.dist_shard and world_size > 1:
            # With padding in __iter__, each rank gets ceil(n/world_size).
            # If n == 0, return 0 to avoid a phantom step.
            if self.ensure_equal_batch_counts:
                return 0 if n == 0 else (n + world_size - 1) // world_size
            # If you explicitly disable equal batch counts, we still report ceil
            # to avoid under-reporting. Prefer leaving ensure_equal_batch_counts=True for train.
            return 0 if n == 0 else (n + world_size - 1) // world_size
        return n

    # Allow trainers (e.g., PyTorch Lightning) to reshuffle deterministically each epoch.
    # Lightning will call this if present.
    def set_epoch(self, epoch: int):
        # Mix epoch into the base seed (64-bit golden ratio constant for diffusion).
        mix = (epoch * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        self._rng.manual_seed(self._base_seed ^ mix)