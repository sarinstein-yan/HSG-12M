import numpy as np
import torch
from typing import Iterator, List, Sequence


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
        self._rng.manual_seed(int(seed))
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
            usable = len(idxs) - (len(idxs) % world_size)
            if usable <= 0:
                usable = len(idxs)  # don't drop everything if batches < world_size
            idxs = idxs[:usable]

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
            usable = n - (n % world_size) if self.ensure_equal_batch_counts else n
            return max(int(usable / world_size), 0)
        return n

    # # Optional: allow trainers/callbacks to vary batch order per epoch deterministically
    # def set_epoch(self, epoch: int):
    #     # Mix seed with epoch for reproducible reshuffles
    #     base = int(self._rng.initial_seed())
    #     self._rng.manual_seed(base ^ (epoch + 2025))