'''
Custom Callback for DDP-Safe Monitoring
'''
import time
import torch
import pytorch_lightning as pl


class DDPMonitorCallback(pl.Callback):
    """Logs: throughput_graph_per_sec, peak_mem_mb_sum_ranks_max, peak_mem_mb_per_graph.
       Exposes properties: avg_throughput, peak_gpu_mem_sum, peak_gpu_mem_per_graph."""
    def __init__(self):
        super().__init__()
        self._tput_sum = 0.0
        self._epochs = 0
        self._sum_mb_max = 0.0
        self._mb_per_graph_max = 0.0

    @property
    def avg_throughput(self):  # graphs/sec
        return self._tput_sum / self._epochs if self._epochs else 0.0

    @property
    def peak_gpu_mem_sum(self):  # MB
        return self._sum_mb_max

    @property
    def peak_gpu_mem_per_graph(self):  # MB/graph
        return self._mb_per_graph_max

    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            idx = getattr(pl_module.device, "index", torch.cuda.current_device())
            torch.cuda.reset_peak_memory_stats(idx)
        self.t0 = time.monotonic()
        self.graphs_epoch = 0
        self.max_graphs_batch = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = getattr(batch, "num_graphs", None)
        if n is None:
            try:
                n = sum(getattr(b, "num_graphs", 1) for b in batch)
            except Exception:
                n = 1
        n = int(n)
        self.graphs_epoch += n
        if n > self.max_graphs_batch:
            self.max_graphs_batch = n

    def on_train_epoch_end(self, trainer, pl_module):
        dt = max(time.monotonic() - self.t0, 1e-9)

        g_local = torch.tensor(self.graphs_epoch, device=pl_module.device, dtype=torch.float32)
        g_all = pl_module.all_gather(g_local)

        peak_mb = 0.0
        if torch.cuda.is_available():
            idx = getattr(pl_module.device, "index", torch.cuda.current_device())
            peak_mb = torch.cuda.max_memory_allocated(idx) / (1024.0**2)
        mb_local = torch.tensor(float(peak_mb), device=pl_module.device, dtype=torch.float32)
        mb_all = pl_module.all_gather(mb_local)

        bsz_local = torch.tensor(max(1, self.max_graphs_batch), device=pl_module.device, dtype=torch.float32)
        mb_per_graph_local = mb_local / bsz_local
        mb_per_graph_all = pl_module.all_gather(mb_per_graph_local)

        if trainer.is_global_zero:
            throughput_graph_per_sec = g_all.sum().item() / dt
            peak_mem_mb_sum_ranks = mb_all.sum().item()
            peak_mem_mb_sum_ranks_max = max(self._sum_mb_max, peak_mem_mb_sum_ranks)
            peak_mem_mb_per_graph = mb_per_graph_all.max().item()

            # accumulate / update for export
            self._tput_sum += throughput_graph_per_sec
            self._epochs += 1
            self._sum_mb_max = peak_mem_mb_sum_ranks_max
            self._mb_per_graph_max = max(self._mb_per_graph_max, peak_mem_mb_per_graph)

            # logs
            pl_module.log("perf/throughput_graph_per_sec", throughput_graph_per_sec, rank_zero_only=True)
            pl_module.log("perf/peak_mem_mb_sum_ranks_max", peak_mem_mb_sum_ranks_max, rank_zero_only=True)
            pl_module.log("perf/peak_mem_mb_per_graph", peak_mem_mb_per_graph, rank_zero_only=True)