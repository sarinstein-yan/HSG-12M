import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Dict

import networkx as nx
import numpy as np
import torch
from joblib import Parallel, delayed
from torch_geometric.data import OnDiskDataset, InMemoryDataset


class HSGOnDisk(OnDiskDataset):
    """HSG dataset stored in an on‑disk database with *dense* labels.

    Unlike the original implementation, the label remapping (old scattered
    class‑IDs → dense **0‥C‑1**) is handled *here* during processing so that
    every downstream consumer – including
    :class:`HSGInMemory` – can just load graphs without any
    further bookkeeping.
    """

    #: Total number of *raw* class files that may be present
    NUM_CLASSES: int = 1401

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __init__(
        self,
        root: str,
        subset: Optional[str] = "one-band",
        *,
        n_jobs: int = -1,
        backend: str = "sqlite",
        transform=None,
        pre_filter=None,
    ) -> None:
        self._raw_root = Path(root).expanduser().resolve()
        self.subset = (subset or "one-band").lower()
        if self.subset in {"all", "all-static"}:
            self.subset = "all"
        self.n_jobs = n_jobs

        # processed data live in a subset‑specific folder so we can build
        # several variants side‑by‑side
        processed_root = self._raw_root / f"processed_{self.subset}"

        super().__init__(
            root=str(processed_root),
            transform=transform,
            pre_filter=pre_filter,
            backend=backend,
        )

    # ------------------------------------------------------------------
    # torch_geometric.Dataset hooks
    # ------------------------------------------------------------------
    @property
    def raw_file_names(self) -> List[str]:  # type: ignore[override]
        files = [
            "ParamTable.npz",
            "HSG-topology-mask.pkl",
            *[f"raw/class_{i}.npz" for i in range(self.NUM_CLASSES)],
        ]
        return files

    # processed_file_names is inherited from OnDiskDataset ("<backend>.db")

    # ------------------------------------------------------------------
    # Label mapping helpers
    # ------------------------------------------------------------------
    _label_map: Dict[int, int] = {}  # class‑level cache (subset‑specific)

    @classmethod
    def _set_label_map(cls, mapping: Dict[int, int]) -> None:
        """Store the *dense* label map on the class so that the (parallel)
        helpers can access it without serialising the whole instance.
        """
        cls._label_map = mapping

    # ------------------------------------------------------------------
    # Graph attribute stripping
    # ------------------------------------------------------------------
    @staticmethod
    def _process_edge_pts(graph: nx.MultiGraph) -> nx.MultiGraph:
        """Return a *new* graph with compact node / edge attributes.

        • node attribute ``x``       → np.float32, shape = (4,)
        • edge attribute ``edge_attr`` → np.float32, shape = (13,)
        """
        g = graph.copy()
        _PTS_IDXS = np.arange(1, 6)

        node_pos, node_pot, node_dos = {}, {}, {}
        for nid, nd in g.nodes(data=True):
            pos = np.asarray(nd.pop("pos"), dtype=np.float32).reshape(-1)  # (2,)
            pot = np.float32(nd.pop("potential", 0.0))
            dos = np.float32(nd.pop("dos", 0.0))
            nd["x"] = np.array([*pos, pot, dos], dtype=np.float32)
            node_pos[nid], node_pot[nid], node_dos[nid] = pos, pot, dos

        for u, v, ed in g.edges(data=True):
            weight = np.float32(ed.pop("weight"))
            if "pts" in ed:  # common case
                pts = ed.pop("pts")
                idx = np.round(_PTS_IDXS * (len(pts) - 1) / 6).astype(int)
                pts5 = pts[idx].astype(np.float32).reshape(-1)
                avg_pot = np.float32(ed.pop("avg_potential", 0.5 * (node_pot[u] + node_pot[v])))
                avg_dos = np.float32(ed.pop("avg_dos", 0.5 * (node_dos[u] + node_dos[v])))
            else:  # fallback – straight line
                mid_xy = 0.5 * (node_pos[u] + node_pos[v])
                pts5 = np.tile(mid_xy, 5).astype(np.float32)
                avg_pot = np.float32(0.5 * (node_pot[u] + node_pot[v]))
                avg_dos = np.float32(0.5 * (node_dos[u] + node_dos[v]))

            ed["edge_attr"] = np.concatenate(([weight, avg_pot, avg_dos], pts5), dtype=np.float32)
        return g

    @classmethod
    def _graph_to_data(cls, g: nx.MultiGraph, class_id: int) -> "torch_geometric.data.Data":
        """Convert NetworkX → PyG ``Data`` with *dense* label already applied."""
        import torch_geometric
        from torch_geometric.utils import from_networkx

        dense_label = cls._label_map[class_id]
        g2 = cls._process_edge_pts(g)
        data = from_networkx(g2, group_node_attrs=["x"], group_edge_attrs=["edge_attr"])
        data.y = torch.tensor([dense_label], dtype=torch.long)
        return data

    # ------------------------------------------------------------------
    # Subset selection
    # ------------------------------------------------------------------
    def _select_class_ids(self, metas: np.ndarray) -> np.ndarray:
        """Return the list of *class IDs* to process for the chosen subset."""
        band_counts = np.array([m["number_of_bands"] for m in metas])

        match self.subset:
            case "one-band" | "none":
                return np.where(band_counts == 1)[0]
            case "two-band":
                return np.where(band_counts == 2)[0]
            case "three-band":
                return np.where(band_counts == 3)[0]
            case "all":
                return np.arange(len(metas))
            case "topology":
                return np.arange(len(metas))  # mask applied later
        raise ValueError(f"Unknown subset '{self.subset}'.")

    # ------------------------------------------------------------------
    # Heavy lifting – executed *once* per subset when the DB is absent
    # ------------------------------------------------------------------
    def process(self) -> None:  # type: ignore[override]
        """Convert raw *.npz* files into a serialised on‑disk DB *with dense labels*."""
        from tqdm import tqdm

        # --------------------------------------------------------------
        # 1) global metadata & subset filtering
        # --------------------------------------------------------------
        param = np.load(self._raw_root / "ParamTable.npz", allow_pickle=True)
        metas = param["metas"]
        class_ids = self._select_class_ids(metas)

        # dense label map *for this subset*
        label_map = {int(old): new for new, old in enumerate(sorted(map(int, class_ids)))}
        self._set_label_map(label_map)

        # optional topology masks (only for the "topology" subset)
        topo_masks: Optional[Sequence[Sequence[int]]] = None
        if self.subset == "topology":
            topo_masks = pickle.load(open(self._raw_root / "HSG-topology-mask.pkl", "rb"))

        # ensure DB exists before we start parallel inserts
        _ = self.db

        # --------------------------------------------------------------
        threader = Parallel(n_jobs=self.n_jobs, prefer="threads")
        processor = Parallel(n_jobs=self.n_jobs, backend="loky")
        total_graphs = 0

        for cid in tqdm(class_ids, desc="Processing classes"):
            class_npz = self._raw_root / "raw" / f"class_{int(cid)}.npz"
            arr = np.load(class_npz, allow_pickle=True)
            pickled_graphs = arr["graphs_pickle"]

            # unpickle all graphs in parallel (I/O‑bound → threads)
            nx_graphs: List[nx.MultiGraph] = threader(delayed(pickle.loads)(buf) for buf in pickled_graphs)

            # optional topology filter (cheap)
            allowed: Optional[set[int]] = None
            if topo_masks is not None:
                allowed = set(map(int, topo_masks[int(cid)]))

            selected_graphs = (
                g for idx, g in enumerate(nx_graphs) if allowed is None or idx in allowed
            )

            # expensive NX → PyG conversion (CPU‑bound → processes)
            data_list = processor(
                delayed(self._graph_to_data)(g, int(cid)) for g in selected_graphs
            )

            # user hooks
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            # bulk insert → DB
            self.extend(data_list, batch_size=1024)
            total_graphs += len(data_list)

        if self.log:
            print(f"[{self.__class__.__name__}] Stored {total_graphs} graphs → {self.processed_paths[0]}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def num_classes(self) -> int:  # type: ignore[override]
        """Number of *dense* classes for *this subset*."""
        return len(self._label_map) if self._label_map else self.NUM_CLASSES


# ======================================================================
#                Memory‑resident variant
# ======================================================================

class HSGInMemory(InMemoryDataset):
    """Load the *dense‑labelled* on‑disk DB into system RAM."""

    def __init__(
        self,
        root: str | Path,
        subset: Optional[str] = "one-band",
        *,
        backend: str = "sqlite",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.subset = (subset or "one-band").lower()
        self.backend = backend
        super().__init__(
            root=Path(root).expanduser().resolve(),
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        # pull the cached, collated tensors into memory
        self.load(self.processed_paths[0])

    # ------------------------------------------------------------------
    # Dataset hooks
    # ------------------------------------------------------------------
    @property
    def processed_file_names(self) -> List[str]:  # type: ignore[override]
        return [f"data_{self.subset}.pt"]

    def process(self) -> None:  # type: ignore[override]
        """Populate the cache by copying *already dense* graphs from disk."""
        on_disk = HSGOnDisk(
            root=str(self.root),
            subset=self.subset,
            backend=self.backend,
            transform=None,
            pre_filter=None,
        )

        data_list = [on_disk.get(i) for i in range(len(on_disk))]

        # user hooks
        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # collate & cache using the convenient helper
        self.save(data_list, self.processed_paths[0])

        # cache num‑classes for quick access
        self._num_classes = on_disk.num_classes

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def num_classes(self) -> int:  # type: ignore[override]
        return getattr(self, "_num_classes", None) or int(self.data.y.max().item() + 1)


if __name__ == "__main__":
    # Test the dataset
    print("Processing one-band subset...")
    ds = HSGInMemory(root="/mnt/ssd/nhsg12m", subset="one-band")
    print("one-band subset processed.")
    print(f"Number of classes: {ds.num_classes}")
    print(f"Number of graphs: {len(ds)}")
    print(f"Raw files: {ds.raw_file_names}")
    print(f"Processed files: {ds.processed_file_names}")

    print("Processing two-band subset...")
    ds = HSGInMemory(root="/mnt/ssd/nhsg12m", subset="two-band")
    print("two-band subset processed.")
    print(f"Number of classes: {ds.num_classes}")
    print(f"Number of graphs: {len(ds)}")
    print(f"Raw files: {ds.raw_file_names}")
    print(f"Processed files: {ds.processed_file_names}")

    print("Processing three-band subset...")
    ds = HSGInMemory(root="/mnt/ssd/nhsg12m", subset="three-band")
    print("three-band subset processed.")
    print(f"Number of classes: {ds.num_classes}")
    print(f"Number of graphs: {len(ds)}")
    print(f"Raw files: {ds.raw_file_names}")
    print(f"Processed files: {ds.processed_file_names}")

    print("Processing all subset...")
    ds = HSGInMemory(root="/mnt/ssd/nhsg12m", subset="all")
    print("all subset processed.")
    print(f"Number of classes: {ds.num_classes}")
    print(f"Number of graphs: {len(ds)}")
    print(f"Raw files: {ds.raw_file_names}")
    print(f"Processed files: {ds.processed_file_names}")

    print("Processing topology subset...")
    ds = HSGOnDisk(root="/mnt/ssd/nhsg12m", subset="topology")
    print("topology subset processed.")
    print(f"Number of classes: {ds.num_classes}")
    print(f"Number of graphs: {len(ds)}")
    print(f"Raw files: {ds.raw_file_names}")
    print(f"Processed files: {ds.processed_file_names}")