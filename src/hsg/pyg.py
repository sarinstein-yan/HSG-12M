from __future__ import annotations

"""PyG **On‑disk** dataset wrapper for the 12‑million‑graph *HSG* corpus.

Highlights
==========
* **Subset‑aware** – choose ``one‑band``, ``two‑band``, ``three‑band``, ``all`` or
  ``topology`` and only the relevant raw files are downloaded and processed.
* **Lazy metadata bootstrap** – on first construction we fetch the two tiny
  metadata assets (*HSG-generator-meta.npz* + *HSG‑topology‑mask.pkl*) **before** PyG
  inspects :pyattr:`raw_file_names`.  This lets us discover which of the
  ``class_XXX.npz`` blobs are actually required.
* **Dynamic `raw_file_names`** – the property now lists *only* the files needed
  for the chosen subset so PyG never complains about missing placeholders.
* **Subset‑scoped root folder** – each subset lives in its own sub‑directory
  (``<root>/<subset>/``) so you can keep multiple variants side‑by‑side without
  collisions.

Usage
-----
```python
from hsg_dataset import HSGOnDisk

dset = HSGOnDisk("~/data/HSG", subset="two-band", n_jobs=8)
print(len(dset), "graphs", dset.num_classes, "classes")
```
"""

import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import networkx as nx
import numpy as np
import pickle
import requests
from urllib.request import urlretrieve
import torch
from joblib import Parallel, delayed
from tqdm import tqdm
from torch_geometric.data import OnDiskDataset, InMemoryDataset

# ---------------------------------------------------------------------------
# Helper utilities (stand‑alone – no torch_geometric imports here)
# ---------------------------------------------------------------------------

def _download_url(url: str, dst: Path, *, desc: str = "", overwrite: bool = False) -> None:
    """Stream *url* → *dst* with a tqdm progress‑bar."""
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with tqdm.wrapattr(r.raw, "read", total=total, desc=desc or dst.name) as raw:
            with open(dst, "wb") as f:
                shutil.copyfileobj(raw, f)


def _dataverse_file_map() -> Dict[str, int]:
    """Return *filename → file‑ID* mapping for the Harvard Dataverse record."""
    api = (
        "https://dataverse.harvard.edu/api/datasets/:persistentId"
        "?persistentId=doi:10.7910/DVN/PYDSSQ"
    )
    js = requests.get(api, timeout=30).json()
    files = js["data"]["latestVersion"]["files"]
    return {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in files}


# ---------------------------------------------------------------------------
# Main dataset class
# ---------------------------------------------------------------------------
class HSGOnDisk(OnDiskDataset):
    """On‑disk PyG wrapper around the HSG‑12M corpus (1401 classes)."""

    #: total count of *possible* raw class archives on Dataverse
    TOTAL_NUM_CLASSES: int = 1401

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(
        self,
        root: str | Path,
        subset: str | None = "one-band",
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

        # ------------------------------------------------------------------
        # 0) make sure the two *tiny* metadata files exist locally
        # ------------------------------------------------------------------
        GH   = "https://raw.githubusercontent.com/sarinstein-yan/HSG-12M/main/assets"
        meta_files = ["HSG-generator-meta.npz", "HSG-topology-mask.pkl"]
        for fname in meta_files:
            fpath = self._raw_root / fname
            if not fpath.exists():
                fpath.parent.mkdir(parents=True, exist_ok=True)
                print(f"[HSG] downloading {fname} …")
                urlretrieve(f"{GH}/{fname}", fpath)

        # ------------------------------------------------------------------
        # 1) parse HSG-generator-meta → decide which class_*.npz files we need
        # ------------------------------------------------------------------
        param = np.load(self._raw_root / "HSG-generator-meta.npz", allow_pickle=True)
        metas = param["metas"]

        # *this* is the call you wanted – identical logic to _select_class_ids
        self._required_cids = self._select_class_ids(metas)

        # Build the raw-file list for this subset
        self._raw_file_names_subset = [
            "HSG-generator-meta.npz",
            "HSG-topology-mask.pkl",
            *[f"raw/class_{int(cid)}.npz" for cid in self._required_cids],
        ]

        # processed data live in a subset-specific folder
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
        """Only the files needed for *this* subset."""
        return self._raw_file_names_subset

    # processed_file_names is inherited from OnDiskDataset ("<backend>.db")

    # ------------------------------------------------------------------
    # Label mapping helpers (shared across workers)
    # ------------------------------------------------------------------
    _label_map: Dict[int, int] = {}

    @classmethod
    def _set_label_map(cls, mapping: Dict[int, int]) -> None:
        cls._label_map = mapping

    # ------------------------------------------------------------------
    # Graph attribute stripping
    # ------------------------------------------------------------------
    @staticmethod
    def _process_edge_pts(graph: nx.MultiGraph) -> nx.MultiGraph:
        """Return a *new* graph with compact node / edge attributes."""
        g = graph.copy()
        _IDX = np.arange(1, 6)

        node_pos, node_pot, node_dos = {}, {}, {}
        for nid, nd in g.nodes(data=True):
            pos = np.asarray(nd.pop("pos"), dtype=np.float32).reshape(-1)
            pot = np.float32(nd.pop("potential", 0.0))
            dos = np.float32(nd.pop("dos", 0.0))
            nd["x"] = np.array([*pos, pot, dos], dtype=np.float32)
            node_pos[nid], node_pot[nid], node_dos[nid] = pos, pot, dos

        for u, v, ed in g.edges(data=True):
            w = np.float32(ed.pop("weight"))
            if "pts" in ed:
                pts = ed.pop("pts")
                idx = np.round(_IDX * (len(pts) - 1) / 6).astype(int)
                pts5 = pts[idx].astype(np.float32).reshape(-1)
                avg_pot = np.float32(ed.pop("avg_potential", 0.5 * (node_pot[u] + node_pot[v])))
                avg_dos = np.float32(ed.pop("avg_dos", 0.5 * (node_dos[u] + node_dos[v])))
            else:
                mid = 0.5 * (node_pos[u] + node_pos[v])
                pts5 = np.tile(mid, 5).astype(np.float32)
                avg_pot = np.float32(0.5 * (node_pot[u] + node_pot[v]))
                avg_dos = np.float32(0.5 * (node_dos[u] + node_dos[v]))
            ed["edge_attr"] = np.concatenate(([w, avg_pot, avg_dos], pts5), dtype=np.float32)
        return g

    @classmethod
    def _graph_to_data(cls, g: nx.MultiGraph, class_id: int):
        from torch_geometric.utils import from_networkx

        dense = cls._label_map[class_id]
        g2 = cls._process_edge_pts(g)
        data = from_networkx(g2, group_node_attrs=["x"], group_edge_attrs=["edge_attr"])
        data.y = torch.tensor([dense], dtype=torch.long)
        return data

    # ------------------------------------------------------------------
    # Subset helpers
    # ------------------------------------------------------------------
    def _select_class_ids(self, metas: np.ndarray) -> np.ndarray:
        bands = np.array([m["number_of_bands"] for m in metas])
        match self.subset:
            case "one-band" | "none":
                return np.where(bands == 1)[0]
            case "two-band":
                return np.where(bands == 2)[0]
            case "three-band":
                return np.where(bands == 3)[0]
            case "all":
                return np.arange(len(metas))
            case "topology":
                return np.arange(len(metas))  # mask later
        raise ValueError(f"Unknown subset: {self.subset}")

    # ------------------------------------------------------------------
    # Heavy lifting – executed once per subset when the DB is absent
    # ------------------------------------------------------------------
    def process(self) -> None:  # type: ignore[override]
        """Convert raw *.npz* files → on-disk DB (dense labels)."""
        from tqdm import tqdm

        # --------------------------------------------------------------
        # subset-specific metadata                            (unchanged)
        # --------------------------------------------------------------
        param = np.load(self._raw_root / "HSG-generator-meta.npz", allow_pickle=True)
        metas = param["metas"]

        # we already know the class-ID list from __init__
        class_ids = self._required_cids

        # dense label map *for this subset*
        label_map = {int(old): new for new, old in enumerate(sorted(map(int, class_ids)))}
        self._set_label_map(label_map)

        # optional topology masks
        topo_masks: Optional[Sequence[Sequence[int]]] = None
        if self.subset == "topology":
            topo_masks = pickle.load(open(self._raw_root / "HSG-topology-mask.pkl", "rb"))

        # make sure DB exists before parallel inserts
        _ = self.db

        threader = Parallel(n_jobs=self.n_jobs, prefer="threads")
        worker = Parallel(n_jobs=self.n_jobs, backend="loky")
        total = 0

        for cid in tqdm(self._required_cids, desc="Processing classes"):
            arr = np.load(self._raw_dir / f"class_{cid}.npz", allow_pickle=True)
            pickled = arr["graphs_pickle"]
            nx_graphs = threader(delayed(pickle.loads)(buf) for buf in pickled)

            allowed: Optional[Set[int]] = None
            if topo_masks is not None:
                allowed = set(map(int, topo_masks[cid]))
            selected = (
                g for idx, g in enumerate(nx_graphs) if allowed is None or idx in allowed
            )

            data_list: List[Data] = worker(
                delayed(self._graph_to_data)(g, cid) for g in selected
            )
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.extend(data_list, batch_size=1024)
            total += len(data_list)

        if self.log:
            print(f"[{self.__class__.__name__}] Stored {total} graphs → {self.processed_paths[0]}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def num_classes(self) -> int:  # type: ignore[override]
        return len(self._label_map)

    # ------------------------------------------------------------------
    # Download logic
    # ------------------------------------------------------------------
    def download(self) -> None:  # type: ignore[override]
        """Download *only* the files listed in :pyattr:`raw_file_names`."""
        self._raw_dir.mkdir(parents=True, exist_ok=True)

        # small assets are guaranteed present by _ensure_metadata(); we just have to
        # fetch the potentially large class archives.
        to_get = [p for p in self._raw_file_list if p.startswith("class_")]
        if not to_get:
            return  # nothing to do (rare)

        fname2id = _dataverse_file_map()
        for fname in tqdm(to_get, desc="Downloading .npz blobs"):
            cid = int(fname.split("_")[1].split(".")[0])
            dst = self._raw_dir / fname
            if dst.exists():
                continue
            fid = fname2id[fname]
            url = f"https://dataverse.harvard.edu/api/access/datafile/{fid}?format=original"
            _download_url(url, dst, desc=fname)
        time.sleep(0.2)  # flush dir cache on some filesystems

    # ------------------------------------------------------------------
    # Bootstrap helper (download small metadata assets *early*)
    # ------------------------------------------------------------------
    def _ensure_metadata(self) -> None:
        """Fetch *HSG-generator-meta* and the topology mask into :pyattr:`_raw_dir`."""
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        gh = "https://raw.githubusercontent.com/sarinstein-yan/HSG-12M/main/assets"
        _download_url(f"{gh}/HSG-generator-meta.npz", self._raw_dir / "HSG-generator-meta.npz")
        _download_url(f"{gh}/HSG-topology-mask.pkl", self._raw_dir / "HSG-topology-mask.pkl")


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