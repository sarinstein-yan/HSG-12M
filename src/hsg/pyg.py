import os.path as osp
import pickle
import time
from pathlib import Path
import numpy as np
import networkx as nx
from urllib.request import urlretrieve
import torch
from torch_geometric.data import Data, OnDiskDataset, InMemoryDataset
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Dict, List, Optional, Sequence, Set
from easyDataverse import Dataverse


def _select_class_ids(subset: str, metas: np.ndarray) -> np.ndarray:
    bands = np.array([m["number_of_bands"] for m in metas])
    match subset:
        case "one-band" | None:
            return np.where(bands == 1)[0].tolist()
        case "two-band":
            return np.where(bands == 2)[0].tolist()
        case "three-band":
            return np.where(bands == 3)[0].tolist()
        case "all":
            return np.arange(len(metas)).tolist()
        case "topology":
            return np.arange(len(metas)).tolist()
        case _:
            raise ValueError(f"Unknown subset: {subset}. Expected one of: "\
                             "'one-band', 'two-band', 'three-band', 'all', 'topology'")

class HSGOnDisk(OnDiskDataset):
    """On‑disk PyG wrapper around the HSG‑12M corpus (1401 classes)."""
    def __init__(
        self,
        root: str | Path = ".",
        subset: str | None = "one-band",
        *,
        n_jobs: int = -1,
        backend: str = "sqlite",
        transform=None,
        pre_filter=None,
    ) -> None:
        
        self.root = Path(root).expanduser().resolve()
        self.meta_dir = self.root / "meta"

        self.subset = (subset or "one-band").lower()
        if self.subset in {"all", "all-static"}:
            self.subset = "all"
        self.n_jobs = n_jobs

        # download two meta files if they are missing
        GH   = "https://raw.githubusercontent.com/sarinstein-yan/HSG-12M/main/assets"
        meta_files = ["HSG-generator-meta.npz", "HSG-topology-mask.pkl"]
        for fname in meta_files:
            fpath = self.meta_dir / fname
            if not fpath.exists():
                fpath.parent.mkdir(parents=True, exist_ok=True)
                print(f"[HSG] downloading {fname} …")
                urlretrieve(f"{GH}/{fname}", fpath)

        # parse HSG-generator-meta and decide which class_*.npz files we need
        param = np.load(self.meta_dir / "HSG-generator-meta.npz", allow_pickle=True)
        metas = param["metas"]
        self.required_classes = _select_class_ids(self.subset, metas)

        self.server_url = "https://dataverse.harvard.edu"
        self.dataset_pid = "doi:10.7910/DVN/PYDSSQ"

        # Build the raw-file list for this subset
        self._raw_files = [f"class_{int(cid)}.npz" for cid in self.required_classes]

        # # processed data live in a subset-specific folder
        # processed_root = self.root / f"processed_{self.subset}"

        super().__init__(
            root=self.root,
            transform=transform,
            pre_filter=pre_filter,
            backend=backend,
        )

    # torch_geometric.Dataset hooks
    @property
    def raw_file_names(self) -> List[str]:  # type: ignore[override]
        """Only the files needed for *this* subset."""
        return self._raw_files

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"processed_{self.subset}")

    # processed_file_names is inherited from OnDiskDataset ("<backend>.db")

    # Label mapping helpers (shared across workers)
    _label_map: Dict[int, int] = {}

    @classmethod
    def _set_label_map(cls, mapping: Dict[int, int]) -> None:
        cls._label_map = mapping

    def download(self):
        """Download *only* the files listed in :pyattr:`raw_file_names`."""
        to_download =  []
        for fname in self._raw_files:
            fpath = osp.join(self.raw_dir, fname)
            if not osp.exists(fpath):
                to_download.append(f'raw/{fname}')
        
        if not to_download:
            return

        print(f"[HSG] Downloading {len(to_download)} files ({to_download}) …")
        dv = Dataverse(self.server_url)
        hsg = dv.load_dataset(
            pid=self.dataset_pid, 
            filedir=self.root,
            filenames=to_download,
            download_files=True,
        )

    # Graph attributes
    @staticmethod
    def _process_edge_pts(graph: nx.MultiGraph) -> nx.MultiGraph:
        """Return a *new* graph with compact node / edge attributes."""
        g = graph.copy()
        # _IDX = np.arange(1, 6)

        node_pos, node_pot, node_dos = {}, {}, {}
        for nid, nd in g.nodes(data=True):
            pos = np.asarray(nd.pop("pos"), dtype=np.float32).reshape(-1)
            pot = np.float32(nd.pop("potential", 0.0))
            dos = np.float32(nd.pop("dos", 0.0))
            nd["pos"] = pos
            nd["x"] = np.array([*pos, pot, dos], dtype=np.float32)
            node_pos[nid], node_pot[nid], node_dos[nid] = pos, pot, dos

        for u, v, ed in g.edges(data=True):
            w = np.float32(ed.pop("weight"))
            pu, pv = node_pos[u], node_pos[v]
            displacement = np.float32(np.linalg.norm(pv - pu))
            # sinuosity = w / displacement if displacement > 1e-6 else 1.0
            if "pts" in ed:
                pts = ed.pop("pts")
                # idx = np.round(_IDX * (len(pts) - 1) / 6).astype(int)
                # pts5 = pts[idx].astype(np.float32).reshape(-1)
                mid = pts[len(pts) // 2].astype(np.float32)
                avg_pot = np.float32(ed.pop("avg_potential", 0.5 * (node_pot[u] + node_pot[v])))
                avg_dos = np.float32(ed.pop("avg_dos", 0.5 * (node_dos[u] + node_dos[v])))
            else:
                mid = 0.5 * (pu + pv)
                # pts5 = np.tile(mid, 5).astype(np.float32)
                avg_pot = np.float32(0.5 * (node_pot[u] + node_pot[v]))
                avg_dos = np.float32(0.5 * (node_dos[u] + node_dos[v]))
            ed["edge_attr"] = np.concatenate(([w, displacement], mid, [avg_pot, avg_dos]), dtype=np.float32)
        return g

    @classmethod
    def _graph_to_data(cls, g: nx.MultiGraph, class_id: int):
        from torch_geometric.utils import from_networkx

        dense = cls._label_map[class_id]
        g2 = cls._process_edge_pts(g)
        data = from_networkx(g2, group_node_attrs=["x"], group_edge_attrs=["edge_attr"])
        data.y = torch.tensor([dense], dtype=torch.long)
        return data

    def process(self) -> None:  # type: ignore[override]
        """Convert raw *.npz* files → on-disk DB (dense labels)."""
        # dense label map *for this subset*
        label_map = {int(old): new for new, old in enumerate(sorted(self.required_classes))}
        self._set_label_map(label_map)

        # optional topology masks
        topo_masks: Optional[Sequence[Sequence[int]]] = None
        if self.subset == "topology":
            topo_masks = pickle.load(open(self.root / "HSG-topology-mask.pkl", "rb"))

        # make sure DB exists before parallel inserts
        _ = self.db

        threader = Parallel(n_jobs=self.n_jobs, prefer="threads")
        worker = Parallel(n_jobs=self.n_jobs, backend="loky")
        total = 0

        for cid in tqdm(self.required_classes, desc="Processing classes"):
            data = np.load(osp.join(self.raw_dir, f"class_{cid}.npz"), allow_pickle=True)
            pickled = data["graphs_pickle"]
            nx_graphs = threader(delayed(pickle.loads)(buf) for buf in pickled)

            allowed: Optional[Set[int]] = None
            if topo_masks is not None:
                allowed = set(map(int, topo_masks[cid]))
            selected = (
                g for idx, g in enumerate(nx_graphs) if allowed is None or idx in allowed
            )

            data_list: List[Data] = threader(
                delayed(self._graph_to_data)(g, cid) for g in selected
            )
            if self.pre_filter is not None:
                data_list = [self.pre_filter(d) for d in data_list]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.extend(data_list, batch_size=1024)
            total += len(data_list)

        if self.log:
            print(f"[{self.__class__.__name__}] Stored {total} graphs → {self.processed_paths[0]}")

    # Convenience
    @property
    def num_classes(self) -> int:  # type: ignore[override]
        return len(self._label_map)


class HSGInMemory(InMemoryDataset):
    """In‑memory PyG wrapper around the HSG‑12M corpus (1401 classes)."""
    def __init__(
        self,
        root: str | Path = ".",
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

    # Dataset hooks
    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"processed_{self.subset}")

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

        # data_list = [on_disk.get(i) for i in range(len(on_disk))]
        data_list = []
        for i in tqdm(range(len(on_disk)), desc="processing graphs"):
            data_list.append(on_disk.get(i))

        # user hooks
        if self.pre_filter is not None:
            data_list = [self.pre_filter(d) for d in data_list]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        # collate & cache using the convenient helper
        self.save(data_list, self.processed_paths[0])

        # cache num‑classes for quick access
        self._num_classes = on_disk.num_classes

    # Convenience
    @property
    def num_classes(self) -> int:  # type: ignore[override]
        return getattr(self, "_num_classes", None) or int(self.data.y.max().item() + 1)


# import shutil
# import requests

# # Helper utilities (stand‑alone – no torch_geometric imports here)
# def _download_url(url: str, dst: Path, *, desc: str = "", overwrite: bool = False) -> None:
#     """Stream *url* → *dst* with a tqdm progress‑bar."""
#     if dst.exists() and not overwrite:
#         return
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     with requests.get(url, stream=True, timeout=60) as r:
#         r.raise_for_status()
#         total = int(r.headers.get("content-length", 0))
#         with tqdm.wrapattr(r.raw, "read", total=total, desc=desc or dst.name) as raw:
#             with open(dst, "wb") as f:
#                 shutil.copyfileobj(raw, f)


# def _dataverse_file_map() -> Dict[str, int]:
#     api = (
#         "https://dataverse.harvard.edu/api/datasets/:persistentId"
#         "?persistentId=doi:10.7910/DVN/PYDSSQ"
#     )
#     js = requests.get(api, timeout=30).json()
#     files = js["data"]["latestVersion"]["files"]
#     return {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in files}