import os.path as osp
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Iterable
import numpy as np
import networkx as nx
from urllib.request import urlretrieve
import torch
from torch_geometric.data import Data, OnDiskDataset, InMemoryDataset
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from joblib import Parallel, delayed
from easyDataverse import Dataverse

# -----------------------------
# Module-private helpers (shared by both datasets)
# -----------------------------

_GH_META = "https://raw.githubusercontent.com/sarinstein-yan/HSG-12M/main/assets"
_META_FILES = ["HSG-generator-meta.npz", "HSG-topology-mask.pkl"]

def _normalize_subset(subset: Optional[str]) -> str:
    s = (subset or "one-band").lower()
    return "all" if s in {"all", "all-static"} else s

def _ensure_meta(meta_dir: Path) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    for fname in _META_FILES:
        fpath = meta_dir / fname
        if not fpath.exists():
            print(f"[HSG] downloading {fname} …")
            urlretrieve(f"{_GH_META}/{fname}", fpath)

def _select_class_ids(subset: str, metas: np.ndarray) -> List[int]:
    bands = np.array([m["number_of_bands"] for m in metas])
    if subset in ("one-band", None):
        ids = np.where(bands == 1)[0].tolist()
    elif subset == "two-band":
        ids = np.where(bands == 2)[0].tolist()
    elif subset == "three-band":
        ids = np.where(bands == 3)[0].tolist()
    elif subset in ("all", "topology"):
        ids = np.arange(len(metas)).tolist()
    else:
        raise ValueError(f"Unknown subset: {subset}. Expected one of: "
                         "'one-band', 'two-band', 'three-band', 'all', 'topology'")
    return ids

def _load_required(meta_dir: Path, subset: str) -> tuple[list[int], dict[int, int]]:
    param = np.load(meta_dir / "HSG-generator-meta.npz", allow_pickle=True)
    metas = param["metas"]
    required = _select_class_ids(subset, metas)
    label_map = {int(old): new for new, old in enumerate(sorted(required))}
    return required, label_map

def _load_topo_masks(meta_dir: Path, subset: str) -> Optional[Sequence[Sequence[int]]]:
    if subset != "topology":
        return None
    with open(meta_dir / "HSG-topology-mask.pkl", "rb") as f:
        return pickle.load(f)

def _build_raw_files(required_classes: Iterable[int]) -> List[str]:
    return [f"class_{int(cid)}.npz" for cid in required_classes]

def _download_required_raw_files(root: Path, raw_files: list[str],
                                 server_url: str, dataset_pid: str) -> None:
    # Mirror 'raw/<fname>' layout under root for easyDataverse
    to_download = []
    for fname in raw_files:
        fpath = osp.join(root, "raw", fname)
        if not osp.exists(fpath):
            to_download.append(f"raw/{fname}")
    if not to_download:
        return
    print(f"[HSG] Downloading {len(to_download)} files …")
    dv = Dataverse(server_url)
    dv.load_dataset(pid=dataset_pid, filedir=root, filenames=to_download, download_files=True)

# ----- Graph attribute packing & conversion -----
def _process_edge_pts(graph: nx.MultiGraph) -> nx.MultiGraph:
    g = graph.copy()
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

def _graph_to_data(g: nx.MultiGraph, class_id: int, label_map: Dict[int, int]) -> Data:
    g2 = _process_edge_pts(g)
    data = from_networkx(g2, group_node_attrs=["x"], group_edge_attrs=["edge_attr"])
    data.y = torch.tensor([label_map[class_id]], dtype=torch.long)
    # caching properties during build
    data.n_nodes = torch.tensor([data.num_nodes], dtype=torch.int32)
    data.n_edges = torch.tensor([data.num_edges], dtype=torch.int32)
    return data

def _apply_user_hooks(data_list: List[Data], pre_filter, pre_transform) -> List[Data]:
    if pre_filter is not None:
        data_list = [d for d in data_list if pre_filter(d)]
    if pre_transform is not None:
        data_list = [pre_transform(d) for d in data_list]
    return data_list

def _build_data_list_for_class(cid: int, raw_dir: str, allowed: Optional[Set[int]],
                               label_map: Dict[int, int], worker: Parallel) -> List[Data]:
    arr = np.load(osp.join(raw_dir, f"class_{cid}.npz"), allow_pickle=True)
    pickled = arr["graphs_pickle"]
    nx_graphs = worker(delayed(pickle.loads)(buf) for buf in pickled)
    if allowed is not None:
        selected = (g for idx, g in enumerate(nx_graphs) if idx in allowed)
    else:
        selected = nx_graphs
    data_list: List[Data] = worker(delayed(_graph_to_data)(g, cid, label_map) for g in selected)
    return data_list

# -----------------------------
# On-disk dataset
# -----------------------------
class HSGOnDisk(OnDiskDataset):
    """On‑disk PyG wrapper around the HSG‑12M corpus (1401 classes)."""
    def __init__(
        self,
        root: str | Path = ".",
        subset: Optional[str] = "one-band",
        *,
        n_jobs: int = -1,
        backend: str = "sqlite",
        transform=None,
        pre_filter=None,
        pre_transform=None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.meta_dir = self.root / "meta"
        self.subset = _normalize_subset(subset)
        self.n_jobs = n_jobs
        self.server_url = "https://dataverse.harvard.edu"
        self.dataset_pid = "doi:10.7910/DVN/PYDSSQ"

        _ensure_meta(self.meta_dir)
        self.required_classes, self._label_map = _load_required(self.meta_dir, self.subset)
        self._raw_files = _build_raw_files(self.required_classes)

        super().__init__(
            root=self.root,
            transform=transform,
            pre_filter=pre_filter,
            pre_transform=pre_transform,
            backend=backend,
        )

    # hooks
    @property
    def raw_file_names(self) -> List[str]:
        return self._raw_files

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"processed_{self.subset}")

    def download(self):
        _download_required_raw_files(self.root, self._raw_files, self.server_url, self.dataset_pid)

    def process(self) -> None:
        topo_masks = _load_topo_masks(self.meta_dir, self.subset)

        # Ensure DB exists before inserts
        _ = self.db

        worker = Parallel(n_jobs=self.n_jobs, prefer="threads")
        total = 0
        for cid in tqdm(self.required_classes, desc="Processing classes"):
            allowed: Optional[Set[int]] = None
            if topo_masks is not None:
                allowed = set(map(int, topo_masks[cid]))
            data_list = _build_data_list_for_class(cid, self.raw_dir, allowed, self._label_map, worker)
            data_list = _apply_user_hooks(data_list, self.pre_filter, self.pre_transform)
            self.extend(data_list, batch_size=2048)
            total += len(data_list)
        if self.log:
            print(f"[{self.__class__.__name__}] Stored {total} graphs → {self.processed_paths[0]}")

    # Convenience
    @property
    def num_classes(self) -> int:
        return len(self._label_map)

# -----------------------------
# In-memory dataset (now independent of HSGOnDisk)
# -----------------------------
class HSGInMemory(InMemoryDataset):
    """In‑memory PyG wrapper around the HSG‑12M corpus (1401 classes)."""
    def __init__(
        self,
        root: str | Path = ".",
        subset: Optional[str] = "one-band",
        *,
        n_jobs: int = -1,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.meta_dir = self.root / "meta"
        self.subset = _normalize_subset(subset)
        self.n_jobs = n_jobs
        self.server_url = "https://dataverse.harvard.edu"
        self.dataset_pid = "doi:10.7910/DVN/PYDSSQ"

        _ensure_meta(self.meta_dir)
        self.required_classes, self._label_map = _load_required(self.meta_dir, self.subset)
        self._raw_files = _build_raw_files(self.required_classes)

        super().__init__(
            root=self.root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        # pull cached tensors
        self.load(self.processed_paths[0])

    # hooks
    @property
    def raw_file_names(self) -> List[str]:
        return self._raw_files

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, f"processed_{self.subset}")

    @property
    def processed_file_names(self) -> List[str]:
        return [f"data_{self.subset}.pt"]

    def download(self):
        _download_required_raw_files(self.root, self._raw_files, self.server_url, self.dataset_pid)

    def process(self) -> None:
        topo_masks = _load_topo_masks(self.meta_dir, self.subset)
        worker = Parallel(n_jobs=self.n_jobs, prefer="threads")

        all_data: List[Data] = []
        for cid in tqdm(self.required_classes, desc="Processing classes (in-memory)"):
            allowed: Optional[Set[int]] = None
            if topo_masks is not None:
                allowed = set(map(int, topo_masks[cid]))
            data_list = _build_data_list_for_class(cid, self.raw_dir, allowed, self._label_map, worker)
            data_list = _apply_user_hooks(data_list, self.pre_filter, self.pre_transform)
            all_data.extend(data_list)

        # collate & cache
        self.save(all_data, self.processed_paths[0])
        if self.log:
            print(f"[{self.__class__.__name__}] Collated {len(all_data)} graphs → {self.processed_paths[0]}")

    # Convenience
    @property
    def num_classes(self) -> int:
        return len(self._label_map)




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