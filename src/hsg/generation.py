import os
import glob
import time
import pickle
import numpy as np
import sympy as sp
import networkx as nx
import itertools as it
from functools import partial
from joblib import Parallel, delayed
# from multiprocessing.pool import ThreadPool
import poly2graph as p2g
from pathlib import Path
from tqdm import tqdm
from typing import Sequence, Tuple, Dict, List, Any

__all__ = [
    "HSG_Generator",
]



def _get_single_param_arr(
    real_coeff_walk: List[float],
    imag_coeff_walk: List[float],
) -> np.ndarray:
    # Convert walks to numpy arrays
    assert not np.iscomplex(real_coeff_walk).any(), \
        "[get_single_param_arr] real_coeff_walk should not be complex"
    assert not np.iscomplex(imag_coeff_walk).any(), \
        "[get_single_param_arr] imag_coeff_walk should not be complex"
    real_vals = np.array(real_coeff_walk)
    imag_vals = 1j*np.array(imag_coeff_walk)
    # Sum real + imag parts to form complex parameter grid
    param_arr = real_vals[:, None] + imag_vals[None, :]
    return param_arr


def _one_poly(
    z: sp.Symbol,
    E: sp.Symbol,
    D: int, # Total hopping range, D = p+q
    params: Tuple[sp.Symbol, ...], # Free coefficients (e.g., a, b)
    s: int, # Number of bands
    q: int, # Max positive z exponent in the base term
    E_deg_list: Tuple[int, ...], # Exponents for E in additional terms
    param_pos: Tuple[int, ...], # z exponents where params are inserted
) -> Tuple[Tuple[int, ...], str, sp.Poly, Dict]:
    # Compute complementary exponent range
    p = D - q
    z_degs = [d for d in range(-p + 1, q) if d != 0]

    # Start with the base polynomial -E^s + z^q + z^{-p}
    expr = -E**s + z**q + z**(-p)
    param_map = dict(zip(param_pos, params))

    # Add additional terms based on E_deg_list and parameter positions
    for z_deg, E_deg in zip(z_degs, E_deg_list):
        if E_deg > 0 and z_deg in param_map:
            expr += param_map[z_deg] * E**E_deg * z**z_deg
        elif E_deg > 0:
            expr += E**E_deg * z**z_deg
        elif z_deg in param_map:
            expr += param_map[z_deg] * z**z_deg

    # Expand, convert to Poly, and gather metadata
    expr = sp.expand(expr)
    poly = sp.Poly(expr, z, 1/z, E)

    meta = {
        'parameter_symbols': tuple(str(p) for p in params),
        'generator_symbols': tuple(str(g) for g in poly.gens),
        'latex': sp.latex(expr),
        'sympy_repr': sp.srepr(poly),
        'number_of_bands': s,
        'max_left_hopping': q,
        'max_right_hopping': p,
        # Add construction details to metadata for clarity
        'intermediate_z_degrees': tuple(z_degs),
        'E_deg_assignments': E_deg_list,
        'param_degree_placements': param_pos,
    }

    # Build de-duplication key and its reciprocal
    flags = [1 if d in param_pos else 0 for d in z_degs]
    key = list(zip(z_degs, E_deg_list, flags))
    key_reciprocal = [(-d, e, f) for (d, e, f) in reversed(key)]
    final_key = min(tuple(key), tuple(key_reciprocal))

    return final_key, str(expr), poly, meta


# --- HSG-topology helper functions --- #
def _simple_copy_with_multiplicity(g_multi: nx.MultiGraph):
    """Collapse a (multi)graph into a simple Graph, recording multiplicity."""
    if not g_multi.is_multigraph():
        return g_multi          # already simple, nothing to do

    g_simple = nx.Graph() if isinstance(g_multi, nx.MultiGraph) else nx.DiGraph()
    g_simple.add_nodes_from(g_multi.nodes(data=True))

    for u, v, _k, data in g_multi.edges(keys=True, data=True):
        if g_simple.has_edge(u, v):
            g_simple[u][v]["m"] += 1          # bump multiplicity
        else:
            g_simple.add_edge(u, v, **data, m=1)
    return g_simple

def wl_hash_safe(g, iters=3):
    """WL hash that tolerates MultiGraphs by collapsing them first."""
    g_for_hash = _simple_copy_with_multiplicity(g)
    return nx.weisfeiler_lehman_graph_hash(
        g_for_hash,
        iterations=iters,
        edge_attr="m"  # include multiplicity in the label
    )

def unique_indices_by_hash(graphs, iters=3, n_jobs=-1):
    hashes = Parallel(n_jobs=n_jobs, batch_size=256)(
        delayed(wl_hash_safe)(g, iters=iters) for g in graphs
    )
    first_seen = {}
    for idx, h in enumerate(hashes):
        first_seen.setdefault(h, idx)
    return list(first_seen.values())

def get_topology_mask(class_idx, data_dir, wl_hash_iters=3):
    path = os.path.join(data_dir, f"class_{class_idx}.npz")
    graphs_pickle = np.load(path, allow_pickle=True)["graphs_pickle"]
    graphs = [pickle.loads(g) for g in graphs_pickle]
    return unique_indices_by_hash(graphs, iters=wl_hash_iters)


class HSG_Generator:
    """
    A class to generate and manage datasets of Hamiltonian Spectral Graphs (HSG)
    derived from characteristic polynomials.

    It systematically samples polynomial classes based on parameters such as
    hopping range (D = p+q) and the number of energy bands (s). The generation
    process respects mathematical symmetries (e.g., z-reciprocity, where
    P(z) is equivalent to z^(p+q)P(1/z)) to avoid redundant or physically
    equivalent classes, ensuring a unique set of foundational polynomials.

    For each unique polynomial class, graph samples are generated by systematically
    varying two free complex coefficients (denoted as 'a' and 'b').
    For instance, coefficients might be varied from -10-5i to +10+5i, using
    a grid of real and imaginary values (e.g., 13 real and 7 imaginary steps,
    leading to (13*7)^2 = 8281 samples per class).
    """
    def __init__(
        self,
        root: str = './',
        z: sp.Symbol = sp.Symbol('z'),
        E: sp.Symbol = sp.Symbol('E'),
        params: Sequence[sp.Symbol] = (sp.Symbol('a'), sp.Symbol('b')),
        hopping_range: Sequence[int] | int = [4, 5, 6],
        num_bands: Sequence[int] = [1, 2, 3],
        real_coeff_walk: Sequence[float] = np.array([-10, -5, -2, -1, -0.5, -0.1, 
                                                     0, 0.1, 0.5, 1, 2, 5, 10]),
        imag_coeff_walk: Sequence[float] = np.array([-5, -2, -1, 0, 1, 2, 5]),
        n_jobs: int = -1,
    ) -> None:
        
        self.root_dir = Path(root).expanduser().resolve()
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        
        # Ensure hopping ranges is a list
        if isinstance(hopping_range, int):
            self.hopping_ranges = [hopping_range]
        else:
            self.hopping_ranges = list(hopping_range)

        # Ensure num_bands is a list
        if isinstance(num_bands, int):
            self.num_bands = [num_bands]
        else:
            self.num_bands = list(num_bands)

        # Assign core attributes
        self.z = z
        self.E = E
        self.params = tuple(params)
        self.n_jobs = n_jobs

        # Validity check for parameters
        self.num_params = len(self.params)
        for D_val in self.hopping_ranges:
            assert D_val >= 2, f"hopping_range ({D_val}) must be >= 2"
            # D-2 refers to the number of available intermediate z exponents.
            assert self.num_params <= D_val - 2 if D_val > 2 else self.num_params == 0, \
                f"num_params (= {self.num_params}) must not exceed D-2 " \
                f"({D_val-2 if D_val > 2 else 'N/A, D<=2 should have 0 params for this rule'})"
        
        self.real_coeff_walk = np.array(real_coeff_walk)
        self.imag_coeff_walk = np.array(imag_coeff_walk)
        self.single_param_arr = _get_single_param_arr(
            self.real_coeff_walk,
            self.imag_coeff_walk,
        )
        self.single_param_sweep = self.single_param_arr.ravel()

        # This assumes two free parameters (a,b)
        # If num_params in HSG_Generator is different, this needs generalization
        # For now, hardcoding for 2 parameters.
        if self.num_params == 2:
            p1_vals_mesh, p2_vals_mesh = np.meshgrid(self.single_param_sweep, self.single_param_sweep)
            self.param_value_pairs = np.asarray([p1_vals_mesh.ravel(), p2_vals_mesh.ravel()])
        elif self.num_params == 1:
            self.param_value_pairs = np.asarray([self.single_param_sweep.ravel()]) # Only one parameter
        elif self.num_params == 0:
            self.param_value_pairs = np.array([]) # No parameter values
        else: # More than 2 params, not covered by example
            raise ValueError("Parameter generation for >2 free coefficients not explicitly defined by example.")

        # Generate unique polynomial classes
        print(f"[init] Generating and de-duplicating polynomial classes...")
        self.all_exprs, self.all_metas = self.generate_char_poly_classes()
        print(f"[init] Generated {len(self.all_exprs)} unique polynomial classes.")
        # Save metadata to a file
        self.save_meta_files(file_name="HSG-generator-meta.npz")


    def generate_char_poly_classes(self) -> Tuple[List[str], List[Dict]]:
        all_exprs = []  # accumulator for expression strings
        all_metas = []  # accumulator for corresponding metadata

        for D in self.hopping_ranges:
            # Assemble tasks for this D
            tasks = []
            for s in self.num_bands:
                for q in range(1, D):
                    p = D - q
                    degs = [d for d in range(-p + 1, q) if d != 0]
                    for E_deg_list in it.product(range(s), repeat=D-2):
                        for pos in it.combinations(degs, len(self.params)):
                            tasks.append((s, q, E_deg_list, pos))

            print(f"[Hopping range = {D}] raw samples: {len(tasks)}")
            build = partial(_one_poly, self.z, self.E, D, self.params)

            # Parallel execution of polynomial construction
            results = Parallel(n_jobs=self.n_jobs, backend='loky')(
                delayed(build)(*t) for t in tasks
            )
            if not results:
                continue  # skip if no results

            # Unzip results and remove duplicates
            keys, expr_strs, polys, metas = zip(*results)
            keys_arr = np.asarray(keys)
            unique_keys, unique_indices = np.unique(
                keys_arr, return_index=True, axis=0
            )

            exprs = [expr_strs[i] for i in unique_indices]
            mlist = [metas[i] for i in unique_indices]
            print(f"[Hopping range = {D}] unique samples: {len(exprs)}")

            # Extend accumulators
            all_exprs.extend(exprs)
            all_metas.extend(mlist)

        return all_exprs, all_metas


    def save_meta_files(
        self,
        file_name: str = "HSG-generator-meta.npz",
    ) -> None:

        meta_dir = os.path.join(self.root_dir, "meta")
        os.makedirs(meta_dir, exist_ok=True)
        if not file_name.endswith('.npz'):
            file_name += '.npz'
        
        np.savez(
            os.path.join(meta_dir, file_name),
            polys=np.array(self.all_exprs, dtype=object),
            metas=np.array(self.all_metas, dtype=object),
            param_vals=self.param_value_pairs,
        )
        print(f"Meta data saved to {os.path.join(meta_dir, file_name)}")


    def generate_dataset(
        self,
        class_idx: int,
        num_partition: int = 10,
        short_edge_threshold: int = 30,
        save_dir: str = "raw",
        class_data: str = "meta/HSG-generator-meta.npz",
    ) -> None:
        """
        Generate and save spectral graph data for a single polynomial class using a
        pre-defined parameter table.

        This method loads a specific polynomial expression (identified by `class_idx`)
        and its associated metadata from the `class_data_file` (e.g., 'HSG-generator-meta.npz').
        It retrieves the grid of parameter value pairs (e.g., for coefficients 'a' and 'b')
        also stored in this file. Typically, this grid contains numerous combinations,
        such as (13 real * 7 imag)^2 = 8281 pairs, for systematic exploration.

        For each pair of coefficient values, they are substituted into the polynomial.
        The `poly2graph.CharPolyClass.spectral_graph` method is then used to compute
        the corresponding spectral graph.

        To handle potentially large numbers of graph generations, the parameter sweep
        can be divided into `num_partitions` for processing. The results for each
        partition are temporarily saved and then consolidated into a single compressed
        NPZ file for the class (e.g., `class_{class_idx}.npz`) in `save_dir`.
        This final file contains all generated graphs for the class, their corresponding
        coefficient values, and the class metadata.

        Parameters
        ----------
        class_idx : int
            Index of the polynomial class in the `class_data_file` to process.
        class_data_file : str, optional
            Path to the NPZ file containing polynomial expressions, metadata, and
            parameter value sweeps (e.g., 'HSG-generator-meta.npz'), by default "./HSG-generator-meta.npz".
        save_dir : str, optional
            Directory to save the generated dataset for this class, by default "./raw_class_data".
        num_partitions : int, optional
            Number of partitions to split the parameter sweep into for processing, by default 10.
        short_edge_threshold : int, optional
            Threshold for `poly2graph`'s short edge detection, by default 30.
        """
        save_dir = os.path.join(self.root_dir, save_dir)
        class_data = os.path.join(self.root_dir, class_data)
        # Define working symbols and parameter dict
        k, z, E, a, b = sp.symbols('k z E a b')
        params = {a, b}

        # Load param table with all parameter combinations
        data = np.load(class_data, allow_pickle=True)
        polys = data['polys']       # array of expression strings
        metas = data['metas']       # array of metadata dicts
        param1_vals, param2_vals = data['param_vals']

        # Split parameter sweeps into partitions
        p1_parts = np.array_split(param1_vals, num_partition)
        p2_parts = np.array_split(param2_vals, num_partition)

        print(f"[Class {class_idx}]: {polys[class_idx]}")
        cp = p2g.CharPolyClass(polys[class_idx], k, z, E, params)
        os.makedirs(save_dir, exist_ok=True)

        batcher = Parallel()
        for i, (v1, v2) in enumerate(zip(p1_parts, p2_parts)):
            print(f"[Class {class_idx}] Partition {i} - computing...")
            t0 = time.perf_counter()
            graphs, _ = cp.spectral_graph(
                {a: v1, b: v2}, 
                short_edge_threshold=short_edge_threshold,
            )

            # Save pickled graphs and their parameter values
            out = os.path.join(save_dir, f'class_{class_idx}_part_{i}.npz')
            graphs_ser = batcher(
                delayed(pickle.dumps)(g) for g in graphs
            )
            np.savez_compressed(
                out,
                graphs_pickle=graphs_ser,
                a_vals=v1,
                b_vals=v2,
            )
            dt = (time.perf_counter() - t0) / 60
            print(
                f"[Class {class_idx}] Partition {i} saved → {out}"
                f" ({len(graphs_ser)} graphs, took {dt:.4f} min)"
            )
        
        # Combine all partitions into a single NPZ
        all_pickles = []
        for i in range(num_partition):
            d = np.load(
                os.path.join(save_dir, f'class_{class_idx}_part_{i}.npz'), 
                allow_pickle=True
                )
            # .tolist() because np.savez loads object arrays as dtype=object
            all_pickles.extend(d['graphs_pickle'].tolist())
            os.remove(os.path.join(save_dir, f'class_{class_idx}_part_{i}.npz'))

        out = os.path.join(save_dir, f'class_{class_idx}.npz')
        np.savez_compressed(
            out,
            graphs_pickle=np.array(all_pickles, dtype=object),
            y=class_idx,
            a_vals=param1_vals,
            b_vals=param2_vals,
            **metas[class_idx],
        )
        print(f"[Class {class_idx}] Combined dataset saved → {out}")


    def load_class(
        self,
        class_idx: int,
        data_dir: str = "raw",
    ) -> Tuple[List[nx.Graph], np.ndarray, np.ndarray, Dict]:
        """
        Load the NPZ file for a specific class and extract the graphs and metadata.

        Parameters
        ----------
        class_idx : int
            Index of the polynomial class to load.
        data_dir : str, optional
            Directory containing the NPZ files, by default "raw".

        Returns
        -------
        Tuple[List[nx.Graph], np.ndarray, np.ndarray, Dict]
            A tuple containing:
            - List of graphs for the specified class.
            - Array of 'a' values.
            - Array of 'b' values.
            - Metadata dictionary for the class.
        """
        path = os.path.join(self.root_dir, data_dir, f"class_{class_idx}.npz")
        data = np.load(path, allow_pickle=True)
        graphs_pickle = data["graphs_pickle"]
        graphs = [pickle.loads(g) for g in graphs_pickle]
        y = data["y"]
        a_vals = data["a_vals"]
        b_vals = data["b_vals"]
        keys = data.files
        metas = {k: data[k] for k in keys 
                 if k not in ["graphs_pickle", "y", "a_vals", "b_vals"]}
        return graphs, y, a_vals, b_vals, metas


    # --- Temporal Graph Selection Functions --- #
    def get_temporal_mask(self) -> List[List[int]]:
        """
        Build the temporal masks that identify every 1-D parameter-sweep
        (“temporal graph”) inside a single class.

        Under the default settings two complex-valued free coefficients
        (a,b) are swept over the same 13 × 7 mesh

            Re ∈ self.real_coeff_walk   (13 values)
            Im ∈ self.imag_coeff_walk   (7 values)

        giving 91 distinct values per coefficient and 91 × 91 = 8 281
        (a,b) pairs in row-major order:

            (a_idx, b_idx)  →  flat_idx = a_idx*91 + b_idx

        A temporal graph is defined by **fixing one part (Re or Im) of one
        coefficient** and sweeping the complementary part of the *other*
        coefficient, so every mask is either 13-long (real sweep) or
        7-long (imag sweep).

        Returns
        -------
        masks : List[List[int]]
            Each inner list is the list of flat indices (into the
            8 281-length parameter pair array) that belong to one temporal
            sequence.
        """
        if self.num_params != 2:
            raise ValueError(
                "Temporal masks are only defined for exactly two free "
                "coefficients (a, b) in the current dataset design."
            )

        n_real, n_imag = self.single_param_arr.shape          # 13, 7
        param_grid_size = n_real * n_imag                     # 91
        masks: List[List[int]] = []

        # Helper: map (real_idx, imag_idx) -> flat index in the 91-element list
        def _coef_flat(r: int, im: int) -> int:
            return r * n_imag + im

        # ---- 1.  Vary *b* while *a* is fixed --------------------------------
        for a_r in range(n_real):
            for a_i in range(n_imag):
                a_flat = _coef_flat(a_r, a_i)

                # (a fixed, Im[b] fixed) ── sweep Re[b]  ← length 13
                for b_i in range(n_imag):
                    seq = [
                        a_flat * param_grid_size + _coef_flat(b_r, b_i)
                        for b_r in range(n_real)
                    ]
                    masks.append(seq)

                # (a fixed, Re[b] fixed) ── sweep Im[b]  ← length 7
                for b_r_fixed in range(n_real):
                    seq = [
                        a_flat * param_grid_size + _coef_flat(b_r_fixed, b_i)
                        for b_i in range(n_imag)
                    ]
                    masks.append(seq)

        # ---- 2.  Vary *a* while *b* is fixed --------------------------------
        for b_r in range(n_real):
            for b_i in range(n_imag):
                b_flat = _coef_flat(b_r, b_i)

                # (b fixed, Im[a] fixed) ── sweep Re[a]  ← length 13
                for a_i in range(n_imag):
                    seq = [
                        _coef_flat(a_r, a_i) * param_grid_size + b_flat
                        for a_r in range(n_real)
                    ]
                    masks.append(seq)

                # (b fixed, Re[a] fixed) ── sweep Im[a]  ← length 7
                for a_r_fixed in range(n_real):
                    seq = [
                        _coef_flat(a_r_fixed, a_i) * param_grid_size + b_flat
                        for a_i in range(n_imag)
                    ]
                    masks.append(seq)

        # Sanity check (can be removed in production)
        # Expected: 3 640 masks, 1 274 length-13, 2 366 length-7
        # assert len(masks) == 3640
        return masks


    def get_temporal_graphs_by_class(
        self,
        class_idx: int,
        data_dir: str = "./raw",
    ):
        """
        Load the NPZ file for a specific class and extract the temporal graphs.

        Parameters
        ----------
        class_idx : int
            Index of the polynomial class to load.
        data_dir : str, optional
            Directory containing the NPZ files, by default "./raw".

        Returns
        -------
        List[nx.Graph]
            List of temporal graphs for the specified class.
        """
        path = os.path.join(data_dir, f"class_{class_idx}.npz")
        data = np.load(path, allow_pickle=True)
        graphs_pickle = data["graphs_pickle"]
        graphs = [pickle.loads(g) for g in graphs_pickle]
        temporal_masks = self.get_temporal_mask()
        temporal_graphs = [graphs[mask] for mask in temporal_masks]
        return temporal_graphs


    # --- Topology Mask Generation Functions --- #
    def generate_topology_mask(
        self,
        class_indices: Sequence[int] = range(1401),
        wl_hash_iters: int = 3,
        meta_dir: str = "meta",
        raw_dir: str = "raw",
        file_name: str = "HSG-topology-mask.pkl",
    ) -> None:
        """Generate a topology mask for a specific class and save it."""
        meta_dir = os.path.join(self.root_dir, meta_dir)
        mask_file = os.path.join(meta_dir, file_name)
        if os.path.exists(mask_file):
            print(f"Topology mask file already exists: {mask_file}")
            return self.load_topology_mask(mask_file)

        os.makedirs(meta_dir, exist_ok=True)
        raw_dir = os.path.join(self.root_dir, raw_dir)
        topology_masks = []
        for class_idx in tqdm(class_indices, desc="Generating topology masks"):
            mask = get_topology_mask(class_idx, raw_dir, wl_hash_iters)
            topology_masks.append(mask)
        
        with open(mask_file, "wb") as f:
            pickle.dump(topology_masks, f)
        print(f"Topology masks saved to {mask_file}")

        return topology_masks

    def load_topology_mask(
        self,
        mask_file: str = "meta/HSG-topology-mask.pkl",
    ) -> List[List[int]]:
        """Load the topology mask from a file."""
        mask_file = os.path.join(self.root_dir, mask_file)
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Topology mask file not found: {mask_file}")
        with open(mask_file, "rb") as f:
            topology_masks = pickle.load(f)
        return topology_masks


    # --- General Subset Selection Functions --- #
    @staticmethod
    def filter_metas(
        metas: List[Dict],
        number_of_bands=None,
        max_left_hopping=None,
        max_right_hopping=None,
        intermediate_z_degrees=None,
        E_deg_assignments=None,
        param_degree_placements=None,
    ) -> List[int]:
        """Return the indices of *metas* that match *all* supplied criteria.

        Examples
        --------
        >>> idx = HSG_Generator.filter_metas(
        ...     metas,
        ...     number_of_bands=2,
        ...     max_left_hopping=3,
        ... )

        Any keyword found in ``criteria`` must also be a key of each metadata
        dict; the entry is retained only if ``meta[key] == value`` for *every*
        supplied *(key,value)* pair.  Passing no criteria returns ``range(len(metas))``.
        """
        filters = {
            'number_of_bands': number_of_bands,
            'max_left_hopping': max_left_hopping,
            'max_right_hopping': max_right_hopping,
            'intermediate_z_degrees': intermediate_z_degrees,
            'E_deg_assignments': E_deg_assignments,
            'param_degree_placements': param_degree_placements,
        }
        matched = []
        for i, m in enumerate(metas):
            if all(desired is None or m.get(key) == desired
                   for key, desired in filters.items()):
                matched.append(i)
        return matched

    def select_subset(
        self,
        metas: List[Dict],
        input_dir: str = "./raw",
        output_dir: str = "./subset",
        a_vals_filter: Sequence[float] | None = None,
        b_vals_filter: Sequence[float] | None = None,
        **class_filters: Any,
    ) -> str:
        """Assemble a study subset from pre‑generated ``class_{cid}.npz`` files.

        Parameters
        ----------
        metas : list[dict]
            The list returned by :py:meth:`generate_char_poly_classes` (or the
            loaded ``metas`` array from *HSG-generator-meta.npz*).
        input_dir : str, default ``"./raw"``
            Folder containing *class_{cid}.npz* archives.
        output_dir : str, default current directory
            Folder to which the combined subset is written.
        a_vals_filter, b_vals_filter : Sequence[float] | None
            Optional whitelist of parameter values.  If supplied, only graphs
            whose sampled ``a``/``b`` lie in the corresponding list are
            included.
        **class_filters
            Any keywords accepted by :py:meth:`filter_metas` to restrict which
            *classes* are loaded (e.g. ``number_of_bands=3``).

        Returns
        -------
        str
            Path to the *npz* file containing the subset.  The file name is
            constructed from the filtering options so that repeated calls do
            not overwrite each other.
        """
        # ---- pick classes
        cls_ids = self.filter_metas(metas, **class_filters)
        if not cls_ids:
            raise ValueError("No classes match the supplied filters.")

        combined_pickles: list = []
        combined_a: list = []
        combined_b: list = []
        combined_metas: list = []

        for cid in cls_ids:
            fpath = os.path.join(input_dir, f"class_{cid}.npz")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"Expected file not found: {fpath}")

            data = np.load(fpath, allow_pickle=True)
            pkl, av, bv = data["graphs_pickle"], data["a_vals"], data["b_vals"]
            mask = np.ones(len(av), dtype=bool)
            if a_vals_filter is not None:
                mask &= np.isin(av, a_vals_filter)
            if b_vals_filter is not None:
                mask &= np.isin(bv, b_vals_filter)

            combined_pickles.extend(pkl[mask].tolist())
            combined_a.extend(av[mask].tolist())
            combined_b.extend(bv[mask].tolist())
            combined_metas.extend([metas[cid]] * int(mask.sum()))

        # ---- file name summarising filters
        fname_parts = ["subset"]
        for k, v in class_filters.items():
            fname_parts.append(f"{k}-{v}")
        if a_vals_filter is not None:
            fname_parts.append("a-" + "_".join(map(str, a_vals_filter)))
        if b_vals_filter is not None:
            fname_parts.append("b-" + "_".join(map(str, b_vals_filter)))
        fname = "_".join(fname_parts) + ".npz"
        out_path = os.path.join(output_dir, fname)

        # ---- save
        np.savez_compressed(
            out_path,
            graphs_pickle=np.array(combined_pickles, dtype=object),
            a_vals=np.array(combined_a),
            b_vals=np.array(combined_b),
            metas=np.array(combined_metas, dtype=object),
        )
        return out_path