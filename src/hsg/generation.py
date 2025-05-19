import os
import glob
import time
import pickle
from functools import partial
from typing import (Any, Dict, List, Mapping, MutableSequence, Sequence,
                    Tuple)

import itertools as it

import numpy as np
import sympy as sp
from joblib import Parallel, delayed

import poly2graph as p2g

# ──────────────────────────────────────────────────────────────────────────────
# HSG_Generator
# ──────────────────────────────────────────────────────────────────────────────

class HSG_Generator:
    """Generate, store, and post‑process *Hopping‑range Spectral Graph* (HSG)
    datasets.

    A *characteristic polynomial* (``CharPoly``) of a 1‑D tight‑binding model
    with maximal hopping range **D** has the generic form

    .. math::

        -E^b + z^{q} + z^{-(D-q)} + \sum_{d=-(D-q)+1}^{q-1} c_d E^{e_d} z^{d},

    where ``z = exp(i k)``, :math:`E` is the eigen‑energy, the integer
    ``b`` (``num_bands``) sets the number of energy bands, and the exponent
    pair ``(d, e_d)`` together with the complex coefficient ``c_d`` encodes a
    hopping amplitude.  The generator exhaustively enumerates all such
    polynomials for the supplied parameter space, deduplicates reciprocal
    duplicates, converts them to *poly2graph* graph instances, and stores the
    result in compressed ``npz`` archives that are small enough to ship to ML
    pipelines.

    Parameters
    ----------
    z, E
        SymPy symbols used for :math:`z` and energy :math:`E`.
    hopping_range : int | Sequence[int]
        Maximum hopping distance *D*.  A sequence enables a sweep over several
        ``D`` values in one call.
    params : Sequence[sympy.Symbol]
        Symbols that will replace some of the *unit* hopping parameters
        :math:`c_d` by free parameters (typically noted ``a``, ``b`` …).
    num_bands : int | Sequence[int], default ``(1, 2, 3)``
        Number(s) of energy bands :math:`b` to build.
    n_jobs : int, default ``-1``
        Degree of parallelism for :pyclass:`joblib.Parallel`; ``-1`` uses all
        logical CPUs.

    Notes
    -----
    * The constructor *only* validates that the requested symbol count does not
      exceed the available degrees of freedom (``len(params) ≤ D‑2``).
    * Heavy lifting (polynomial construction, graph building) is delegated to
      :pyclass:`joblib.Parallel`, so the class scales well to many‑core
      machines.
    """

    # ────────────────────────────────────────────────────────────────────────
    # Construction helpers
    # ────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        z: sp.Symbol,
        E: sp.Symbol,
        hopping_range: Sequence[int] | int,
        params: Sequence[sp.Symbol],
        num_bands: Sequence[int] | int = (1, 2, 3),
        n_jobs: int = -1,
    ) -> None:
        # Normalise iterable inputs ------------------------------------------------
        self.hopping_ranges: List[int] = (
            [hopping_range] if isinstance(hopping_range, int) else list(hopping_range)
        )
        self.num_bands: List[int] = (
            [num_bands] if isinstance(num_bands, int) else list(num_bands)
        )

        # Core attributes ----------------------------------------------------------
        self.z = z
        self.E = E
        self.params: Tuple[sp.Symbol, ...] = tuple(params)
        self.n_jobs = n_jobs

        # Parameter‑vs‑model sanity checks -----------------------------------------
        self.num_params = len(self.params)
        for D in self.hopping_ranges:
            if D < 2:
                raise ValueError("hopping_range must be ≥ 2 (got {D})")
            max_params = D - 2
            if self.num_params > max_params:
                raise ValueError(
                    f"num_params (={self.num_params}) exceeds D‑2 (={max_params})"
                )

    # ────────────────────────────────────────────────────────────────────────
    # Private – build a single polynomial
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _one_poly(
        z: sp.Symbol,
        E: sp.Symbol,
        D: int,
        params: Tuple[sp.Symbol, ...],
        b: int,
        q: int,
        E_deg_list: Tuple[int, ...],
        param_pos: Tuple[int, ...],
    ) -> Tuple[Tuple[int, int, int], str, sp.Poly, Dict[str, Any]]:
        """Helper for :pymeth:`generate_char_poly_classes` – **build exactly
        one** SymPy :pyclass:`~sympy.Poly` plus metadata and a reciprocal
        de‑duplication key.
        """
        # Complementary exponent range -------------------------------------------
        p = D - q
        z_degs = [d for d in range(-p + 1, q) if d != 0]

        # Base polynomial:  −E^b + z^q + z^{‑p} -----------------------------------
        expr = -E ** b + z ** q + z ** (-p)
        param_map = dict(zip(param_pos, params))

        # Add user‑controlled and/or *unit* hopping terms -------------------------
        for z_deg, E_deg in zip(z_degs, E_deg_list):
            has_param = z_deg in param_map
            if E_deg > 0 and has_param:
                expr += param_map[z_deg] * E ** E_deg * z ** z_deg
            elif E_deg > 0:
                expr += E ** E_deg * z ** z_deg
            elif has_param:
                expr += param_map[z_deg] * z ** z_deg

        # Expand & convert --------------------------------------------------------
        expr = sp.expand(expr)
        poly = sp.Poly(expr, z, 1 / z, E)

        # Metadata – rich enough for later filtering -----------------------------
        meta: Dict[str, Any] = {
            "parameter_symbols": tuple(str(p) for p in params),
            "generator_symbols": tuple(str(g) for g in poly.gens),
            "latex": sp.latex(expr),
            "sympy_repr": sp.srepr(poly),
            "number_of_bands": b,
            "max_left_hopping": q,
            "max_right_hopping": p,
            "intermediate_z_degrees": tuple(z_degs),
            "E_deg_assignments": E_deg_list,
            "param_degree_placements": param_pos,
        }

        # Reciprocal de‑duplication key ------------------------------------------
        flags = [1 if d in param_pos else 0 for d in z_degs]
        key = list(zip(z_degs, E_deg_list, flags))
        key_reciprocal = [(-d, e, f) for d, e, f in reversed(key)]
        final_key = min(tuple(key), tuple(key_reciprocal))

        return final_key, str(expr), poly, meta

    # ────────────────────────────────────────────────────────────────────────
    # Public – polynomial enumeration
    # ────────────────────────────────────────────────────────────────────────

    def generate_char_poly_classes(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Enumerate *all* distinct characteristic polynomial classes for the
        configured parameter space.

        De‑duplicates reciprocal pairs on the fly and aggregates expressions as
        raw strings as well as rich metadata dicts.
        """
        all_exprs: List[str] = []
        all_metas: List[Dict[str, Any]] = []

        for D in self.hopping_ranges:
            tasks: List[Tuple[Any, ...]] = []
            for b in self.num_bands:
                for q in range(1, D):
                    p = D - q
                    degs = [d for d in range(-p + 1, q) if d != 0]
                    for E_deg_list in it.product(range(b), repeat=D - 2):
                        for pos in it.combinations(degs, len(self.params)):
                            tasks.append((b, q, E_deg_list, pos))

            print(f"[D = {D}] raw samples: {len(tasks)}")
            build = partial(self._one_poly, self.z, self.E, D, self.params)

            results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(build)(*t) for t in tasks
            )
            if not results:
                continue

            # Unpack & reciprocal‑dedup -----------------------------------------
            keys, expr_strs, _, metas = zip(*results)
            keys_arr = np.asarray(keys)
            _, unique_idx = np.unique(keys_arr, axis=0, return_index=True)

            all_exprs.extend(expr_strs[i] for i in unique_idx)
            all_metas.extend(metas[i] for i in unique_idx)
            print(f"[D = {D}] unique classes: {len(unique_idx)}")

        return all_exprs, all_metas

    # ────────────────────────────────────────────────────────────────────────
    # Dataset generation for *one* class id
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def generate_dataset(
        class_idx: int,
        class_data: str = "./ParamTable.npz",
        save_dir: str = "./raw",
        num_partition: int = 10,
        short_edge_threshold: int = 30,
    ) -> None:
        """Generate a *spectral‑graph* dataset for a **single** polynomial class.

        The heavy lifting is delegated to :pyclass:`poly2graph.CharPolyClass`.
        All graphs are serialised via :pyfunc:`pickle.dumps` and saved into an
        ``npz`` file that additionally contains the exact parameter values used
        for each graph.
        """
        # — Working symbols ------------------------------------------------------
        k, z, E, a, b = sp.symbols("k z E a b")
        params = {a, b}

        # — Load characteristic polynomials + parameter sweeps ------------------
        data = np.load(class_data, allow_pickle=True)
        polys: np.ndarray = data["polys"]
        metas: np.ndarray = data["metas"]
        param1_vals, param2_vals = data["param_vals"]

        # — Split parameter grid into roughly equal partitions ------------------
        p1_parts = np.array_split(param1_vals, num_partition)
        p2_parts = np.array_split(param2_vals, num_partition)

        print(f"[Class {class_idx}] poly: {polys[class_idx]}")
        cp = p2g.CharPolyClass(polys[class_idx], k, z, E, params)
        os.makedirs(save_dir, exist_ok=True)

        batcher = Parallel()  # local thread‑pool for pickle.dumps

        for i, (v1, v2) in enumerate(zip(p1_parts, p2_parts)):
            print(f"[Class {class_idx}] Partition {i} …")
            t0 = time.perf_counter()
            graphs, _ = cp.spectral_graph(
                {a: v1, b: v2}, short_edge_threshold=short_edge_threshold
            )

            # — Write partition --------------------------------------------------
            out = os.path.join(save_dir, f"class_{class_idx}_part_{i}.npz")
            graphs_ser = batcher(delayed(pickle.dumps)(g) for g in graphs)
            np.savez_compressed(
                out,
                graphs_pickle=graphs_ser,
                a_vals=v1,
                b_vals=v2,
            )
            dt = (time.perf_counter() - t0) / 60
            print(
                f"[Class {class_idx}] Partition {i} → {out} "
                f"({len(graphs_ser)} graphs, {dt:.2f} min)"
            )

        # — Consolidate all partitions -----------------------------------------
        all_pickles: List[bytes] = []
        for i in range(num_partition):
            part_path = os.path.join(save_dir, f"class_{class_idx}_part_{i}.npz")
            d = np.load(part_path, allow_pickle=True)
            all_pickles.extend(d["graphs_pickle"].tolist())
            os.remove(part_path)

        out = os.path.join(save_dir, f"class_{class_idx}.npz")
        np.savez_compressed(
            out,
            graphs_pickle=np.array(all_pickles, dtype=object),
            y=class_idx,
            a_vals=param1_vals,
            b_vals=param2_vals,
            **metas[class_idx],
        )
        print(f"[Class {class_idx}] dataset → {out}")

    # ────────────────────────────────────────────────────────────────────────
    # Metadata utilities
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def filter_metas(
        metas: Sequence[Mapping[str, Any]],
        **criteria: Any,
    ) -> List[int]:
        """Return *indices* of ``metas`` that satisfy **all** key–value filters.

        Examples
        --------
        >>> idx = HSG_Generator.filter_metas(
        ...     metas,
        ...     number_of_bands=2,
        ...     max_left_hopping=3,
        ... )

        Passing ``None`` for a criterion disables filtering on that key. Keys
        that are missing from an entry never match and therefore remove that
        candidate.
        """
        selected: List[int] = []
        for i, m in enumerate(metas):
            if all(v is None or m.get(k) == v for k, v in criteria.items()):
                selected.append(i)
        return selected

    # ────────────────────────────────────────────────────────────────────────
    # Dataset consolidation – flexible subset selector
    # ────────────────────────────────────────────────────────────────────────

    def select_subset(
        self,
        metas: List[Dict[str, Any]],
        *,
        input_dir: str = "./raw",
        output_dir: str = ".",
        meta_filters: Mapping[str, Any] | None = None,
        a_vals_filter: Sequence[Any] | None = None,
        b_vals_filter: Sequence[Any] | None = None,
    ) -> str:
        """Consolidate all *graph* partitions that match the supplied filters.

        Parameters
        ----------
        metas : list of dict
            Global metadata list obtained from
            :pymeth:`generate_char_poly_classes`.
        input_dir, output_dir : str, optional
            Where to read partition files from and where to write the
            consolidated archive to.
        meta_filters : mapping, optional
            Dictionary of *exact* key–value pairs that an entry must satisfy to
            be included.  ``None`` disables metadata filtering.
        a_vals_filter, b_vals_filter : sequence, optional
            Restrict the parameter sweeps along *a* and/or *b* to the provided
            discrete values.

        Returns
        -------
        str
            Absolute path of the consolidated ``npz`` file.
        """
        if meta_filters is None:
            meta_filters = {}

        cls_ids = self.filter_metas(metas, **meta_filters)
        if not cls_ids:
            raise ValueError("No classes match the given meta_filters.")

        combined_pickles: List[bytes] = []
        combined_a: MutableSequence[Any] = []
        combined_b: MutableSequence[Any] = []
        combined_metas: MutableSequence[Dict[str, Any]] = []

        for cid in cls_ids:
            pattern = os.path.join(input_dir, f"class_{cid}_part_*.npz")
            for part in sorted(glob.glob(pattern)):
                data = np.load(part, allow_pickle=True)
                pkl, av, bv = data["graphs_pickle"], data["a_vals"], data["b_vals"]

                mask = np.ones(len(av), dtype=bool)
                if a_vals_filter is not None:
                    mask &= np.isin(av, a_vals_filter)
                if b_vals_filter is not None:
                    mask &= np.isin(bv, b_vals_filter)

                combined_pickles.extend(pkl[mask].tolist())
                combined_a.extend(av[mask].tolist())
                combined_b.extend(bv[mask].tolist())
                combined_metas.extend([metas[cid]] * mask.sum())

        # — Derive a descriptive output name ------------------------------------
        fname = ["subset"]
        for key, val in (meta_filters or {}).items():
            if val is not None:
                fname.append(f"{key}_{val}")
        if a_vals_filter is not None:
            fname.append("a_" + "_".join(map(str, a_vals_filter)))
        if b_vals_filter is not None:
            fname.append("b_" + "_".join(map(str, b_vals_filter)))
        out_path = os.path.join(output_dir, "_".join(fname) + ".npz")

        np.savez_compressed(
            out_path,
            graphs_pickle=np.array(combined_pickles, dtype=object),
            a_vals=np.array(combined_a),
            b_vals=np.array(combined_b),
            metas=np.array(combined_metas, dtype=object),
        )
        return os.path.abspath(out_path)

    # ────────────────────────────────────────────────────────────────────────
    # Support routine – build global parameter grid table
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def generate_ParamTable(
        save_dir: str,
        file_name: str,
        exprs: List[str],
        metas: List[Dict[str, Any]],
        real_coeff_walk: Sequence[float],
        imag_coeff_walk: Sequence[complex],
    ) -> None:
        """Pre‑compute the dense parameter table used by
        :pymeth:`generate_dataset`.
        """
        real_vals = np.asarray(real_coeff_walk)
        imag_vals = np.asarray(imag_coeff_walk)

        param_grid = real_vals[:, None] + imag_vals[None, :]
        param_vals = param_grid.ravel()

        p1, p2 = np.meshgrid(param_vals, param_vals)
        param_vals_pair = np.asarray([p1.ravel(), p2.ravel()])

        os.makedirs(save_dir, exist_ok=True)
        if not file_name.endswith(".npz"):
            file_name += ".npz"
        path = os.path.join(save_dir, file_name)
        np.savez(
            path,
            polys=np.array(exprs, dtype=object),
            metas=np.array(metas, dtype=object),
            param_vals=param_vals_pair,
        )
        print(f"Parameter table → {path}")
