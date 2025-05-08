import os
import glob
import time
import pickle
from functools import partial
import numpy as np
import sympy as sp
import itertools as it
from joblib import Parallel, delayed
from multiprocessing.pool import ThreadPool
import poly2graph as p2g
from typing import Sequence, Tuple, Dict, List


class NHSG12M:
    """
    A class to generate polynomial-based spectral graph datasets
    and manage their combination and parameter table generation.

    Attributes
    ----------
    z : sp.Symbol
        Symbol for the complex variable z=exp(iK).
    E : sp.Symbol
        Symbol for the energy variable E.
    hopping_ranges : List[int] or int
        One or more hopping ranges (D values) to generate polynomials for.
    params : Tuple[sp.Symbol, ...]
        Symbols used as parameters in the polynomial construction.
    num_bands : List[int] or int
        List of number of energy bands (b) to consider.
    n_jobs : int
        Number of parallel jobs to run (-1 uses all available cores).
    """

    def __init__(
        self,
        z: sp.Symbol,
        E: sp.Symbol,
        hopping_range: Sequence[int] | int,
        params: Sequence[sp.Symbol],
        num_bands: Sequence[int] = [1, 2, 3],
        n_jobs: int = -1,
    ):
        """
        Initialize the P2GCreator.

        Parameters
        ----------
        z : sp.Symbol
            Symbol for z in the polynomials.
        E : sp.Symbol
            Symbol for E in the polynomials.
        hopping_range : int or list of int
            Single D value or list of D values to generate polynomials.
        params : Sequence[sp.Symbol]
            Sequence of parameter symbols (e.g., a, b).
        num_bands : Sequence[int], optional
            List of number of bands (b) to consider, by default [1, 2, 3].
        n_jobs : int, optional
            Number of parallel jobs for computations, by default -1.
        """
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
        self.num_bands = num_bands
        self.n_jobs = n_jobs

        # Validity check for parameters
        self.num_params = len(self.params)
        for D in self.hopping_ranges:
            assert D >= 2, f"hopping_range ({D}) must be ≥ 2"
            assert self.num_params <= D - 2, \
                f"num_params (= {self.num_params}) must not exceed D-2 ({D}-2)"

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
    ) -> Tuple[Tuple[int, int, int], str, sp.Poly, Dict]:
        """
        Build a single characteristic polynomial expression, its Sympy Poly,
        and metadata describing construction parameters.

        Parameters
        ----------
        z : sp.Symbol
            Symbol for z.
        E : sp.Symbol
            Symbol for energy E.
        D : int
            Hopping range parameter (total range of z exponents).
        params : Tuple[sp.Symbol]
            Symbols for the coefficients to include.
        b : int
            Number of bands (power of E in the base term).
        q : int
            Maximum positive z exponent in the base term.
        E_deg_list : Tuple[int]
            Exponents for E in the additional terms.
        param_pos : Tuple[int]
            z exponents at which parameter symbols are inserted.

        Returns
        -------
        final_key : tuple
            Lexicographically minimal key for de-duplication (z-degree, E-degree, flag).
        expr_str : str
            String representation of the expanded polynomial.
        poly : sp.Poly
            Sympy Poly object over (z, 1/z, E).
        meta : dict
            Metadata with construction details and latex repr.
        """
        # Compute complementary exponent range
        p = D - q
        z_degs = [d for d in range(-p + 1, q) if d != 0]

        # Start with the base polynomial -E^b + z^q + z^{-p}
        expr = -E**b + z**q + z**(-p)
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
            'number_of_bands': b,
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

    def generate_char_poly_classes(
        self
    ) -> Tuple[List[str], List[Dict]]:
        """
        Generate unique characteristic polynomial classes across all specified D values.

        Iterates each D in hopping_ranges, builds all (b, q, E_deg_list, param_pos)
        combinations, runs them in parallel, removes reciprocal duplicates, and
        returns combined lists of expression strings and metadata dicts.

        Returns
        -------
        all_exprs : List[str]
            Unique polynomial expressions as strings.
        all_metas : List[Dict]
            Metadata dictionaries matching each expression.
        """
        all_exprs = []  # accumulator for expression strings
        all_metas = []  # accumulator for corresponding metadata

        for D in self.hopping_ranges:
            # Assemble tasks for this D
            tasks = []
            for b in self.num_bands:
                for q in range(1, D):
                    p = D - q
                    degs = [d for d in range(-p + 1, q) if d != 0]
                    for E_deg_list in it.product(range(b), repeat=D-2):
                        for pos in it.combinations(degs, len(self.params)):
                            tasks.append((b, q, E_deg_list, pos))

            print(f"[Hopping range = {D}] raw samples: {len(tasks)}")
            build = partial(self._one_poly, self.z, self.E, D, self.params)

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

    @staticmethod
    def generate_dataset(
        class_idx: int,
        class_data: str = "./ParamTable.npz",
        save_dir: str = "./poly_class_dataset",
        num_partition: int = 10,
        short_edge_threshold: int = 30,
    ) -> None:
        """
        Generate and save spectral graph data for one polynomial class.

        Reads 'polys' and 'param_vals' from class_data NPZ, splits parameters
        into partitions, builds CharPolyClass, and stores pickled graphs.
        """
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

    @staticmethod
    def filter_metas(
        metas: List[Dict],
        number_of_bands=None,
        max_left_hopping=None,
        max_right_hopping=None,
        intermediate_z_degrees=None,
        E_deg_assignments=None,
    ) -> List[int]:
        """
        Filter metadata entries by matching non-None criteria.

        Returns indices of metas where each specified key equals the desired value.
        """
        filters = {
            'number_of_bands': number_of_bands,
            'max_left_hopping': max_left_hopping,
            'max_right_hopping': max_right_hopping,
            'intermediate_z_degrees': intermediate_z_degrees,
            'E_deg_assignments': E_deg_assignments,
        }
        matched = []
        for i, m in enumerate(metas):
            if all(desired is None or m.get(key) == desired
                   for key, desired in filters.items()):
                matched.append(i)
        return matched

    def combine_class_parts(
        self,
        metas: List[Dict],
        input_dir: str = './poly_class_dataset',
        output_dir: str = '.',
        number_of_bands=None,
        max_left_hopping=None,
        max_right_hopping=None,
        intermediate_z_degrees=None,
        E_deg_assignments=None,
        a_vals_filter=None,
        b_vals_filter=None,
    ) -> str:
        """
        Combine all dataset partitions matching metadata filters into one NPZ.

        Searches for files named 'class_<cid>_part_*.npz' in input_dir,
        filters by a_vals_filter and b_vals_filter, and consolidates.

        Returns
        -------
        out_path : str
            Path of the combined NPZ archive.
        """
        cls_ids = self.filter_metas(
            metas,
            number_of_bands,
            max_left_hopping,
            max_right_hopping,
            intermediate_z_degrees,
            E_deg_assignments,
        )

        combined_pickles, combined_a, combined_b, combined_metas = [], [], [], []
        for cid in cls_ids:
            cls_meta = metas[cid]
            pattern = os.path.join(input_dir, f'class_{cid}_part_*.npz')
            parts = sorted(glob.glob(pattern))
            if not parts:
                raise FileNotFoundError(
                    f"No parts found for class {cid} in {input_dir}"
                )
            for part in parts:
                data = np.load(part, allow_pickle=True)
                pkl, av, bv = data['graphs_pickle'], data['a_vals'], data['b_vals']
                mask = np.ones(len(av), bool)
                if a_vals_filter is not None:
                    mask &= np.isin(av, a_vals_filter)
                if b_vals_filter is not None:
                    mask &= np.isin(bv, b_vals_filter)
                combined_pickles.extend(pkl[mask].tolist())
                combined_a.extend(av[mask].tolist())
                combined_b.extend(bv[mask].tolist())
                selected_count = mask.sum()   # or len(av[mask])
                combined_metas.extend([cls_meta] * selected_count)

        # Construct output file_name based on filters
        fname = 'classes_combined'
        if a_vals_filter is not None:
            fname += '_a_' + '_'.join(map(str, a_vals_filter))
        if b_vals_filter is not None:
            fname += '_b_' + '_'.join(map(str, b_vals_filter))
        fname += '.npz'
        out_path = os.path.join(output_dir, fname)

        # Save combined arrays
        np.savez_compressed(
            out_path,
            graphs_pickle=np.array(combined_pickles, dtype=object),
            a_vals=np.array(combined_a),
            b_vals=np.array(combined_b),
            metas=np.array(combined_metas, dtype=object),
        )
        return out_path

    @staticmethod
    def generate_ParamTable(
        save_dir: str,
        file_name: str,
        exprs: List[str],
        metas: List[Dict],
        real_coeff_walk,
        imag_coeff_walk,
    ) -> None:
        """
        Generate and save a parameter table with all combinations of real and imaginary parts.

        Creates a mesh of parameter1 vs parameter2 and stores along with
        provided polynomials and metadata.

        Parameters
        ----------
        save_dir : str
            Directory to save the parameter table.
        file_name : str
            Name of the output file. NPZ format.
        exprs : List[str]
            Polynomial expressions to include.
        metas : List[Dict]
            Corresponding metadata for each expression.
        real_coeff_walk : sequence
            Real coefficient values to sweep.
        imag_coeff_walk : sequence
            Imaginary coefficient values to sweep.
        """
        # Convert walks to numpy arrays
        real_vals = np.array(real_coeff_walk)
        imag_vals = np.array(imag_coeff_walk)

        # Sum real + imag parts to form complex parameter grid
        param_arr = real_vals[:, None] + imag_vals[None, :]
        param_vals = param_arr.ravel()

        # Generate all ordered pairs (param1, param2)
        p1, p2 = np.meshgrid(param_vals, param_vals)
        param_vals_pair = np.asarray([p1.ravel(), p2.ravel()])

        # Save expressions, metadata, and parameter grid
        os.makedirs(save_dir, exist_ok=True)
        if not file_name.endswith('.npz'):
            file_name += '.npz'
        np.savez(
            os.path.join(save_dir, file_name),
            polys=np.array(exprs, dtype=object),
            metas=np.array(metas, dtype=object),
            param_vals=param_vals_pair,
        )