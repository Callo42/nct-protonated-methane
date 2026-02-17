# %%
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import heapq
import numpy as np


# ======================== Physical constants (AU) =========================
# Boltzmann constant in atomic units (Hartree per Kelvin)
KB_AU_PER_K: float = 3.166811563e-6  # Ha/K (CODATA)


# ============================== Type aliases ==============================
# A level is (energy_Ha, occupations)
Level = Tuple[float, Tuple[int, ...]]


# =================== Enumerate the K lowest harmonic levels ===================
def k_lowest_harmonic_levels_au(
    wavenumbers_ha: Iterable[float],
    max_states: int,
) -> List[Level]:
    """Return the lowest `max_states` harmonic levels (energy in Ha, occupations).

    Enumerates the first `max_states` energies of the non-negative integer lattice:
        E(n) = sum_i n_i * w_i,  with n_i ∈ {0,1,2,...}
    including the ground state n=(0,...,0). All energies and inputs are **Hartree**.

    Uses a k-smallest-sums heap-merge across modes, treating each mode as the
    arithmetic progression {0, w_i, 2 w_i, ...}. Complexity ~ O(M * K log K),
    with M = number of modes and K = `max_states`.

    Args:
        wavenumbers_ha: Iterable of mode “frequencies” (Ha), length M, all ≥ 0.
        max_states: Positive integer number of states to return (including ground).

    Returns:
        A list of length `max_states` with elements:
            (energy_Ha: float, occupations: Tuple[int, ...])

    Raises:
        ValueError: If inputs are invalid (empty, negative values, or max_states ≤ 0).
        TypeError: If `wavenumbers_ha` is not an iterable of real finite numbers.

    Examples:
        >>> import numpy as np
        >>> w = np.array([0.004, 0.007], dtype=np.float64)  # Hartree
        >>> levels = k_lowest_harmonic_levels_au(w, max_states=8)
        >>> [round(e, 6) for e, _ in levels]
        [0.0, 0.004, 0.007, 0.008, 0.011, 0.012, 0.014, 0.015]
        >>> levels[0][1]
        (0, 0)
    """
    # Validate wavenumbers
    try:
        w = np.asarray(list(wavenumbers_ha), dtype=np.float64)
    except Exception as exc:  # pragma: no cover
        raise TypeError("`wavenumbers_ha` must be an iterable of real numbers.") from exc

    if w.ndim != 1 or w.size == 0:
        raise ValueError("`wavenumbers_ha` must be a non-empty 1D sequence.")
    if not np.all(np.isfinite(w)):
        raise ValueError("All values in `wavenumbers_ha` must be finite (no NaN/inf).")
    if np.any(w < 0):
        raise ValueError("All values in `wavenumbers_ha` must be non-negative.")
    if not isinstance(max_states, int) or max_states <= 0:
        raise ValueError("`max_states` must be a positive integer.")

    # Start with ground state for zero modes
    results: List[Level] = [(0.0, tuple())]

    def merge_with_mode(
        current: List[Level],
        w_i: float,
        keep_k: int,
    ) -> List[Level]:
        """k-smallest sums of `current` with {j*w_i | j ≥ 0}, using a min-heap.

        Each heap item is (sum_energy, index_in_current, quanta_for_this_mode).
        On pop, push the (j+1) successor for the same base index.
        """
        if keep_k <= 0:
            return []

        heap: List[Tuple[float, int, int]] = []
        for i, (e_base, _) in enumerate(current[:keep_k]):  # seeding beyond k is unnecessary
            heap.append((e_base, i, 0))
        heapq.heapify(heap)

        merged: List[Level] = []
        while heap and len(merged) < keep_k:
            e_sum, i, j = heapq.heappop(heap)
            e_base, occ_base = current[i]
            merged.append((e_sum, occ_base + (j,)))
            heapq.heappush(heap, (e_base + (j + 1) * w_i, i, j + 1))

        return merged

    for w_i in w:
        results = merge_with_mode(results, float(w_i), max_states)

    if len(results) > max_states:
        results = results[:max_states]

    # Ensure full-length occupation tuples
    M = int(w.size)
    fixed: List[Level] = []
    for e, occ in results:
        if len(occ) != M:
            occ = occ + (0,) * (M - len(occ))
        fixed.append((float(e), occ))

    return fixed


# =================== Boltzmann probabilities (atomic units) ===================
def boltzmann_probabilities_au(
    levels: Sequence[Level],
    temperature_K: float,
) -> np.ndarray:
    """Compute **normalized**, numerically stable Boltzmann probabilities in AU.

    All energies are in Hartree, and k_B is Hartree/K. Numerical stability is
    ensured by shifting energies by the minimum before exponentiation:

        p_i = exp(-(E_i - E_min) / (k_B * T)) / sum_j exp(...)

    Args:
        levels: Sequence of (energy_Ha, occupations).
        temperature_K: Absolute temperature in Kelvin (must be > 0).

    Returns:
        A (N,) NumPy array of probabilities (float64) that sums to 1 (within FP error).

    Raises:
        ValueError: If `levels` is empty or `temperature_K` ≤ 0.
        TypeError: If any energy is non-finite.

    Examples:
        >>> lvls = [(0.0, (0,)), (0.004, (1,))]
        >>> p = boltzmann_probabilities_au(lvls, temperature_K=300.0)
        >>> round(float(p.sum()), 12)
        1.0
        >>> p[0] > p[1]
        True
    """
    if temperature_K <= 0.0:
        raise ValueError("`temperature_K` must be > 0.")
    if len(levels) == 0:
        raise ValueError("`levels` must be non-empty.")

    E = np.array([e for (e, _) in levels], dtype=np.float64)
    if not np.all(np.isfinite(E)):
        raise TypeError("All level energies must be finite (no NaN/inf).")

    E0 = float(E.min())
    kT = KB_AU_PER_K * float(temperature_K)

    exponents = -(E - E0) / kT
    f = np.exp(exponents)

    Z = float(f.sum())
    if Z == 0.0:
        # Underflow safety: distribute mass uniformly across minimum-energy states
        probs = np.zeros_like(f)
        is_min = (E == E0)
        probs[is_min] = 1.0 / float(np.sum(is_min))
        return probs

    return f / Z



# ========================= Optional unit helpers =========================

if __name__ == "__main__":
    from neuralvib.utils.convert import convert_inverse_cm_to_hartree

    # Suppose your original list was in cm^-1; convert to Hartree first (optional helper).
    w_cm = np.array(
        [
            199.9, 839.4, 1296.7, 1303.5, 1477.7, 1499.9,
            1586.7, 2417.9, 2707.7, 3001.0, 3132.9, 3224.4,
        ],
        dtype=np.float64,
    )
    w_ha = convert_inverse_cm_to_hartree(w_cm)  # <-- All atomic units from here on

    # 1) Lowest 136 states in Hartree
    levels_20 = k_lowest_harmonic_levels_au(w_ha, max_states=136)
    # Ground state checks
    assert levels_20[0][0] == 0.0
    assert levels_20[0][1] == tuple([0]*len(w_ha))
    # Monotone non-decreasing energies
    energies_ha = np.array([e for e, _ in levels_20])
    assert np.all(np.diff(energies_ha) >= -1e-12)

    p = boltzmann_probabilities_au(levels_20, temperature_K=1000.0)
    # Expected outputs (deterministic):
    print(round(float(p.sum()), 12))                # -> 1.0
    print([round(float(x), 6) for x in p[:5]])     # First 5 probabilities, descending
    assert np.all(np.diff(p) <= 1e-12)             # Non-increasing with energy
    print(p)

# %%
    import sys; sys.path.append("../../")

# %%
