"""
Numerical comparison of the rank-computation methods exposed by
``Ranking_exp`` (in ``Figures/Ranking_exp.py``).

Methods checked
---------------
* ``calc_rank()``                : numerical -- builds K_exp, solves
                                   ``(K_exp + (1/c_til) I) a_exp = y_exp``
                                   with ``np.linalg.solve``, then post-processes
                                   with the analytical D_ij/c~~ formulas and
                                   multiplies the final array by ``-1``.
* ``calc_rank_analytic()``       : refactored = ``calc_rank_original()`` +
                                   ``calc_rank_perturbative()``.
* ``calc_rank_analytic_prev()``  : older single-method analytical form.
* ``calc_rank_original()``       : unperturbed rank only (sinh ratio).
* ``calc_rank_perturbative()``   : perturbative ``rank_til`` only.

For each set of parameters we report ``max|Δ|`` and ``np.allclose`` for:

  1. ``calc_rank_analytic``       vs ``calc_rank_analytic_prev``
     -- pure refactor sanity check; should agree to machine precision.
  2. ``calc_rank_analytic``       vs ``calc_rank_original + calc_rank_perturbative``
     -- definitional identity.
  3. ``calc_rank``                vs ``calc_rank_analytic``
     -- direct comparison (no sign flip).
  4. ``calc_rank``                vs ``-calc_rank_analytic``
     -- comparison after accounting for the ``*-1`` at the end of
     ``calc_rank``.

Run from the ``Figures/`` directory:

    python compare_rank_methods.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make sure we import the local Ranking_exp.py, not a sibling copy.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from Ranking_exp import Ranking_exp


def make_sim(
    alpha: float,
    *,
    n: int = 9,
    p: int = 6,
    q: int = 4,
    k_s: float = 1.0,
    k_d: float = 0.0,
    c_reg: float = 1e10,
) -> Ranking_exp:
    """Build a Ranking_exp instance with k_o derived from alpha (matches Figure 4.1)."""
    k_o = ((1.0 - alpha) / 2.0) * (k_s - k_d) + k_d
    return Ranking_exp(n=n, k_o=k_o, k_s=k_s, k_d=k_d, p=p, q=q, c_reg=c_reg)


def _fmt_vec(x: np.ndarray) -> str:
    return np.array2string(x, precision=6, suppress_small=False, floatmode="fixed")


def _compare(label: str, a: np.ndarray, b: np.ndarray, atol: float = 1e-8, rtol: float = 1e-6) -> bool:
    diff = np.abs(np.asarray(a) - np.asarray(b))
    ok = bool(np.allclose(a, b, atol=atol, rtol=rtol))
    flag = "OK " if ok else "DIFF"
    print(f"  [{flag}] {label:<55s} max|diff|={diff.max():.3e}   allclose={ok}")
    return ok


def run_one(
    alpha: float,
    *,
    n: int = 9,
    p: int = 6,
    q: int = 4,
    k_s: float = 1.0,
    k_d: float = 0.0,
    c_reg: float = 1e10,
    show_arrays: bool = False,
) -> dict[str, bool]:
    print(
        f"\n=== alpha = {alpha:+.2f}   "
        f"(n={n}, p={p}, q={q}, k_s={k_s}, k_d={k_d}, c_reg={c_reg:g}) ==="
    )
    sim = make_sim(alpha, n=n, p=p, q=q, k_s=k_s, k_d=k_d, c_reg=c_reg)
    print(
        f"  derived: alpha={sim.alpha:+.6f}  alpha'={sim.alpha_prime:+.6f}  "
        f"lamb_val={sim.lamb_val:+.6f}  c_til={sim.c_til:g}"
    )

    r_num = sim.calc_rank()
    r_ana_new = sim.calc_rank_analytic()
    r_ana_prev = sim.calc_rank_analytic_prev()
    r_orig = sim.calc_rank_original()
    r_pert = sim.calc_rank_perturbative()

    if show_arrays:
        print("  calc_rank               =", _fmt_vec(r_num))
        print("  calc_rank_analytic      =", _fmt_vec(r_ana_new))
        print("  calc_rank_analytic_prev =", _fmt_vec(r_ana_prev))
        print("  calc_rank_original      =", _fmt_vec(r_orig))
        print("  calc_rank_perturbative  =", _fmt_vec(r_pert))

    results: dict[str, bool] = {}
    results["analytic_vs_analytic_prev"] = _compare(
        "calc_rank_analytic   vs   calc_rank_analytic_prev", r_ana_new, r_ana_prev
    )
    results["analytic_vs_orig_plus_pert"] = _compare(
        "calc_rank_analytic   vs   original + perturbative", r_ana_new, r_orig + r_pert
    )
    results["calc_rank_vs_analytic"] = _compare(
        "calc_rank            vs   calc_rank_analytic     ", r_num, r_ana_new
    )
    results["calc_rank_vs_neg_analytic"] = _compare(
        "calc_rank            vs  -calc_rank_analytic     ", r_num, -r_ana_new
    )
    return results


def main() -> int:
    np.set_printoptions(linewidth=140)
    print("Comparison of rank methods in Figures/Ranking_exp.py")
    print("=" * 78)

    alphas = (0.1, 0.2, 0.4, 0.6, 0.8)
    summary: list[tuple[float, dict[str, bool]]] = []
    for alpha in alphas:
        res = run_one(alpha, show_arrays=True)
        summary.append((alpha, res))

    print("\n" + "=" * 78)
    print("Summary across alphas:")
    headers = [
        ("analytic_vs_analytic_prev", "analytic == analytic_prev"),
        ("analytic_vs_orig_plus_pert", "analytic == orig + pert"),
        ("calc_rank_vs_analytic", "calc_rank == analytic"),
        ("calc_rank_vs_neg_analytic", "calc_rank == -analytic"),
    ]
    for key, title in headers:
        all_ok = all(res[key] for _, res in summary)
        per_alpha = ", ".join(
            f"a={a:+.2f}:{'OK' if res[key] else 'DIFF'}" for a, res in summary
        )
        print(f"  {title:<32s}  -> {'ALL OK' if all_ok else 'MISMATCH'}   ({per_alpha})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
