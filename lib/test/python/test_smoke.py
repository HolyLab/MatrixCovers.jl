"""Smoke test for the `matrixcovers` Python package.

Reference values are those of MatrixCovers.jl on the same inputs.
"""
import sys

import numpy as np

import matrixcovers as mc

failures = []


def check(name, cond):
    if cond:
        print(f"ok: {name}")
    else:
        print(f"FAIL: {name}")
        failures.append(name)


def close(x, ref, rtol=1e-8):
    return np.allclose(x, ref, rtol=rtol, atol=0)


def raises(name, exc, f, test=lambda e: True):
    try:
        f()
    except exc as e:
        check(name, test(e))
    except Exception as e:  # noqa: BLE001
        print(f"  unexpected {type(e).__name__}: {e}")
        check(name, False)
    else:
        check(name, False)


As = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
Aa = np.array([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0], [0.5, 0.0, 2.0]])
V = np.array([[3.0, 0.0, 1.0], [0.0, 2.0, 5.0], [4.0, 1.0, 0.0]])

# --- Heuristic hard covers ------------------------------------------------------

a = mc.symcover(As)
check("symcover", close(a, [2.0000000000000004, 1.7320508075702776, 1.4142135623730951]))
check("symcover covers", mc.iscover(a, As))

ab = mc.cover(Aa)
check("cover", close(ab[0], [1.2256529170433434, 4.359106692464643, 0.8171019446955622])
      and close(ab[1], [1.376428801426651, 1.6317833313077115, 2.4476749969615677]))
check("cover covers", mc.iscover(ab[0], Aa, ab[1]))
a2, b2 = mc.cover(Aa, start="geomean")
check("cover(start='geomean')", close(a2, [1.2579400817731892, 4.317698277786294, 0.7900840937475638]))
a3, b3 = mc.cover(Aa, [1.0, 2.0, 3.0])
check("cover with fixed rows", close(a3, [1.0, 2.0, 3.0]) and close(b3, [3.0, 2.5, 3.0]))

# Arrays of another dtype or memory order are converted.
check("integer input", close(mc.symcover(np.array([[4, 1], [1, 4]])), [2.0, 2.0]))
check("C- and Fortran-order input agree",
      close(mc.cover(np.ascontiguousarray(Aa))[0], mc.cover(np.asfortranarray(Aa))[0], rtol=0))

# --- Minimal hard covers ----------------------------------------------------------

a, stats = mc.symcover_min(As)
check("symcover_min", close(a, [2.0, 1.7320508075688787, 1.4142135623730965], rtol=1e-6))
check("symcover_min stats", isinstance(stats, mc.SolverStats) and stats.converged is True
      and stats.iterations > 0 and stats.residual <= stats.tolerance)
a_member, _ = mc.symcover_min(As, penalty=mc.Penalty.abslog2)
a_string, _ = mc.symcover_min(As, penalty="abslog2", linsolve="dense")
check("penalty as member or string", close(a_member, a) and close(a_string, a, rtol=1e-6))

a, b, stats = mc.cover_min(Aa)
check("cover_min", close(a, [1.225652917043481, 4.359106692464615, 0.8171019446954835], rtol=1e-6)
      and close(b, [1.3764288014266595, 1.6317833313075425, 2.447674996961822], rtol=1e-6)
      and stats.converged)

a, b, stats = mc.cover_transversal(Aa)
check("cover_transversal", close(a, [1.2256529170433457, 4.359106692464852, 0.8171019446956469], rtol=1e-6)
      and close(b, [1.3764288014267252, 1.6317833313078742, 2.447674996961562], rtol=1e-6)
      and stats.converged)
a, b, stats = mc.cover_transversal(Aa, penalty="powermean2", target=0.1, maxbalance=True)
check("cover_transversal(powermean2)", mc.iscover(a, Aa, b, rtol=1e-12) and stats.converged)

# --- Soft covers --------------------------------------------------------------------

a, stats = mc.soft_symcover(As)
check("soft_symcover", close(a, [1.562232778093002, 1.3444663804879606, 1.0842729416188959], rtol=1e-6)
      and stats.converged)
a, stats = mc.soft_symcover(As, penalty="abslog2")
check("soft_symcover(abslog2)", close(a, [1.363879622416882, 0.9130259591914891, 0.6330568425802215], rtol=1e-6))
check("abslog2 soft stats are None", stats == mc.SolverStats(None, None, None, None))
a, b, stats = mc.soft_cover(Aa)
check("soft_cover", close(a, [1.2507036306044128, 3.9635731170047523, 0.7270572991132858], rtol=1e-6)
      and stats.converged)
a, b, stats = mc.noise_cover(V)
check("noise_cover", close(a, [0.8074142945783362, 1.625852803134376, 1.0352995238411953], rtol=1e-6)
      and close(b, [1.6667022698244278, 0.750460466860714, 1.06855679786747], rtol=1e-6))
check("noise_cover balances rows", close((V / np.outer(a, b) ** 2).sum(axis=1), [3.0, 3.0, 3.0], rtol=1e-8))

# --- Starting points ----------------------------------------------------------------

check("initialize_symcover",
      close(mc.initialize_symcover(As, strategy="diagonal"), [3.0468307578901657, 2.039648902655506, 1.414213562373095]))
a, b = mc.initialize_cover(Aa, strategy=mc.Strategy.unconstrained)
check("initialize_cover", close(a, [1.5879291144626209, 4.310302762201199, 0.9167913016882971]))

# --- Gram covers, predicates, objectives ----------------------------------------------

check("gramcover", close(mc.gramcover(Aa), [6.103277807866859, 5.38516480713451, 5.385164807134511]))
check("gramcover(w)", close(mc.gramcover(Aa, [1.0, 2.0, 3.0]), [8.587782018658846, 7.3484692283495425, 7.280109889280529]))
check("gramcover(W)", close(mc.gramcover(Aa, As), [13.099411626608232, 12.660254037851402, 15.756630355027324]))
s = mc.gramcover(Aa, As, v=mc.symcover(As))
check("gramcover(W, v)", close(s, [13.099411626608232, 12.660254037851402, 15.756630355027324]))

a = mc.symcover(As)
ref = [10.961277846688837, 21.646276228291487, 4.905797024033771, 4.054094048064017,
       6.055480822655063, 8.082527846687071]
check("cover_objective, every penalty",
      close([mc.cover_objective(a, As, penalty=p) for p in mc.Penalty], ref, rtol=1e-10))
ab = mc.cover(Aa)
check("max_coverage_ratio", abs(mc.max_coverage_ratio(ab[0], Aa, ab[1]) - 1) < 1e-12)
a_min, b_min, _ = mc.cover_min(Aa)
ld = mc.level_dispersion(ab[0], a_min, Aa, ab[1], b_min)
check("level_dispersion", isinstance(ld, mc.LevelDispersion) and abs(ld.level) < 1e-10 and ld.dispersion < 1e-10)

# --- Error paths ------------------------------------------------------------------------

raises("unknown penalty string names valid members", ValueError,
       lambda: mc.symcover_min(As, penalty="not-a-penalty"),
       lambda e: all(n in str(e) for n in ("abslog1", "abslog2", "powermean2")))
raises("extension-only penalty raises JLWError code 2", mc.JLWError,
       lambda: mc.symcover_min(As, penalty="abslinear2"),
       lambda e: e.code == 2 and "Ipopt" in e.message)
raises("abslog1 names HiGHS", mc.JLWError,
       lambda: mc.soft_cover(Aa, penalty="abslog1"),
       lambda e: e.code == 2 and "HiGHS" in e.message)
raises("shape mismatch raises JLWError code 3", mc.JLWError,
       lambda: mc.iscover(np.ones(5), As), lambda e: e.code == 3)
raises("option for another penalty is rejected", mc.JLWError,
       lambda: mc.soft_cover(Aa, penalty="abslog2", maxiter=3), lambda e: e.code == 2)
raises("linsolve of another solver is rejected", mc.JLWError,
       lambda: mc.cover_min(Aa, linsolve="cg"), lambda e: e.code == 2)
raises("cover_transversal rejects target for abslog2", TypeError,
       lambda: mc.cover_transversal(Aa, target=0.5))
raises("noise_cover reports an impossible zero pattern", mc.JLWError,
       lambda: mc.noise_cover(np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 0.0]])),
       lambda e: "no scaling" in e.message)

if failures:
    print(f"\n{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("\nALL CHECKS PASSED")
