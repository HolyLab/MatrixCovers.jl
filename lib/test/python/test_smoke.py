"""Smoke test for the bundled `matrixcovers` wheel."""
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


A_sym = np.array([[4.0, 1.0], [1.0, 4.0]])
A_asym = np.array([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]])

# symcover / iscover
a = mc.symcover(A_sym)
check("symcover matches reference", np.allclose(a, [2.0, 2.0], rtol=1e-8))
check("symcover result covers A", mc.iscover(a, A_sym, rtol=1e-8))

# symcover_min
a_min = mc.symcover_min(A_sym)
check("symcover_min matches reference", np.allclose(a_min, [2.0, 2.0], rtol=1e-6))

# penalty accepted as a member and as a string
a_min_member = mc.symcover_min(A_sym, penalty=mc.Penalty.abslog2)
a_min_str = mc.symcover_min(A_sym, penalty="abslog2")
check("symcover_min(penalty=Penalty.abslog2) matches reference", np.allclose(a_min_member, [2.0, 2.0], rtol=1e-6))
check("symcover_min(penalty='abslog2') matches reference", np.allclose(a_min_str, [2.0, 2.0], rtol=1e-6))

# cover / cover_min
a_asym, b_asym = mc.cover(A_asym)
check(
    "cover matches reference",
    np.allclose(a_asym, [1.2544610775677627, 3.475905976749231], rtol=1e-6)
    and np.allclose(b_asym, [1.7261686708831454, 1.621762761307448, 2.3914651906272066], rtol=1e-6),
)

a_cmin, b_cmin = mc.cover_min(A_asym)
check(
    "cover_min matches reference",
    np.allclose(a_cmin, [1.1986299952850965, 3.2535823504068366], rtol=1e-6)
    and np.allclose(b_cmin, [1.8441211421157804, 1.6685716376905815, 2.502857440802694], rtol=1e-6),
)

# soft_symcover / soft_cover
a_soft = mc.soft_symcover(A_sym, penalty="abslog2")
check("soft_symcover(abslog2) matches reference", np.allclose(a_soft, [1.414213562373095, 1.414213562373095], rtol=1e-6))

# The default seed makes the multistart result reproducible.
a_soft_default = mc.soft_symcover(A_sym)
check("soft_symcover default matches reference",
      np.allclose(a_soft_default, [1.8439088914584778, 1.8439088914585735], rtol=1e-6))

a_soft_l1 = mc.soft_symcover(A_sym, penalty="abslinear1")
check("soft_symcover(abslinear1) matches reference", np.allclose(a_soft_l1, [2.0, 2.0], rtol=1e-6))

a_sc, b_sc = mc.soft_cover(A_asym)
check(
    "soft_cover default matches reference",
    np.allclose(a_sc, [1.0817791286952234, 2.7823842907260183], rtol=1e-6)
    and np.allclose(b_sc, [1.7867556808278862, 1.8232808727324803, 2.3172250172537487], rtol=1e-6),
)

# cover_objective
for penalty in ("abslog1", "abslog2", "abslinear1", "abslinear2"):
    val = mc.cover_objective(a, A_sym, penalty=penalty)
    check(f"cover_objective({penalty}) is finite", np.isfinite(val))

# gramcover
s = mc.gramcover(a_asym, b_asym, A_asym)
G = A_asym.T @ A_asym
check("gramcover covers A'*A", np.all(s[:, None] * s[None, :] >= np.abs(G) - 1e-8))

# error paths

try:
    mc.symcover_min(A_sym, penalty="not-a-penalty")
    check("unknown penalty string raises ValueError", False)
except ValueError as e:
    check("unknown penalty string raises ValueError", True)
    check("unknown penalty string names valid members",
          all(name in str(e) for name in ("abslog1", "abslog2", "abslinear1", "abslinear2")))

try:
    from matrixcovers import _lowlevel
    _lowlevel.matrixcovers_symcover_min(
        _lowlevel.CMatrix_borrowed_Float64.from_numpy(np.asfortranarray(A_sym)),
        99,
        _lowlevel.COpt_Int64.from_optional(None),
        1,
    )
    check("bad low-level penalty enum raises JLWError", False)
except mc.JLWError as e:
    check("bad low-level penalty enum raises JLWError", e.code == 2)

try:
    mc.iscover(np.zeros(5), A_sym)
    check("shape-mismatched call raises cleanly", False)
except mc.JLWError as e:
    check("shape-mismatched call raises cleanly", e.code == 3)

try:
    mc.symcover_min(A_sym, penalty="abslinear2")
    check("extension-only penalty raises JLWError code 2", False)
except mc.JLWError as e:
    check("extension-only penalty raises JLWError code 2", e.code == 2)

try:
    mc.symcover_min(A_sym, penalty="abslinear1")
    check("symcover_min(abslinear1) raises JLWError code 2 mentioning the extension", False)
except mc.JLWError as e:
    check(
        "symcover_min(abslinear1) raises JLWError code 2 mentioning the extension",
        e.code == 2 and "extension" in e.message,
    )

if failures:
    print(f"\n{len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print("\nALL CHECKS PASSED")
