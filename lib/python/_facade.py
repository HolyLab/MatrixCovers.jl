"""Python bindings for MatrixCovers.jl.

Inputs are converted to `float64` arrays and outputs are newly allocated
`numpy.ndarray`s. Invalid options raise `ValueError`; errors reported by the
compiled library raise `JLWError`.

Penalties are `"abslog1"`, `"abslog2"`, `"abslinear1"`, and `"abslinear2"`.
The `_min` functions support only `"abslog2"`; other penalties require Julia
package extensions that are not included in the compiled library.
"""
from . import _lowlevel
import numpy as np

from ._lowlevel import JLWError

_PENALTY_CODES = {
    "abslog1": 1,
    "abslog2": 2,
    "abslinear1": 3,
    "abslinear2": 4,
}


def _penalty_code(penalty):
    try:
        return _PENALTY_CODES[penalty]
    except KeyError:
        raise ValueError(
            f"unknown penalty {penalty!r}; expected one of {sorted(_PENALTY_CODES)}"
        ) from None


_LINSOLVE_CODES = {"auto": 1, "dense": 2, "lsqr": 3}


def _linsolve_code(linsolve):
    try:
        return _LINSOLVE_CODES[linsolve]
    except KeyError:
        raise ValueError(
            f"unknown linsolve {linsolve!r}; expected one of {sorted(_LINSOLVE_CODES)}"
        ) from None


def _sentinel_int(value):
    return -1 if value is None else int(value)


def _sentinel_float(value):
    return -1.0 if value is None else float(value)


def _as_matrix(A):
    return np.asfortranarray(A, dtype=np.float64)


def _as_vector(v):
    return np.ascontiguousarray(v, dtype=np.float64)


def symcover(A, *, maxiter=None):
    """Heuristic symmetric hard cover: `a` with `a[i]*a[j] >= abs(A[i, j])`."""
    _A = _as_matrix(A)
    a = np.zeros(_A.shape[0], dtype=np.float64)
    _lowlevel.mc_symcover(
        _lowlevel.CMatrix_Float64.from_numpy(_A),
        _sentinel_int(maxiter),
        _lowlevel.CVector_Float64.from_numpy(a),
    )
    return a


def cover(A, *, maxiter=None):
    """Heuristic hard cover: `(a, b)` with `a[i]*b[j] >= abs(A[i, j])`."""
    _A = _as_matrix(A)
    m, n = _A.shape
    a = np.zeros(m, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    _lowlevel.mc_cover(
        _lowlevel.CMatrix_Float64.from_numpy(_A),
        _sentinel_int(maxiter),
        _lowlevel.CVector_Float64.from_numpy(a),
        _lowlevel.CVector_Float64.from_numpy(b),
    )
    return a, b


def symcover_min(A, *, penalty="abslog2", maxiter=None, linsolve="auto"):
    """phi-minimal symmetric hard cover of `A`."""
    _A = _as_matrix(A)
    a = np.zeros(_A.shape[0], dtype=np.float64)
    _lowlevel.mc_symcover_min(
        _penalty_code(penalty),
        _lowlevel.CMatrix_Float64.from_numpy(_A),
        _sentinel_int(maxiter),
        _linsolve_code(linsolve),
        _lowlevel.CVector_Float64.from_numpy(a),
    )
    return a


def cover_min(A, *, penalty="abslog2", maxiter=None, linsolve="auto"):
    """phi-minimal hard cover of `A`."""
    _A = _as_matrix(A)
    m, n = _A.shape
    a = np.zeros(m, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    _lowlevel.mc_cover_min(
        _penalty_code(penalty),
        _lowlevel.CMatrix_Float64.from_numpy(_A),
        _sentinel_int(maxiter),
        _linsolve_code(linsolve),
        _lowlevel.CVector_Float64.from_numpy(a),
        _lowlevel.CVector_Float64.from_numpy(b),
    )
    return a, b


def soft_symcover(A, *, penalty="abslinear2", maxiter=None, starts=None, sigma=None, seed=0):
    """Symmetric soft cover of `A` minimizing the penalty, with no coverage constraint."""
    _A = _as_matrix(A)
    a = np.zeros(_A.shape[0], dtype=np.float64)
    _lowlevel.mc_soft_symcover(
        _penalty_code(penalty),
        _lowlevel.CMatrix_Float64.from_numpy(_A),
        _sentinel_int(maxiter),
        _sentinel_int(starts),
        _sentinel_float(sigma),
        int(seed),
        _lowlevel.CVector_Float64.from_numpy(a),
    )
    return a


def soft_cover(A, *, penalty="abslinear2", maxiter=None, starts=None, sigma=None, seed=0):
    """Asymmetric soft cover of `A` minimizing the penalty, with no coverage constraint."""
    _A = _as_matrix(A)
    m, n = _A.shape
    a = np.zeros(m, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    _lowlevel.mc_soft_cover(
        _penalty_code(penalty),
        _lowlevel.CMatrix_Float64.from_numpy(_A),
        _sentinel_int(maxiter),
        _sentinel_int(starts),
        _sentinel_float(sigma),
        int(seed),
        _lowlevel.CVector_Float64.from_numpy(a),
        _lowlevel.CVector_Float64.from_numpy(b),
    )
    return a, b


def soft_symcover_min(A, *, penalty="abslog2", maxiter=None):
    """phi-minimal symmetric soft cover of `A`, with no coverage constraint."""
    _A = _as_matrix(A)
    a = np.zeros(_A.shape[0], dtype=np.float64)
    _lowlevel.mc_soft_symcover_min(
        _penalty_code(penalty),
        _lowlevel.CMatrix_Float64.from_numpy(_A),
        _sentinel_int(maxiter),
        _lowlevel.CVector_Float64.from_numpy(a),
    )
    return a


def soft_cover_min(A, *, penalty="abslog2", maxiter=None):
    """phi-minimal asymmetric soft cover of `A`, with no coverage constraint."""
    _A = _as_matrix(A)
    m, n = _A.shape
    a = np.zeros(m, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    _lowlevel.mc_soft_cover_min(
        _penalty_code(penalty),
        _lowlevel.CMatrix_Float64.from_numpy(_A),
        _sentinel_int(maxiter),
        _lowlevel.CVector_Float64.from_numpy(a),
        _lowlevel.CVector_Float64.from_numpy(b),
    )
    return a, b


def iscover(a, A, b=None, *, rtol=0.0, atol=0.0):
    """Whether `a`, `b` cover `A`: `a[i]*b[j] >= abs(A[i, j])*(1 - rtol) - atol`.

    `b=None` (the default) tests the symmetric cover `a*a'`, and requires `A`
    to be square.
    """
    _A = _as_matrix(A)
    _a = _as_vector(a)
    if b is None:
        _result = _lowlevel.mc_iscover_sym(
            _lowlevel.CVector_Float64.from_numpy(_a),
            _lowlevel.CMatrix_Float64.from_numpy(_A),
            float(rtol), float(atol),
        )
    else:
        _b = _as_vector(b)
        _result = _lowlevel.mc_iscover(
            _lowlevel.CVector_Float64.from_numpy(_a),
            _lowlevel.CVector_Float64.from_numpy(_b),
            _lowlevel.CMatrix_Float64.from_numpy(_A),
            float(rtol), float(atol),
        )
    return bool(_result.value)


def cover_objective(a, A, b=None, *, penalty="abslog2"):
    """`sum(phi(abs(A[i,j]) / (a[i]*b[j])))`; `b=None` tests the symmetric cover `a*a'`."""
    _A = _as_matrix(A)
    _a = _as_vector(a)
    if b is None:
        _result = _lowlevel.mc_cover_objective_sym(
            _penalty_code(penalty),
            _lowlevel.CVector_Float64.from_numpy(_a),
            _lowlevel.CMatrix_Float64.from_numpy(_A),
        )
    else:
        _b = _as_vector(b)
        _result = _lowlevel.mc_cover_objective(
            _penalty_code(penalty),
            _lowlevel.CVector_Float64.from_numpy(_a),
            _lowlevel.CVector_Float64.from_numpy(_b),
            _lowlevel.CMatrix_Float64.from_numpy(_A),
        )
    return float(_result.value)


def gramcover(a, b, A, *, w=None, W=None):
    """Symmetric cover of a (weighted) Gram matrix of `A`, from a cover `(a, b)` of `A`.

    `G = A'*A` when neither `w` nor `W` is given, `G = A'*diag(w)*A` for a
    vector `w`, and `G = A'*W*A` for a matrix `W`. Passing both `w` and `W`
    raises `ValueError`.
    """
    if w is not None and W is not None:
        raise ValueError("pass at most one of `w` or `W`, not both")
    _a = _as_vector(a)
    _b = _as_vector(b)
    _A = _as_matrix(A)
    s = np.zeros(_A.shape[1], dtype=np.float64)
    if w is not None:
        _w = _as_vector(w)
        _lowlevel.mc_gramcover_weighted(
            _lowlevel.CVector_Float64.from_numpy(_a),
            _lowlevel.CVector_Float64.from_numpy(_b),
            _lowlevel.CMatrix_Float64.from_numpy(_A),
            _lowlevel.CVector_Float64.from_numpy(_w),
            _lowlevel.CVector_Float64.from_numpy(s),
        )
    elif W is not None:
        _W = _as_matrix(W)
        _lowlevel.mc_gramcover_matrix(
            _lowlevel.CVector_Float64.from_numpy(_a),
            _lowlevel.CVector_Float64.from_numpy(_b),
            _lowlevel.CMatrix_Float64.from_numpy(_A),
            _lowlevel.CMatrix_Float64.from_numpy(_W),
            _lowlevel.CVector_Float64.from_numpy(s),
        )
    else:
        _lowlevel.mc_gramcover(
            _lowlevel.CVector_Float64.from_numpy(_a),
            _lowlevel.CVector_Float64.from_numpy(_b),
            _lowlevel.CMatrix_Float64.from_numpy(_A),
            _lowlevel.CVector_Float64.from_numpy(s),
        )
    return s


__all__ = [
    "JLWError",
    "symcover", "cover",
    "symcover_min", "cover_min",
    "soft_symcover", "soft_cover",
    "soft_symcover_min", "soft_cover_min",
    "iscover", "cover_objective",
    "gramcover",
]
