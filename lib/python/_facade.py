"""Python bindings for MatrixCovers.jl.

Matrix and vector arguments are converted to Fortran-order/contiguous
`float64` arrays; outputs are newly allocated `numpy.ndarray`s. `Penalty` and
`Linsolve` are enum classes: a keyword typed against one accepts a member
(`Penalty.abslog2`), its name as a string (`"abslog2"`), or the underlying
int. An unrecognized name or value raises `ValueError`; errors reported by the
compiled library raise `JLWError`.
"""
from . import _lowlevel  # noqa: F401
import numpy as np  # noqa: F401

from ._lowlevel import (
    JLWStatus,
    JLWResult_Bool,
    COpt_Int64,
    CString_owned,
    CVector_owned_Float64,
    JLWResult_CVector_owned_Float64,
    CVector_borrowed_Float64,
    COpt_Float64,
    JLWResult_Float64,
    CMatrix_borrowed_Float64,
    Linsolve,
    Penalty,
    JLWError,
    _enum_coerce,
)


def _as_matrix(A):
    return np.asfortranarray(A, dtype=np.float64)


def _as_vector(v):
    return np.ascontiguousarray(v, dtype=np.float64)


def soft_cover_ab(A, *, penalty=Penalty.abslinear2, maxiter=None, starts=None, sigma=None, seed=0):
    """Asymmetric soft cover of `A` minimizing the penalty, with no coverage
constraint, as `vcat(a, b)`; `a` has length `size(A, 1)`. `abslog1` and
`abslog2` accept only `maxiter`; `abslinear1` and `abslinear2` additionally
accept `starts`, `sigma`, and `seed`."""
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _penalty = _enum_coerce(Penalty, penalty)
    _maxiter = COpt_Int64.from_optional(maxiter)
    _starts = COpt_Int64.from_optional(starts)
    _sigma = COpt_Float64.from_optional(sigma)
    _r = _lowlevel.matrixcovers_soft_cover_ab(_A, _penalty, _maxiter, _starts, _sigma, seed)
    try:
        _out = np.array(_r.value.as_numpy(), copy=True)
    finally:
        _r.value.free()
    return _out

def iscover_sym(a, A, *, rtol=0.0, atol=0.0):
    """Whether `a` (playing the role of both `a` and `b`) covers `A`: `a[i]*a[j] >= abs(A[i, j])`."""
    _a = CVector_borrowed_Float64.from_numpy(_as_vector(a))
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _r = _lowlevel.matrixcovers_iscover_sym(_a, _A, rtol, atol)
    return _r.value

def soft_symcover_min(A, *, penalty=Penalty.abslog2, maxiter=None):
    """ϕ-minimal symmetric soft cover of `A`, with no coverage constraint. Only
`abslog2` is solved natively; `MatrixCovers` does not implement `abslog1` for
this entrypoint, and the `abslinear` penalties need the JuMP/Ipopt extension."""
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _penalty = _enum_coerce(Penalty, penalty)
    _maxiter = COpt_Int64.from_optional(maxiter)
    _r = _lowlevel.matrixcovers_soft_symcover_min(_A, _penalty, _maxiter)
    try:
        _out = np.array(_r.value.as_numpy(), copy=True)
    finally:
        _r.value.free()
    return _out

def gramcover_weighted(a, b, A, w):
    """Symmetric cover of the weighted Gram matrix `A'*Diagonal(w)*A`, from a cover `(a, b)` of `A`."""
    _a = CVector_borrowed_Float64.from_numpy(_as_vector(a))
    _b = CVector_borrowed_Float64.from_numpy(_as_vector(b))
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _w = CVector_borrowed_Float64.from_numpy(_as_vector(w))
    _r = _lowlevel.matrixcovers_gramcover_weighted(_a, _b, _A, _w)
    try:
        _out = np.array(_r.value.as_numpy(), copy=True)
    finally:
        _r.value.free()
    return _out

def soft_cover_min_ab(A, *, penalty=Penalty.abslog2, maxiter=None):
    """ϕ-minimal asymmetric soft cover of `A`, with no coverage constraint, as
`vcat(a, b)`; `a` has length `size(A, 1)`. Only `abslog2` is solved natively;
`MatrixCovers` does not implement `abslog1` for this entrypoint, and the
`abslinear` penalties need the JuMP/Ipopt extension."""
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _penalty = _enum_coerce(Penalty, penalty)
    _maxiter = COpt_Int64.from_optional(maxiter)
    _r = _lowlevel.matrixcovers_soft_cover_min_ab(_A, _penalty, _maxiter)
    try:
        _out = np.array(_r.value.as_numpy(), copy=True)
    finally:
        _r.value.free()
    return _out

def symcover_min(A, *, penalty=Penalty.abslog2, maxiter=None, linsolve=Linsolve.auto):
    """ϕ-minimal symmetric hard cover of `A`. Only `abslog2` is solved natively;
`abslog1` needs the JuMP/HiGHS extension and the `abslinear` penalties need the
JuMP/Ipopt extension, neither linked into this library."""
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _penalty = _enum_coerce(Penalty, penalty)
    _maxiter = COpt_Int64.from_optional(maxiter)
    _linsolve = _enum_coerce(Linsolve, linsolve)
    _r = _lowlevel.matrixcovers_symcover_min(_A, _penalty, _maxiter, _linsolve)
    try:
        _out = np.array(_r.value.as_numpy(), copy=True)
    finally:
        _r.value.free()
    return _out

def cover_objective_ab(a, b, A, *, penalty=Penalty.abslog2):
    """`sum(penalty(abs(A[i,j]) / (a[i]*b[j])))`."""
    _a = CVector_borrowed_Float64.from_numpy(_as_vector(a))
    _b = CVector_borrowed_Float64.from_numpy(_as_vector(b))
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _penalty = _enum_coerce(Penalty, penalty)
    _r = _lowlevel.matrixcovers_cover_objective_ab(_a, _b, _A, _penalty)
    return _r.value

def cover_min_ab(A, *, penalty=Penalty.abslog2, maxiter=None, linsolve=Linsolve.auto):
    """ϕ-minimal hard cover of `A`, as `vcat(a, b)`; `a` has length `size(A, 1)`. Only
`abslog2` is solved natively; `abslog1` needs the JuMP/HiGHS extension and the
`abslinear` penalties need the JuMP/Ipopt extension, neither linked into this
library."""
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _penalty = _enum_coerce(Penalty, penalty)
    _maxiter = COpt_Int64.from_optional(maxiter)
    _linsolve = _enum_coerce(Linsolve, linsolve)
    _r = _lowlevel.matrixcovers_cover_min_ab(_A, _penalty, _maxiter, _linsolve)
    try:
        _out = np.array(_r.value.as_numpy(), copy=True)
    finally:
        _r.value.free()
    return _out

def soft_symcover(A, *, penalty=Penalty.abslinear2, maxiter=None, starts=None, sigma=None, seed=0):
    """Symmetric soft cover of `A` minimizing the penalty, with no coverage
constraint. `abslog1` and `abslog2` accept only `maxiter`; `abslinear1` and
`abslinear2` additionally accept `starts`, `sigma`, and `seed` (which seeds the
multistart perturbation stream)."""
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _penalty = _enum_coerce(Penalty, penalty)
    _maxiter = COpt_Int64.from_optional(maxiter)
    _starts = COpt_Int64.from_optional(starts)
    _sigma = COpt_Float64.from_optional(sigma)
    _r = _lowlevel.matrixcovers_soft_symcover(_A, _penalty, _maxiter, _starts, _sigma, seed)
    try:
        _out = np.array(_r.value.as_numpy(), copy=True)
    finally:
        _r.value.free()
    return _out

def gramcover_matrix(a, b, A, W):
    """Symmetric cover of the weighted Gram matrix `A'*W*A`, from a cover `(a, b)` of `A`."""
    _a = CVector_borrowed_Float64.from_numpy(_as_vector(a))
    _b = CVector_borrowed_Float64.from_numpy(_as_vector(b))
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _W = CMatrix_borrowed_Float64.from_numpy(_as_matrix(W))
    _r = _lowlevel.matrixcovers_gramcover_matrix(_a, _b, _A, _W)
    try:
        _out = np.array(_r.value.as_numpy(), copy=True)
    finally:
        _r.value.free()
    return _out

def iscover_ab(a, b, A, *, rtol=0.0, atol=0.0):
    """Whether `a`, `b` cover `A`: `a[i]*b[j] >= abs(A[i, j])`."""
    _a = CVector_borrowed_Float64.from_numpy(_as_vector(a))
    _b = CVector_borrowed_Float64.from_numpy(_as_vector(b))
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _r = _lowlevel.matrixcovers_iscover_ab(_a, _b, _A, rtol, atol)
    return _r.value

def cover_objective_sym(a, A, *, penalty=Penalty.abslog2):
    """`sum(penalty(abs(A[i,j]) / (a[i]*a[j])))`."""
    _a = CVector_borrowed_Float64.from_numpy(_as_vector(a))
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _penalty = _enum_coerce(Penalty, penalty)
    _r = _lowlevel.matrixcovers_cover_objective_sym(_a, _A, _penalty)
    return _r.value

def symcover(A, *, maxiter=None):
    """Heuristic symmetric hard cover: `a` with `a[i]*a[j] >= abs(A[i, j])`."""
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _maxiter = COpt_Int64.from_optional(maxiter)
    _r = _lowlevel.matrixcovers_symcover(_A, _maxiter)
    try:
        _out = np.array(_r.value.as_numpy(), copy=True)
    finally:
        _r.value.free()
    return _out

def cover_ab(A, *, maxiter=None):
    """Heuristic hard cover of `A`, as `vcat(a, b)`; `a` has length `size(A, 1)`."""
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _maxiter = COpt_Int64.from_optional(maxiter)
    _r = _lowlevel.matrixcovers_cover_ab(_A, _maxiter)
    try:
        _out = np.array(_r.value.as_numpy(), copy=True)
    finally:
        _r.value.free()
    return _out

def _gramcover_plain(a, b, A):
    """Symmetric cover of the Gram matrix `A'*A`, from a cover `(a, b)` of `A`."""
    _a = CVector_borrowed_Float64.from_numpy(_as_vector(a))
    _b = CVector_borrowed_Float64.from_numpy(_as_vector(b))
    _A = CMatrix_borrowed_Float64.from_numpy(_as_matrix(A))
    _r = _lowlevel.matrixcovers_gramcover(_a, _b, _A)
    try:
        _out = np.array(_r.value.as_numpy(), copy=True)
    finally:
        _r.value.free()
    return _out


# Hand-written public API. Each of these splits or dispatches across the
# generated entrypoints above, whose own names gain a leading underscore
# where reused (`gramcover` -> `_gramcover_plain`) to free the public name.

def cover(A, *, maxiter=None):
    """Heuristic hard cover of `A`: `(a, b)` with `a[i]*b[j] >= abs(A[i, j])`."""
    out = cover_ab(A, maxiter=maxiter)
    m = np.shape(A)[0]
    return out[:m], out[m:]


def cover_min(A, *, penalty=Penalty.abslog2, maxiter=None, linsolve=Linsolve.auto):
    """ϕ-minimal hard cover of `A`: `(a, b)`. Only `abslog2` is solved natively."""
    out = cover_min_ab(A, penalty=penalty, maxiter=maxiter, linsolve=linsolve)
    m = np.shape(A)[0]
    return out[:m], out[m:]


def soft_cover(A, *, penalty=Penalty.abslinear2, maxiter=None, starts=None, sigma=None, seed=0):
    """Asymmetric soft cover of `A` minimizing the penalty, with no coverage constraint."""
    out = soft_cover_ab(A, penalty=penalty, maxiter=maxiter, starts=starts, sigma=sigma, seed=seed)
    m = np.shape(A)[0]
    return out[:m], out[m:]


def soft_cover_min(A, *, penalty=Penalty.abslog2, maxiter=None):
    """ϕ-minimal asymmetric soft cover of `A`, with no coverage constraint."""
    out = soft_cover_min_ab(A, penalty=penalty, maxiter=maxiter)
    m = np.shape(A)[0]
    return out[:m], out[m:]


def iscover(a, A, b=None, *, rtol=0.0, atol=0.0):
    """Whether `a`, `b` cover `A`: `a[i]*b[j] >= abs(A[i, j])`.

    `b=None` (the default) tests the symmetric cover `a*a'`, and requires `A`
    to be square.
    """
    if b is None:
        return iscover_sym(a, A, rtol=rtol, atol=atol)
    return iscover_ab(a, b, A, rtol=rtol, atol=atol)


def cover_objective(a, A, b=None, *, penalty=Penalty.abslog2):
    """`sum(penalty(abs(A[i,j]) / (a[i]*b[j])))`; `b=None` tests the symmetric cover `a*a'`."""
    if b is None:
        return cover_objective_sym(a, A, penalty=penalty)
    return cover_objective_ab(a, b, A, penalty=penalty)


def gramcover(a, b, A, *, w=None, W=None):
    """Symmetric cover of a (weighted) Gram matrix of `A`, from a cover `(a, b)` of `A`.

    `G = A'*A` when neither `w` nor `W` is given, `G = A'*diag(w)*A` for a
    vector `w`, and `G = A'*W*A` for a matrix `W`. Passing both `w` and `W`
    raises `ValueError`.
    """
    if w is not None and W is not None:
        raise ValueError("pass at most one of `w` or `W`, not both")
    if w is not None:
        return gramcover_weighted(a, b, A, w)
    if W is not None:
        return gramcover_matrix(a, b, A, W)
    return _gramcover_plain(a, b, A)


__all__ = [
    "JLWError", "Penalty", "Linsolve",
    "symcover", "cover",
    "symcover_min", "cover_min",
    "soft_symcover", "soft_cover",
    "soft_symcover_min", "soft_cover_min",
    "iscover", "cover_objective",
    "gramcover",
]
