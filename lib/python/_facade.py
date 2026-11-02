"""The public Python API of matrixcovers.

`_generated.py` holds one wrapper per `@api` entrypoint of the compiled
library. This module builds the public API on top of it:

- the solvers return their convergence statistics as a `SolverStats`;
- `iscover`, `cover_objective`, `max_coverage_ratio`, and `level_dispersion`
  take an optional `b`, covering both the symmetric and the asymmetric forms;
- `gramcover` selects the weighted form from the shape of `W`;
- `cover` accepts a fixed row scale `a`;
- `cover_transversal` selects its solver from `penalty`.

The generated entrypoints these replace remain reachable as
`_generated.<name>`.
"""
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import _generated
from ._generated import (  # noqa: F401
    Feasible,
    Linsolve,
    Penalty,
    Start,
    Strategy,
    SymStrategy,
    initialize_cover,
    initialize_symcover,
    symcover,
)
from ._lowlevel import JLWError  # noqa: F401


@dataclass(frozen=True)
class SolverStats:
    """Convergence statistics of an optimizing solver.

    `converged` reports whether the result meets the solver's stopping
    criterion, `iterations` counts the iterations of its main loop, and
    `residual` is the quantity the criterion compares with `tolerance`. All
    four are `None` for a solver without a stopping criterion (the `abslog2`
    soft covers, which are a single linear solve).
    """

    converged: Optional[bool]
    iterations: Optional[int]
    residual: Optional[float]
    tolerance: Optional[float]


LevelDispersion = namedtuple("LevelDispersion", ["level", "dispersion"])


def _split1(r):
    return r[0], SolverStats(*r[1:])


def _split2(r):
    return r[0], r[1], SolverStats(*r[2:])


def _reject(fname, why, **kwargs):
    given = [k for k, v in kwargs.items() if v is not None]
    if given:
        raise TypeError(f"{fname}: {', '.join(given)} {why}")


# --- Heuristic hard covers ---------------------------------------------------

def cover(A, a=None, *, maxiter=None, cgiter=None, start=None):
    """Heuristic hard cover `(a, b)` of `A`: `a[i]*b[j] >= abs(A[i, j])`.

    `start` (`"auto"`, `"transversal"`, `"unconstrained"`, or `"geomean"`)
    selects the kind of cover; `maxiter` and `cgiter` default to values that
    depend on `start`.

    With a row scale `a`, return the tightest cover with `a` held fixed:
    `(a, b)` with `b[j] = max_i abs(A[i, j]) / a[i]`.
    """
    if a is not None:
        _reject("cover", "cannot be combined with a fixed row scale `a`",
                maxiter=maxiter, cgiter=cgiter, start=start)
        a = np.ascontiguousarray(a, dtype=np.float64)
        return a, _generated.cover_given_rows(A, a)
    return _generated.cover(A, maxiter=maxiter, cgiter=cgiter,
                            start=Start.auto if start is None else start)


# --- Minimal hard covers ------------------------------------------------------

def symcover_min(A, *, penalty=Penalty.abslog2, theta=None, maxouter=None, maxiter=None,
                 tol=None, linsolve=Linsolve.auto, fillbudget=None, flopbudget=None):
    """The symmetric hard cover of `A` that minimizes `penalty`: `(a, stats)`.

    Only `abslog2` is solved natively; `abslog1` needs JuMP/HiGHS and the
    `abslinear` penalties need JuMP/Ipopt, neither linked into this library.
    `theta`, `maxouter`, `maxiter`, `tol`, `linsolve` (`"auto"`, `"dense"`,
    `"woodbury"`, or `"lsqr"`), `fillbudget`, and `flopbudget` are the solver
    options of `MatrixCovers.symcover_min`; `None` selects its default.
    """
    return _split1(_generated.symcover_min(
        A, penalty=penalty, theta=theta, maxouter=maxouter, maxiter=maxiter, tol=tol,
        linsolve=linsolve, fillbudget=fillbudget, flopbudget=flopbudget))


def cover_min(A, *, penalty=Penalty.abslog2, theta=None, maxouter=None, maxiter=None,
              tol=None, linsolve=Linsolve.auto, fillbudget=None, flopbudget=None):
    """The hard cover of `A` that minimizes `penalty`: `(a, b, stats)`.

    Penalties and options are as for `symcover_min`.
    """
    return _split2(_generated.cover_min(
        A, penalty=penalty, theta=theta, maxouter=maxouter, maxiter=maxiter, tol=tol,
        linsolve=linsolve, fillbudget=fillbudget, flopbudget=flopbudget))


def cover_transversal(A, *, penalty=Penalty.abslog2, target=None, maxbalance=None,
                      theta=None, maxouter=None, maxiter=None, tol=None, linsolve=None,
                      fillbudget=None, flopbudget=None):
    """A hard cover `(a, b, stats)` of the square matrix `A` that is tight on its
    maximum-product transversals.

    With `penalty` `abslog2` (the default), the cover minimizes the `abslog2`
    objective among those tight on the transversals; the options are those of
    `cover_min` (`target` and `maxbalance` do not apply). With `powermean1` or
    `powermean2`, it pushes the coverage ratios of the other nonzero entries
    toward `target` (default 0.01), and `maxbalance=True` max-balances the
    scales within each fully indecomposable block; `tol`, `maxiter`,
    `fillbudget`, and `flopbudget` apply (`theta`, `maxouter`, and `linsolve`
    do not).
    """
    penalty = _generated._enum_coerce(Penalty, penalty)
    if penalty == Penalty.abslog2:
        _reject("cover_transversal", "apply only to the powermean penalties",
                target=target, maxbalance=maxbalance)
        return _split2(_generated.cover_transversal(
            A, theta=theta, maxouter=maxouter, maxiter=maxiter, tol=tol,
            linsolve=Linsolve.auto if linsolve is None else linsolve,
            fillbudget=fillbudget, flopbudget=flopbudget))
    _reject("cover_transversal", "apply only to penalty abslog2",
            theta=theta, maxouter=maxouter, linsolve=linsolve)
    return _split2(_generated.cover_transversal_powermean(
        A, penalty=penalty, target=0.01 if target is None else target,
        maxbalance=False if maxbalance is None else maxbalance, tol=tol, maxiter=maxiter,
        fillbudget=fillbudget, flopbudget=flopbudget))


# --- Soft covers --------------------------------------------------------------

def soft_symcover(A, *, penalty=Penalty.powermean2, maxiter=None, tol=None, newton=None,
                  maxnewton=None, maxpolish=None, linsolve=Linsolve.auto,
                  fillbudget=None, flopbudget=None):
    """The symmetric soft cover of `A` that minimizes `penalty` without a
    coverage constraint: `(a, stats)`.

    `powermean1`, `powermean2` (the default), and `abslog2` are solved
    natively; `abslog1` needs JuMP/HiGHS and the `abslinear` penalties need
    JuMP/Ipopt, neither linked into this library. `maxiter`, `tol`, `newton`,
    `maxnewton`, and `maxpolish` apply to the `powermean` penalties, whose
    `linsolve` is `"auto"`, `"dense"`, `"cholesky"`, or `"cg"`; `fillbudget`
    and `flopbudget` apply to `abslog2`, whose `linsolve` is as for
    `symcover_min`.
    """
    return _split1(_generated.soft_symcover(
        A, penalty=penalty, maxiter=maxiter, tol=tol, newton=newton, maxnewton=maxnewton,
        maxpolish=maxpolish, linsolve=linsolve, fillbudget=fillbudget, flopbudget=flopbudget))


def soft_cover(A, *, penalty=Penalty.powermean2, maxiter=None, tol=None, newton=None,
               maxnewton=None, maxpolish=None, linsolve=Linsolve.auto,
               fillbudget=None, flopbudget=None):
    """The soft cover of `A` that minimizes `penalty` without a coverage
    constraint: `(a, b, stats)`. Penalties and options are as for
    `soft_symcover`.
    """
    return _split2(_generated.soft_cover(
        A, penalty=penalty, maxiter=maxiter, tol=tol, newton=newton, maxnewton=maxnewton,
        maxpolish=maxpolish, linsolve=linsolve, fillbudget=fillbudget, flopbudget=flopbudget))


def noise_cover(V, *, shrink=0.0, maxiter=None, tol=None, newton=None, maxnewton=None,
                maxpolish=None, linsolve=Linsolve.auto):
    """Scales `(a, b, stats)` that normalize a data matrix whose entries have
    noise variances estimated by `V`: every row and column of
    `Y / np.outer(a, b)` has average noise variance one.

    `shrink` blends `V` with its rank-one fit; the other options are the
    `powermean` options of `soft_cover`.
    """
    return _split2(_generated.noise_cover(
        V, shrink=shrink, maxiter=maxiter, tol=tol, newton=newton, maxnewton=maxnewton,
        maxpolish=maxpolish, linsolve=linsolve))


# --- Gram covers --------------------------------------------------------------

def gramcover(A, W=None, *, v=None):
    """A vector `s` whose outer product covers the Gram matrix `A.T @ W @ A`.

    `W` may be omitted (`A.T @ A`), a vector of weights `w` (`A.T @ diag(w) @ A`),
    or a general square matrix. For a matrix `W`, `v` is a nonnegative vector
    with `abs(W[i, k]) <= v[i]*v[k]`; by default it is a symmetric cover of
    `abs(W)`.
    """
    if W is None:
        _reject("gramcover", "applies only to a matrix W", v=v)
        return _generated.gramcover(A)
    ndim = np.ndim(W)
    if ndim == 1:
        _reject("gramcover", "applies only to a matrix W", v=v)
        return _generated.gramcover_weighted(A, W)
    if ndim == 2:
        if v is None:
            return _generated.gramcover_matrix(A, W)
        return _generated.gramcover_matrix_v(A, W, v)
    raise ValueError(f"gramcover: W must be a vector or a matrix, got {ndim} dimensions")


# --- Predicates and objectives ------------------------------------------------

def iscover(a, A, b=None, *, rtol=0.0, atol=0.0):
    """Whether `a` and `b` cover `A`: `a[i]*b[j] >= abs(A[i, j])*(1 - rtol) - atol`.

    Without `b`, test the symmetric cover `np.outer(a, a)`.
    """
    if b is None:
        return _generated.iscover_sym(a, A, rtol=rtol, atol=atol)
    return _generated.iscover_ab(a, b, A, rtol=rtol, atol=atol)


def max_coverage_ratio(a, A, b=None):
    """The largest coverage ratio `abs(A[i, j]) / (a[i]*b[j])` over the nonzero
    entries of `A`; `a` and `b` cover `A` when it is at most one.

    Without `b`, use the symmetric cover `np.outer(a, a)`.
    """
    if b is None:
        return _generated.max_coverage_ratio_sym(a, A)
    return _generated.max_coverage_ratio_ab(a, b, A)


def level_dispersion(a, aref, A, b=None, bref=None):
    """Compare `np.outer(a, b)` with `np.outer(aref, bref)` on the nonzero entries
    of `A`: the mean (`level`) and standard deviation (`dispersion`) of the log
    ratio of the two products.

    Without `b` and `bref`, compare the symmetric products of `a` and `aref`.
    """
    if (b is None) != (bref is None):
        raise TypeError("level_dispersion: pass both b and bref, or neither")
    if b is None:
        return LevelDispersion(*_generated.level_dispersion_sym(a, aref, A))
    return LevelDispersion(*_generated.level_dispersion_ab(a, b, aref, bref, A))


def cover_objective(a, A, b=None, *, penalty=Penalty.abslog2):
    """`sum(penalty(log(a[i]*b[j]) - log(abs(A[i, j]))))` over the nonzero
    entries of `A`. Without `b`, use the symmetric cover `np.outer(a, a)`.
    """
    if b is None:
        return _generated.cover_objective_sym(a, A, penalty=penalty)
    return _generated.cover_objective_ab(a, b, A, penalty=penalty)


__all__ = [
    "Feasible",
    "JLWError",
    "LevelDispersion",
    "Linsolve",
    "Penalty",
    "SolverStats",
    "Start",
    "Strategy",
    "SymStrategy",
    "cover",
    "cover_min",
    "cover_objective",
    "cover_transversal",
    "gramcover",
    "initialize_cover",
    "initialize_symcover",
    "iscover",
    "level_dispersion",
    "max_coverage_ratio",
    "noise_cover",
    "soft_cover",
    "soft_symcover",
    "symcover",
    "symcover_min",
]
