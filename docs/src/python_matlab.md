# Python and MATLAB

MatrixCovers is also available as a compiled library with Python and MATLAB
interfaces, neither of which requires a Julia installation. Both are built for
Linux on x86-64 only, and are attached as assets to each
[release](https://github.com/HolyLab/MatrixCovers.jl/releases) starting with
2.0.1.

## Python

The Python package requires Python 3.8 or later, NumPy 1.20 or later, and glibc
2.35 or later (Ubuntu 22.04+, Debian 12+, Fedora 36+; RHEL and Rocky 9 are too
old). Because the wheel bundles the Julia runtime, it is about 40 MB to download
and 120 MB once installed. To install it, replace `X.Y.Z` with the release
version:

```sh
pip install https://github.com/HolyLab/MatrixCovers.jl/releases/download/vX.Y.Z/matrixcovers-X.Y.Z-py3-none-manylinux_2_35_x86_64.whl
```

```python
import numpy as np
import matrixcovers as mc

A = np.array([[4.0, 2.0], [2.0, 16.0]])
a = mc.symcover(A)           # array([2., 4.]); a[i]*a[j] >= abs(A[i, j])
mc.iscover(a, A)             # True

M = np.array([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]])
a, b, stats = mc.cover_min(M)
stats.converged              # True
```

The module provides the Julia functions under the same names: `symcover`,
`cover`, `symcover_min`, `cover_min`, `cover_transversal`, `soft_symcover`,
`soft_cover`, `noise_cover`, `initialize_symcover`, `initialize_cover`,
`gramcover`, `iscover`, `cover_objective`, `max_coverage_ratio`, and
`level_dispersion`. `help(mc.cover_min)` shows the documentation for each.
A few conventions differ from Julia:

- Options are keyword arguments, and those left at `None` take the Julia
  defaults. Enumerated options, including the penalty, accept either a member of
  the corresponding enum (`Penalty`, `Linsolve`, `Start`, `SymStrategy`,
  `Strategy`, `Feasible`) or its name as a string, such as `"powermean1"`.
- The functions that check or score a cover take the matrix second:
  `iscover(a, A, b=None)`, where Julia uses `iscover(a, b, A)`. The same order
  applies to `cover_objective`, `max_coverage_ratio`, and `level_dispersion`.
- Solvers that optimize return a `SolverStats` as their last output, with fields
  `converged`, `iterations`, `residual`, and `tolerance`. All four are `None`
  for the `abslog2` soft covers, which are a single linear solve.
- Arrays of any numeric dtype and memory order are accepted and converted to
  `float64`.
- Errors raise `matrixcovers.JLWError`, whose `.code` and `.message` describe
  the failure.

## MATLAB

The MATLAB package requires R2021a or later on Linux. Download
`matrixcovers-matlab-glnxa64.tar.gz` from the release, unpack it, and add the
directory containing `+matrixcovers` and `matrixcovers-bundle` to the MATLAB
path. The functions are in the `matrixcovers` namespace:

```matlab
A = [4 2; 2 16];
a = matrixcovers.symcover(A);
[a, b, converged] = matrixcovers.cover_min([1 2 3; 6 5 4]);
```

`help matrixcovers.symcover` shows the documentation for each function. Solver
statistics are returned as extra outputs (`converged`, `iterations`, `residual`,
`tolerance`), which are `[]` when the solver does not compute them. Where Julia selects the symmetric or nonsymmetric form of a
function by its arguments, MATLAB uses separate names, such as `iscover_sym` and
`iscover_ab`, `cover_given_rows` for `cover(A, a)`, and `gramcover_weighted` and
`gramcover_matrix` for the weighted forms of `gramcover`.

## Limitations

Both interfaces call the same compiled library, which differs from the Julia
package in the following ways:

- Computations use `Float64` only.
- Inputs must be dense; convert sparse matrices before passing them.
- The `AbsLog{1}` and `AbsLinear` penalties are unavailable, because they depend
  on JuMP with HiGHS or Ipopt, which are not compiled in. Requesting them raises
  an error that names the missing extension.
- `gramcover` accepts a weight vector or matrix, but not a `Cholesky`
  factorization.
- Solver warnings, such as "may not minimize the objective," are not printed.
  Check the `converged` statistic instead.
