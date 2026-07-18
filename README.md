# MatrixCovers

<!--- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/MatrixCovers.jl/stable/) --->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/MatrixCovers.jl/dev/)
[![Build Status](https://github.com/HolyLab/MatrixCovers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/MatrixCovers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/MatrixCovers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/MatrixCovers.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package computes **covers** of matrices: non-negative vectors `a` (and `b`)
such that `a[i] * b[j] >= abs(A[i, j])` for all `i`, `j`.  Covers are the
natural scale-covariant representation of a matrix — under row/column diagonal
scaling they transform exactly as the matrix entries do — making them a useful
building block for scale-invariant numerical analysis.

Fast O(mn) heuristics (`symcover`, `cover`) are provided for everyday use, along
with *soft* covers (`soft_symcover`, `soft_cover`) that penalize under-coverage
rather than forbid it.  Objective-minimal hard covers (`symcover_min`,
`cover_min`) minimize a penalty subject to the coverage constraint: the default
squared-log-excess penalty is solved natively, with no external solver, while
the other penalties are available when JuMP and HiGHS (or Ipopt) are loaded.

See the [documentation](https://HolyLab.github.io/MatrixCovers.jl/dev/)
for motivation, examples, and a full API reference.
