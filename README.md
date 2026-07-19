# MatrixCovers

<!--- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/MatrixCovers.jl/stable/) --->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/MatrixCovers.jl/dev/)
[![Build Status](https://github.com/HolyLab/MatrixCovers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/MatrixCovers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/MatrixCovers.jl/graph/badge.svg?token=trG4HXo9N4)](https://codecov.io/gh/HolyLab/MatrixCovers.jl)
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

## Example

```julia
julia> using MatrixCovers, LinearAlgebra

julia> A = [4.0 2.0; 2.0 16.0];

julia> a = symcover(A)      # a[i] * a[j] >= abs(A[i, j])
2-element Vector{Float64}:
 2.0
 4.0

julia> a * a'               # dominates A entrywise
2×2 Matrix{Float64}:
 4.0   8.0
 8.0  16.0

julia> iscover(a, A)
true
```

Covers are scale-covariant: rescaling the matrix rescales the cover the same way.

```julia
julia> D = Diagonal([10.0, 0.5]);

julia> symcover(D * A * D) ≈ D * a
true
```

Non-symmetric matrices get separate row and column scales from `cover`, and
`cover_min`/`symcover_min` trade the fast heuristic for a cover that minimizes a
penalty subject to the same constraint:

```julia
julia> M = [1.0 2.0 3.0; 6.0 5.0 4.0];

julia> a, b = cover(M);

julia> iscover(a, b, M)
true

julia> aq, bq = cover_min(AbsLog{2}(), M);   # minimal, solved natively

julia> cover_objective(AbsLog{2}(), aq, bq, M) <= cover_objective(AbsLog{2}(), a, b, M)
true
```

See the [documentation](https://HolyLab.github.io/MatrixCovers.jl/dev/)
for motivation, examples, and a full API reference.
