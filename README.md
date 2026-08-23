# MatrixCovers

<!--- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/MatrixCovers.jl/stable/) --->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/MatrixCovers.jl/dev/)
[![Build Status](https://github.com/HolyLab/MatrixCovers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/MatrixCovers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/MatrixCovers.jl/graph/badge.svg?token=trG4HXo9N4)](https://codecov.io/gh/HolyLab/MatrixCovers.jl)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package computes **covers** of matrices: non-negative vectors `a` and `b`
such that `a[i] * b[j] >= abs(A[i, j])` for all `i`, `j`.  Covers are the
natural scale-covariant representation of a matrix, making them a useful
building block for scale-invariant numerical analysis. In particular, 
$`\hat A = A ./ (a b^T)`$ is scale-invariant, and because $`|\hat A[i, j]| \le 1`$
for all `i` and `j`, this simple construct finds applications that range from
[statistical normalization](https://en.wikipedia.org/wiki/Normalization_(statistics))
of data to the design of well-behaved numerical algorithms (thanks, e.g., 
to [bounds on $`\hat A`$'s eigenvalues](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem)).

The package provides O(mn) heuristics (`symcover`, `cover`), *soft* covers that
penalize violations (`soft_symcover`, `soft_cover`), and objective-minimal hard
covers (`symcover_min`, `cover_min`). The default squared-log penalty uses a
built-in solver; other penalties use JuMP with HiGHS or Ipopt.

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

Covers are scale-covariant:

```julia
julia> D = Diagonal([10.0, 0.5]);

julia> symcover(D * A * D) ≈ D * a
true
```

For nonsymmetric matrices, `cover` returns separate row and column scales.
`cover_min` minimizes a penalty subject to the coverage constraint:

```julia
julia> M = [1.0 2.0 3.0; 6.0 5.0 4.0];

julia> a, b = cover(M);

julia> iscover(a, b, M)
true

julia> aq, bq = cover_min(AbsLog{2}(), M);

julia> cover_objective(AbsLog{2}(), aq, bq, M) <= cover_objective(AbsLog{2}(), a, b, M)
true
```

See the [documentation](https://HolyLab.github.io/MatrixCovers.jl/dev/) for the
algorithm guide and API reference.
