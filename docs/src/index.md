```@meta
CurrentModule = ScaleInvariantAnalysis
```

# ScaleInvariantAnalysis

This package computes **covers** of matrices.  Given a matrix `A`, a cover is a
matrix `C` that can be defined as `C = a * b'`, where `a` and `b` are non-negative vectors.
`C` must satisfy

```math
C_{ij} \;\geq\; |A_{ij}| \quad \text{for all } i, j.
```

For a symmetric matrix the cover is symmetric (`b = a`), so a single vector
suffices: `a[i] * a[j] >= abs(A[i, j])`.

## Why covers?

Covers are the natural **scale-covariant** representation of a matrix.  If you
rescale rows by a positive diagonal factor `D_r` and columns by `D_c`, the
optimal cover transforms as `a → D_r * a`, `b → D_c * b` — exactly mirroring
how the matrix entries change.  Scalar summaries like `norm(A)` or
`maximum(abs, A)` do not have this property and therefore implicitly encode an
arbitrary choice of units.

A concrete example: a 3×3 matrix whose rows and columns correspond to physical
variables with different units (position in meters, velocity in m/s, force
in N):

```jldoctest coverones
julia> using ScaleInvariantAnalysis

julia> A = [1e6 1e3 1.0; 1e3 1.0 1e-3; 1.0 1e-3 1e-6];

julia> round.(symcover(A); digits=6)
3-element Vector{Float64}:
 1000.0
    1.0
    0.001
```

The cover `a` captures the natural per-variable scale.  The normalized matrix
`A ./ (a .* a')` is all-ones and scale-invariant.

## Penalty functions

A cover is valid as long as every constraint is satisfied, but tighter covers
better capture the scaling of `A`.  Cover quality is measured through the ratios
`r_{ij} = |A[i,j]| / (a[i] * b[j])`: a hard cover has every `r ≤ 1`, and `r = 1`
means the constraint is exactly tight.  A **penalty function** `ϕ` turns those
ratios into a scalar objective

```math
\sum_{i,j} \phi\!\left(\frac{|A_{ij}|}{a_i\, b_j}\right),
```

which the solvers minimize.  Two penalty families are provided:

- [`AbsLog`](@ref)`{p}` — `ϕ(r) = |log r|^p` (and `ϕ(0) = 0`).  Convex in log
  space; the natural penalty for *hard* covers, where `r ≤ 1` and `|log r|`
  is the log-excess of a constraint.  `AbsLog{1}` sums the log-excesses (L1),
  `AbsLog{2}` sums their squares (L2).
- [`AbsLinear`](@ref)`{p}` — `ϕ(r) = |1 - r|^p`.  Continuous at `r = 0`
  (`ϕ(0) = 1`), so zero entries of `A` contribute a bounded penalty.  This is the
  penalty used by the *soft* covers, where `r > 1` (an uncovered entry) is
  allowed but penalized.

[`cover_objective`](@ref) evaluates either penalty for a given cover:

```jldoctest quality; filter = r"(\d+\.\d{6})\d+" => s"\1"
julia> using ScaleInvariantAnalysis

julia> A = [4.0 2.0; 2.0 16.0];

julia> a = symcover(A)
2-element Vector{Float64}:
 2.0
 4.0

julia> cover_objective(AbsLog{1}(), a, A)   # sum of log-excesses (L1)
2.772588722239781

julia> cover_objective(AbsLog{2}(), a, A)   # sum of squared log-excesses (L2)
3.843624111345611
```

Both objectives are zero if and only if every constraint is exactly tight.

## Choosing a cover algorithm

| Function | Symmetric | Constraint | Objective minimized | Requires |
|---|---|---|---|---|
| [`symcover`](@ref) | yes | hard (`r ≤ 1`) | heuristic | — |
| [`cover`](@ref) | no | hard (`r ≤ 1`) | heuristic | — |
| [`symcover_min`](@ref) | yes | hard (`r ≤ 1`) | `AbsLog{2}` (or `AbsLog{1}`, `AbsLinear`) | native for `AbsLog{2}`; else JuMP |
| [`cover_min`](@ref) | no | hard (`r ≤ 1`) | `AbsLog{2}` (or `AbsLog{1}`, `AbsLinear`) | native for `AbsLog{2}`; else JuMP |
| [`soft_symcover`](@ref) | yes | soft (penalized) | `AbsLinear{2}` (or `AbsLog`, `AbsLinear{1}`) | — |
| [`soft_cover`](@ref) | no | soft (penalized) | `AbsLinear{2}` (or `AbsLinear{1}`) | — |
| [`soft_symcover_min`](@ref) | yes | soft (penalized) | `AbsLog{2}`, `AbsLinear` | JuMP |
| [`soft_cover_min`](@ref) | no | soft (penalized) | `AbsLog{2}` only | native |

**[`symcover`](@ref), [`cover`](@ref), [`soft_symcover`](@ref), and
[`soft_cover`](@ref) are recommended for production use.**  The hard-cover
heuristics run in O(mn) time for an ``m\times n`` matrix and often land within a
few percent of the objective-minimal cover (see the quality tests involving
`test/testmatrices.jl`); the soft covers add a scale-covariant multistart to
escape poor local minima of the non-convex `AbsLinear` objective.

### Objective-minimal hard covers

[`symcover_min`](@ref) and [`cover_min`](@ref) return a cover that minimizes the
chosen penalty subject to the hard constraint.  For the default `AbsLog{2}`
penalty they are solved natively (no external solver) by penalty-continuation
with a damped semismooth Newton iteration:

```jldoctest qmin; filter = r"(\d+\.\d{6})\d+" => s"\1"
julia> using ScaleInvariantAnalysis

julia> A = [1 2 3; 6 5 4];

julia> a, b = cover(A);          # fast heuristic

julia> aq, bq = cover_min(AbsLog{2}(), A);   # AbsLog{2}-minimal, native

julia> a * b'
2×3 Matrix{Float64}:
 2.16541  2.03444  3.0
 6.0      5.63709  8.31251

julia> aq * bq'
2×3 Matrix{Float64}:
 2.21042  2.0      3.0
 6.0      5.42884  8.14325
```

The native solver is near-exact (relative objective excess typically a few
``\times 10^{-7}``, growing slowly with problem size) and orders of magnitude
faster than a general-purpose convex solver.  The other penalties — `AbsLog{1}`
(a linear program) and the non-convex `AbsLinear` variants — are solved through
[JuMP](https://jump.dev/) and are loaded on demand as a package extension:

```julia
using JuMP, HiGHS   # HiGHS for AbsLog penalties
using ScaleInvariantAnalysis

a    = symcover_min(AbsLog{1}(), A)   # L1-minimal symmetric hard cover
a, b = cover_min(AbsLog{1}(), A)      # L1-minimal general hard cover
```

The [`soft_symcover_min`](@ref) soft solver is likewise JuMP-backed (HiGHS for
`AbsLog{2}`, Ipopt for the `AbsLinear` penalties).

`AbsLog{1}` minimization is deliberately left to JuMP rather than given a native
solver.  With the hard constraint `r ≥ 1` the penalty `|log r|` equals the
log-excess `α_i + α_j - log|A_{ij}|` (writing `α = log a`), so the problem is an
exact **linear program** — minimize `∑ (α_i + α_j)` subject to
`α_i + α_j ≥ log|A_{ij}|`.  The penalty-continuation Newton scheme that makes the
`AbsLog{2}` quadratic near-exact does not transfer to a linear objective; a native
path would mean reimplementing a robust LP solver, which HiGHS already provides
cheaply.  The L1 optimum is also non-unique (a whole face of the feasible
polytope rather than an isolated point), so a native solver would additionally
need a canonical selection rule to be deterministic and scale-covariant.  For
these reasons the `AbsLog{1}` hard covers require `JuMP` + `HiGHS`.

### Starting points: initialize and refine

The `AbsLinear` penalties are **non-convex**.  A solver handed one of them descends into
whichever local minimum lies below its starting point, so the start is a genuine input,
not a hint — two starts can return two different covers, both correct answers to "a local
minimum of this objective."  The package makes that structure explicit rather than hiding
it behind a default:

- **Initializers** name the starting points.  [`initialize_symcover`](@ref) and
  [`initialize_cover`](@ref) take a `strategy` — `:hardcover`, `:geomean`, `:leaveout`,
  `:diagfeasible` — and return that point.  Each is a property of `A` alone; no objective
  is involved, so an initializer takes no penalty.  By default the result is inflated until
  it covers `A`, which is what the hard-cover solvers require; `feasible=false` returns the
  point raw, which is what the soft covers want.
- **Refiners** improve a starting point in place.  [`symcover_min!`](@ref) and
  [`cover_min!`](@ref) validate the start, then optimize from it.  Which basin they reach
  is the caller's choice, by construction.
- **Solvers** bundle the two.  [`symcover_min`](@ref) and [`cover_min`](@ref) refine every
  start on a menu (the `strategies` keyword) and return the best cover by
  [`cover_objective`](@ref), so their result depends on `A` and not on an initialization the
  caller never chose.

```julia
using JuMP, Ipopt   # Ipopt for the AbsLinear penalties
using ScaleInvariantAnalysis

a = symcover_min(AbsLinear{2}(), A)                      # multistart over the whole menu
a = symcover_min(AbsLinear{2}(), A; strategies=(:geomean,))   # or commit to one start

a0 = initialize_symcover(A; strategy=:geomean)           # or drive it yourself
symcover_min!(AbsLinear{2}(), a0, A)
```

The same menu supplies the starting points of the [`soft_symcover`](@ref) and
[`soft_cover`](@ref) multistarts, so it is worth meeting once.  Those add randomized
perturbations of a base point to fill out their `starts` budget, which is multistart policy
rather than a named point, and so is not part of the menu.

For the convex `AbsLog` penalties the start cannot change the minimum *value*, and the
refiners accept one only so that the two families share an interface.  `AbsLog{2}` has a
unique minimizer, so its result is start-independent outright; `AbsLog{1}` has a flat face
of equally-good optima, so the start may select which member of that family comes back.

## Index of available tools

```@index
Modules = [ScaleInvariantAnalysis]
```

## Reference documentation

```@autodocs
Modules = [ScaleInvariantAnalysis]
Private = false
```
