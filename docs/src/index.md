```@meta
CurrentModule = ScaleInvariantAnalysis
```

# ScaleInvariantAnalysis

This package computes *covers* of matrices.  Given a matrix `A`, a cover (more
specifically, a *hard cover*) is a matrix `C` that can be defined as
`C = a * b'`, where `a` and `b` are non-negative vectors. `C` must satisfy

```math
C_{ij} \;\geq\; |A_{ij}| \quad \text{for all } i, j.
```

For a symmetric matrix the cover is symmetric (`b = a`), so a single vector
suffices: `a[i] * a[j] >= abs(A[i, j])`.

An *optimal cover* is one for which `C` is "as tight as possible" in bounding
`A` (equivalently, no larger than strictly necessary), by criteria that will be
described below.

This package also supports *soft covers*: these satisfy
`C[i, j] ⪆ abs(A[i, j])`, meaning that `C` matches or exceeds `A` at most indexes
but not necessarily all; this intuitive notion will be made concrete
below.

## Why covers?

Covers provide a natural **scale-covariant** "summary" of a matrix.  If you
rescale rows by a positive diagonal factor `D_r` and columns by `D_c`, the
optimal cover transforms as `a → D_r * a`, `b → D_c * b`, exactly mirroring
how the matrix entries change.  Scalar summaries like `norm(A)` or
`maximum(abs, A)` do not have this property and therefore implicitly encode an
arbitrary choice of units.

Most users will employ matrices that store pure numbers, and this package works
well with such matrices. But to emphasize the scale-covariance, we'll start with
an example of a 3×3 matrix whose rows and columns correspond to physical
variables with different units — position in meters, velocity in m/s, force in
Newtons.  Loading [Unitful](https://github.com/PainterQubits/Unitful.jl) lets
the matrix carry those units itself:

```jldoctest coverunits
julia> using ScaleInvariantAnalysis, Unitful

julia> L, V, F = u"m", u"m/s", u"N";  # length, velocity, force

julia> A = [1e6/L^2   1e3/(L*V)  1.0/(L*F)
            1e3/(L*V) 1.0/V^2    1e-3/(V*F)
            1.0/(L*F) 1e-3/(V*F) 1e-6/F^2];

julia> a = symcover(A);

julia> round.(typeof.(a), a; digits=6)
3-element Vector{Quantity{Float64}}:
 1000.0 m^-1
    1.0 s m^-1
    0.001 N^-1
```

`A[i,j]` has units `1/(u[i]*u[j])` (modeling a Hessian matrix for functions of
parameter vectors with units `u[i]`), so `a[i]` comes back with gradient-like
units of `1/u[i]`: the cover names each variable's natural scale outright, here
1 mm, 1 m/s, and 1 kN. Had we expressed the original matrix in those units
directly, we would have gotten the equivalent cover stated in those units.

Normalizing by the cover cancels the units along with the magnitudes, leaving a
matrix that is all-ones, dimensionless, and scale-invariant:

```jldoctest coverunits
julia> round.(A ./ (a .* a'); digits=6)
3×3 Matrix{Float64}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
```

It is worth noting that this yields 1 only for entries where the cover bound is
*tight*; had `A` been, say, diagonal, then `A ./ (a .* a')` would also be diagonal.

A cover exists only when the units of `A` factor as
`unit(A[i,j]) == unit(a[i])*unit(b[j])`, and a matrix that fails this is rejected with a
`DimensionMismatch` naming the entries that conflict.  The requirement is not one
this package adds: without it the terms in a row of `A*x` carry different units,
so `A*x` is undefined for every `x`. If a matrix can be used in matrix-vector
multiplication, it has a cover.

## Penalty functions

A cover is valid as long as every constraint is satisfied, but tighter covers
better capture the scaling of `A`.  Cover quality is measured through the ratios
`r_{ij} = |A[i,j]| / (a[i] * b[j])`: a hard cover has every `r ≤ 1`, and `r = 1`
means the constraint is exactly tight.  A **penalty function** `ϕ` combines those
ratios into a scalar objective

```math
\sum_{i,j} \phi\!\left(\frac{|A_{ij}|}{a_i\, b_j}\right),
```

which the solvers minimize.  Two penalty families are provided:

- [`AbsLog`](@ref)`{p}` — `ϕ(r) = |log r|^p` (and `ϕ(0) = 0`).  Convex in log
  space, which makes them a favorable (and therefore default) penalty for *hard*
  covers, where `r ≤ 1` and `|log r|` is the log-excess of a constraint.
  `AbsLog{1}` sums the log-excesses (L1), `AbsLog{2}` sums their squares (L2).
  Their principal disadvantage is the discontinuity at `r = 0`.
- [`AbsLinear`](@ref)`{p}` — `ϕ(r) = |1 - r|^p`.  Non-convex, but unlike
  `AbsLog` these are continuous at `r = 0` (`ϕ(0) = 1`), so zero entries of `A`
  contribute a bounded penalty.  This is the penalty used by default for the
  *soft* covers, where `r > 1` (an uncovered entry) is allowed but penalized.

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

You can override the default penalty by supplying it as an argument to the solvers.

## Choosing a cover algorithm

| Function | Symmetric | Constraint | Default (or alternative) objective | Requires |
|---|---|---|---|---|
| [`symcover`](@ref) | yes | hard (`r ≤ 1`) | heuristic | — |
| [`cover`](@ref) | no | hard (`r ≤ 1`) | heuristic | — |
| [`symcover_min`](@ref) | yes | hard (`r ≤ 1`) | `AbsLog{2}` (or `AbsLog{1}`, `AbsLinear`) | native for `AbsLog{2}`; else JuMP |
| [`cover_min`](@ref) | no | hard (`r ≤ 1`) | `AbsLog{2}` (or `AbsLog{1}`, `AbsLinear`) | native for `AbsLog{2}`; else JuMP |
| [`soft_symcover`](@ref) | yes | soft (penalized) | `AbsLinear{2}` (or `AbsLog`, `AbsLinear{1}`) | — |
| [`soft_cover`](@ref) | no | soft (penalized) | `AbsLinear{2}` (or `AbsLinear{1}`) | — |
| [`soft_symcover_min`](@ref) | yes | soft (penalized) | `AbsLog{2}`, `AbsLinear` | JuMP |
| [`soft_cover_min`](@ref) | no | soft (penalized) | `AbsLog{2}`, `AbsLinear` | native for `AbsLog{2}`; else JuMP |

**[`symcover`](@ref), [`cover`](@ref), and any native implementation can be recommended for production use,**
possibly with relaxed convergence bounds.
The heuristic solvers are particularly fast: they run in ``O(mn)`` time for an
``m\times n`` matrix and often land within a few percent of the
objective-minimal cover (see the quality tests involving
`test/testmatrices.jl`). Native solvers (both hard and soft) are intermediate, still roughly ``O(mn)`` but
requiring many iterations (and for non-convex cases, multiple start points by default) for convergence;
still, they are much faster than their JuMP-counterparts, which are provided mainly as a reference.

### Objective-minimal covers

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

julia> round(cover_objective(AbsLog{2}(), a, b, A); digits=6)
1.146646

julia> round(cover_objective(AbsLog{2}(), aq, bq, A); digits=6)
1.141281
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

### Uniqueness

The `AbsLog{2}()` penalty generally has a unique minimum, with one exception:
row/column scaling `a → γ*a`, `b → b/γ` does not affect `C` and thus invisible
to the objective function. For non-symmetric (i.e., not `symcover`) problems,
the scaling of each is pinned separately by the balance convention
`∑ n_i log a[i] = ∑ m_j log b[j]`, where `n_i`, `m_j` are the the nonzero counts
of row `i` and column `j`, respectively. This convention is not scale-invariant
but has no impact on the cover itself.

Other penalties may be more degenerate. The `AbsLog{1}()` penalty is identical
over a whole face of the feasible polytope, and its members are genuinely
different covers — the products `a[i]*b[j]` differ — that merely happen to score
the same objective. To make the result deterministic, we select the one that
additionally minimizes the `AbsLog{2}` objective.

`AbsLinear` penalties typically have isolated minima, so are not as degenerate
as `AbsLog{1}()`, but these minima occur in separate basins. There is no
guarantee of global optimality.

### Starting points: initialize and refine

For objectives with multiple minima, the solver starts from a specified point and descends.
At a lower level, this package's interface is organized in three layers:

- **Initializers** name the starting points.  [`initialize_symcover`](@ref) and
  [`initialize_cover`](@ref) take a `strategy` — `:geomean`, `:leaveout`, `:diagfeasible`,
  or `:hardcover` — and return that point.  Each is a property of `A` alone; no objective
  is involved, so an initializer takes no penalty.  A second keyword, `feasible`, says how
  the point is brought up to covering `A`: `:inflate` (the default) scales it bodily by one
  common factor, `:boost` raises only the rows touching a violated entry, and `:none`
  leaves it as it is.  The hard-cover solvers need a cover, so they take one of the first
  two; the soft covers want `:none`, since forcing the geometric mean to cover `A` would
  spoil the very property that makes it the soft `AbsLog{2}` optimum.

  The two feasible routes land on the boundary at different points, hence in different
  basins — which is why the choice is a named part of the start rather than an internal
  detail.  The heuristic [`cover`](@ref) is itself a composition of these: the geometric
  mean, boosted, then tightened.
- **Refiners** improve a starting point in place.  [`symcover_min!`](@ref),
  [`cover_min!`](@ref), and [`soft_symcover_min!`](@ref) validate the start, then optimize
  from it.  Which basin they reach is the caller's choice, by construction.  The hard
  refiners require a start that covers `A`; the soft one does not, since its objective
  constrains nothing.
- **Solvers** bundle the two.  [`symcover_min`](@ref), [`cover_min`](@ref), and
  [`soft_symcover_min`](@ref) refine every start on a menu (the `strategies` keyword) and
  return the best cover by [`cover_objective`](@ref), so their result depends on `A` and not
  on an initialization the caller never chose.

For finer control, you can run these manually:

```julia
using JuMP, Ipopt   # Ipopt for the AbsLinear penalties
using ScaleInvariantAnalysis

a = symcover_min(AbsLinear{2}(), A)                      # multistart over the whole menu
a = symcover_min(AbsLinear{2}(), A; strategies=(:geomean,))   # or commit to one start

a0 = initialize_symcover(A; strategy=:geomean)           # or drive it yourself
symcover_min!(AbsLinear{2}(), a0, A)
```

The same menu supplies the starting points of the [`soft_symcover`](@ref) and
[`soft_cover`](@ref) multistarts, adding (by default) a few randomized
perturbations of a base point up to a user-controllable number of `starts`.

For the convex `AbsLog` penalties the start cannot change the result, and the refiners
accept one only so that the two families share an interface.

## Index of available tools

```@index
Modules = [ScaleInvariantAnalysis]
```

## Reference documentation

```@autodocs
Modules = [ScaleInvariantAnalysis]
Private = false
```
