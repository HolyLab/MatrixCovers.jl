```@meta
CurrentModule = MatrixCovers
```

# MatrixCovers

This package computes *covers* of matrices.  Given a matrix `A`, a cover (more
specifically, a *hard cover*) is a matrix `C` that can be defined as
`C = a * b'`, where `a` and `b` are non-negative vectors. `C` must satisfy

```math
C_{ij} \;\geq\; |A_{ij}| \quad \text{for all } i, j.
```

For a symmetric matrix the cover is symmetric (`b = a`), so a single vector
suffices: `a[i] * a[j] >= abs(A[i, j])`.

A *minimal cover* chooses `C` "as tight as possible" in bounding `A`, by
criteria that will be described below.

The package also supports *soft covers*, which penalize uncovered entries instead
of requiring every inequality to hold.

## Why covers?

Covers provide a natural **scale-covariant** "summary" of a matrix.  If you
rescale rows by a positive diagonal factor `D_r` and columns by `D_c`, the
optimal cover transforms as `a → D_r * a`, `b → D_c * b`, so the product `a * b'`
transforms identically to `A`.  Scalar summaries like `norm(A)` or
`maximum(abs, A)` do not have this property and therefore implicitly encode an
arbitrary choice of units.

While most users will employ matrices that store pure numbers, we'll start with
an example of a 3×3 matrix whose rows and columns correspond to *physical
variables with different units* — position in meters, velocity in m/s, force in
Newtons.  Loading [Unitful](https://github.com/PainterQubits/Unitful.jl) lets
the matrix express those units directly:

```jldoctest coverunits
julia> using MatrixCovers, Unitful

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

`A[i,j]` has units `1/(u[i]*u[j])`, as in a Hessian whose parameters have units
`u[i]`. The cover has units `1/u[i]` and identifies scales of 1 mm, 1 m/s, and
1 kN. Had we expressed `A` in those units, we would have gotten the equivalent cover.

Normalizing by the cover cancels the units along with the magnitudes, leaving a
matrix that is all-ones, dimensionless, and scale-invariant:

```jldoctest coverunits
julia> round.(A ./ (a .* a'); digits=6)
3×3 Matrix{Float64}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
```

An entry is 1 only where the cover bound is tight, and this is not guaranteed for all matrices.
For example, given diagonal `A`, the normalized matrix is also diagonal.

A cover exists only when the units of `A` factor as
`unit(A[i,j]) == unit(a[i])*unit(b[j])`, and a matrix that fails this is rejected with a
`DimensionMismatch`. Without it, the terms in a row of `A*x` have incommensurate
units and cannot be added, so `A*x` is undefined for every `x`.
If a matrix can be used in matrix-vector multiplication, it has a cover.

## Penalty functions

A cover is valid as long as every constraint is satisfied, but tighter covers
better capture the scaling of `A`.  Cover quality is measured through the ratios
`r[i, j] = |A[i,j]| / (a[i] * b[j])`: a hard cover has every `0 ≤ r[i, j] ≤ 1`,
and `r[i, j] == 1` means the constraint is exactly tight.

A **penalty function** `ϕ` combines those ratios into a scalar objective

```math
\sum_{i,j} \phi\!\left(\frac{|A_{ij}|}{a_i\, b_j}\right),
```

which the solvers minimize.  Two penalty families are provided:

- [`AbsLog`](@ref)`{p}` — `ϕ(r) = |log r|^p` (and `ϕ(0) = 0`).  Convex in log
  space, which makes them a favorable (and therefore default) penalty for *hard*
  covers, where `r ≤ 1` and `|log r|` is the log-excess of a constraint.
  `AbsLog{1}` sums the log-excesses (L1), `AbsLog{2}` sums their squares (L2).
  Their principal disadvantage is the divergence and discontinuity at `r = 0`.
- [`AbsLinear`](@ref)`{p}` — `ϕ(r) = |1 - r|^p`.  Non-convex, but unlike
  `AbsLog` these are finite and continuous at `r = 0` (`ϕ(0) = 1`), so zero entries of `A`
  contribute a bounded penalty.  This is the penalty used by default for the
  *soft* covers, where `r > 1` (an uncovered entry) is allowed but penalized.

[`cover_objective`](@ref) evaluates either penalty for a given cover:

```jldoctest quality; filter = r"(\d+\.\d{6})\d+" => s"\1"
julia> using MatrixCovers

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
| [`soft_symcover`](@ref) | yes | soft (penalized) | `AbsLinear{2}` (or `AbsLog`, `AbsLinear{1}`) | native for `AbsLog`; else — |
| [`soft_cover`](@ref) | no | soft (penalized) | `AbsLinear{2}` (or `AbsLog`, `AbsLinear{1}`) | native for `AbsLog`; else — |
| [`soft_symcover_min`](@ref) | yes | soft (penalized) | `AbsLog{2}`, `AbsLinear` | native for `AbsLog{2}`; else JuMP |
| [`soft_cover_min`](@ref) | no | soft (penalized) | `AbsLog{2}`, `AbsLinear` | native for `AbsLog{2}`; else JuMP |

For hard covers, [`symcover`](@ref) trades optimality for speed while
[`symcover_min`](@ref) minimizes the selected objective. The soft solvers differ
in their convergence guarantees:

- [`soft_symcover`](@ref) and [`soft_cover`](@ref) use native coordinate descent
  and multistart. For nonconvex or nonsmooth penalties they may stop at a fixed
  point that is not a local minimum.
- [`soft_symcover_min`](@ref) and [`soft_cover_min`](@ref) find a local minimum.
  `AbsLog{2}` is native; `AbsLinear` requires JuMP and Ipopt; `AbsLog{1}` is not
  implemented.

Under `AbsLog{2}` the objective is convex with a unique minimizer, so both soft
tiers return the same result. Under `AbsLog{1}`, only the native coordinate-descent
solvers are available.

The heuristic solvers run in ``O(mn)`` time for an ``m\times n`` matrix and
often come within a few percent of the minimum objective. Native iterative
solvers are also roughly ``O(mn)`` per iteration. JuMP-based methods are slower
and primarily useful when the native methods do not support the selected
penalty.

### Covariance of the heuristics

[`symcover`](@ref) and [`cover`](@ref), the two heuristic solvers, *are not universally covariant*.  Both
are covariant when every row and column of
`A` has the same pattern of nonzeros, notably for any dense `A` lacking zero entries.  But on an *irregular* sparse support they are only
approximately covariant. A symmetric three-node path is enough to show it:

```jldoctest
julia> using MatrixCovers, LinearAlgebra

julia> A = [1.0 1 0; 1 1 1; 0 1 1];   # rows 1 and 3 supported on 2 columns, row 2 on all 3

julia> d = [1.0, 6.0, 0.5]; D = Diagonal(d);

julia> a1 = symcover(A); a2 = symcover(D * A * D);

julia> P1 = (d .* a1) * (d .* a1)'; P2 = a2 * a2';   # does scaling commute with cover-computation?

julia> round.(extrema(P2 ./ P1); digits=3)           # not for the heuristic solver
(1.0, 1.077)
```

Both covers are valid, but the difference matters when covariance is required. Use
[`symcover_min`](@ref) or [`cover_min`](@ref), whose minimizer is scale-covariant

### Objective-minimal covers

[`symcover_min`](@ref) and [`cover_min`](@ref) return a cover that minimizes the
chosen penalty subject to the hard constraint.  For the default `AbsLog{2}`
penalty they are solved natively (no external solver) by penalty-continuation
with a damped semismooth Newton iteration:

```jldoctest qmin; filter = r"(\d+\.\d{6})\d+" => s"\1"
julia> using MatrixCovers

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

The native solver typically has relative objective error of a few
``\times 10^{-7}``, growing slowly with problem size. The other penalties — `AbsLog{1}`
(a linear program) and the non-convex `AbsLinear` variants — are solved through
[JuMP](https://jump.dev/) and are loaded on demand as a package extension:

```jldoctest jumpmin
julia> using MatrixCovers, JuMP, HiGHS   # HiGHS for the AbsLog penalties

julia> S = [4 1 0; 1 1 5; 0 5 2];      # symmetric

julia> round.(symcover_min(AbsLog{1}(), S); digits=6)   # L1-minimal symmetric hard cover
3-element Vector{Float64}:
 2.0
 1.0
 5.0

julia> A = [1 2 3; 6 5 4];

julia> a, b = cover_min(AbsLog{1}(), A);   # L1-minimal general hard cover

julia> round.(a * b'; digits=6)            # tight on four of the six entries
2×3 Matrix{Float64}:
 2.4  2.0  3.0
 6.0  5.0  7.5
```

The solver returns values good to roughly solver tolerance, so these examples
round before displaying.

[`soft_symcover_min`](@ref) and [`soft_cover_min`](@ref) solve `AbsLog{2}`
natively and use JuMP with Ipopt for `AbsLinear`. They do not accept
`AbsLog{1}`; the soft `AbsLog{1}` covers are available through [`soft_symcover`](@ref) and
[`soft_cover`](@ref), which are native.

### Uniqueness

The `AbsLog{2}()` penalty generally has a unique minimum, with one exception:
row/column scaling `a → γ*a`, `b → b/γ` does not affect `C` and is thus
invisible to the objective function. For non-symmetric (i.e., not `symcover`) problems,
the scaling of each is pinned by the balance convention
`∑ n_i log a[i] = ∑ m_j log b[j]`, where `n_i`, `m_j` are the nonzero counts
of row `i` and column `j`, respectively. The gauge freedom, and hence this
convention, acts independently on each connected component of the bipartite
support graph of `A` (rows and columns as vertices, stored nonzeros as edges),
so the sums are taken within each component separately. This convention is not
scale-invariant but has no impact on the cover itself.

The `AbsLog{1}()` minimum can be a face of the feasible polytope containing
different covers with the same objective. The implementation returns the member
that also minimizes the `AbsLog{2}` objective.

`AbsLinear` penalties typically have isolated minima, so are not as degenerate
as `AbsLog{1}()`, but these minima occur in separate basins. There is no
guarantee of global optimality.

### Starting points: initialize and refine

For objectives with multiple minima, the result can depend on the starting
point. The interface has three layers:

- **Initializers** construct starting points. [`initialize_symcover`](@ref) and
  [`initialize_cover`](@ref) take a `strategy` — `:geomean`, `:leaveout`, `:diagfeasible`,
  or `:hardcover` — and return that point.  Each is a property of `A` alone; no objective
  is involved, so an initializer takes no penalty. A second keyword, `feasible`,
  controls how the point is made to cover `A`: `:inflate` (the default) applies one
  common factor, `:boost` raises only the rows touching a violated entry, and `:none`
  leaves it unchanged. Hard-cover solvers use one of the first two; soft-cover
  solvers use `:none`.

  The two feasible routes reach different points on the boundary and can enter
  different basins. The heuristic [`cover`](@ref) computes the geometric mean,
  boosts it, and then tightens it.
- **Refiners** improve a starting point in place, and are the `!`-suffixed forms of the
  solvers: [`symcover_min!`](@ref), [`cover_min!`](@ref), [`soft_symcover!`](@ref),
  [`soft_cover!`](@ref), [`soft_symcover_min!`](@ref), and [`soft_cover_min!`](@ref)
  validate a caller-provided start, then optimize from it. Hard refiners require
  a cover; soft refiners accept starts built with `feasible=:none`.
- **Solvers** bundle the two.  [`symcover_min`](@ref), [`cover_min`](@ref),
  [`soft_symcover`](@ref), [`soft_cover`](@ref), [`soft_symcover_min`](@ref), and
  [`soft_cover_min`](@ref) refine a set of starts (the `strategies` keyword or the
  multistart list) and return the best result by [`cover_objective`](@ref).

In general, a plain form chooses among several starts, while a `!` form refines
the supplied start. [`symcover!`](@ref) and [`cover!`](@ref) are initializers and
overwrite their vector arguments.

For finer control, you can run these manually:

```jldoctest manualstart
julia> using MatrixCovers, JuMP, Ipopt   # Ipopt for the AbsLinear penalties

julia> S = [4 1 0; 1 1 5; 0 5 2];

julia> round.(symcover_min(AbsLinear{2}(), S); digits=6)   # use all default starts
3-element Vector{Float64}:
 2.0
 1.0
 5.0

julia> round.(symcover_min(AbsLinear{2}(), S; strategies=(:geomean,)); digits=6)   # use one strategy
3-element Vector{Float64}:
 2.0
 1.0
 5.0

julia> a0 = initialize_symcover(S; strategy=:geomean);     # construct a start explicitly

julia> symcover_min!(AbsLinear{2}(), a0, S);

julia> round.(a0; digits=6)
3-element Vector{Float64}:
 2.0
 1.0
 5.0
```

The same menu supplies the starting points of the [`soft_symcover`](@ref) and
[`soft_cover`](@ref) multistarts, adding (by default) a few randomized
perturbations of a base point up to a user-controllable number of `starts`.

For the convex `AbsLog` penalties the start cannot change the result, and the refiners
accept one only so that the two families share an interface.

## Consuming one factor alone: gauges and Gram covers

For asymmetric covers, only the products `a[i]*b[j]` are determined by the
problem; the split into the pair is fixed by the balance convention described
under [Uniqueness](@ref). That convention makes the split *deterministic*, but
it is still a convention, and it is **not covariant** under one-sided
rescaling: if `a*b'` covers `A`, then `a*(D*b)'` covers `A*D` — but the
balanced representative of the rescaled problem is `(γ*a, D*b/γ)` for a
per-component constant `γ ≠ 1` that depends on `D`.

This matters when covers are composed. For example, the
[Levenberg-Marquardt algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
uses products `J'*J` of a Jacobian `J`. If `a*b'` covers `J`, then
`(a'*a) * b * b'` covers `J'*J`, but its tightness and invariance depends on the balance
convention for `a` and `b`.

To do better, this package provides the Gram cover `s = ` [`gramcover`](@ref)`(a, b, J[, W])`,
a symmetric cover of `J'*W*J` built from the asymmetric cover of `J`.
Built this way, `s` covaries with right-scaling of `J`.

```jldoctest gauge
julia> using MatrixCovers, LinearAlgebra

julia> J = [1.0 2; 3 4; 5 6];

julia> D = Diagonal([100.0, 1.0]);        # reparametrize the second frame

julia> a1, b1 = cover(J); a2, b2 = cover(J * D);   # `cover` is covariant because J has no zeros; `cover_min` is safer

julia> r = b2 ./ (D.diag .* b1); all(x -> x ≈ first(r), r)
true

julia> first(r) ≈ 1                       # bare-factor consumers see this constant
false

julia> s1 = gramcover(a1, b1, J); s2 = gramcover(a2, b2, J * D);

julia> s2 ≈ D.diag .* s1                  # the Gram cover co-varies exactly
true
```

## Worked example: roundoff in `A \ b`

A cover provides scales for each variable with which to measure a solution independently
of the parameterization.

Solving `x = A \ b` is *contravariant*: rescaling `A → D*A*D` and `b → D*b` sends
`x → x ./ d`, while the cover is covariant, `a → d .* a`.  The products `x .* a` are
therefore unchanged, and `∑ᵢ |xᵢ * aᵢ|` is a measure of the solution's size that is the
same in every frame.

That quantity can be estimated from the magnitudes of `A` and `b` alone, without
forming `x` at all:

```jldoctest roundoff
julia> using MatrixCovers, LinearAlgebra

julia> A = [1e6 1e3; 1e3 4.0];

julia> b = [1.5e3, 6.0];

julia> a = symcover(A);

julia> round.(a; digits=6)
2-element Vector{Float64}:
 1000.0
    2.0

julia> mag = sum(abs(bi / ai) for (bi, ai) in zip(b, a))
4.5
```

The cover reports natural scales of 1000 and 2, and `mag` estimates the size of the
solution measured against them — here within a factor of 1.5 of the truth:

```jldoctest roundoff
julia> x = A \ b;

julia> sum(abs.(x .* a))
3.0
```

Both numbers are scale-invariant, so the estimate is unchanged by any diagonal
rescaling of the problem:

```jldoctest roundoff
julia> d = [0.05, 3.0];

julia> Ad, bd = d .* A .* d', d .* b;

julia> ad = symcover(Ad);

julia> sum(abs(bi / ai) for (bi, ai) in zip(bd, ad))
4.5
```

This makes `eps(mag)` a scale-invariant estimate of the roundoff floor of the sum.
For a well-conditioned `A`, the error of the `Float64` solve meets that floor:

```jldoctest roundoff
julia> xbig = big.(A) \ big.(b);

julia> abs(sum(abs.(x .* a)) - sum(abs.(Float64.(xbig) .* a))) <= 2 * eps(mag)
true
```

The estimate is built from magnitudes only, so it knows nothing about the conditioning
of `A` or about cancellation during the solve.  When `A` is ill-conditioned the true
error sits far above the floor:

```jldoctest roundoff
julia> Aill = [1.0 -0.9999; -0.9999 1.0];

julia> bill = [0.75, 7.0];

julia> aill = symcover(Aill);

julia> magill = sum(abs(bi / ai) for (bi, ai) in zip(bill, aill));

julia> xill = Aill \ bill;

julia> xbigill = big.(Aill) \ big.(bill);

julia> err = abs(sum(abs.(xill .* aill)) - sum(abs.(Float64.(xbigill) .* aill)));

julia> err > 1e6 * eps(magill)
true
```

Folding in the condition number of the *normalized* matrix `A ./ (a .* a')` — itself
scale-invariant, since normalizing cancels the frame — restores a usable bound:

```jldoctest roundoff
julia> κ = cond(Aill ./ (aill .* aill'));

julia> err <= 1e3 * eps(κ * magill)
true
```

## Index of available tools

```@index
Modules = [MatrixCovers]
```

## Reference documentation

```@autodocs
Modules = [MatrixCovers]
Private = false
```
