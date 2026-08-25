```@meta
CurrentModule = MatrixCovers
```

# MatrixCovers

Given a matrix `A`, a *hard cover* is `C = a * b'`, where `a` and `b` are
nonnegative vectors satisfying

```math
C_{ij} \;\geq\; |A_{ij}| \quad \text{for all } i, j.
```

For symmetric `A`, a single vector suffices (`b = a`). A *minimal cover*
minimizes a chosen penalty, while a *soft cover* penalizes violations instead
of enforcing every inequality.

## Why covers?

Covers provide the "natural scales" of a matrix.  If you
rescale rows by a positive diagonal factor `D_r` and columns by `D_c`, the
optimal cover transforms as `a → D_r * a`, `b → D_c * b`, so the product `a * b'`
transforms identically to `A`.  Moreover, `Ahat = A ./ (a * b')` is scale-invariant.  Scalar metrics like `norm(A)` or
`maximum(abs, A)` implicitly encode an
arbitrary choice of units, but applying them to `Ahat` rather than `A` fixes this deficiency.

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

Here `A[i,j]` has units `1/(u[i]*u[j])`, as in a Hessian. Its cover identifies
scales of 1 mm, 1 m/s, and 1 kN. Normalization removes both units and the magnitudes affected by choice of units:

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
`unit(A[i,j]) == unit(a[i])*unit(b[j])`. But this is not an onerous requirement, as it is the same one that lets expressions like `A*x` be well-defined. 
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

Two penalty families are provided:

- [`AbsLog`](@ref)`{p}`: `ϕ(r) = |log r|^p`, with `ϕ(0) = 0`. It is convex in
  log space and is the default for hard covers.
- [`AbsLinear`](@ref)`{p}`: `ϕ(r) = |1-r|^p`. It is nonconvex, finite at zero,
  and is the default for soft covers.

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

Pass a penalty as the first solver argument to override the default.

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

For hard covers, [`symcover`](@ref) and [`cover`](@ref) are fast heuristics;
their `_min` counterparts minimize the selected objective. For soft covers:

- [`soft_symcover`](@ref) and [`soft_cover`](@ref) use native coordinate descent
  and multistart. For nonconvex or nonsmooth penalties they may stop at a fixed
  point that is not a local minimum.
- [`soft_symcover_min`](@ref) and [`soft_cover_min`](@ref) find a local minimum.
  `AbsLog{2}` is native; `AbsLinear` requires JuMP and Ipopt; `AbsLog{1}` is not
  implemented.

Under `AbsLog{2}`, the objective is convex, so both soft tiers reach the same minimum.
The heuristics cost ``O(mn)``; native iterative solvers cost roughly ``O(mn)``
per iteration.

### Covariance of the heuristics

The heuristic solvers are exactly covariant when every row and column has the
same nonzero pattern, including dense matrices without zeros. On irregular
sparse support they may be only approximately covariant:

```jldoctest
julia> using MatrixCovers, LinearAlgebra

julia> A = [1.0 1 0; 1 1 1; 0 1 1];

julia> d = [1.0, 6.0, 0.5]; D = Diagonal(d);

julia> a1 = symcover(A); a2 = symcover(D * A * D);

julia> P1 = (d .* a1) * (d .* a1)'; P2 = a2 * a2';

julia> round.(extrema(P2 ./ P1); digits=3)
(1.0, 1.077)
```

Use [`symcover_min`](@ref) or [`cover_min`](@ref) when exact covariance is
required.

### Objective-minimal covers

[`symcover_min`](@ref) and [`cover_min`](@ref) minimize the chosen penalty
subject to the hard constraint. The built-in `AbsLog{2}` solver uses penalty
continuation with a damped semismooth Newton iteration:

```jldoctest qmin; filter = r"(\d+\.\d{6})\d+" => s"\1"
julia> using MatrixCovers

julia> A = [1 2 3; 6 5 4];

julia> a, b = cover(A);

julia> aq, bq = cover_min(AbsLog{2}(), A);

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

`AbsLog{1}` and `AbsLinear` use [JuMP](https://jump.dev/) with HiGHS and Ipopt,
respectively:

```jldoctest jumpmin
julia> using MatrixCovers, JuMP, HiGHS

julia> S = [4 1 0; 1 1 5; 0 5 2];

julia> round.(symcover_min(AbsLog{1}(), S); digits=6)
3-element Vector{Float64}:
 2.0
 1.0
 5.0

julia> A = [1 2 3; 6 5 4];

julia> a, b = cover_min(AbsLog{1}(), A);

julia> round.(a * b'; digits=6)
2×3 Matrix{Float64}:
 2.4  2.0  3.0
 6.0  5.0  7.5
```

[`soft_symcover_min`](@ref) and [`soft_cover_min`](@ref) solve `AbsLog{2}`
natively and use JuMP with Ipopt for `AbsLinear`. They do not accept
`AbsLog{1}`; use [`soft_symcover`](@ref) or [`soft_cover`](@ref) instead.

### Uniqueness

For asymmetric covers, `a → γ*a`, `b → b/γ` leaves `a*b'` unchanged. The package
chooses a unique representative using
`∑ n_i log a[i] = ∑ m_j log b[j]`, where `n_i`, `m_j` are the nonzero counts
of row `i` and column `j`. The convention is applied separately to each
connected component of the bipartite support graph. It affects the factors but
not their products.

`AbsLog{2}` has a unique minimizer except when the support pattern leaves a
scaling freedom, as in `[0 1; 1 0]`, where every `a` with `a[1]*a[2] = 1` is
optimal. If `AbsLog{1}()` has multiple minima, the implementation chooses the one
with the smallest `AbsLog{2}` objective. `AbsLinear` may have several local minima.

### Starting points: initialize and refine

For objectives with multiple minima, the result can depend on its starting
point. The interface separates initialization, refinement, and multistart
selection:

- **Initializers** [`initialize_symcover`](@ref) and [`initialize_cover`](@ref)
  build a named `strategy`. Their `feasible` keyword selects uniform inflation,
  selective boosting, or no feasibility step.
- **Refiners** are the `!`-suffixed forms of the
  solvers: [`symcover_min!`](@ref), [`cover_min!`](@ref), [`soft_symcover!`](@ref),
  [`soft_cover!`](@ref), [`soft_symcover_min!`](@ref), and [`soft_cover_min!`](@ref)
  optimize a supplied point. Hard refiners require a cover; soft refiners do not.
- **Solvers** [`symcover_min`](@ref), [`cover_min`](@ref),
  [`soft_symcover`](@ref), [`soft_cover`](@ref), [`soft_symcover_min`](@ref), and
  [`soft_cover_min`](@ref) refine several starts and return the best objective.

Plain forms choose their starts; `!` forms refine the supplied start, except
[`symcover!`](@ref) and [`cover!`](@ref), which are in-place heuristics.

```jldoctest manualstart
julia> using MatrixCovers, JuMP, Ipopt

julia> S = [4 1 0; 1 1 5; 0 5 2];

julia> round.(symcover_min(AbsLinear{2}(), S); digits=6)
3-element Vector{Float64}:
 2.0
 1.0
 5.0

julia> round.(symcover_min(AbsLinear{2}(), S; strategies=(:geomean,)); digits=6)
3-element Vector{Float64}:
 2.0
 1.0
 5.0

julia> a0 = initialize_symcover(S; strategy=:geomean);

julia> symcover_min!(AbsLinear{2}(), a0, S);

julia> round.(a0; digits=6)
3-element Vector{Float64}:
 2.0
 1.0
 5.0
```

For convex `AbsLog` penalties, the start does not change the result.

## Consuming one factor alone: gauges and Gram covers

The balanced factors of an asymmetric cover are deterministic but not
individually covariant under one-sided scaling. This matters when consuming one
factor, for example when covering `J'*J` from a cover of `J`.

[`gramcover`](@ref)`(a, b, J[, W])` constructs a symmetric cover of `J'*W*J`
that covaries with right-scaling of `J`.

```jldoctest gauge
julia> using MatrixCovers, LinearAlgebra

julia> J = [1.0 2; 3 4; 5 6];

julia> D = Diagonal([100.0, 1.0]);

julia> a1, b1 = cover(J); a2, b2 = cover(J * D);

julia> r = b2 ./ (D.diag .* b1); all(x -> x ≈ first(r), r)
true

julia> first(r) ≈ 1
false

julia> s1 = gramcover(a1, b1, J); s2 = gramcover(a2, b2, J * D);

julia> s2 ≈ D.diag .* s1
true
```

## Worked example: roundoff in `A \ b`

For `x = A \ b`, diagonal rescaling sends `x → x ./ d` and a symmetric cover
`a → d .* a`. Thus `sum(abs.(x .* a))` is invariant. The quantity can be
estimated without solving for `x`:

```jldoctest roundoff
julia> using MatrixCovers, LinearAlgebra

julia> A = [1e6 1e3; 1e3 4.0];

julia> b = [1.5e3, 6.0];

julia> a = symcover(A);

julia> round.(a; digits=6)
2-element Vector{Float64}:
 1000.0
    2.0

julia> mag = sum(abs(bi / ai) for (bi, ai) in zip(b, a));

julia> round(mag; digits=6)
4.5
```

Here `mag` is within a factor of 1.5 of the scaled solution norm:

```jldoctest roundoff
julia> x = A \ b;

julia> round(sum(abs.(x .* a)); digits=6)
3.0
```

The estimate is unchanged by diagonal rescaling:

```jldoctest roundoff
julia> d = [0.05, 3.0];

julia> Ad, bd = d .* A .* d', d .* b;

julia> ad = symcover(Ad);

julia> round(sum(abs(bi / ai) for (bi, ai) in zip(bd, ad)); digits=6)
4.5
```

For well-conditioned `A`, `eps(mag)` estimates the roundoff floor:

```jldoctest roundoff
julia> xbig = big.(A) \ big.(b);

julia> abs(sum(abs.(x .* a)) - sum(abs.(Float64.(xbig) .* a))) <= 2 * eps(mag)
true
```

For ill-conditioned `A`, the error can be much larger:

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

The condition number of the normalized matrix provides a corresponding bound:

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
