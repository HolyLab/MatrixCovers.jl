```@meta
CurrentModule = MatrixCovers
```

# Solver invariants

A cover is useful as a set of natural scales only if it does not depend on
arbitrary choices: the units in which `A` is written, or which of several
equally good covers a solver happens to find. This page states when the covers
computed by this package have these properties.

## Covariance

A cover is *covariant* if rescaling `A → D_r * A * D_c` by positive diagonal
matrices rescales the cover as `a → D_r * a`, `b → D_c * b`, so that `a * b'`
changes exactly as `A` does. Covariance is what makes the normalized matrix
`Ahat = A ./ (a .* b')` independent of units.

[`symcover`](@ref) is exactly covariant whenever every connected component of
the support contains a nonzero diagonal entry. In the example below, the cover
of the rescaled matrix `D * A * D` equals the rescaled cover of `A`, so every
ratio of their products is 1:

```jldoctest covariance
julia> using MatrixCovers, LinearAlgebra

julia> A = [1.0 1 0; 1 1 1; 0 1 1];

julia> d = [1.0, 6.0, 0.5]; D = Diagonal(d);

julia> a1 = symcover(A); a2 = symcover(D * A * D);

julia> P1 = (d .* a1) * (d .* a1)'; P2 = a2 * a2';

julia> round.(extrema(P2 ./ P1); digits=3)
(1.0, 1.0)
```

For symmetric support components without a nonzero diagonal entry, use
[`symcover_min`](@ref) when covariance is required.

[`cover`](@ref) is covariant on any support: positive diagonal rescaling of
`A` rescales the cover products identically wherever `A` is nonzero. This holds
for both the default transversal start (square, structurally nonsingular `A`)
and the `:unconstrained` start, and for any number of conjugate-gradient refinement
iterations (`cgiter`):

```jldoctest covariance
julia> e = [2.0, 0.3, 5.0]; E = Diagonal(e);

julia> function covariance_spread(cgiter; start=:auto)
           r1, c1 = cover(A; cgiter, start)
           r2, c2 = cover(D * A * E; cgiter, start)
           Q1 = (d .* r1) * (e .* c1)'
           Q2 = r2 * c2'
           return round.(extrema(Q2 ./ Q1); digits=3)
       end;

julia> covariance_spread(0)
(1.0, 1.0)

julia> covariance_spread(4)
(1.0, 1.0)
```

The `start=:geomean` alternative begins from the row and column geometric
means. Unless the support is rank one, that starting point depends on the units
of `A`, so the cover becomes covariant only as the refinement converges:

```jldoctest covariance
julia> covariance_spread(0; start=:geomean)
(0.846, 1.236)

julia> covariance_spread(4; start=:geomean)
(1.0, 1.0)
```

The [`PowerMean`](@ref) soft covers are covariant in exact arithmetic, since the
objective depends only on the ratios `|A[i,j]|/(a[i]*b[j])`. In floating point,
the hard part is the log products `log(a[i]*b[j])` across clusters of rows and
columns joined to the rest of the support only by entries with tiny `r^p`. The
solvers' polish resolves these to about the `shift` reported in
[`PowerMeanStats`](@ref MatrixCovers.PowerMeanStats), typically within a small
multiple of `tol/p` at `p = 2`; larger residuals are possible for large `p` or for
long chains of equally weak joins.

## Uniqueness

An asymmetric cover is determined only up to a scalar factor: `a → λ*a`,
`b → b/λ` leaves `a*b'` unchanged. The package chooses a unique representative,
or *gauge*, by requiring `∑ n_i log a[i] = ∑ m_j log b[j]`, where `n_i` and
`m_j` are the nonzero counts of row `i` and column `j`. This balancing
condition is applied separately to each connected component of the bipartite
support graph. It affects the factors but not their products.

Apart from this scalar factor, whether the minimizing cover is unique depends
on the penalty:

- `AbsLog{2}` has a unique minimizer except when the support pattern leaves a
  scaling freedom, as in `[0 1; 1 0]`, where every `a` with `a[1]*a[2] = 1` is
  optimal.
- `PowerMean` has unique products `a[i]*b[j]` on the support, and a symmetric
  matrix has a symmetric minimizer; where a symmetric support leaves a scaling
  freedom, the two sides of the bipartition are balanced as above.
- `AbsLog{1}` can have multiple minima; the implementation returns the one with
  the smallest `AbsLog{2}` objective.
- `AbsLinear` may have several local minima (see
  [Starting points: initialize and refine](@ref)).

## Using a single factor

Covariance applies to the product `a * b'`, not to each factor separately.
Under one-sided scaling `J → J * D`, the column factor of the balanced cover of
`J` is rescaled by `D` and also by a scalar factor from the gauge. This matters
when only one factor is used, for example to cover `J'*J` from a cover of `J`.

[`gramcover`](@ref)`(J[, W])` avoids the issue: it constructs a symmetric cover
of `J'*W*J` directly from `J` and `W`, without a cover of `J`, and covaries with
right-scaling of `J`. Alternatively, [`cover`](@ref)`(J, a)` gives the tightest
cover of `J` for a fixed row scale `a`. In the example below, the column
factors of `cover(J)` and `cover(J * D)` differ by `D` times a scalar that is
not 1, whereas the Gram covers differ by `D` alone:

```jldoctest gauge
julia> using MatrixCovers, LinearAlgebra

julia> J = [1.0 2; 3 4; 5 6];

julia> D = Diagonal([100.0, 1.0]);

julia> a1, b1 = cover(J); a2, b2 = cover(J * D);

julia> r = b2 ./ (D.diag .* b1); all(x -> x ≈ first(r), r)
true

julia> first(r) ≈ 1
false

julia> s1 = gramcover(J); s2 = gramcover(J * D);

julia> s2 ≈ D.diag .* s1
true
```
