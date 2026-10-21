```@meta
CurrentModule = MatrixCovers
```

# Solver invariants

## Covariance

[`symcover`](@ref) is exactly covariant whenever every connected component of
the support contains a nonzero diagonal entry:

```jldoctest covariance
julia> using MatrixCovers, LinearAlgebra

julia> A = [1.0 1 0; 1 1 1; 0 1 1];

julia> d = [1.0, 6.0, 0.5]; D = Diagonal(d);

julia> a1 = symcover(A); a2 = symcover(D * A * D);

julia> P1 = (d .* a1) * (d .* a1)'; P2 = a2 * a2';

julia> round.(extrema(P2 ./ P1); digits=3)
(1.0, 1.0)
```

[`cover`](@ref) is covariant on any support: positive diagonal rescaling of
`A` rescales the cover products identically wherever `A` is nonzero.
This holds for both the default transversal start (square, structurally
nonsingular `A`) and the `:covariant` start, and for any number of
conjugate-gradient refinement iterations (`cgiter`):

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

The `start=:geomean` alternative begins from the row and column geometric means.
Unless the support is rank one, that point depends on the units `A` is written in, so the cover becomes scale-covariant only as the refinement converges:

```jldoctest covariance
julia> covariance_spread(0; start=:geomean)
(0.846, 1.236)

julia> covariance_spread(4; start=:geomean)
(1.0, 1.0)
```

For symmetric support components without a nonzero diagonal entry, use
[`symcover_min`](@ref) when covariance is required.

## Uniqueness

For asymmetric covers, `a → γ*a`, `b → b/γ` leaves `a*b'` unchanged. The package
chooses a unique representative using
`∑ n_i log a[i] = ∑ m_j log b[j]`, where `n_i`, `m_j` are the nonzero counts
of row `i` and column `j`. The convention is applied separately to each
connected component of the bipartite support graph. It affects the factors but
not their products.

`AbsLog{2}` has a unique minimizer except when the support pattern leaves a
scaling freedom, as in `[0 1; 1 0]`, where every `a` with `a[1]*a[2] = 1` is
optimal. `PowerMean` has unique products `a[i]*b[j]` on the support, and a
symmetric matrix has a symmetric minimizer; where a symmetric support leaves a
scaling freedom, the two sides of the bipartition are balanced as above. If
`AbsLog{1}()` has multiple minima, the implementation chooses the one with the
smallest `AbsLog{2}` objective. `AbsLinear` may have several local minima.

## Consuming one factor alone: gauges and Gram covers

The balanced factors of an asymmetric cover are deterministic but not
individually covariant under one-sided scaling. This matters when consuming one
factor, for example when covering `J'*J` from a cover of `J`.

[`gramcover`](@ref)`(J[, W])` constructs a symmetric cover of `J'*W*J`
directly from `J` and `W`, without a cover of `J`, and covaries with
right-scaling of `J`. [`cover`](@ref)`(J, a)` gives the tightest cover of `J`
for a fixed row scale `a`.

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

