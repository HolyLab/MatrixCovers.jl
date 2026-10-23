```@meta
CurrentModule = MatrixCovers
```

# MatrixCovers

MatrixCovers computes rank-one bounds on the magnitudes of the entries of a
matrix. Given a matrix `A`, a *hard cover* is `C = a * b'`, where `a` and `b`
are nonnegative vectors satisfying

```math
C_{ij} \;\geq\; |A_{ij}| \quad \text{for all } i, j.
```

For symmetric `A`, a single vector suffices (`b = a`). A matrix has many hard
covers; a *minimal cover* is the one that minimizes a chosen penalty on the gap
between `C` and `|A|`. A *soft cover* penalizes violations of the inequality
instead of enforcing it.

## Installation

```julia
julia> using Pkg; Pkg.add("MatrixCovers")
```

## Why covers?

Covers provide the "natural scales" of a matrix. Many quantities computed from
a matrix, such as `norm(A)` or `maximum(abs, A)`, depend on the units in which
its rows and columns are expressed. Changing those units rescales the rows by a
positive diagonal factor `D_r` and the columns by `D_c`. A cover changes with
them, `a → D_r * a` and `b → D_c * b`, so the product `a * b'` transforms
exactly as `A` does (see [Covariance](@ref) for the precise conditions). The
normalized matrix `Ahat = A ./ (a * b')` is therefore unaffected by the choice
of units, and so are norms and other metrics computed from `Ahat`.

Most matrices store pure numbers, but units make this property concrete. In the
3×3 matrix below, the rows and columns correspond to *physical variables with
different units*: position in meters, velocity in m/s, and force in newtons.
Loading [Unitful](https://github.com/PainterQubits/Unitful.jl) lets the matrix
express those units directly:

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

As in a Hessian, `A[i,j]` has units `1/(u[i]*u[j])`, where `u[i]` is the unit
of variable `i`. The cover identifies scales of 1 mm, 1 m/s, and 1 kN. Dividing
by the cover removes both the units and the magnitudes that depend on them:

```jldoctest coverunits
julia> round.(A ./ (a .* a'); digits=6)
3×3 Matrix{Float64}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
```

Every entry here equals 1 because this cover is tight on every entry, which is
not typical. An entry of the normalized matrix has magnitude 1 only where the
cover is tight; for a diagonal `A`, for example, the normalized matrix is also
diagonal.

A cover exists only when the units of `A` factor as
`unit(A[i,j]) == unit(a[i])*unit(b[j])`. This is the same requirement that
makes `A*x` well defined, so any matrix that can be used in matrix-vector
multiplication has a cover.

## Conditioning and the error in `A \ y`

The condition number `cond(A)` is the usual guide to the accuracy of solving
`A x = y`: a backward-stable method loses a factor of about `cond(A)` in
relative accuracy. But `cond(A)` depends on how the rows and columns of `A` are
scaled, and when the ill-conditioning comes from scaling, the rule of thumb can
overestimate the error by many orders of magnitude. A cover separates the
scaling from the rest of the matrix.

The matrix below has a smallest singular value near `η`, so its condition
number is about `1/η`:

```jldoctest conditioning
julia> using MatrixCovers, LinearAlgebra

julia> η = 1e-8;

julia> A = [1 η 0; η 1 1; 1 0 0]
3×3 Matrix{Float64}:
 1.0     1.0e-8  0.0
 1.0e-8  1.0     1.0
 1.0     0.0     0.0

julia> round.(svdvals(A); sigdigits=3)
3-element Vector{Float64}:
 1.41
 1.41
 5.0e-9

julia> round(eps() * cond(A); sigdigits=2)
6.3e-8
```

The rule of thumb therefore predicts a relative error near `6e-8`. Yet the
computed solution is far more accurate than that:

```jldoctest conditioning
julia> y = [1.0, 2.0, 3.0];

julia> x = A \ y;

julia> xbig = big.(A) \ big.(y);

julia> norm(x - xbig) / norm(xbig) < 10 * eps()
true
```

The reason is that `A` is a row and column rescaling of a well-conditioned
matrix, and `cond(A)` measures the scaling as well as the matrix. The cover of
`A` finds the scales, and dividing by the cover removes them:

```jldoctest conditioning
julia> a, b = cover(A);

julia> round.(a; sigdigits=3), round.(b; sigdigits=3)
([0.0001, 10000.0, 0.0001], [10000.0, 0.0001, 0.0001])

julia> Ahat = A ./ (a .* b')
3×3 Matrix{Float64}:
 1.0      1.0  0.0
 1.0e-16  1.0  1.0
 1.0      0.0  0.0

julia> round.(svdvals(Ahat); sigdigits=3)
3-element Vector{Float64}:
 1.8
 1.25
 0.445
```

The cover is tight on `A[1,2]`, `A[2,3]`, and `A[3,1]`, one entry in each row
and each column. Such a set of entries is called a *transversal*, and a cover
that leaves the transversals of `A` loose can fail to remove the scaling (see
[Transversals and conditioning](@ref)).

Because rescaling the rows and columns of `A` rescales its cover identically
(see [Covariance](@ref)), `Ahat` and its singular values do not depend on the
units of `A`. A cover removes only the ill-conditioning that comes from
scaling: nearly dependent rows or columns remain, and show up as small singular
values of `Ahat`.

To see why the solve is accurate, express the error in scale-invariant terms.
`A x = y` is equivalent to `Ahat * (b .* x) = y ./ a`, and the matrix and both
vectors in the latter form are scale-invariant. The natural measure of the size
of `x` (and of its error) is therefore `norm(b .* x)`, and of `y`,
`norm(y ./ a)`. In these terms the rule of thumb involves `cond(Ahat)`, which is
about 4, rather than `cond(A)`, which is about 3e8. The large condition number
of `A` comes from its scaling, which does not affect the error measured this
way.

`cond(Ahat)` governs perturbations of `A` bounded in norm. The affiliated
manuscript derives the corresponding result for perturbations bounded entry by
entry, which follows from classical results applied to the scale-invariant
form. Suppose the computed solution `x̂` exactly solves `(A + ΔA) x̂ = y` with
`|ΔA[i,j]| ≤ ε * a[i] * b[j]`, a perturbation of each entry by at most a
fraction `ε` of its cover value. Then

```math
\frac{\lVert b \odot (\hat x - x) \rVert_2}{\lVert b \odot x \rVert_2}
\;\le\; \frac{\epsilon\, c}{1 - \epsilon\, c},
\qquad
c = \sqrt{n}\, \bigl\lVert\, |\hat A^{-1}|\, e \,\bigr\rVert_2,
```

provided that `ε * c < 1`, where `e` is the vector of ones and `⊙` denotes
elementwise multiplication. Like `cond(Ahat)`, the constant `c` depends only on
`Ahat`. For a dense square matrix, `\` uses LU factorization with partial
pivoting, which yields a perturbation of this form with `ε` a modest multiple
of `eps()` when the entries of the computed factors do not grow much relative
to the cover.

For this matrix `c` is about 6.5, so the bound is far smaller than
`eps() * cond(A)`, and the error in the cover-weighted norm satisfies it:

```jldoctest conditioning
julia> c = sqrt(3) * norm(abs.(inv(Ahat)) * ones(3));

julia> round(c; sigdigits=3)
6.48

julia> norm(b .* (x - xbig)) / norm(b .* xbig) <= c * eps()
true
```

## Making a symmetric matrix positive definite

Tikhonov regularization and modified Newton methods shift a symmetric matrix,
`H + τ*I`, to make it positive definite. The identity has no units, so the
same `τ` perturbs different variables by amounts that depend on their units.
Consider a matrix with two blocks whose variables have different scales:

```jldoctest posdef
julia> using MatrixCovers, LinearAlgebra

julia> H0 = [1.0 2; 2 1];

julia> H = cat(1e6 * H0, H0; dims=(1, 2))
4×4 Matrix{Float64}:
 1.0e6  2.0e6  0.0  0.0
 2.0e6  1.0e6  0.0  0.0
 0.0    0.0    1.0  2.0
 0.0    0.0    2.0  1.0

julia> round(eigmin(H); sigdigits=6)
-1.0e6
```

The identity shift needs `τ ≥ 1e6` to make `H` positive semidefinite, which
swamps the entries of the second block. Replacing `I` with the diagonal of a
cover, `H + τ * Diagonal(a .^ 2)`, shifts each variable in proportion to its
own scale. The smallest sufficient `τ` is `-eigmin(Hhat)` for the
cover-normalized matrix `Hhat`:

```jldoctest posdef
julia> a = symcover(H);

julia> Hhat = H ./ (a .* a');

julia> τ = round(-eigmin(Hhat); digits=6)
0.5

julia> round.(τ .* a .^ 2; sigdigits=3)
4-element Vector{Float64}:
 1.0e6
 1.0e6
 1.0
 1.0
```

The shift `τ * Diagonal(a .^ 2)` perturbs both blocks by the same relative
amount, whatever the units. Measured in the cover norm (the spectral norm of
the cover-normalized perturbation), no symmetric perturbation smaller than this
shift makes `H` positive semidefinite.

When computing `eigmin(Hhat)` is too expensive, a bound suffices. Every entry
of `Hhat` has magnitude at most one, so its eigenvalues lie in `[-q, q]`, where
`q` is the largest number of nonzeros in any row of `H`. For a hard cover, the
shift with `τ = q` therefore always makes `H` positive semidefinite, and any
larger `τ` makes it positive definite. Here `q = 2`, and the eigenvalues of
`Hhat` are `-0.5` and `1.5`, so `τ = 2` gives a positive definite matrix:

```jldoctest posdef
julia> isposdef(H + 2 * Diagonal(a .^ 2))
true
```

## Which function to use

For most uses, [`symcover`](@ref) (symmetric matrices) and [`cover`](@ref)
(general matrices) suffice: they cost ``O(mn)`` and are covariant under
rescaling. [`symcover_min`](@ref) and [`cover_min`](@ref) compute covers that
minimize a chosen penalty, at greater cost (see
[Objective-minimal covers](@ref)). A minimal cover is not necessarily better
for conditioning: on the matrix of
[Conditioning and the error in `A \ y`](@ref), [`cover_min`](@ref) leaves
`cond(Ahat)` near `3e8`, because it is not tight on the transversal.
[`cover_transversal`](@ref) minimizes the penalty among the covers that are.

For applications that standardize data rather than precondition a solver,
consider [`soft_cover`](@ref) and [`soft_symcover`](@ref). A soft cover need not
satisfy every inequality `|A[i,j]| ≤ a[i] * b[j]`; with the default
`PowerMean{2}` penalty, the ratios `|A[i,j]| / (a[i] * b[j])` over the nonzero
entries of each row and column have root-mean-square equal to one (see
[Penalty functions](@ref)).
