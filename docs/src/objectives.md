```@meta
CurrentModule = MatrixCovers
```

# Cover objectives

## Penalty functions

Any hard cover is valid, but tighter covers better capture the scaling of `A`.
The tightness of a cover is measured through the ratios
`r[i, j] = |A[i,j]| / (a[i] * b[j])`: a hard cover has every `0 ≤ r[i, j] ≤ 1`,
and `r[i, j] == 1` means the constraint is exactly tight.

A **penalty function** `ϕ` combines those ratios into a scalar objective:

```math
\sum_{i,j} \phi\!\left(\frac{|A_{ij}|}{a_i\, b_j}\right).
```

Three penalty families are provided:

- [`AbsLog`](@ref)`{p}`: `ϕ(r) = |log r|^p`, with `ϕ(0) = 0`. It is convex in
  log space and is the default for hard covers.
- [`PowerMean`](@ref)`{p}`: `ϕ(r) = (r^p - 1 - p log r)/p`, with `ϕ(0) = 0`. It
  is strictly convex in log space and is the default for soft covers. Its
  minimizer makes the mean of `r^p` over the nonzero entries of every row and
  column equal to one.
- [`AbsLinear`](@ref)`{p}`: `ϕ(r) = |1-r|^p`. It is nonconvex but finite and
  continuous at zero, so an entry that shrinks to zero drops out of the soft
  cover continuously.

[`cover_objective`](@ref) evaluates any penalty for a given cover:

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

To override a solver's default penalty, pass a penalty as its first argument.

## Functions and their objectives

| Function | Symmetric | Constraint | Default (or alternative) objective | Requires |
|---|---|---|---|---|
| [`symcover`](@ref) | yes | hard (`r ≤ 1`) | heuristic | — |
| [`cover`](@ref) | no | hard (`r ≤ 1`) | heuristic | — |
| [`symcover_min`](@ref) | yes | hard (`r ≤ 1`) | `AbsLog{2}` (or `AbsLog{1}`, `AbsLinear`) | native for `AbsLog{2}`; else JuMP |
| [`cover_min`](@ref) | no | hard (`r ≤ 1`) | `AbsLog{2}` (or `AbsLog{1}`, `AbsLinear`) | native for `AbsLog{2}`; else JuMP |
| [`cover_transversal`](@ref) | no | hard, tight on a maximum-product transversal | `AbsLog{2}` (or `PowerMean{p}` with a target ratio, optionally max-balanced within blocks) | — |
| [`soft_symcover`](@ref) | yes | soft (penalized) | `PowerMean{2}` (or `PowerMean{p}`, `AbsLog`, `AbsLinear`) | native for `PowerMean`, `AbsLog{2}`; else JuMP |
| [`soft_cover`](@ref) | no | soft (penalized) | `PowerMean{2}` (or `PowerMean{p}`, `AbsLog`, `AbsLinear`) | native for `PowerMean`, `AbsLog{2}`; else JuMP |

For hard covers, [`symcover`](@ref) and [`cover`](@ref) are fast heuristics;
their `_min` counterparts minimize the selected objective.
[`cover_transversal`](@ref) minimizes the `AbsLog{2}` objective among the hard
covers that are tight on a maximum-product transversal (see
[Transversals and conditioning](@ref)); with a `PowerMean{p}` penalty and a
`target` ratio `t`, it instead pushes the ratios of the entries off the
transversal toward `t`, minimizing `∑ t^p φ(r/t)` over them with `φ` the
`PowerMean{p}` penalty. Adding `maxbalance=true` replaces that objective within
each fully indecomposable block by its `p → ∞` limit, max-balancing: across
every partition of a block's rows, the largest ratio in one direction equals
the largest in the other, which also minimizes the block's largest ratio. The
`PowerMean{p}` objective then fixes only the scales between blocks. A soft cover has no constraint to
satisfy, so there is no separate heuristic: [`soft_symcover`](@ref) and
[`soft_cover`](@ref) always minimize the objective.

The method used to minimize the objective depends on the penalty:

- `PowerMean` (native) uses power-mean scaling iterations (damped simultaneous
  updates for the symmetric problem, overrelaxed alternating updates for the
  asymmetric one), followed by damped Newton steps when those iterations
  converge slowly.
- `AbsLog{2}` (native) is a single linear least-squares solve for soft covers;
  for hard covers, see [Objective-minimal covers](@ref).
- `PowerMean{p}` with a target ratio, for [`cover_transversal`](@ref) (native),
  uses an active-set Newton method that keeps every iterate a hard cover
  tight on the transversal. With `maxbalance=true`, a parametric shortest-path
  algorithm max-balances each block first, at a cost comparable to that of
  finding the transversal for dense and for typical sparse matrices, and the
  Newton method acts on one scale per block.
- `AbsLog{1}` is a linear program solved with JuMP and HiGHS.
- `AbsLinear` is not convex; several starts are each refined to a local minimum
  with JuMP and Ipopt, and the best is returned.

The heuristics cost ``O(mn)``, except that the transversal start of
[`cover`](@ref) also solves an assignment problem by shortest augmenting paths.
The native iterative solvers cost roughly ``O(mn)`` per iteration.

## Objective-minimal covers

[`symcover_min`](@ref) and [`cover_min`](@ref) minimize the chosen penalty
subject to the hard constraint. The built-in `AbsLog{2}` solver uses an
augmented-Lagrangian iteration, with Newton steps between multiplier updates.
For the matrix below, the heuristic [`cover`](@ref) is already close to
minimal: the minimal cover lowers the objective by less than 1%.

```jldoctest qmin; filter = r"(\d+\.\d{6})\d+" => s"\1"
julia> using MatrixCovers

julia> A = [1 2 3; 6 5 4];

julia> a, b = cover(A);

julia> aq, bq = cover_min(AbsLog{2}(), A);

julia> a * b'
2×3 Matrix{Float64}:
 2.17597  2.03823  3.0
 6.0      5.62019  8.27216

julia> aq * bq'
2×3 Matrix{Float64}:
 2.21042  2.0      3.0
 6.0      5.42884  8.14325

julia> round(cover_objective(AbsLog{2}(), a, b, A); digits=6)
1.146449

julia> round(cover_objective(AbsLog{2}(), aq, bq, A); digits=6)
1.141281
```

`AbsLog{1}` and `AbsLinear` use [JuMP](https://jump.dev/) with HiGHS and Ipopt,
respectively, and are available once those packages are loaded:

```jldoctest jumpmin
julia> using MatrixCovers, JuMP, HiGHS

julia> S = [4 1 0; 1 1 5; 0 5 2];

julia> a, stats = symcover_min(AbsLog{1}(), S);

julia> round.(a; digits=6)
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

[`soft_symcover`](@ref) and [`soft_cover`](@ref) solve `PowerMean` and
`AbsLog{2}` natively, and use JuMP with HiGHS for `AbsLog{1}` and with Ipopt for
`AbsLinear`.

## Transversals and conditioning

Minimizing a penalty does not ensure that the cover-normalized matrix
`Ahat = A ./ (a .* b')` is well conditioned. What matters for conditioning is
whether the cover is tight on a suitable *transversal*, which plays the role
for a general matrix that the diagonal plays for a symmetric positive-definite
one.

For a symmetric positive-definite matrix `A`, `a * a'` covers `A` exactly when
`a[i] ≥ sqrt(A[i,i])` for every `i`. The diagonal entries require this, and it
suffices because `abs(A[i,j]) ≤ sqrt(A[i,i] * A[j,j])`. The minimal symmetric
cover is therefore `a = sqrt.(diag(A))`, which is tight on the whole diagonal
and gives `Ahat` a unit diagonal; [`symcover`](@ref) and [`symcover_min`](@ref)
both return it, to within roundoff. A general square matrix has no
distinguished diagonal, but transversals provide an analog that matters for the
asymmetric covers computed by [`cover`](@ref), [`cover_min`](@ref), and
[`cover_transversal`](@ref).

A *transversal* of an `n×n` matrix is a set of `n` entries with exactly one in
each row and each column: `A[1,σ[1]], A[2,σ[2]], …, A[n,σ[n]]` for some
permutation `σ`. Permuting the columns turns any transversal into the main
diagonal, since `A[:, σ][i, i] == A[i, σ[i]]`, so a transversal is a diagonal
up to the ordering of the columns; some authors call it a *diagonal* or
*generalized diagonal*. "Transversal" is the usual term in the literature on
sparse direct solvers. It comes from combinatorics, where a
[transversal](https://en.wikipedia.org/wiki/Transversal_(combinatorics)) of a
collection of sets picks a distinct element from each set; here each row
contributes the set of columns of its nonzero entries. The products
`prod(A[i,σ[i]] for i in 1:n)` are, up to sign, the terms of `det(A)`, so a
nonsingular matrix has at least one transversal of nonzero entries. A matrix
with such a transversal is *structurally nonsingular*. A *maximum-product
transversal* is one for which `prod(abs(A[i,σ[i]]) for i in 1:n)` is largest.
By the inequality above, the main diagonal of a symmetric positive-definite
matrix is a maximum-product transversal.

Transversals limit how well conditioned `Ahat` can be. If every transversal of
`Ahat` contains an entry of magnitude at most `t`, then the smallest singular
value of `Ahat` is at most `(n+1)/2 * t`, however well its rows and columns are
otherwise balanced.

For any hard cover, `prod(a) * prod(b)` is at least the product of the
magnitudes on any transversal of `A`, with equality exactly when
`abs(Ahat[i,σ[i]]) == 1` on every entry of that transversal. Since
`det(Ahat) = det(A) / (prod(a) * prod(b))`, the hard covers that are tight on a
maximum-product transversal are those that maximize `abs(det(Ahat))`. Such
scalings are a standard preprocessing step for Gaussian elimination, where they
favor the maximum-product transversal as the pivots (M. Olschowka and
A. Neumaier, *Linear Algebra Appl.* 240:131–151, 1996; I. S. Duff and
J. Koster, *SIAM J. Matrix Anal. Appl.* 22:973–996, 2001).

A cover that minimizes a penalty need not be tight on any transversal. The
matrix of [Conditioning and the error in `A \ y`](@ref) has a single
transversal of nonzero entries, `A[1,2]`, `A[2,3]`, and `A[3,1]`, which
includes the small entry `A[1,2] = η`. The `AbsLog{2}`-minimal cover does not
rescale `A` at all, so `Ahat` keeps the condition number of `A`. In contrast,
[`cover_transversal`](@ref) and the default start of [`cover`](@ref) are tight
on the transversal:

```jldoctest transversal
julia> using MatrixCovers, LinearAlgebra

julia> η = 1e-8;

julia> A = [1 η 0; η 1 1; 1 0 0];

julia> diag(A[:, [2, 3, 1]])   # the transversal as a diagonal
3-element Vector{Float64}:
 1.0e-8
 1.0
 1.0

julia> scaledcond(a, b) = round(cond(A ./ (a .* b')); sigdigits=3);

julia> a, b = cover_min(A); scaledcond(a, b)
2.83e8

julia> a, b = cover_transversal(A); scaledcond(a, b)
4.05

julia> a, b = cover(A); scaledcond(a, b)
4.05
```

For a square, structurally nonsingular matrix, [`cover`](@ref) is tight on a
transversal by default (`start=:transversal`). The `:unconstrained` and `:geomean`
starts, and the `_min` solvers, do not guarantee this. Tightness on a
transversal prevents only this one cause of ill-conditioning, and it comes at
the cost of a somewhat larger objective than that of the minimal cover.
