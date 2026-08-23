# Objective-minimal hard covers. The default AbsLog{2} penalty is solved natively
# here; the other penalties are provided by the MatrixCoversJuMPExt and MatrixCoversIpoptExt extensions,
# whose entry points are declared as stubs below.

# ============================================================
# Public interface
# ============================================================

"""
    a = symcover_min(ϕ, A; kwargs...)
    a = symcover_min(A; kwargs...)

Return the ϕ-minimal symmetric hard cover of `A`: the vector `a` minimizing
`∑_{i,j} ϕ(|A[i,j]|/(a[i]*a[j]))` subject to `a[i]*a[j] >= |A[i,j]|` for every nonzero
entry of `A`. The no-ϕ form defaults to `AbsLog{2}()`, matching [`symcover`](@ref).

Supported ϕ values:
- `AbsLog{2}()`: solved natively (no external solver). Accepts keyword arguments
  `κs` (the penalty-continuation schedule, default `(1e2, 1e4, 1e6, 1e8)`),
  `maxiter` (Newton steps per stage, default `40`), and `linsolve` (the inner
  linear solve). `:dense` factorizes the reweighted normal equations densely,
  at O(n³) per Newton step. `:woodbury` solves the same equations as a sparse
  correction of the complete-support ones: the matrix is a sparse symmetric
  positive-definite matrix `C` plus `e*eᵀ`. Well-conditioned penalty stages
  apply that sum without forming it and solve by Jacobi-preconditioned
  conjugate gradients; the rest take a sparse Cholesky of `C` and a
  Sherman–Morrison update. Both are exact to rounding. `:woodbury` requires
  `Float64` arithmetic, a support missing at most `n ÷ 4` entries in any row,
  and at most `4n` zero entries in total, and raises an `ArgumentError`
  otherwise. `:lsqr` uses matrix-free LSQR (per-iteration cost O(nnz),
  intended for large sparse supports), right preconditioned in `Float64` by
  the diagonal of the unweighted normal matrix, joined by the rows the penalty
  currently weights — through a sparse Cholesky — once diagonal scaling alone
  would leave the system ill conditioned; this keeps its iteration count from
  growing with the penalty strength. `:auto`
  selects `:woodbury` where its requirements hold and `:dense` elsewhere.
  `linsolve` defaults to `:auto` for dense `A`; the
  `SparseMatrixCSC`/`Symmetric`/`Hermitian` sparse methods default to `:lsqr`
  instead, since neither factorization of the reweighted normal equations is
  the right solve when `nnz ≪ n²`.
- `AbsLog{1}()`: requires JuMP and HiGHS.
- `AbsLinear{1}()`, `AbsLinear{2}()`: requires JuMP and Ipopt. These objectives are
  nonconvex. Each strategy in `strategies` is refined, and the best local
  minimum is returned. A strategy that cannot produce a start is skipped.

The `AbsLog` penalties are convex in the log-scales. `AbsLog{2}` has a unique
minimizer. When the `AbsLog{1}` optimum is a face, the method returns the member
that minimizes the `AbsLog{2}` objective.

The native `AbsLog{2}` solver runs its penalty continuation in `Float64` when `A`
works in a narrower type, whose resolution the continuation's tolerances outrun, and
returns the cover in the element type `A` calls for.

!!! note
    Even the native solver is more expensive than the [`symcover`](@ref) heuristic.

See also: [`cover_min`](@ref), [`symcover`](@ref), [`symcover_min!`](@ref).
"""
function symcover_min end
symcover_min(A::AbstractMatrix; kwargs...) = symcover_min(AbsLog{2}(), A; kwargs...)

"""
    a, b = cover_min(ϕ, A)
    a, b = cover_min(A)

Return the ϕ-minimal asymmetric hard cover of `A`: vectors `a`, `b` minimizing
`∑_{i,j} ϕ(|A[i,j]|/(a[i]*b[j]))` subject to `a[i]*b[j] >= |A[i,j]|` for every nonzero
entry of `A`. The split between `a` and `b` is set within each support component
by the balance convention
`∑ nzaᵢ log a[i] = ∑ nzbⱼ log b[j]` (`nzaᵢ`, `nzbⱼ` = nonzero counts of row `i`,
column `j`). The no-ϕ form defaults to `AbsLog{2}()`, matching
[`cover`](@ref).

Supported ϕ values:
- `AbsLog{2}()`: solved natively (no external solver). Accepts keyword arguments
  `κs` (the penalty-continuation schedule, default `(1e2, 1e4, 1e6, 1e8)`),
  `maxiter` (Newton steps per stage, default `40`), and `linsolve` (the inner
  linear solve). `:dense` factorizes the reweighted normal equations densely,
  at O((m+n)³) per Newton step. `:woodbury` solves the same equations as a
  sparse correction of the complete-support ones: the matrix is a sparse
  symmetric positive-definite matrix `C` plus a rank-two term. Well-conditioned
  penalty stages apply that sum without forming it and solve by
  Jacobi-preconditioned conjugate gradients; the rest take a sparse Cholesky of
  `C` and a Woodbury update. Both are exact to rounding. `:woodbury` requires
  `Float64` arithmetic, a support missing at most `min(m, n) ÷ 4` entries in
  any row or column, and at most `4·max(m, n)` zero entries in total, and
  raises an `ArgumentError` otherwise. `:lsqr` uses matrix-free
  LSQR (per-iteration cost O(nnz), intended for large sparse supports), right
  preconditioned in `Float64` by the diagonal of the unweighted normal matrix,
  joined by the rows the penalty currently weights — through a sparse Cholesky
  — once diagonal scaling alone would leave the system ill conditioned; this
  keeps its iteration count from growing with the penalty strength.
  `:auto` selects `:woodbury` where its requirements hold and `:dense`
  elsewhere. `linsolve` defaults to `:auto` for dense `A`; the
  `SparseMatrixCSC` sparse method defaults to `:lsqr` instead, since neither
  factorization of the reweighted normal equations is the right solve when
  `nnz ≪ n²`.
- `AbsLog{1}()`: requires JuMP and HiGHS.
- `AbsLinear{1}()`, `AbsLinear{2}()`: requires JuMP and Ipopt. These objectives are
  nonconvex. Each strategy in `strategies` is refined, and the best local
  minimum is returned.

The `AbsLog` penalties are convex in the log-scales. `AbsLog{2}` has a unique
minimizer. When the `AbsLog{1}` optimum is a face, the method returns the member
that minimizes the `AbsLog{2}` objective.

The native `AbsLog{2}` solver runs its penalty continuation in `Float64` when `A`
works in a narrower type, whose resolution the continuation's tolerances outrun, and
returns the cover in the element type `A` calls for.

!!! note
    Even the native solver is more expensive than the [`cover`](@ref) heuristic.

See also: [`symcover_min`](@ref), [`cover`](@ref), [`cover_min!`](@ref).
"""
function cover_min end
cover_min(A::AbstractMatrix; kwargs...) = cover_min(AbsLog{2}(), A; kwargs...)

"""
    a = symcover_min!(ϕ, a, A; kwargs...)
    a = symcover_min!(a, A; kwargs...)

Refine the starting cover `a` into the ϕ-minimal symmetric hard cover of `A`, in
place. This is the second half of the initialize/refine pair: `a` must already be
a starting point, as produced by [`initialize_symcover`](@ref) (or by
[`symcover`](@ref)). The no-ϕ form defaults to `AbsLog{2}()`, matching
[`symcover_min`](@ref), whose keyword arguments and supported ϕ values these
methods share.

`a` must be strictly positive on every row of `A` that carries support, and must
cover `A` — `a[i]*a[j] >= abs(A[i,j])` — to within the roundoff of the log-domain
arithmetic; otherwise an `ArgumentError` is raised. Scales on rows carrying no
support are inert: whatever they hold on input, they are zero on output.

The `AbsLog` result is independent of the start. For `AbsLog{1}`, ties are broken
by the `AbsLog{2}` objective. Local minima under `AbsLinear` can depend on the
start.

See also: [`initialize_symcover`](@ref), [`symcover_min`](@ref), [`cover_min!`](@ref).
"""
function symcover_min! end
symcover_min!(a::AbstractVector, A::AbstractMatrix; kwargs...) =
    symcover_min!(AbsLog{2}(), a, A; kwargs...)

"""
    a, b = cover_min!(ϕ, a, b, A; kwargs...)
    a, b = cover_min!(a, b, A; kwargs...)

Refine the starting cover `(a, b)` into the ϕ-minimal asymmetric hard cover of
`A`, in place. This is the asymmetric counterpart of [`symcover_min!`](@ref), and
carries the same contract on the start: strict positivity on every supported row
and column, coverage of `A` to within roundoff, and inert scales on the
unsupported rows and columns. The no-ϕ form defaults to `AbsLog{2}()`, matching
[`cover_min`](@ref), whose keyword arguments and supported ϕ values these methods
share.

The product `a[i]*b[j]` is unchanged by `a -> c*a`, `b -> b/c`, so the start is
read only up to that gauge: `(a, b)` and `(2a, b/2)` give the same result, and
the result itself is pinned to the balance convention of [`cover_min`](@ref).

See also: [`initialize_cover`](@ref), [`cover_min`](@ref), [`symcover_min!`](@ref).
"""
function cover_min! end
cover_min!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...) =
    cover_min!(AbsLog{2}(), a, b, A; kwargs...)

# Symmetric AbsLog{2} hard cover via a one-sided quadratic penalty on the
# log-residuals z_ij = α_i + α_j - log|A_ij| (α = log a):
#
#   f_κ(α) = ∑_{ij ∈ support} w(z_ij) z_ij²,   w = 1 for z ≥ 0, κ for z < 0.
#
# As κ → ∞ the minimizer approaches the constrained (hard-cover) optimum. Each κ
# stage runs a damped semismooth Newton iteration: freeze the weights at the
# current α, solve the reweighted normal equations `B α = f` (an SDD system with
# the sparsity of the nonzero-pattern graph), and take a backtracking line
# search toward that point (which ensures convergence). A final uniform shift
# makes the cover exactly feasible.
function symcover_min(::AbsLog{2}, A::AbstractMatrix; kwargs...)
    a, _ = _symcover_min_abslog2(A; kwargs...)
    return a
end

# Asymmetric AbsLog{2} hard cover via the same one-sided quadratic penalty as
# `symcover_min`, on stacked log-scales x = (α; β) (α = log a over rows, β = log b
# over columns) with residuals z_ij = α_i + β_j - log|A_ij|. The row and column
# scales share a gauge freedom (α_i, β_j) → (α_i + s, β_j - s), one dimension per
# connected component of the support, that leaves every residual unchanged; during
# the solve the global one is fixed by adding v0*v0ᵀ, v0 = [ones(m); -ones(n)], to
# the normal equations, and afterwards the result is shifted, per component, to the
# balance convention ∑ nzaᵢ αᵢ = ∑ nzbⱼ βⱼ (nzaᵢ, nzbⱼ = nonzero counts of row i,
# column j, summed within the component) so it is deterministic — see
# `_cover_min_abslog2` for how the remaining per-component gauges are lifted and shifted.
function cover_min(::AbsLog{2}, A::AbstractMatrix; kwargs...)
    a, b, _ = _cover_min_abslog2(A; kwargs...)
    return a, b
end

# The AbsLog{2} objective is convex in the log-scales, so the continuation converges
# to the same cover from any start; the start is honored (it replaces the cold
# unweighted solve as the first iterate) but is not observable in the result.
function symcover_min!(::AbsLog{2}, a::AbstractVector, A::AbstractMatrix; kwargs...)
    _prepare_symcover_start!(a, A)
    anew, _ = _symcover_min_abslog2(A; start=a, kwargs...)
    a .= anew
    return a
end

function cover_min!(::AbsLog{2}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...)
    _prepare_cover_start!(a, b, A)
    anew, bnew, _ = _cover_min_abslog2(A; start=(a, b), kwargs...)
    a .= anew
    b .= bnew
    return a, b
end

# Multistart drivers for nonconvex AbsLinear objectives. The `*_min!` kernels live
# in MatrixCoversIpoptExt; the main package owns the starts and selection.
#
# Use the same roundoff-tolerant selection rule as the soft-cover multistarts.
function symcover_min(ϕ::AbsLinear, A::AbstractMatrix; strategies=SYMCOVER_MIN_STRATEGIES)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min requires a square matrix"))
    isempty(strategies) &&
        throw(ArgumentError("symcover_min: `strategies` must name at least one starting cover"))
    T = float(real(eltype(A)))
    starts = [similar(Array{T}, ax) for _ in strategies]
    # A strategy for which `A` admits no start forfeits its slot; only a menu that yields
    # no start at all leaves nothing to refine.
    built = [_initialize_symcover!(a, A, strategy, :inflate) for (a, strategy) in zip(starts, strategies)]
    covers = [symcover_min!(ϕ, a, A) for (a, ok) in zip(starts, built) if ok]
    isempty(covers) &&
        throw(ArgumentError("symcover_min: no strategy in $(string(strategies)) yields a starting cover of `A`"))
    return covers[_multistart_select([cover_objective(ϕ, a, A) for a in covers])]
end

function cover_min(ϕ::AbsLinear, A::AbstractMatrix; strategies=COVER_MIN_STRATEGIES)
    isempty(strategies) &&
        throw(ArgumentError("cover_min: `strategies` must name at least one starting cover"))
    covers = [initialize_cover(A; strategy) for strategy in strategies]
    for (a, b) in covers
        cover_min!(ϕ, a, b, A)
    end
    return covers[_multistart_select([cover_objective(ϕ, a, b, A) for (a, b) in covers])]
end

# ============================================================
# Internal helpers
# ============================================================

# Log-domain slack allowed of a start supplied to the `*_min!` refiners, in units of
# `eps(T)` scaled by the magnitudes entering the residual. The heuristics reach the
# coverage boundary through log-domain updates and so land on it only to within their
# own roundoff — a fresh `symcover` violates `a[i]*a[j] >= abs(A[i,j])` by a fraction
# of one such unit — and an exact test would reject them. This bound accepts that
# while still rejecting a start that misses coverage by any margin a solver would see.
const START_FEASIBILITY_ULPS = 64

_start_slack(lv::T, li::T, lj::T) where {T} =
    START_FEASIBILITY_ULPS * eps(T) * max(oneunit(T), abs(lv), abs(li), abs(lj))

# Shared prologue of the `symcover_min!` kernels: check that the caller's start is a
# cover of `A`, discard the inert scales on unsupported rows, and move the start onto
# the coverage boundary exactly, so every kernel begins from a feasible point.
function _prepare_symcover_start!(a::AbstractVector, A::AbstractMatrix, fname=:symcover_min!)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("$fname requires a square matrix"))
    require_abs_symmetric(A, fname)
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(ax))"))
    T = float(eltype(a))
    supp = fill!(similar(a, Bool), false)
    foreach_support_sym(A) do i, j, v
        supp[i] = true
        supp[j] = true
    end
    for i in ax
        supp[i] || (a[i] = zero(eltype(a)))
    end
    foreach_support_sym(A) do i, j, v
        (isfinite(a[i]) && a[i] > zero(a[i])) ||
            throw(ArgumentError("symcover_min! requires a start with finite positive scale on every supported row, got a[$(string(i))] = $(string(a[i]))"))
        (isfinite(a[j]) && a[j] > zero(a[j])) ||
            throw(ArgumentError("symcover_min! requires a start with finite positive scale on every supported row, got a[$(string(j))] = $(string(a[j]))"))
        lv, li, lj = log(T(v)), log(T(a[i])), log(T(a[j]))
        lv - li - lj <= _start_slack(lv, li, lj) ||
            throw(ArgumentError("symcover_min! requires a start that covers `A`, but a[$(string(i))]*a[$(string(j))] = $(string(a[i] * a[j])) < $(string(v)) = abs(A[$(string(i)),$(string(j))]); see initialize_symcover"))
    end
    return inflate_feasible!(a, A)
end

# Shared prologue of the `cover_min!` kernels; the asymmetric counterpart of
# `_prepare_symcover_start!`. The start is additionally pinned to the balance
# convention (imposed within each connected component of the support), so the
# refiners read it only up to the per-component row/column gauge.
function _prepare_cover_start!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(string(eachindex(b))), axes(A, 2)=$(string(axes(A, 2)))"))
    T = float(promote_type(eltype(a), eltype(b)))
    suppa = fill!(similar(a, Bool), false)
    suppb = fill!(similar(b, Bool), false)
    foreach_support(A) do i, j, v
        suppa[i] = true
        suppb[j] = true
    end
    for i in eachindex(a)
        suppa[i] || (a[i] = zero(eltype(a)))
    end
    for j in eachindex(b)
        suppb[j] || (b[j] = zero(eltype(b)))
    end
    foreach_support(A) do i, j, v
        (isfinite(a[i]) && a[i] > zero(a[i])) ||
            throw(ArgumentError("cover_min! requires a start with finite positive scale on every supported row, got a[$(string(i))] = $(string(a[i]))"))
        (isfinite(b[j]) && b[j] > zero(b[j])) ||
            throw(ArgumentError("cover_min! requires a start with finite positive scale on every supported column, got b[$(string(j))] = $(string(b[j]))"))
        lv, li, lj = log(T(v)), log(T(a[i])), log(T(b[j]))
        lv - li - lj <= _start_slack(lv, li, lj) ||
            throw(ArgumentError("cover_min! requires a start that covers `A`, but a[$(string(i))]*b[$(string(j))] = $(string(a[i] * b[j])) < $(string(v)) = abs(A[$(string(i)),$(string(j))]); see initialize_cover"))
    end
    _balance_cover!(a, b, A)
    return inflate_feasible!(a, b, A)
end


# Inner linear solve for the AbsLog{2} MMC Newton steps.
#
# `:dense` forms and factorizes the reweighted normal equations densely, at O(n³)
# per step.
#
# `:woodbury` splits the same matrix as `C + U Uᵀ`, where `C` is sparse (its
# off-diagonal pattern is the zero set `Z` of `A` together with the currently
# violated entries `V`) and symmetric positive definite, and `U` has one column
# (symmetric) or two (asymmetric). `C` is assembled sparsely on every such solve, and
# two sub-paths then take it, both exact to rounding, which is what the
# sign-stability stopping test in the continuation loop requires. A sparse Cholesky
# of `C` plus a Woodbury update — Sherman–Morrison, in the one-column symmetric case
# — costs far less than the dense factorization whenever `A` is close to fully
# supported. Alternatively `C + U Uᵀ` is applied as `C·x` plus the low-rank term, at
# O(nnz(C)) per application; Gershgorin on `(κ−1)·L_V` against the complete-support
# diagonal gives `1 + (κ−1)·2·maxdeg(V)/n` as an estimate of its condition number
# (the sharp bound is a small multiple of that), and while the estimate stays under
# `WOODBURY_CG_KAPPA`, Jacobi-preconditioned conjugate gradients converge to rounding
# in a few hundred such applications — cheaper than a factorization whose fill, on the
# near-random violated pattern of the early stages, approaches dense. Above it the
# factorization runs, as it does for any CG run that exhausts its iteration cap.
#
# `C` is positive definite because the complete-support matrix contributes `n` (or
# `m`) to each diagonal while the zero set subtracts a signless Laplacian `L_Z` with
# λmax(L_Z) ≤ 2·maxdeg(Z); requiring at most a quarter of a row to be zero keeps the
# difference bounded below by half the diagonal. A second requirement is about cost
# rather than definiteness: `Z` enters every matvec and every factorization, so the
# path is taken only while the total number of zeros is O(n). CHOLMOD is the sparse
# factorization behind it, and it is reliable only in `Float64`, so that is the only
# working type the path accepts.
#
# `:auto` takes `:woodbury` where it applies and `:dense` otherwise.
#
# `:lsqr` forces the matrix-free path, whose per-iteration cost is O(nnz); it is the
# intended solve for large sparse supports (where nnz ≪ n²) and is used by the
# structured/sparse methods. In `Float64` it is right preconditioned, which is what
# keeps its iteration count from growing as the continuation raises κ. The
# preconditioner is `M = diag(RᵀR) + (κ−1)·Σ_{e∈V} rₑ·rₑᵀ`: the diagonal of the
# unweighted normal matrix, together with the exact contribution of the rows LSQR
# weights by κ. All of the κ-dependence of `RᵀWR` sits in those rows, and every
# generalized eigenvalue of `(RᵀWR, M)` is a mediant of eigenvalues of
# `(RᵀR, diag(RᵀR))` and so lies in their range. `M = K·Kᵀ` and LSQR runs on
# `√W·R·K⁻ᵀ` in the variable `y = Kᵀ·x`. While the same condition-number estimate,
# taken against the unweighted diagonal, stays under `LSQR_PRECOND_KAPPA` the
# violated rows are left out and `K` is the diagonal `sqrt.(diag(RᵀR))`, applied
# without forming anything; above it they are included and `K` is the permuted sparse
# Cholesky factor of `M`, applied through the CHOLMOD factor components `F.PtL` and
# `F.UP`. `Kᵀ` is never needed as a product: the warm start `Kᵀ·x₀` is `K⁻¹·(M·x₀)`,
# which those same components and one sparse matrix-vector product supply.

# Condition-number estimate above which a Woodbury solve is factorized rather than
# iterated: past it conjugate gradients need more applications than the sparse
# Cholesky costs.
const WOODBURY_CG_KAPPA = 1000

# Condition-number estimate above which the LSQR preconditioner takes in the rows the
# penalty currently weights; below it diagonal scaling alone leaves the system well
# enough conditioned, and no factorization is formed.
const LSQR_PRECOND_KAPPA = 1000

# `(C + U·Uᵀ) x = f` solved from a factorization `F` of the sparse `C`, by the
# Woodbury identity `x = y − Y·((I + Uᵀ·Y) \ (Uᵀ·y))` with `y = C\f` and `Y = C\U`.
# One multi-right-hand-side solve of `[f U]` supplies both, and the capacitance is
# `k×k` for `U` of `k` columns: `k = 1` for the symmetric gauge `e`, where this is
# Sherman–Morrison, and `k = 2` for the asymmetric row and column indicators. `rhs`
# is the `size(U, 1)×(k+1)` buffer the block right-hand side is staged in.
function _woodbury_solve!(x, F, U, f, rhs)
    k = size(U, 2)
    copyto!(view(rhs, :, 1), f)
    copyto!(view(rhs, :, 2:k+1), U)
    sol = F \ rhs
    y = view(sol, :, 1)
    Y = view(sol, :, 2:k+1)
    K = U' * Y
    for i in axes(K, 1)
        K[i, i] += oneunit(eltype(K))
    end
    g = K \ (U' * y)
    copyto!(x, y)
    return mul!(x, Y, g, -1, 1)
end

# Jacobi-preconditioned conjugate gradients for the symmetric positive-definite
# Woodbury system `B x = f`, with `Bmul!(y, x)` applying `B` and `dg` holding its
# diagonal. `x` carries the warm start in and the iterate out; `r`, `z`, `d`, `Ad`
# are work vectors of the same length. Returns `(iters, converged)`.
#
# The Newton step has to be exact to rounding for the stage's sign-stability
# stopping test to mean what it says, so `tol` sits at the level of `eps` and a run
# that exhausts `maxiter` reports failure instead of a partial answer; the caller
# then falls back to the factorization, which is exact.
#
# `r` is carried by a recurrence that drifts from `f − B x`, so success is never
# declared on it: a claim of convergence is confirmed against a freshly computed
# residual, and a disagreement restarts the iteration there. The confirming
# application counts against `maxiter` like any other.
function _pcg!(Bmul!, x, dg, f, r, z, d, Ad, maxiter::Int, tol)
    iters = 0
    Bmul!(r, x)
    @. r = f - r
    nrm = norm(r)
    fresh = true          # `r` holds `f − B x`, not the recurrence's estimate of it
    @. z = r / dg
    copyto!(d, z)
    rz = dot(r, z)
    while iters < maxiter
        if nrm <= tol
            fresh && return iters, true
            iters += 1
            Bmul!(Ad, x)
            @. r = f - Ad
            nrm = norm(r)
            fresh = true
            @. z = r / dg
            copyto!(d, z)
            rz = dot(r, z)
            continue
        end
        iters += 1
        Bmul!(Ad, d)
        dAd = dot(d, Ad)
        dAd > 0 || break
        a = rz / dAd
        @. x += a * d
        @. r -= a * Ad
        fresh = false
        nrm = norm(r)
        @. z = r / dg
        rznew = dot(r, z)
        @. d = z + (rznew / rz) * d
        rz = rznew
    end
    return iters, fresh && nrm <= tol
end

# Matrix-free LSQR (Paige & Saunders) for the weighted least-squares problem
# `min ‖M x - b‖` underlying the reweighted normal equations `MᵀM x = Mᵀb`.
# `Amul!(y, x)` overwrites `y` with `M*x`; `Atmul!(z, y)` overwrites `z` with
# `Mᵀ*y`. Warm-started from `x0`. LSQR is used in preference to CG on the normal
# equations because it works with the condition number of `M` (≈ √κ at penalty
# strength κ) rather than that of `MᵀM` (≈ κ); at κ = 1e8 the squared conditioning
# breaks CG while LSQR stays accurate.
#
# A `Float64` caller passes a right-preconditioned operator, so `x` is then the
# preconditioned variable and `M` is `√W·R·K⁻ᵀ`; see the description of `:lsqr` in the
# inner-solve overview above. Other working types pass `√W·R` itself.
#
# The penalty least-squares problem is inconsistent (its optimal residual is
# nonzero), so the stopping test is on the normal-equations residual
# ‖Mᵀ(b - Mx)‖ ≤ atol · ‖M‖ · ‖b - Mx‖, both estimated from the bidiagonalization
# scalars (‖Mᵀr‖ = ϕbar·α·|c|, ‖r‖ = ϕbar, ‖M‖ from the Frobenius norm of the
# bidiagonal). Returns `(x, iters)`.
function _lsqr(Amul!, Atmul!, b::AbstractVector{T}, x0::AbstractVector{T};
               atol=5000 * eps(T), maxiter::Int=2 * (length(b) + length(x0)) + 100) where {T}
    x = copy(x0)
    u = similar(b)
    Amul!(u, x)
    @. u = b - u
    β = norm(u)
    β > 0 && (u ./= β)
    v = similar(x0)
    Atmul!(v, u)
    α = norm(v)
    α > 0 && (v ./= α)
    w = copy(v)
    tmpm = similar(u)
    tmpn = similar(v)
    ϕbar = β
    ρbar = α
    anorm2 = α^2          # Frobenius norm² of the lower bidiagonal ≈ ‖M‖²
    (iszero(β) || iszero(α)) && return x, 0   # x0 already optimal
    iters = 0
    for k in 1:maxiter
        iters = k
        # Golub-Kahan bidiagonalization step.
        Amul!(tmpm, v)
        @. u = tmpm - α * u
        β = norm(u)
        β > 0 && (u ./= β)
        Atmul!(tmpn, u)
        @. v = tmpn - β * v
        α = norm(v)
        α > 0 && (v ./= α)
        anorm2 += β^2 + α^2
        # Orthogonal transformation applied to the bidiagonal system.
        ρ = hypot(ρbar, β)
        iszero(ρ) && break
        c = ρbar / ρ
        s = β / ρ
        θ = s * α
        ρbar = -c * α
        ϕ = c * ϕbar
        ϕbar = s * ϕbar
        @. x += (ϕ / ρ) * w
        @. w = v - (θ / ρ) * w
        # Stop when the normal-equations residual is negligible relative to ‖M‖‖r‖,
        # or when the least-squares residual itself has vanished (consistent system).
        arnorm = ϕbar * α * abs(c)
        rnorm = abs(ϕbar)
        (arnorm <= atol * sqrt(anorm2) * rnorm || iszero(rnorm) || iszero(β)) && break
    end
    return x, iters
end

# The two layouts the AbsLog{2} continuation reads its support through. Both present
# the same abstract object: a set of support entries, each pairing two unknowns `(p, q)`
# with a value `c = log|A_ij|` and contributing a residual `z = x[p] + x[q] - c`.
#
# Every support entry is represented once: in the symmetric case an off-diagonal entry
# and its mirror are the single pair `(p, q)` with `p < q`, the diagonal is `(p, p)`,
# and the `symmetric` flag of `SupportSystem` records that an off-diagonal entry stands
# for two residuals. Its multiplicity is therefore `mult = (symmetric && p != q) ? 2 : 1`,
# and every weighted quantity below carries it.

# One stored element per support entry. Suits a support that is sparse relative to the
# grid, which is what the LSQR and dense paths face.
struct EdgeList{T}
    edges::Vector{Tuple{Int,Int}}   # support entries as pairs of unknowns
    cvals::Vector{T}                # log|A_ij| per stored entry
end

# The values on a dense grid, with `-Inf` (the image of a zero entry under `log`)
# marking a position outside the support. The sweeps then run as contiguous column
# passes that vectorize, at the cost of visiting the whole grid; the Woodbury path,
# which is taken only when the zero set is thin, is where that trade pays.
#
# Symmetric: `C` is `n×n` and only the upper triangle (`i ≤ j`) is read, entry `(i, j)`
# coupling unknowns `i` and `j`. Asymmetric: `C` is `m×n`, entry `(i, j)` coupling
# unknowns `i` and `m + j`.
struct Grid{T}
    C::Matrix{T}
end

# The support of `A` as the linear system the AbsLog{2} continuation solves. The
# unknowns are stacked log-scales `x[1:N]` — the row scales alone for a symmetric
# problem, the row scales followed by the column scales for an asymmetric one.
struct SupportSystem{T,S}
    N::Int
    supp::S                         # support layout: `EdgeList` or `Grid`
    symmetric::Bool                 # an off-diagonal entry stands for both orientations
    hassupp::BitVector              # unknowns carrying at least one support entry
    dfull::Vector{T}                # complete-support diagonal (Woodbury path only)
    zedges::Vector{Tuple{Int,Int}}  # zero set, as pairs of unknowns (Woodbury path only)
    U::Matrix{T}                    # low-rank block of `B + v0·v0ᵀ = C + U·Uᵀ` (Woodbury path only)
    v0::Vector{T}                   # gauge: dense adds `v0·v0ᵀ`, LSQR appends the row `v0ᵀx = 0`
    function SupportSystem{T,S}(N, supp, symmetric, hassupp, dfull, zedges, U, v0) where {T,S}
        return new{T,S}(N, supp, symmetric, hassupp, dfull, zedges, U, v0)
    end
end
SupportSystem{T}(N, supp::S, args...) where {T,S} = SupportSystem{T,S}(N, supp, args...)

# The support sweeps, one method per layout. `symmetric` carries the multiplicity
# convention described above; on a `Grid` it also selects the layout's geometry.

# Objective `f_κ(x) = Σ_e mult_e·w_e·z_e²`, `w_e = κ` where `z_e < 0` and 1 elsewhere.
function _fκ(x, κ, supp::EdgeList{T}, symmetric::Bool) where {T}
    edges, cvals = supp.edges, supp.cvals
    v = zero(T)
    for (e, (p, q)) in enumerate(edges)
        z = x[p] + x[q] - cvals[e]
        v += ((symmetric && p != q) ? 2 : 1) * (z < 0 ? T(κ) : oneunit(T)) * z^2
    end
    return v
end

function _fκ(x, κ, supp::Grid{T}, symmetric::Bool) where {T}
    C = supp.C
    m, n = size(C)
    κT = T(κ)
    v = zero(T)
    if symmetric
        for j in 1:n
            xj = x[j]
            cj = view(C, 1:j-1, j)
            xi = view(x, 1:j-1)
            vj = zero(T)
            @simd for i in eachindex(cj, xi)
                c = cj[i]
                z = xi[i] + xj - c
                w = ifelse(z < 0, κT, oneunit(T))
                vj += ifelse(isfinite(c), w * z^2, zero(T))
            end
            c = C[j, j]
            z = 2xj - c
            w = ifelse(z < 0, κT, oneunit(T))
            v += 2vj + ifelse(isfinite(c), w * z^2, zero(T))
        end
    else
        xr = view(x, 1:m)
        for j in 1:n
            xj = x[m+j]
            cj = view(C, :, j)
            vj = zero(T)
            @simd for i in eachindex(cj, xr)
                c = cj[i]
                z = xr[i] + xj - c
                w = ifelse(z < 0, κT, oneunit(T))
                vj += ifelse(isfinite(c), w * z^2, zero(T))
            end
            v += vj
        end
    end
    return v
end

# The objective and the violated set at `x` from one sweep: the line search needs the
# value and the stage's stopping test needs to know whether the set still matches
# `pat`, and both read the same residuals.
function _fκpat(x, κ, pat, supp::EdgeList{T}, symmetric::Bool) where {T}
    edges, cvals = supp.edges, supp.cvals
    v = zero(T)
    same = true
    for (e, (p, q)) in enumerate(edges)
        z = x[p] + x[q] - cvals[e]
        viol = z < 0
        v += ((symmetric && p != q) ? 2 : 1) * (viol ? T(κ) : oneunit(T)) * z^2
        same &= viol == pat[e]
    end
    return v, same
end

function _fκpat(x, κ, pat, supp::Grid{T}, symmetric::Bool) where {T}
    C = supp.C
    m, n = size(C)
    κT = T(κ)
    v = zero(T)
    ndiff = 0
    if symmetric
        for j in 1:n
            xj = x[j]
            cj = view(C, 1:j-1, j)
            pj = view(pat, 1:j-1, j)
            xi = view(x, 1:j-1)
            vj = zero(T)
            dj = 0
            @simd for i in eachindex(cj, pj, xi)
                c = cj[i]
                fin = isfinite(c)
                z = xi[i] + xj - c
                w = ifelse(z < 0, κT, oneunit(T))
                vj += ifelse(fin, w * z^2, zero(T))
                dj += ifelse((fin & (z < 0)) == pj[i], 0, 1)
            end
            c = C[j, j]
            fin = isfinite(c)
            z = 2xj - c
            w = ifelse(z < 0, κT, oneunit(T))
            v += 2vj + ifelse(fin, w * z^2, zero(T))
            ndiff += dj + ifelse((fin & (z < 0)) == pat[j, j], 0, 1)
        end
    else
        xr = view(x, 1:m)
        for j in 1:n
            xj = x[m+j]
            cj = view(C, :, j)
            pj = view(pat, :, j)
            vj = zero(T)
            dj = 0
            @simd for i in eachindex(cj, pj, xr)
                c = cj[i]
                fin = isfinite(c)
                z = xr[i] + xj - c
                w = ifelse(z < 0, κT, oneunit(T))
                vj += ifelse(fin, w * z^2, zero(T))
                dj += ifelse((fin & (z < 0)) == pj[i], 0, 1)
            end
            v += vj
            ndiff += dj
        end
    end
    return v, ndiff == 0
end

# Storage for the violated set of one solve, in the layout's own shape. `Matrix{Bool}`
# rather than `BitMatrix` so that the column views the `Grid` sweeps take vectorize.
_violation_pattern(supp::EdgeList) = falses(length(supp.edges))
_violation_pattern(supp::Grid) = fill(false, size(supp.C))

# Shift to exact feasibility: the smallest γ ≥ 0 with `x[p] + x[q] + 2γ ≥ c` on the
# whole support.
function _boost_shift(x, supp::EdgeList{T}, symmetric::Bool) where {T}
    edges, cvals = supp.edges, supp.cvals
    γ = zero(T)
    for (e, (p, q)) in enumerate(edges)
        γ = max(γ, (cvals[e] - x[p] - x[q]) / 2)
    end
    return γ
end

function _boost_shift(x, supp::Grid{T}, symmetric::Bool) where {T}
    C = supp.C
    m, n = size(C)
    γ = zero(T)
    if symmetric
        for j in 1:n
            xj = x[j]
            cj = view(C, 1:j-1, j)
            xi = view(x, 1:j-1)
            gj = zero(T)
            @simd for i in eachindex(cj, xi)
                gj = max(gj, (cj[i] - xi[i] - xj) / 2)
            end
            γ = max(γ, gj, (C[j, j] - 2xj) / 2)
        end
    else
        xr = view(x, 1:m)
        for j in 1:n
            xj = x[m+j]
            cj = view(C, :, j)
            gj = zero(T)
            @simd for i in eachindex(cj, xr)
                gj = max(gj, (cj[i] - xr[i] - xj) / 2)
            end
            γ = max(γ, gj)
        end
    end
    return γ
end

# One Woodbury assembly sweep: the right-hand side `f`, the violated set `vedges` with
# its per-unknown count `degV`, and the diagonal `dg` of `B` on top of its
# already-initialized zero-set correction. `dκ = κ - 1`, or 0 for the cold unweighted
# solve, which `κ === nothing` marks and in which no entry counts as violated.
#
# The vectorized pass over a column writes `vpat` and accumulates `f`; a scalar scan of
# the same column then collects the violated entries. `w·c` is formed with `ifelse` on
# `isfinite(c)`: `0 * -Inf` is NaN.
function _assemble_woodbury!(f, dg, degV, vedges, vpat, x, κ, supp::Grid{T},
                             symmetric::Bool, dκ) where {T}
    C = supp.C
    m, n = size(C)
    weighted = κ !== nothing
    κT = weighted ? T(κ) : oneunit(T)
    if symmetric
        for j in 1:n
            xj = x[j]
            cj = view(C, 1:j-1, j)
            xi = view(x, 1:j-1)
            fi = view(f, 1:j-1)
            vj = view(vpat, 1:j-1, j)
            fq = zero(T)
            @simd for i in eachindex(cj, xi, fi, vj)
                c = cj[i]
                fin = isfinite(c)
                viol = weighted & fin & (xi[i] + xj - c < 0)
                w = ifelse(viol, κT, oneunit(T))
                wc = ifelse(fin, w * c, zero(T))
                fi[i] += wc
                fq += wc
                vj[i] = viol
            end
            c = C[j, j]
            fin = isfinite(c)
            viol = weighted & fin & (2xj - c < 0)
            w = ifelse(viol, κT, oneunit(T))
            f[j] += fq + ifelse(fin, w * c, zero(T))
            vpat[j, j] = viol
            for i in eachindex(vj)
                vj[i] || continue
                push!(vedges, (i, j))
                degV[i] += 1
                degV[j] += 1
                dg[i] += dκ
                dg[j] += dκ
            end
            # A symmetric diagonal entry sits at both ends of its own residual, so it
            # lands on `dg[j]` twice while counting once in `degV`.
            if vpat[j, j]
                push!(vedges, (j, j))
                degV[j] += 1
                dg[j] += 2dκ
            end
        end
    else
        xr = view(x, 1:m)
        fr = view(f, 1:m)
        for j in 1:n
            q = m + j
            xj = x[q]
            cj = view(C, :, j)
            vj = view(vpat, :, j)
            fq = zero(T)
            @simd for i in eachindex(cj, xr, fr, vj)
                c = cj[i]
                fin = isfinite(c)
                viol = weighted & fin & (xr[i] + xj - c < 0)
                w = ifelse(viol, κT, oneunit(T))
                wc = ifelse(fin, w * c, zero(T))
                fr[i] += wc
                fq += wc
                vj[i] = viol
            end
            f[q] += fq
            for i in eachindex(vj)
                vj[i] || continue
                push!(vedges, (i, q))
                degV[i] += 1
                degV[q] += 1
                dg[i] += dκ
                dg[q] += dκ
            end
        end
    end
    return f
end

# The AbsLog{2} penalty continuation on `sys`: a sequence of stages of increasing κ,
# each a sequence of reweighted Newton steps with a backtracking line search, starting
# from `x0` (or from the cold unweighted solve when `x0 === nothing`). Returns the
# stacked log-scales and the `stats` NamedTuple the callers pass on. With `boost`, the
# result is shifted uniformly to exact feasibility `x[p] + x[q] ≥ c` on the support.
#
# The objective counts each support entry with its multiplicity, `f_κ(x) =
# Σ_e mult_e·w_e·(x[p] + x[q] - c_e)²`, so a symmetric problem is weighted on the full
# grid rather than on one triangle. The normal equations assembled below are that
# system scaled by ½ — uniformly, so they have the same solution: a support entry puts
# `w` on `B[p,p]` and `B[p,q]` and `w·c` on `f[p]`, and the same again transposed when
# `q != p`. A symmetric diagonal entry, whose row of `R` is `2·e_p`, thereby collects
# `2w` on `B[p,p]` and `w·c` on `f[p]`.
#
# The layout of `sys.supp` selects the linear-solve path: a `Grid` is built exactly
# when the Woodbury path applies, and the LSQR and dense paths run on an `EdgeList`.
function _abslog2_continuation(sys::SupportSystem{T}, x0;
                               κs, maxiter::Int, linsolve::Symbol, boost::Bool) where {T}
    N = sys.N
    supp = sys.supp
    v0 = sys.v0
    U = sys.U
    use_woodbury = supp isa Grid
    ne = supp isa EdgeList ? length(supp.edges) : 0
    use_lsqr = linsolve === :lsqr
    # CHOLMOD, which factors the LSQR preconditioner, is reliable only in Float64;
    # other working types run the plain matrix-free iteration.
    use_precond = use_lsqr && T === Float64
    # Number of residuals a stored entry stands for.
    symmetric = sys.symmetric
    mult = (p, q) -> (symmetric && p != q) ? 2 : 1
    # Diagonal of `C` before any entry is violated: the complete-support value, less
    # one per zero entry at each of its ends (twice over for a symmetric zero on the
    # diagonal, whose signless Laplacian row counts it at both).
    czero = copy(sys.dfull)
    for (p, q) in sys.zedges
        czero[p] -= oneunit(T)
        czero[q] -= oneunit(T)
    end
    # Each Newton step freezes the weights at the current `x` and solves the reweighted
    # least-squares problem `min ‖√W (R x - c)‖`, `(R x)_e = x[p] + x[q]`, whose normal
    # equations are the signless Laplacian system `B x = f`. The dense path forms
    # `B + v0·v0ᵀ` and factorizes it (a support-free variable gets an identity row; a
    # minimal scale-relative ridge lifts what the gauge term leaves singular — the
    # bipartite null space of a symmetric support such as `[0 1; 1 0]`, and the extra
    # gauge each connected component beyond the first carries in the asymmetric case).
    # The Woodbury path solves the same regularized system exactly, splitting
    # `B + v0·v0ᵀ` as `C + U·Uᵀ` around the complete-support matrix and correcting `C`
    # for the zero set and the violated entries. The LSQR path applies `√W R` and its
    # transpose matrix-free, with the gauge as an appended row, and warm-starts from
    # the incoming iterate; it solves the least-squares form directly, so its accuracy
    # tracks the conditioning of `√W R` (≈ √κ) rather than that of `B` (≈ κ).
    f = zeros(T, N)
    ws = zeros(T, ne)       # √weight per stored entry, frozen during one solve
    cv = zeros(T, ne + 1)   # √weight · log|A_ij|, with a trailing 0 gauge target
    # Entries the frozen weights of the current solve treat as violated. A full Newton
    # step that leaves this pattern intact has landed on the stage's minimizer.
    vpat = _violation_pattern(supp)
    vedges = Tuple{Int,Int}[]                # the violated entries of the current solve
    degV = zeros(Int, use_woodbury ? N : 0)  # violated entries per unknown
    dg = zeros(T, use_woodbury ? N : 0)      # diagonal of `B`, for the ridge and the CG preconditioner
    # Diagonal of the unweighted normal matrix of the gauge-augmented system,
    # `RᵀR + v0·v0ᵀ`, the base of the LSQR preconditioner: each stored entry puts its
    # multiplicity at each of its ends, and an entry whose row of `R` is `2·e_p` puts 4.
    # A variable with neither support nor gauge is given 1 so the preconditioner stays
    # positive definite.
    dpart = zeros(T, use_lsqr ? N : 0)
    if supp isa EdgeList && use_lsqr
        for (p, q) in supp.edges
            if p == q
                dpart[p] += 4 * oneunit(T)
            else
                w = mult(p, q) * oneunit(T)
                dpart[p] += w
                dpart[q] += w
            end
        end
        for p in 1:N
            dpart[p] += v0[p]^2
        end
        for p in 1:N
            dpart[p] > 0 || (dpart[p] = oneunit(T))
        end
    end
    mdiag = zeros(T, use_lsqr ? N : 0)   # the violated rows' diagonal, per unit of κ−1
    Mi = Int[]                           # COO triplets of the preconditioner
    Mj = Int[]
    Mv = T[]
    px = zeros(T, use_lsqr ? N : 0)      # scale vector recovered from the LSQR variable
    pg = zeros(T, use_lsqr ? N : 0)      # `Rᵀ√W y` before the preconditioner is applied
    # `K` of the diagonal preconditioner, which is κ-independent and so built once.
    psqrt = use_lsqr ? sqrt.(dpart) : T[]
    # COO triplets of `C`, refilled whenever a Woodbury solve is factorized.
    Ci = Int[]
    Cj = Int[]
    Cv = T[]
    rhs = zeros(T, use_woodbury ? N : 0, size(U, 2) + 1)
    dmin = use_woodbury ? minimum(sys.dfull) : oneunit(T)
    cgx = zeros(T, use_woodbury ? N : 0)
    cgr = zeros(T, use_woodbury ? N : 0)
    cgz = zeros(T, use_woodbury ? N : 0)
    cgd = zeros(T, use_woodbury ? N : 0)
    cgAd = zeros(T, use_woodbury ? N : 0)
    nsolves = Ref(0)
    nlsqr = Ref(0)
    ncg = Ref(0)
    nchol = Ref(0)
    solve_weighted = function (x, κ)
        nsolves[] += 1
        if supp isa Grid
            # `B + v0·v0ᵀ = C + U·Uᵀ` with `C = D − L_Z + (κ−1)·L_V`: the
            # complete-support diagonal `D`, corrected by the zero set `Z` and by the
            # currently violated entries `V`. One sweep over the grid collects the
            # right-hand side, the violated set, and the diagonal of `B`; everything
            # after it is O(|Z| + |V|).
            dκ = κ === nothing ? zero(T) : T(κ) - oneunit(T)
            fill!(f, zero(T))
            copyto!(dg, czero)
            fill!(degV, 0)
            empty!(vedges)
            _assemble_woodbury!(f, dg, degV, vedges, vpat, x, κ, supp, symmetric, dκ)
            # Same ridge as the dense path, so both solve the same regularized system:
            # `U·Uᵀ` puts 1 on every diagonal of `B + v0·v0ᵀ`, and every variable has
            # support here, so no identity row arises.
            dmax = zero(T)
            maxdegV = 0
            for p in 1:N
                dmax = max(dmax, dg[p] + oneunit(T))
                maxdegV = max(maxdegV, degV[p])
            end
            ridge = (dmax > 0 ? dmax : oneunit(T)) * eps(T)
            for p in 1:N
                dg[p] += oneunit(T) + ridge
            end
            # These triplets store `C` in full rather than in one triangle: the same
            # matrix then serves the matvec below and the factorization after it.
            # `sparse` sums the duplicates, and the ridge rides on the diagonal. The
            # zero set's diagonal contribution is already in `czero`, so only its
            # off-diagonal entries are pushed here; the violated entries are assembled
            # afresh on every solve, both diagonal and off-diagonal.
            empty!(Ci)
            empty!(Cj)
            empty!(Cv)
            for p in 1:N
                push!(Ci, p)
                push!(Cj, p)
                push!(Cv, czero[p] + ridge)
            end
            for (p, q) in sys.zedges
                p == q && continue
                push!(Ci, p)
                push!(Cj, q)
                push!(Cv, -oneunit(T))
                push!(Ci, q)
                push!(Cj, p)
                push!(Cv, -oneunit(T))
            end
            for (p, q) in vedges
                push!(Ci, p)
                push!(Cj, p)
                push!(Cv, dκ)
                push!(Ci, p)
                push!(Cj, q)
                push!(Cv, dκ)
                if q != p
                    push!(Ci, q)
                    push!(Cj, q)
                    push!(Cv, dκ)
                    push!(Ci, q)
                    push!(Cj, p)
                    push!(Cv, dκ)
                end
            end
            C = sparse(Ci, Cj, Cv, N, N)
            # Gershgorin on `(κ−1)·L_V` against the smallest complete-support diagonal
            # estimates the condition number of `B`. While that estimate is small,
            # conjugate gradients on `C·x + U·(Uᵀx)` reach the same answer in a few
            # hundred O(nnz(C)) applications, which is far cheaper than a factorization
            # whose fill on the near-random violated pattern of the early stages
            # approaches dense.
            κest = oneunit(T) + dκ * 2 * maxdegV / dmin
            if κest <= WOODBURY_CG_KAPPA
                copyto!(cgx, x)
                Bmul! = function (yy, xx)
                    mul!(yy, C, xx)
                    # The columns of `U` are indicator vectors, so the low-rank term is
                    # a block sum broadcast back over the same block.
                    for k in axes(U, 2)
                        s = zero(T)
                        for p in 1:N
                            u = U[p, k]
                            iszero(u) || (s += u * xx[p])
                        end
                        for p in 1:N
                            u = U[p, k]
                            iszero(u) || (yy[p] += u * s)
                        end
                    end
                    return yy
                end
                it, ok = _pcg!(Bmul!, cgx, dg, f, cgr, cgz, cgd, cgAd,
                               50 + 20 * ceil(Int, sqrt(κest)), 100 * eps(T) * norm(f))
                ncg[] += it
                ok && return copy(cgx)
            end
            nchol[] += 1
            return _woodbury_solve!(zeros(T, N), cholesky(Symmetric(C)), U, f, rhs)
        elseif use_lsqr
            edges = supp.edges
            cvals = supp.cvals
            dκ = κ === nothing ? zero(T) : T(κ) - oneunit(T)
            empty!(vedges)
            fill!(mdiag, zero(T))
            for (e, (p, q)) in enumerate(edges)
                c = cvals[e]
                viol = κ !== nothing && (x[p] + x[q] - c) < 0
                vpat[e] = viol
                sw = sqrt(mult(p, q) * (viol ? T(κ) : oneunit(T)))
                ws[e] = sw
                cv[e] = sw * c
                if viol && use_precond
                    push!(vedges, (p, q))
                    if p == q
                        mdiag[p] += 4 * oneunit(T)
                    else
                        w = mult(p, q) * oneunit(T)
                        mdiag[p] += w
                        mdiag[q] += w
                    end
                end
            end
            g = ne + 1   # index of the appended gauge row
            if use_precond
                # Diagonal scaling alone leaves a conditioning that grows with κ once
                # the violated rows dominate a variable's diagonal; past that point
                # they enter the preconditioner in full, and its Cholesky pays for
                # itself in the iterations it removes.
                κest = oneunit(T)
                for p in 1:N
                    κest = max(κest, oneunit(T) + dκ * 2 * mdiag[p] / dpart[p])
                end
                if κest <= LSQR_PRECOND_KAPPA
                    # `K` is diagonal here, so it is applied by a scaling and nothing
                    # is assembled or factorized.
                    Dmul! = function (y, yv)
                        @. px = yv / psqrt
                        for (e, (p, q)) in enumerate(edges)
                            y[e] = ws[e] * (px[p] + px[q])
                        end
                        y[g] = dot(v0, px)
                        return y
                    end
                    Dtmul! = function (z, y)
                        fill!(pg, zero(T))
                        for (e, (p, q)) in enumerate(edges)
                            t = ws[e] * y[e]
                            pg[p] += t
                            pg[q] += t
                        end
                        @. pg += v0 * y[g]
                        @. z = pg / psqrt
                        return z
                    end
                    soly, it = _lsqr(Dmul!, Dtmul!, cv, psqrt .* x)
                    nlsqr[] += it
                    return soly ./ psqrt
                end
                empty!(Mi)
                empty!(Mj)
                empty!(Mv)
                for p in 1:N
                    push!(Mi, p)
                    push!(Mj, p)
                    push!(Mv, dpart[p])
                end
                for (p, q) in vedges
                    if p == q
                        push!(Mi, p)
                        push!(Mj, p)
                        push!(Mv, 4 * dκ)
                    else
                        w = mult(p, q) * dκ
                        push!(Mi, p)
                        push!(Mj, p)
                        push!(Mv, w)
                        push!(Mi, q)
                        push!(Mj, q)
                        push!(Mv, w)
                        push!(Mi, p)
                        push!(Mj, q)
                        push!(Mv, w)
                        push!(Mi, q)
                        push!(Mj, p)
                        push!(Mv, w)
                    end
                end
                Msp = sparse(Mi, Mj, Mv, N, N)
                MF = cholesky(Symmetric(Msp))
                Kc = MF.PtL
                Uc = MF.UP
                # CHOLMOD exposes no in-place solve for a factor component, so each
                # application returns a fresh vector; the transpose product copies it
                # into the buffer LSQR hands over, which is the only copy avoidable here.
                Pmul! = function (y, yv)
                    xv = Uc \ yv
                    for (e, (p, q)) in enumerate(edges)
                        y[e] = ws[e] * (xv[p] + xv[q])
                    end
                    y[g] = dot(v0, xv)
                    return y
                end
                Ptmul! = function (z, y)
                    fill!(pg, zero(T))
                    for (e, (p, q)) in enumerate(edges)
                        t = ws[e] * y[e]
                        pg[p] += t
                        pg[q] += t
                    end
                    @. pg += v0 * y[g]
                    copyto!(z, Kc \ pg)
                    return z
                end
                mul!(px, Msp, x)
                soly, it = _lsqr(Pmul!, Ptmul!, cv, Kc \ px)
                nlsqr[] += it
                return (Uc \ soly)::Vector{T}
            end
            Amul! = function (y, xx)
                for (e, (p, q)) in enumerate(edges)
                    y[e] = ws[e] * (xx[p] + xx[q])
                end
                y[g] = dot(v0, xx)
                return y
            end
            Atmul! = function (z, y)
                fill!(z, zero(T))
                for (e, (p, q)) in enumerate(edges)
                    t = ws[e] * y[e]
                    z[p] += t
                    z[q] += t
                end
                @. z += v0 * y[g]
                return z
            end
            sol, it = _lsqr(Amul!, Atmul!, cv, x)
            nlsqr[] += it
            return sol
        else
            edges = supp.edges
            cvals = supp.cvals
            fill!(f, zero(T))
            B = v0 * v0'
            for (e, (p, q)) in enumerate(edges)
                c = cvals[e]
                viol = κ !== nothing && (x[p] + x[q] - c) < 0
                vpat[e] = viol
                w = viol ? T(κ) : oneunit(T)
                f[p] += w * c
                B[p, p] += w
                B[p, q] += w
                if q != p
                    f[q] += w * c
                    B[q, q] += w
                    B[q, p] += w
                end
            end
            # A minimal scale-relative ridge on the supported diagonals, sized by the
            # largest of them, lifts the gauge directions `v0·v0ᵀ` does not pin: the
            # bipartite null space of a symmetric support, and the independent gauge
            # each connected component of an asymmetric support beyond the first
            # carries. The right-hand side is orthogonal to every gauge null vector, so
            # the ridge leaves the recovered scales essentially unperturbed, and the
            # gauge it fixes is unobservable — no product a_i·b_j spans two components.
            # Support-free variables get an identity row.
            dmax = zero(T)
            for p in 1:N
                dmax = max(dmax, B[p, p])
            end
            ridge = (dmax > 0 ? dmax : oneunit(T)) * eps(T)
            for p in 1:N
                B[p, p] = sys.hassupp[p] ? B[p, p] + ridge : oneunit(T)
            end
            return Symmetric(B) \ f
        end
    end
    x = x0 === nothing ? solve_weighted(zeros(T, N), nothing) : x0
    for κ in κs
        fcur = _fκ(x, κ, supp, symmetric)
        for _ in 1:maxiter
            xnew = solve_weighted(x, κ)
            t = one(T)
            xt = xnew
            fnew, stable = _fκpat(xt, κ, vpat, supp, symmetric)
            while fnew > fcur && t > 500_000 * eps(T)
                t /= 2
                xt = x .+ t .* (xnew .- x)
                fnew = _fκ(xt, κ, supp, symmetric)
                stable = false
            end
            x = xt
            # `f_κ` is convex and the dense and Woodbury steps solve its quadratic model
            # exactly, so a whole step that leaves the violated set unchanged has reached
            # the stage's minimizer: the gradient there is the model's, which is zero.
            # The `:lsqr` solves are inexact and carry no such guarantee.
            !use_lsqr && stable && break
            fcur - fnew <= 5000 * eps(T) * max(fcur, one(T)) && break
            fcur = fnew
        end
    end
    # Uniform boost to exact feasibility: x[p] + x[q] ≥ log|A_ij| on the support.
    # `boost=false` leaves the iterate untouched, for the soft objective, which
    # imposes no coverage constraint and whose optimum the boost would move off.
    if boost
        γ = _boost_shift(x, supp, symmetric)
        for p in 1:N
            x[p] += γ
        end
    end
    return x, (; nsolves=nsolves[], lsqriters=nlsqr[], cgiters=ncg[],
               cholsolves=nchol[],
               linsolve=(use_lsqr ? :lsqr : use_woodbury ? :woodbury : :dense))
end

# Worker for `symcover_min(::AbsLog{2})`. Returns `(a, stats)` where `stats` is a
# NamedTuple `(; nsolves, lsqriters, cgiters, cholsolves, linsolve)` recording the
# number of inner linear solves, the total LSQR and conjugate-gradient iterations (0 on
# paths that run neither), how many Woodbury solves fell to the sparse factorization,
# and which path ran.
# `linsolve` reports the path that ran: `:dense`, `:woodbury`, or `:lsqr`.
# A working type narrower than `Float64` is solved in `Float64` and the cover converted
# back, since the continuation's tolerances assume double precision.
# `start`, when given, is a positive cover of `A`
# indexed like `axes(A, 1)` and supplies the first iterate in place of the cold
# unweighted solve; the objective is convex, so it changes the path but not the result.
function _symcover_min_abslog2(A::AbstractMatrix; κs=(1e2, 1e4, 1e6, 1e8),
                               maxiter::Int=40, linsolve::Symbol=:auto, start=nothing,
                               boost::Bool=true, fname=:symcover_min)
    linsolve in (:auto, :dense, :lsqr, :woodbury) ||
        throw(ArgumentError("linsolve must be :auto, :dense, :lsqr, or :woodbury; got :$linsolve"))
    # The shared entry to the native solve, reached from every sym `*_min` method,
    # so the precondition is checked once here rather than at each of them.
    require_abs_symmetric(A, fname)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min requires a square matrix"))
    # The problem only ever depends on abs.(A), a real quantity, so the working type
    # stays real even for complex A (e.g. a complex Hermitian) — Complex has no total
    # order, and the reweighted Newton solve below compares residuals with `<`/`min`.
    T = float(real(eltype(A)))
    # The continuation's tolerances are multiples of `eps(T)` — the decrease test at
    # `5000*eps(T)`, the line-search floor at `500_000*eps(T)` — and the objective
    # `f_κ` itself must resolve differences of that order at κ up to 1e8. Both assume
    # double precision: at `eps(Float32)` the stages past the first carry no
    # resolvable descent, and the continuation halts far from the constrained optimum.
    # A narrower working type therefore runs the whole solve in `Float64` and the
    # cover is returned in the caller's type. `convert` keeps the wrapper — the
    # `Symmetric`, `Hermitian`, sparse and structured storage all have their own
    # support traversals — and widens a complex eltype to `ComplexF64`.
    if eps(T) > eps(Float64)
        a64, stats = _symcover_min_abslog2(convert(AbstractMatrix{promote_type(eltype(A), Float64)}, A);
                                           κs, maxiter, linsolve, start, boost, fname)
        return T.(a64), stats
    end
    n = length(ax)
    use_lsqr = linsolve === :lsqr
    # The support is measured before it is laid out, so that only the layout the chosen
    # path needs is built. The Newton solve runs on 1-based positions 1:n and is
    # scattered back onto `a` through `ax` so `A`'s own axes are honored.
    G = _sym_support(A, T)
    hassupp = falses(n)
    nsupp = 0              # support entries of `A`, counted in both orientations
    maxzero = 0            # largest number of zeros in any row of `A`
    for (ip, i) in enumerate(ax)
        ns = length(_slots(G, i))
        hassupp[ip] = ns > 0
        nsupp += ns
        maxzero = max(maxzero, n - ns)
    end
    # The Woodbury path splits the normal equations around the complete-support matrix
    # `n·I + e·eᵀ`, so its cost is set by the zero set `Z` rather than by `n`. Two
    # separate conditions gate it. Per row: `n·I − L_Z` is positive definite only
    # while no row carries more than `n ÷ 4` zeros. In total: `Z` is materialized and
    # then traversed by every matvec and every factorization, so the path is worth
    # taking only while `|Z|` stays O(n) — a support that is merely thin per row can
    # still carry Θ(n²) zeros, and the split would then be dense work under a name
    # that promises otherwise.
    nzero = n * n - nsupp
    zbudget = 4 * n
    use_woodbury = false
    if !use_lsqr && linsolve !== :dense
        ok = T === Float64 && maxzero <= n ÷ 4 && nzero <= zbudget
        if linsolve === :woodbury && !ok
            T === Float64 ||
                throw(ArgumentError("linsolve=:woodbury requires Float64 arithmetic, but `A` works in $T; use :dense or :lsqr"))
            maxzero <= n ÷ 4 ||
                throw(ArgumentError("linsolve=:woodbury requires every row of `A` to have at most n ÷ 4 = $(n ÷ 4) zeros; got $maxzero"))
            throw(ArgumentError("linsolve=:woodbury requires `A` to have at most 4n = $zbudget zeros in total; got $nzero"))
        end
        use_woodbury = ok
    end
    # The layout the chosen path reads: the Woodbury sweeps run on the grid, the LSQR
    # and dense sweeps on the list. Both hold one element per unordered pair `{i, j}` —
    # the residuals `z_ij = α_i + α_j - log|A_ij|` of a pair and its mirror are the
    # same, and `SupportSystem` weights the stored entry for both.
    supp = if use_woodbury
        C = fill(T(-Inf), n, n)
        for (ip, i) in enumerate(ax)
            for s in _slots(G, i)
                jp = G.idx[s] - first(ax) + 1
                jp >= ip && (C[ip, jp] = log(G.val[s]))
            end
        end
        Grid{T}(C)
    else
        edges = Tuple{Int,Int}[]
        cvals = T[]
        for (ip, i) in enumerate(ax)
            for s in _slots(G, i)
                jp = G.idx[s] - first(ax) + 1
                if jp >= ip
                    push!(edges, (ip, jp))
                    push!(cvals, log(G.val[s]))
                end
            end
        end
        EdgeList{T}(edges, cvals)
    end
    # Zero set of `A`, one entry per unordered pair. It is the off-diagonal pattern of
    # the sparse `C` the Woodbury path factorizes.
    zedges = Tuple{Int,Int}[]
    if use_woodbury
        mark = falses(n)
        for (ip, i) in enumerate(ax)
            for s in _slots(G, i)
                mark[G.idx[s] - first(ax) + 1] = true
            end
            for jp in ip:n
                mark[jp] || push!(zedges, (ip, jp))
            end
            fill!(mark, false)
        end
    end
    # Nothing here pins a gauge: `v0` is zero, so the dense path adds no rank-one term
    # and the LSQR gauge row is inert. The ridge lifts what singularity remains — the
    # null space of a bipartite support graph such as `[0 1; 1 0]`.
    sys = SupportSystem{T}(n, supp, true, hassupp,
                           use_woodbury ? fill(T(n), n) : T[], zedges,
                           ones(T, use_woodbury ? n : 0, use_woodbury ? 1 : 0),
                           zeros(T, n))
    x0 = start === nothing ? nothing :
         T[hassupp[ip] ? log(T(start[i])) : zero(T) for (ip, i) in enumerate(ax)]
    α, stats = _abslog2_continuation(sys, x0; κs, maxiter, linsolve, boost)
    # Dense scale vector matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    a = similar(Array{T}, ax)
    for (ip, i) in enumerate(ax)
        a[i] = hassupp[ip] ? exp(α[ip]) : zero(T)
    end
    return a, stats
end

# Worker for `cover_min(::AbsLog{2})`. Returns `(a, b, stats)` with `stats` a
# NamedTuple `(; nsolves, lsqriters, cgiters, cholsolves, linsolve)` (see
# `_symcover_min_abslog2`, whose promotion of narrow working types this shares).
# `start`, when given, is a positive cover `(a, b)` indexed like the rows and columns
# of `A`, supplying the first iterate in place of the cold unweighted solve.
function _cover_min_abslog2(A::AbstractMatrix; κs=(1e2, 1e4, 1e6, 1e8),
                            maxiter::Int=40, linsolve::Symbol=:auto, start=nothing,
                            boost::Bool=true)
    linsolve in (:auto, :dense, :lsqr, :woodbury) ||
        throw(ArgumentError("linsolve must be :auto, :dense, :lsqr, or :woodbury; got :$linsolve"))
    axr = axes(A, 1)
    axc = axes(A, 2)
    # The problem only ever depends on abs.(A), a real quantity, so the working type
    # stays real even for complex A (e.g. a complex Hermitian) — Complex has no total
    # order, and the reweighted Newton solve below compares residuals with `<`/`min`.
    T = float(real(eltype(A)))
    # The continuation's tolerances are multiples of `eps(T)` — the decrease test at
    # `5000*eps(T)`, the line-search floor at `500_000*eps(T)` — and the objective
    # `f_κ` itself must resolve differences of that order at κ up to 1e8. Both assume
    # double precision: at `eps(Float32)` the stages past the first carry no
    # resolvable descent, and the continuation halts far from the constrained optimum.
    # A narrower working type therefore runs the whole solve in `Float64` and the
    # cover is returned in the caller's type. `convert` keeps the wrapper — the
    # `Symmetric`, `Hermitian`, sparse and structured storage all have their own
    # support traversals — and widens a complex eltype to `ComplexF64`.
    if eps(T) > eps(Float64)
        a64, b64, stats = _cover_min_abslog2(convert(AbstractMatrix{promote_type(eltype(A), Float64)}, A);
                                             κs, maxiter, linsolve, start, boost)
        return T.(a64), T.(b64), stats
    end
    m = length(axr)
    n = length(axc)
    N = m + n
    use_lsqr = linsolve === :lsqr
    # Each support entry links a row position ip to a column position m+jp. Internal
    # positions 1:m index rows, m+1:m+n index columns, and results are scattered back
    # through axr/axc so A's axes are honored. The support is measured before it is
    # laid out, so that only the layout the chosen path needs is built.
    G = _row_support(A, T)
    nzrow = zeros(Int, m)   # support entries per row, for the balance convention
    nzcol = zeros(Int, n)   # ditto per column
    ne = 0
    for (ip, i) in enumerate(axr)
        for s in _slots(G, i)
            jp = G.idx[s] - first(axc) + 1
            ne += 1
            nzrow[ip] += 1
            nzcol[jp] += 1
        end
    end
    hasrow = nzrow .> 0
    hascol = nzcol .> 0
    # The Woodbury path splits the normal equations around the complete-support
    # matrix `D + u_r·u_rᵀ + u_c·u_cᵀ`, so its cost is set by the zero set `Z` rather
    # than by `N`, and `D − L_Z` is positive definite only while `Z` stays thin. The
    # bound is taken against `min(m, n)`, the smaller of the two diagonal blocks.
    maxzero = 0
    for ip in 1:m
        maxzero = max(maxzero, n - nzrow[ip])
    end
    for jp in 1:n
        maxzero = max(maxzero, m - nzcol[jp])
    end
    zbound = min(m, n) ÷ 4
    # Two separate conditions gate the path. Per row and column: `D − L_Z` is positive
    # definite only while neither carries more than `min(m, n) ÷ 4` zeros. In total:
    # `Z` is materialized and then traversed by every matvec and every factorization,
    # so the path is worth taking only while `|Z|` stays O(m + n) — a support that is
    # merely thin per row can still carry Θ(m·n) zeros, and the split would then be
    # dense work under a name that promises otherwise.
    nzero = m * n - ne
    zbudget = 4 * max(m, n)
    use_woodbury = false
    if !use_lsqr && linsolve !== :dense
        ok = T === Float64 && maxzero <= zbound && nzero <= zbudget
        if linsolve === :woodbury && !ok
            T === Float64 ||
                throw(ArgumentError("linsolve=:woodbury requires Float64 arithmetic, but `A` works in $T; use :dense or :lsqr"))
            maxzero <= zbound ||
                throw(ArgumentError("linsolve=:woodbury requires every row and column of `A` to have at most min(m, n) ÷ 4 = $zbound zeros; got $maxzero"))
            throw(ArgumentError("linsolve=:woodbury requires `A` to have at most 4·max(m, n) = $zbudget zeros in total; got $nzero"))
        end
        use_woodbury = ok
    end
    # The layout the chosen path reads: the Woodbury sweeps run on the grid, the LSQR
    # and dense sweeps on the list. Both hold `log|A_ij|` for each support entry.
    supp = if use_woodbury
        C = fill(T(-Inf), m, n)
        for (ip, i) in enumerate(axr)
            for s in _slots(G, i)
                C[ip, G.idx[s]-first(axc)+1] = log(G.val[s])
            end
        end
        Grid{T}(C)
    else
        edges = Tuple{Int,Int}[]
        cvals = T[]
        for (ip, i) in enumerate(axr)
            for s in _slots(G, i)
                push!(edges, (ip, m + G.idx[s] - first(axc) + 1))
                push!(cvals, log(G.val[s]))
            end
        end
        EdgeList{T}(edges, cvals)
    end
    # Zero set of `A` as stacked-position pairs missing from the support: the
    # off-diagonal pattern of the sparse `C` the Woodbury path factorizes. The
    # complete-support diagonal it corrects is `n` on rows and `m` on columns.
    zedges = Tuple{Int,Int}[]
    dfull = zeros(T, use_woodbury ? N : 0)
    Umat = zeros(T, use_woodbury ? N : 0, use_woodbury ? 2 : 0)
    if use_woodbury
        for ip in 1:m
            dfull[ip] = T(n)
            Umat[ip, 1] = oneunit(T)
        end
        for jp in 1:n
            dfull[m+jp] = T(m)
            Umat[m+jp, 2] = oneunit(T)
        end
        mark = falses(n)
        for (ip, i) in enumerate(axr)
            for s in _slots(G, i)
                mark[G.idx[s] - first(axc) + 1] = true
            end
            for jp in 1:n
                mark[jp] || push!(zedges, (ip, m + jp))
            end
            fill!(mark, false)
        end
    end
    # Row and column scales share the global (e; −e) gauge, which every path pins
    # through `v0`: ±1 on supported variables, 0 on support-free ones (which carry no
    # constraint and are decoupled with an identity row instead). After the solve a
    # closed-form shift, applied within each component, moves the result to the balance
    # convention, so the pinned gauge is not observable.
    v0 = zeros(T, N)
    for ip in 1:m
        hasrow[ip] && (v0[ip] = one(T))
    end
    for jp in 1:n
        hascol[jp] && (v0[m+jp] = -one(T))
    end
    sys = SupportSystem{T}(N, supp, false, vcat(hasrow, hascol), dfull, zedges, Umat, v0)
    x0 = if start === nothing
        nothing
    else
        sa, sb = start
        s0 = zeros(T, N)
        for (ip, i) in enumerate(axr)
            hasrow[ip] && (s0[ip] = log(T(sa[i])))
        end
        for (jp, j) in enumerate(axc)
            hascol[jp] && (s0[m+jp] = log(T(sb[j])))
        end
        s0
    end
    x, stats = _abslog2_continuation(sys, x0; κs, maxiter, linsolve, boost)
    # Shift along the (e; -e) gauges to the balance convention ∑ nzaᵢ αᵢ = ∑ nzbⱼ βⱼ,
    # imposed within each connected component of the support: the gauge acts
    # independently on each component, so a single global shift would leave the
    # per-component splits wherever the ridge (or LSQR's gauge row) put them. It
    # applies whether or not the iterate was boosted: the gauge is a convention, not a
    # constraint, and every cover this package returns satisfies it.
    rowcomp, colcomp, ncomp = _support_components(A)
    Lα = zeros(T, ncomp)
    Lβ = zeros(T, ncomp)
    nec = zeros(Int, ncomp)
    for ip in 1:m
        c = rowcomp[ip]
        c == 0 && continue
        Lα[c] += nzrow[ip] * x[ip]
        nec[c] += nzrow[ip]
    end
    for jp in 1:n
        c = colcomp[jp]
        c == 0 && continue
        Lβ[c] += nzcol[jp] * x[m+jp]
    end
    s = Lβ
    for c in 1:ncomp
        s[c] = (Lβ[c] - Lα[c]) / (2 * nec[c])
    end
    # Dense scale vectors matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    a = similar(Array{T}, axr)
    b = similar(Array{T}, axc)
    for (ip, i) in enumerate(axr)
        a[i] = hasrow[ip] ? exp(x[ip] + s[rowcomp[ip]]) : zero(T)
    end
    for (jp, j) in enumerate(axc)
        b[j] = hascol[jp] ? exp(x[m+jp] - s[colcomp[jp]]) : zero(T)
    end
    return a, b, stats
end

# Workers for the soft (unconstrained) AbsLog{2} covers. The soft objective
# `∑_{i,j∈S} (log a_i + log a_j - log|A_ij|)²` is the hard workers' reweighted
# least-squares problem with every weight held at 1, which is the cold solve they
# already take as their first iterate: `κs=()` runs no penalty continuation, and
# `boost=false` keeps the unconstrained minimizer where it is. It is convex, so one
# linear solve settles it — no iteration and no multistart, unlike the non-convex
# `AbsLinear` soft covers.
#
# Both paths inherit the hard workers' handling of a singular signless Laplacian (the
# `[0 1; 1 0]` support graph among them) and of support-free rows and columns.
_soft_symcover_min_abslog2(A::AbstractMatrix; kwargs...) =
    _symcover_min_abslog2(A; κs=(), boost=false, fname=:soft_symcover_min, kwargs...)
_soft_cover_min_abslog2(A::AbstractMatrix; kwargs...) =
    _cover_min_abslog2(A; κs=(), boost=false, kwargs...)

# Internal exact reference implemented by the MatrixCoversJuMPExt extension; used only to
# cross-check the native `symcover_min(::AbsLog{2})` in the test suite.
function symcover_min_jump end

# Internal exact reference implemented by the MatrixCoversJuMPExt extension; used only to
# cross-check the native `cover_min(::AbsLog{2})` in the test suite.
function cover_min_jump end

# A solve that stops for any reason other than a solved one leaves the model holding
# a point that does not solve the problem posed -- the base of an unbounded ray, or
# whatever the solver last had. Handing that back would be a minimal cover in name
# only, so it is an error. `ALMOST_*` statuses are rejected along with the rest:
# they report a tolerance the caller did not ask for.
#
# Takes the status rather than the model so that both solver extensions can share it
# without the main package depending on JuMP.
function check_solved(status, solver, fname)
    Symbol(status) in (:OPTIMAL, :LOCALLY_SOLVED) ||
        error("$fname: $solver terminated with status $status")
    return nothing
end
