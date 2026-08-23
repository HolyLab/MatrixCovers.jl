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

# Worker for `symcover_min(::AbsLog{2})`. Returns `(a, stats)` where `stats` is a
# NamedTuple `(; nsolves, lsqriters, cgiters, cholsolves, linsolve)` recording the
# number of inner linear solves, the total LSQR and conjugate-gradient iterations (0 on
# paths that run neither), how many Woodbury solves fell to the sparse factorization,
# and which path ran.
# `linsolve` reports the path that ran: `:dense`, `:woodbury`, or `:lsqr`.
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
    n = length(ax)
    use_lsqr = linsolve === :lsqr
    # CHOLMOD, which factors the LSQR preconditioner, is reliable only in Float64;
    # other working types run the plain matrix-free iteration.
    use_precond = use_lsqr && T === Float64
    # Support entries, one per residual z_ij = α_i + α_j - log|A_ij|, with `cvals`
    # holding log|A_ij| alongside. The gather reports each off-diagonal pair in both
    # orientations and the diagonal once, which is the full-grid weighting the
    # objective is defined with. The Newton solve runs on 1-based positions 1:n and
    # is scattered back onto `a` through `ax` so `A`'s own axes are honored.
    G = _sym_support(A, T)
    edges = Tuple{Int,Int}[]
    cvals = T[]
    hassupp = falses(n)
    maxzero = 0            # largest number of zeros in any row of `A`
    for (ip, i) in enumerate(ax)
        slots = _slots(G, i)
        for s in slots
            push!(edges, (ip, G.idx[s] - first(ax) + 1))
            push!(cvals, log(G.val[s]))
        end
        hassupp[ip] = !isempty(slots)
        maxzero = max(maxzero, n - length(slots))
    end
    ne = length(edges)
    # The Woodbury path splits the normal equations around the complete-support matrix
    # `n·I + e·eᵀ`, so its cost is set by the zero set `Z` rather than by `n`. Two
    # separate conditions gate it. Per row: `n·I − L_Z` is positive definite only
    # while no row carries more than `n ÷ 4` zeros. In total: `Z` is materialized and
    # then traversed by every matvec and every factorization, so the path is worth
    # taking only while `|Z|` stays O(n) — a support that is merely thin per row can
    # still carry Θ(n²) zeros, and the split would then be dense work under a name
    # that promises otherwise.
    nzero = n * n - ne
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
    # Zero set of `A` in the same convention as `edges`: both orientations of an
    # off-diagonal pair, the diagonal once. It is the off-diagonal pattern of the
    # sparse `C` the Woodbury path factorizes.
    zedges = Tuple{Int,Int}[]
    # Diagonal of `C` before any entry is violated: `n` from the complete-support
    # matrix, less what `L_Z` puts there — one per zero entry of the row, and one more
    # for a zero on the diagonal, which `L_Z` counts twice.
    czero = fill(T(n), use_woodbury ? n : 0)
    if use_woodbury
        mark = falses(n)
        for (ip, i) in enumerate(ax)
            for s in _slots(G, i)
                mark[G.idx[s] - first(ax) + 1] = true
            end
            for jp in 1:n
                mark[jp] && continue
                push!(zedges, (ip, jp))
                czero[ip] -= oneunit(T)
                ip == jp && (czero[ip] -= oneunit(T))
            end
            fill!(mark, false)
        end
    end
    fκ = function (α, κ)
        v = zero(T)
        for (e, (ip, jp)) in enumerate(edges)
            z = α[ip] + α[jp] - cvals[e]
            v += (z < 0 ? T(κ) : oneunit(T)) * z^2
        end
        return v
    end
    # The objective and the violated set at `α` from one sweep: the line search needs
    # the value and the stage's stopping test needs to know whether the set still
    # matches `pat`, and both read the same residuals.
    fκpat = function (α, κ, pat)
        v = zero(T)
        same = true
        for (e, (ip, jp)) in enumerate(edges)
            z = α[ip] + α[jp] - cvals[e]
            viol = z < 0
            v += (viol ? T(κ) : oneunit(T)) * z^2
            same &= viol == pat[e]
        end
        return v, same
    end
    # Each Newton step freezes the weights at the current α and solves the reweighted
    # least-squares problem `min ‖√W (Rα - c)‖`, `(Rα)_e = α_i + α_j`, whose normal
    # equations are the signless Laplacian system `B α = f`. The dense path forms and
    # factorizes `B` (a support-free variable gets an identity row; a minimal
    # scale-relative ridge lifts the bipartite gauge null space, e.g. the `[0 1; 1 0]`
    # support graph whose signless Laplacian is singular). The Woodbury path solves the
    # same regularized system exactly, splitting `B` as `C + e·eᵀ` around the
    # complete-support matrix `n·I + e·eᵀ` and correcting `C` for the zero set and the
    # violated entries. The LSQR path applies `√W R`
    # and its transpose matrix-free and warm-starts from the incoming iterate; it
    # solves the least-squares form directly, so its accuracy tracks the conditioning
    # of `√W R` (≈ √κ) rather than that of `B` (≈ κ).
    ws = zeros(T, ne)   # √weight per support entry, frozen during one solve
    cv = zeros(T, ne)   # √weight · log|A_ij| (LSQR right-hand side)
    f = zeros(T, n)
    # Entries the frozen weights of the current solve treat as violated. A full Newton
    # step that leaves this pattern intact has landed on the stage's minimizer.
    vpat = falses(ne)
    vedges = Tuple{Int,Int}[]              # the violated entries of the current solve
    degV = zeros(Int, use_woodbury ? n : 0)  # violated entries per row
    dg = zeros(T, use_woodbury ? n : 0)      # diagonal of `B`, for the ridge and the CG preconditioner
    # Diagonal of the unweighted normal matrix `RᵀR`, the base of the LSQR
    # preconditioner: each directed support entry puts 1 at each of its ends, and a
    # diagonal entry, whose row of `R` is `2·e_p`, puts 4. A support-free variable is
    # given 1 so the preconditioner stays positive definite.
    dpart = zeros(T, use_lsqr ? n : 0)
    if use_lsqr
        for (ip, jp) in edges
            if ip == jp
                dpart[ip] += 4 * oneunit(T)
            else
                dpart[ip] += oneunit(T)
                dpart[jp] += oneunit(T)
            end
        end
        for p in 1:n
            dpart[p] > 0 || (dpart[p] = oneunit(T))
        end
    end
    mdiag = zeros(T, use_lsqr ? n : 0)   # the violated rows' diagonal, per unit of κ−1
    Mi = Int[]                           # COO triplets of the preconditioner
    Mj = Int[]
    Mv = T[]
    px = zeros(T, use_lsqr ? n : 0)      # scale vector recovered from the LSQR variable
    pg = zeros(T, use_lsqr ? n : 0)      # `Rᵀ√W y` before the preconditioner is applied
    # `K` of the diagonal preconditioner, which is κ-independent and so built once.
    psqrt = use_lsqr ? sqrt.(dpart) : T[]
    # COO triplets of `C`, refilled whenever a Woodbury solve is factorized.
    Ci = Int[]
    Cj = Int[]
    Cv = T[]
    rhs = zeros(T, use_woodbury ? n : 0, 2)
    Umat = ones(T, use_woodbury ? n : 0, 1)   # the gauge `e`, as the low-rank block
    cgx = zeros(T, use_woodbury ? n : 0)
    cgr = zeros(T, use_woodbury ? n : 0)
    cgz = zeros(T, use_woodbury ? n : 0)
    cgd = zeros(T, use_woodbury ? n : 0)
    cgAd = zeros(T, use_woodbury ? n : 0)
    nsolves = Ref(0)
    nlsqr = Ref(0)
    ncg = Ref(0)
    nchol = Ref(0)
    solve_weighted = function (α, κ)
        nsolves[] += 1
        if use_lsqr
            dκ = κ === nothing ? zero(T) : T(κ) - oneunit(T)
            empty!(vedges)
            fill!(mdiag, zero(T))
            for (e, (ip, jp)) in enumerate(edges)
                c = cvals[e]
                viol = κ !== nothing && (α[ip] + α[jp] - c) < 0
                vpat[e] = viol
                sw = sqrt(viol ? T(κ) : oneunit(T))
                ws[e] = sw
                cv[e] = sw * c
                if viol && use_precond
                    push!(vedges, (ip, jp))
                    if ip == jp
                        mdiag[ip] += 4 * oneunit(T)
                    else
                        mdiag[ip] += oneunit(T)
                        mdiag[jp] += oneunit(T)
                    end
                end
            end
            if use_precond
                # Diagonal scaling alone leaves a conditioning that grows with κ once
                # the violated rows dominate a variable's diagonal; past that point
                # they enter the preconditioner in full, and its Cholesky pays for
                # itself in the iterations it removes.
                κest = oneunit(T)
                for p in 1:n
                    κest = max(κest, oneunit(T) + dκ * 2 * mdiag[p] / dpart[p])
                end
                if κest <= LSQR_PRECOND_KAPPA
                    # `K` is diagonal here, so it is applied by a scaling and nothing
                    # is assembled or factorized.
                    Dmul! = function (y, yv)
                        @. px = yv / psqrt
                        for (e, (ip, jp)) in enumerate(edges)
                            y[e] = ws[e] * (px[ip] + px[jp])
                        end
                        return y
                    end
                    Dtmul! = function (z, y)
                        fill!(pg, zero(T))
                        for (e, (ip, jp)) in enumerate(edges)
                            t = ws[e] * y[e]
                            pg[ip] += t
                            pg[jp] += t
                        end
                        @. z = pg / psqrt
                        return z
                    end
                    soly, it = _lsqr(Dmul!, Dtmul!, cv, psqrt .* α)
                    nlsqr[] += it
                    return soly ./ psqrt
                end
                empty!(Mi)
                empty!(Mj)
                empty!(Mv)
                for p in 1:n
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
                        push!(Mi, p)
                        push!(Mj, p)
                        push!(Mv, dκ)
                        push!(Mi, q)
                        push!(Mj, q)
                        push!(Mv, dκ)
                        push!(Mi, p)
                        push!(Mj, q)
                        push!(Mv, dκ)
                        push!(Mi, q)
                        push!(Mj, p)
                        push!(Mv, dκ)
                    end
                end
                Msp = sparse(Mi, Mj, Mv, n, n)
                MF = cholesky(Symmetric(Msp))
                Kc = MF.PtL
                Uc = MF.UP
                # CHOLMOD exposes no in-place solve for a factor component, so each
                # application returns a fresh vector; the transpose product copies it
                # into the buffer LSQR hands over, which is the only copy avoidable here.
                Pmul! = function (y, yv)
                    xv = Uc \ yv
                    for (e, (ip, jp)) in enumerate(edges)
                        y[e] = ws[e] * (xv[ip] + xv[jp])
                    end
                    return y
                end
                Ptmul! = function (z, y)
                    fill!(pg, zero(T))
                    for (e, (ip, jp)) in enumerate(edges)
                        t = ws[e] * y[e]
                        pg[ip] += t
                        pg[jp] += t
                    end
                    copyto!(z, Kc \ pg)
                    return z
                end
                mul!(px, Msp, α)
                soly, it = _lsqr(Pmul!, Ptmul!, cv, Kc \ px)
                nlsqr[] += it
                return (Uc \ soly)::Vector{T}
            end
            Amul! = function (y, x)
                for (e, (ip, jp)) in enumerate(edges)
                    y[e] = ws[e] * (x[ip] + x[jp])
                end
                return y
            end
            Atmul! = function (z, y)
                fill!(z, zero(T))
                for (e, (ip, jp)) in enumerate(edges)
                    t = ws[e] * y[e]
                    z[ip] += t
                    z[jp] += t
                end
                return z
            end
            sol, it = _lsqr(Amul!, Atmul!, cv, α)
            nlsqr[] += it
            return sol
        elseif use_woodbury
            # `B = C + e·eᵀ` with `C = n·I − L_Z + (κ−1)·L_V`: the complete-support
            # matrix, corrected by the zero set `Z` and by the currently violated
            # entries `V`. One O(nnz) sweep collects the right-hand side, the violated
            # set, and the diagonal of `B`; everything after it is O(|Z| + |V|).
            dκ = κ === nothing ? zero(T) : T(κ) - oneunit(T)
            fill!(f, zero(T))
            copyto!(dg, czero)
            fill!(degV, 0)
            empty!(vedges)
            for (e, (ip, jp)) in enumerate(edges)
                c = cvals[e]
                viol = κ !== nothing && (α[ip] + α[jp] - c) < 0
                vpat[e] = viol
                f[ip] += (viol ? T(κ) : oneunit(T)) * c
                if viol
                    push!(vedges, (ip, jp))
                    degV[ip] += 1
                    dg[ip] += dκ
                    ip == jp && (dg[ip] += dκ)
                end
            end
            # Same ridge as the dense path, so both solve the same regularized system:
            # `e·eᵀ` puts 1 on every diagonal of `B`, and every variable has support
            # here, so no identity row arises.
            dmax = zero(T)
            maxdegV = 0
            for p in 1:n
                dmax = max(dmax, dg[p] + oneunit(T))
                maxdegV = max(maxdegV, degV[p])
            end
            ridge = (dmax > 0 ? dmax : oneunit(T)) * eps(T)
            for p in 1:n
                dg[p] += oneunit(T) + ridge
            end
            # `zedges` and `vedges` carry both orientations of every pair, so these
            # triplets store `C` in full rather than in one triangle: the same matrix
            # then serves the matvec below and the factorization after it. `sparse`
            # sums the duplicates, and the ridge rides on the diagonal.
            empty!(Ci)
            empty!(Cj)
            empty!(Cv)
            for p in 1:n
                push!(Ci, p)
                push!(Cj, p)
                push!(Cv, T(n) + ridge)
            end
            for (p, q) in zedges
                push!(Ci, p)
                push!(Cj, p)
                push!(Cv, -oneunit(T))
                push!(Ci, p)
                push!(Cj, q)
                push!(Cv, -oneunit(T))
            end
            for (p, q) in vedges
                push!(Ci, p)
                push!(Cj, p)
                push!(Cv, dκ)
                push!(Ci, p)
                push!(Cj, q)
                push!(Cv, dκ)
            end
            C = sparse(Ci, Cj, Cv, n, n)
            # Gershgorin on `(κ−1)·L_V` against a diagonal of at least `n` estimates
            # the condition number of `B`. While that estimate is small, conjugate
            # gradients on `C·x + e·(eᵀx)` reach the same answer in a few hundred
            # O(nnz(C)) applications, which is far cheaper than a factorization whose
            # fill, on the near-random violated pattern of the early stages, is close
            # to dense.
            κest = oneunit(T) + dκ * 2 * maxdegV / n
            if κest <= WOODBURY_CG_KAPPA
                copyto!(cgx, α)
                Bmul! = function (y, x)
                    mul!(y, C, x)
                    s = sum(x)
                    y .+= s
                    return y
                end
                it, ok = _pcg!(Bmul!, cgx, dg, f, cgr, cgz, cgd, cgAd,
                               50 + 20 * ceil(Int, sqrt(κest)), 100 * eps(T) * norm(f))
                ncg[] += it
                ok && return copy(cgx)
            end
            nchol[] += 1
            return _woodbury_solve!(zeros(T, n), cholesky(Symmetric(C)), Umat, f, rhs)
        else
            fill!(f, zero(T))
            B = zeros(T, n, n)
            for (e, (ip, jp)) in enumerate(edges)
                c = cvals[e]
                viol = κ !== nothing && (α[ip] + α[jp] - c) < 0
                vpat[e] = viol
                w = viol ? T(κ) : oneunit(T)
                f[ip] += w * c
                B[ip, ip] += w
                B[ip, jp] += w
            end
            # Minimal scale-relative ridge, sized by the largest diagonal, lifts the
            # bipartite gauge null space; support-free variables get an identity row.
            dmax = zero(T)
            for ip in 1:n
                dmax = max(dmax, B[ip, ip])
            end
            ridge = (dmax > 0 ? dmax : oneunit(T)) * eps(T)
            for ip in 1:n
                B[ip, ip] += hassupp[ip] ? ridge : oneunit(T)
            end
            return Symmetric(B) \ f
        end
    end
    α = start === nothing ? solve_weighted(zeros(T, n), nothing) :
        T[hassupp[ip] ? log(T(start[i])) : zero(T) for (ip, i) in enumerate(ax)]
    for κ in κs
        fcur = fκ(α, κ)
        for _ in 1:maxiter
            αnew = solve_weighted(α, κ)
            t = one(T)
            αt = αnew
            fnew, stable = fκpat(αt, κ, vpat)
            while fnew > fcur && t > 500_000 * eps(T)
                t /= 2
                αt = α .+ t .* (αnew .- α)
                fnew = fκ(αt, κ)
                stable = false
            end
            α = αt
            # `f_κ` is convex and the dense and Woodbury steps solve its quadratic model
            # exactly, so a whole step that leaves the violated set unchanged has reached
            # the stage's minimizer: the gradient there is the model's, which is zero.
            # The `:lsqr` solves are inexact and carry no such guarantee.
            !use_lsqr && stable && break
            fcur - fnew <= 5000 * eps(T) * max(fcur, one(T)) && break
            fcur = fnew
        end
    end
    # Uniform boost to exact feasibility: α_i + α_j ≥ log|A_ij| for all support.
    # `boost=false` leaves the iterate untouched, for the soft objective, which
    # imposes no coverage constraint and whose optimum the boost would move off.
    γ = zero(T)
    if boost
        for (e, (ip, jp)) in enumerate(edges)
            γ = max(γ, (cvals[e] - α[ip] - α[jp]) / 2)
        end
    end
    # Dense scale vector matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    a = similar(Array{T}, ax)
    for (ip, i) in enumerate(ax)
        a[i] = hassupp[ip] ? exp(α[ip] + γ) : zero(T)
    end
    return a, (; nsolves=nsolves[], lsqriters=nlsqr[], cgiters=ncg[],
               cholsolves=nchol[],
               linsolve=(use_lsqr ? :lsqr : use_woodbury ? :woodbury : :dense))
end

# Worker for `cover_min(::AbsLog{2})`. Returns `(a, b, stats)` with `stats` a
# NamedTuple `(; nsolves, lsqriters, cgiters, cholsolves, linsolve)` (see
# `_symcover_min_abslog2`).
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
    m = length(axr)
    n = length(axc)
    N = m + n
    use_lsqr = linsolve === :lsqr
    # CHOLMOD, which factors the LSQR preconditioner, is reliable only in Float64;
    # other working types run the plain matrix-free iteration.
    use_precond = use_lsqr && T === Float64
    # Support entries as edges linking a row position ip to a column position m+jp,
    # with `cvals` holding log|A_ij| alongside. Internal positions 1:m index rows,
    # m+1:m+n index columns, and results are scattered back through axr/axc so A's
    # axes are honored.
    G = _row_support(A, T)
    edges = Tuple{Int,Int}[]
    cvals = T[]
    nzrow = zeros(Int, m)   # support entries per row, for the balance convention
    nzcol = zeros(Int, n)   # ditto per column
    for (ip, i) in enumerate(axr)
        for s in _slots(G, i)
            jp = G.idx[s] - first(axc) + 1
            push!(edges, (ip, m + jp))
            push!(cvals, log(G.val[s]))
            nzrow[ip] += 1
            nzcol[jp] += 1
        end
    end
    ne = length(edges)
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
    # Zero set of `A` as stacked-position pairs missing from the support: the
    # off-diagonal pattern of the sparse `C` the Woodbury path factorizes. `czero` is
    # the diagonal of `C` before any entry is violated — the complete-support value
    # `n` on rows and `m` on columns, less one per zero entry at each of its ends.
    zedges = Tuple{Int,Int}[]
    czero = zeros(T, use_woodbury ? N : 0)
    if use_woodbury
        for ip in 1:m
            czero[ip] = T(n)
        end
        for jp in 1:n
            czero[m+jp] = T(m)
        end
        mark = falses(n)
        for (ip, i) in enumerate(axr)
            for s in _slots(G, i)
                mark[G.idx[s] - first(axc) + 1] = true
            end
            for jp in 1:n
                mark[jp] && continue
                q = m + jp
                push!(zedges, (ip, q))
                czero[ip] -= oneunit(T)
                czero[q] -= oneunit(T)
            end
            fill!(mark, false)
        end
    end
    # Gauge vector: ±1 on supported variables, 0 on support-free ones (which carry
    # no constraint and are decoupled with an identity row in `solve_weighted`).
    v0 = zeros(T, N)
    for ip in 1:m
        hasrow[ip] && (v0[ip] = one(T))
    end
    for jp in 1:n
        hascol[jp] && (v0[m+jp] = -one(T))
    end
    fκ = function (x, κ)
        v = zero(T)
        for (e, (p, q)) in enumerate(edges)
            z = x[p] + x[q] - cvals[e]
            v += (z < 0 ? T(κ) : oneunit(T)) * z^2
        end
        return v
    end
    # The objective and the violated set at `x` from one sweep: the line search needs
    # the value and the stage's stopping test needs to know whether the set still
    # matches `pat`, and both read the same residuals.
    fκpat = function (x, κ, pat)
        v = zero(T)
        same = true
        for (e, (p, q)) in enumerate(edges)
            z = x[p] + x[q] - cvals[e]
            viol = z < 0
            v += (viol ? T(κ) : oneunit(T)) * z^2
            same &= viol == pat[e]
        end
        return v, same
    end
    # Each Newton step solves the reweighted least-squares problem for the stacked
    # scales x = (α; β), residuals z_ij = α_i + β_j - log|A_ij|. Row and column scales
    # share the global (e; −e) gauge; every path pins it. The dense path adds the rank-1
    # term v0*v0ᵀ to the normal equations `B x = f` and factorizes (support-free
    # variables get an identity row; a support with more than one connected component
    # carries additional per-component gauges, lifted by the ridge below). The Woodbury
    # path solves the same regularized system exactly: `B + v0·v0ᵀ = C + U·Uᵀ` with `U`
    # the row and column indicators, so a sparse Cholesky of `C` and a rank-two update
    # replace the dense factorization. The LSQR
    # path appends one gauge row `v0ᵀ x = 0` to the least-squares system so `√W R` has
    # full column rank, applies it matrix-free, and warm-starts from the incoming
    # iterate. After the solve a closed-form shift, applied within each component,
    # moves the result to the balance convention, so the pinned gauge is not observable.
    f = zeros(T, N)
    ws = zeros(T, ne)       # √weight per support entry (LSQR path)
    cv = zeros(T, ne + 1)   # √weight · log|A_ij|, with a trailing 0 gauge target
    # Entries the frozen weights of the current solve treat as violated. A full Newton
    # step that leaves this pattern intact has landed on the stage's minimizer.
    vpat = falses(ne)
    vedges = Tuple{Int,Int}[]              # the violated entries of the current solve
    degV = zeros(Int, use_woodbury ? N : 0)  # violated entries per row and per column
    dg = zeros(T, use_woodbury ? N : 0)      # diagonal of `B`, for the ridge and the CG preconditioner
    # Diagonal of the unweighted normal matrix of the gauge-augmented system,
    # `RᵀR + v0·v0ᵀ`: the support degree at each position, plus the gauge row's 1. A
    # support-free variable takes that 1 alone, which keeps the preconditioner
    # positive definite.
    dpart = zeros(T, use_lsqr ? N : 0)
    if use_lsqr
        for (p, q) in edges
            dpart[p] += oneunit(T)
            dpart[q] += oneunit(T)
        end
        for p in 1:N
            dpart[p] += oneunit(T)
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
    rhs = zeros(T, use_woodbury ? N : 0, 3)
    # The row and column indicators, as the low-rank block.
    Umat = zeros(T, use_woodbury ? N : 0, 2)
    if use_woodbury
        for ip in 1:m
            Umat[ip, 1] = oneunit(T)
        end
        for jp in 1:n
            Umat[m+jp, 2] = oneunit(T)
        end
    end
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
        if use_lsqr
            dκ = κ === nothing ? zero(T) : T(κ) - oneunit(T)
            empty!(vedges)
            fill!(mdiag, zero(T))
            for (e, (p, q)) in enumerate(edges)
                c = cvals[e]
                viol = κ !== nothing && (x[p] + x[q] - c) < 0
                vpat[e] = viol
                sw = sqrt(viol ? T(κ) : oneunit(T))
                ws[e] = sw
                cv[e] = sw * c
                if viol && use_precond
                    push!(vedges, (p, q))
                    mdiag[p] += oneunit(T)
                    mdiag[q] += oneunit(T)
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
                    push!(Mi, p)
                    push!(Mj, p)
                    push!(Mv, dκ)
                    push!(Mi, q)
                    push!(Mj, q)
                    push!(Mv, dκ)
                    push!(Mi, p)
                    push!(Mj, q)
                    push!(Mv, dκ)
                    push!(Mi, q)
                    push!(Mj, p)
                    push!(Mv, dκ)
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
        elseif use_woodbury
            # `B + v0·v0ᵀ = C + U·Uᵀ` with `C = D − L_Z + (κ−1)·L_V`,
            # `D = diag(n·1_m, m·1_n)` and `U = [u_r u_c]` the row and column
            # indicators: the complete-support matrix, corrected by the zero set `Z`
            # and by the currently violated entries `V`.
            dκ = κ === nothing ? zero(T) : T(κ) - oneunit(T)
            fill!(f, zero(T))
            copyto!(dg, czero)
            fill!(degV, 0)
            empty!(vedges)
            for (e, (p, q)) in enumerate(edges)
                c = cvals[e]
                viol = κ !== nothing && (x[p] + x[q] - c) < 0
                vpat[e] = viol
                w = viol ? T(κ) : oneunit(T)
                f[p] += w * c
                f[q] += w * c
                if viol
                    push!(vedges, (p, q))
                    degV[p] += 1
                    degV[q] += 1
                    dg[p] += dκ
                    dg[q] += dκ
                end
            end
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
            # Gershgorin on `(κ−1)·L_V` against the smaller diagonal block estimates
            # the condition number of `B`. While that estimate is small the structured
            # matvec plus conjugate gradients reaches the same answer in a few hundred
            # O(N + |Z| + |V|) iterations, which is far cheaper than a factorization
            # whose fill on the near-random violated pattern of the early stages
            # approaches dense.
            # `zedges` and `vedges` carry both ends of every pair, so these triplets
            # store `C` in full rather than in one triangle: the same matrix then
            # serves the matvec below and the factorization after it. `sparse` sums
            # the duplicates, and the ridge rides on the diagonal.
            empty!(Ci)
            empty!(Cj)
            empty!(Cv)
            for p in 1:N
                push!(Ci, p)
                push!(Cj, p)
                push!(Cv, czero[p] + ridge)
            end
            for (p, q) in zedges
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
                push!(Ci, q)
                push!(Cj, q)
                push!(Cv, dκ)
                push!(Ci, p)
                push!(Cj, q)
                push!(Cv, dκ)
                push!(Ci, q)
                push!(Cj, p)
                push!(Cv, dκ)
            end
            C = sparse(Ci, Cj, Cv, N, N)
            # Gershgorin on `(κ−1)·L_V` against the smaller diagonal block estimates
            # the condition number of `B`. While that estimate is small, conjugate
            # gradients on `C·x + u_r·(u_rᵀx) + u_c·(u_cᵀx)` reach the same answer in a
            # few hundred O(nnz(C)) applications, which is far cheaper than a
            # factorization whose fill on the near-random violated pattern of the early
            # stages approaches dense.
            κest = oneunit(T) + dκ * 2 * maxdegV / min(m, n)
            if κest <= WOODBURY_CG_KAPPA
                copyto!(cgx, x)
                Bmul! = function (yy, xx)
                    mul!(yy, C, xx)
                    sr = zero(T)
                    for p in 1:m
                        sr += xx[p]
                    end
                    sc = zero(T)
                    for p in (m+1):N
                        sc += xx[p]
                    end
                    for p in 1:m
                        yy[p] += sr
                    end
                    for p in (m+1):N
                        yy[p] += sc
                    end
                    return yy
                end
                it, ok = _pcg!(Bmul!, cgx, dg, f, cgr, cgz, cgd, cgAd,
                               50 + 20 * ceil(Int, sqrt(κest)), 100 * eps(T) * norm(f))
                ncg[] += it
                ok && return copy(cgx)
            end
            nchol[] += 1
            return _woodbury_solve!(zeros(T, N), cholesky(Symmetric(C)), Umat, f, rhs)
        else
            fill!(f, zero(T))
            B = v0 * v0'
            for (e, (p, q)) in enumerate(edges)
                c = cvals[e]
                viol = κ !== nothing && (x[p] + x[q] - c) < 0
                vpat[e] = viol
                w = viol ? T(κ) : oneunit(T)
                f[p] += w * c
                f[q] += w * c
                B[p, p] += w
                B[q, q] += w
                B[p, q] += w
                B[q, p] += w
            end
            # A support whose bipartite graph splits into k connected components carries k
            # independent (e; −e) gauges; v0*v0ᵀ pins only the global one, leaving k−1
            # singular directions. A minimal scale-relative ridge on the supported
            # diagonals lifts them (the same device the symmetric solver uses for the
            # bipartite null space). The RHS is orthogonal to every gauge null vector, so
            # the ridge leaves the recovered scales essentially unperturbed, and the
            # per-component gauge it fixes is unobservable — no product a_i·b_j spans two
            # components. Support-free variables get an identity row.
            dmax = zero(T)
            for p in 1:N
                dmax = max(dmax, B[p, p])
            end
            ridge = (dmax > 0 ? dmax : oneunit(T)) * eps(T)
            for ip in 1:m
                B[ip, ip] = hasrow[ip] ? B[ip, ip] + ridge : one(T)
            end
            for jp in 1:n
                q = m + jp
                B[q, q] = hascol[jp] ? B[q, q] + ridge : one(T)
            end
            return Symmetric(B) \ f
        end
    end
    x = if start === nothing
        solve_weighted(zeros(T, N), nothing)
    else
        sa, sb = start
        x0 = zeros(T, N)
        for (ip, i) in enumerate(axr)
            hasrow[ip] && (x0[ip] = log(T(sa[i])))
        end
        for (jp, j) in enumerate(axc)
            hascol[jp] && (x0[m+jp] = log(T(sb[j])))
        end
        x0
    end
    for κ in κs
        fcur = fκ(x, κ)
        for _ in 1:maxiter
            xnew = solve_weighted(x, κ)
            t = one(T)
            xt = xnew
            fnew, stable = fκpat(xt, κ, vpat)
            while fnew > fcur && t > 500_000 * eps(T)
                t /= 2
                xt = x .+ t .* (xnew .- x)
                fnew = fκ(xt, κ)
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
    # Uniform boost to exact feasibility: α_i + β_j ≥ log|A_ij| on the support.
    # `boost=false` leaves the iterate untouched, for the soft objective, which
    # imposes no coverage constraint and whose optimum the boost would move off.
    # The balance shift below still applies: the gauge is a convention, not a
    # constraint, and every cover this package returns satisfies it.
    if boost
        γ = zero(T)
        for (e, (p, q)) in enumerate(edges)
            γ = max(γ, (cvals[e] - x[p] - x[q]) / 2)
        end
        for p in 1:N
            x[p] += γ
        end
    end
    # Shift along the (e; -e) gauges to the balance convention ∑ nzaᵢ αᵢ = ∑ nzbⱼ βⱼ,
    # imposed within each connected component of the support: the gauge acts
    # independently on each component, so a single global shift would leave the
    # per-component splits wherever the ridge (or LSQR's gauge row) put them.
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
    return a, b, (; nsolves=nsolves[], lsqriters=nlsqr[], cgiters=ncg[],
                  cholsolves=nchol[],
                  linsolve=(use_lsqr ? :lsqr : use_woodbury ? :woodbury : :dense))
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
