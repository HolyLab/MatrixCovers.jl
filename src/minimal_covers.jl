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
  positive-definite matrix plus `e*eᵀ`, so a sparse Cholesky and a
  Sherman–Morrison update replace the dense factorization. It requires
  `Float64` arithmetic and a support missing at most `n ÷ 4` entries in any
  row, and raises an `ArgumentError` otherwise. `:lsqr` uses matrix-free LSQR
  (per-iteration cost O(nnz), intended for large sparse supports). `:auto`
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
  symmetric positive-definite matrix plus a rank-two term, so a sparse Cholesky
  and a Woodbury update replace the dense factorization. It requires `Float64`
  arithmetic and a support missing at most `min(m, n) ÷ 4` entries in any row
  or column, and raises an `ArgumentError` otherwise. `:lsqr` uses matrix-free
  LSQR (per-iteration cost O(nnz), intended for large sparse supports).
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


# Inner linear solve for the AbsLog{2} MMC Newton steps. `:dense` forms and
# factorizes the reweighted normal equations densely, at O(n³) per step.
# `:woodbury` splits the same matrix as `C + U Uᵀ`, where `C` is sparse (its
# off-diagonal pattern is the zero set of `A` together with the currently violated
# entries) and symmetric positive definite, and `U` has one column (symmetric) or
# two (asymmetric); a sparse Cholesky of `C` plus a Woodbury update then costs far
# less than the dense factorization whenever `A` is close to fully supported.
# `:auto` takes `:woodbury` where it applies and `:dense` otherwise. `:lsqr` forces
# the matrix-free path, whose per-iteration cost is O(nnz); it is the intended
# solve for large sparse supports (where nnz ≪ n²) and is used by the
# structured/sparse methods.
#
# `C` is positive definite because the complete-support matrix contributes `n` (or
# `m`) to each diagonal while the zero set subtracts a signless Laplacian `L_Z`
# with λmax(L_Z) ≤ 2·maxdeg(Z); requiring at most a quarter of a row to be zero
# keeps the difference bounded below by half the diagonal. CHOLMOD is the sparse
# factorization behind it, and it is reliable only in `Float64`, so that is the
# only working type the path accepts.

# True when the residuals of `x` are violated on exactly the entries `pat` marks.
# `edges` and `cvals` are the support list and its `log|A_ij|`, as gathered by the
# workers below.
function _violated_matches(pat, edges, cvals, x)
    for (e, (p, q)) in enumerate(edges)
        ((x[p] + x[q] - cvals[e]) < zero(eltype(cvals))) == pat[e] || return false
    end
    return true
end

# Append one COO triplet of the sparse Woodbury matrix `C`, tracking its diagonal
# in `diagacc` so the ridge can be sized without a second pass over `C`.
function _push_coo!(Ci, Cj, Cv, diagacc, p, q, v)
    push!(Ci, p)
    push!(Cj, q)
    push!(Cv, v)
    p == q && (diagacc[p] += v)
    return nothing
end

# Matrix-free LSQR (Paige & Saunders) for the weighted least-squares problem
# `min ‖M x - b‖` underlying the reweighted normal equations `MᵀM x = Mᵀb`.
# `Amul!(y, x)` overwrites `y` with `M*x`; `Atmul!(z, y)` overwrites `z` with
# `Mᵀ*y`. Warm-started from `x0`. LSQR is used in preference to CG on the normal
# equations because it works with the condition number of `M` (≈ √κ at penalty
# strength κ) rather than that of `MᵀM` (≈ κ); at κ = 1e8 the squared conditioning
# breaks CG while LSQR stays accurate.
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
# NamedTuple `(; nsolves, lsqriters, linsolve)` recording the number of inner linear
# solves, the total LSQR iterations (0 on the dense path), and which path ran.
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
    # The Woodbury path splits the normal equations around the complete-support
    # matrix `n·I + e·eᵀ`, so its cost is set by the zero set `Z` rather than by `n`,
    # and `n·I − L_Z` is positive definite only while `Z` stays thin.
    use_woodbury = false
    if !use_lsqr && linsolve !== :dense
        ok = T === Float64 && maxzero <= n ÷ 4
        if linsolve === :woodbury && !ok
            T === Float64 ||
                throw(ArgumentError("linsolve=:woodbury requires Float64 arithmetic, but `A` works in $T; use :dense or :lsqr"))
            throw(ArgumentError("linsolve=:woodbury requires every row of `A` to have at most n ÷ 4 = $(n ÷ 4) zeros; got $maxzero"))
        end
        use_woodbury = ok
    end
    # Zero set of `A` in the same convention as `edges`: both orientations of an
    # off-diagonal pair, the diagonal once. It is the off-diagonal pattern of the
    # sparse `C` the Woodbury path factorizes.
    zedges = Tuple{Int,Int}[]
    if use_woodbury
        mark = falses(n)
        for (ip, i) in enumerate(ax)
            for s in _slots(G, i)
                mark[G.idx[s] - first(ax) + 1] = true
            end
            for jp in 1:n
                mark[jp] || push!(zedges, (ip, jp))
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
    # COO triplets of `C`, refilled each Woodbury solve; `diagacc` accumulates its
    # diagonal as they are appended.
    Ci = Int[]
    Cj = Int[]
    Cv = T[]
    diagacc = zeros(T, n)
    rhs = zeros(T, n, 2)
    nsolves = Ref(0)
    nlsqr = Ref(0)
    solve_weighted = function (α, κ)
        nsolves[] += 1
        if use_lsqr
            for (e, (ip, jp)) in enumerate(edges)
                c = cvals[e]
                w = κ === nothing ? oneunit(T) : ((α[ip] + α[jp] - c) < 0 ? T(κ) : oneunit(T))
                sw = sqrt(w)
                ws[e] = sw
                cv[e] = sw * c
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
            # entries `V`. `sparse` sums the duplicate triplets.
            fill!(f, zero(T))
            fill!(diagacc, zero(T))
            empty!(Ci)
            empty!(Cj)
            empty!(Cv)
            for p in 1:n
                _push_coo!(Ci, Cj, Cv, diagacc, p, p, T(n))
            end
            for (p, q) in zedges
                _push_coo!(Ci, Cj, Cv, diagacc, p, p, -oneunit(T))
                _push_coo!(Ci, Cj, Cv, diagacc, p, q, -oneunit(T))
            end
            for (e, (ip, jp)) in enumerate(edges)
                c = cvals[e]
                viol = κ !== nothing && (α[ip] + α[jp] - c) < 0
                vpat[e] = viol
                w = viol ? T(κ) : oneunit(T)
                f[ip] += w * c
                if viol
                    _push_coo!(Ci, Cj, Cv, diagacc, ip, ip, w - oneunit(T))
                    _push_coo!(Ci, Cj, Cv, diagacc, ip, jp, w - oneunit(T))
                end
            end
            # Same ridge as the dense path, so both solve the same regularized system:
            # `e·eᵀ` puts 1 on every diagonal of `B`, and every variable has support
            # here, so no identity row arises.
            dmax = zero(T)
            for p in 1:n
                dmax = max(dmax, diagacc[p] + oneunit(T))
            end
            ridge = (dmax > 0 ? dmax : oneunit(T)) * eps(T)
            for p in 1:n
                push!(Ci, p)
                push!(Cj, p)
                push!(Cv, ridge)
            end
            F = cholesky(Symmetric(sparse(Ci, Cj, Cv, n, n)))
            for p in 1:n
                rhs[p, 1] = f[p]
                rhs[p, 2] = oneunit(T)
            end
            # Sherman–Morrison: with y = C\f and u = C\e, (C + e·eᵀ)\f is
            # y − u·(eᵀy)/(1 + eᵀu).
            YU = F \ rhs
            sy = zero(T)
            su = zero(T)
            for p in 1:n
                sy += YU[p, 1]
                su += YU[p, 2]
            end
            r = sy / (oneunit(T) + su)
            return [YU[p, 1] - r * YU[p, 2] for p in 1:n]
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
            fnew = fκ(αnew, κ)
            while fnew > fcur && t > 500_000 * eps(T)
                t /= 2
                fnew = fκ(α .+ t .* (αnew .- α), κ)
            end
            α = α .+ t .* (αnew .- α)
            # `f_κ` is convex and the dense and Woodbury steps solve its quadratic model
            # exactly, so a whole step that leaves the violated set unchanged has reached
            # the stage's minimizer: the gradient there is the model's, which is zero.
            # The `:lsqr` solves are inexact and carry no such guarantee.
            !use_lsqr && isone(t) && _violated_matches(vpat, edges, cvals, α) && break
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
    return a, (; nsolves=nsolves[], lsqriters=nlsqr[],
               linsolve=(use_lsqr ? :lsqr : use_woodbury ? :woodbury : :dense))
end

# Worker for `cover_min(::AbsLog{2})`. Returns `(a, b, stats)` with `stats` a
# NamedTuple `(; nsolves, lsqriters, linsolve)` (see `_symcover_min_abslog2`).
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
    use_woodbury = false
    if !use_lsqr && linsolve !== :dense
        ok = T === Float64 && maxzero <= zbound
        if linsolve === :woodbury && !ok
            T === Float64 ||
                throw(ArgumentError("linsolve=:woodbury requires Float64 arithmetic, but `A` works in $T; use :dense or :lsqr"))
            throw(ArgumentError("linsolve=:woodbury requires every row and column of `A` to have at most min(m, n) ÷ 4 = $zbound zeros; got $maxzero"))
        end
        use_woodbury = ok
    end
    # Zero set of `A` as (row position, column position) pairs missing from the
    # support: the off-diagonal pattern of the sparse `C` the Woodbury path factorizes.
    zedges = Tuple{Int,Int}[]
    if use_woodbury
        mark = falses(n)
        for (ip, i) in enumerate(axr)
            for s in _slots(G, i)
                mark[G.idx[s] - first(axc) + 1] = true
            end
            for jp in 1:n
                mark[jp] || push!(zedges, (ip, jp))
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
    # COO triplets of `C`, refilled each Woodbury solve; `diagacc` accumulates its
    # diagonal as they are appended.
    Ci = Int[]
    Cj = Int[]
    Cv = T[]
    diagacc = zeros(T, N)
    rhs = zeros(T, N, 3)
    nsolves = Ref(0)
    nlsqr = Ref(0)
    solve_weighted = function (x, κ)
        nsolves[] += 1
        if use_lsqr
            for (e, (p, q)) in enumerate(edges)
                c = cvals[e]
                w = κ === nothing ? oneunit(T) : ((x[p] + x[q] - c) < 0 ? T(κ) : oneunit(T))
                sw = sqrt(w)
                ws[e] = sw
                cv[e] = sw * c
            end
            g = ne + 1   # index of the appended gauge row
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
            # and by the currently violated entries `V`. `sparse` sums the duplicate
            # triplets.
            fill!(f, zero(T))
            fill!(diagacc, zero(T))
            empty!(Ci)
            empty!(Cj)
            empty!(Cv)
            for ip in 1:m
                _push_coo!(Ci, Cj, Cv, diagacc, ip, ip, T(n))
            end
            for jp in 1:n
                _push_coo!(Ci, Cj, Cv, diagacc, m + jp, m + jp, T(m))
            end
            for (ip, jp) in zedges
                q = m + jp
                _push_coo!(Ci, Cj, Cv, diagacc, ip, ip, -oneunit(T))
                _push_coo!(Ci, Cj, Cv, diagacc, q, q, -oneunit(T))
                _push_coo!(Ci, Cj, Cv, diagacc, ip, q, -oneunit(T))
                _push_coo!(Ci, Cj, Cv, diagacc, q, ip, -oneunit(T))
            end
            for (e, (p, q)) in enumerate(edges)
                c = cvals[e]
                viol = κ !== nothing && (x[p] + x[q] - c) < 0
                vpat[e] = viol
                w = viol ? T(κ) : oneunit(T)
                f[p] += w * c
                f[q] += w * c
                if viol
                    dw = w - oneunit(T)
                    _push_coo!(Ci, Cj, Cv, diagacc, p, p, dw)
                    _push_coo!(Ci, Cj, Cv, diagacc, q, q, dw)
                    _push_coo!(Ci, Cj, Cv, diagacc, p, q, dw)
                    _push_coo!(Ci, Cj, Cv, diagacc, q, p, dw)
                end
            end
            # Same ridge as the dense path, so both solve the same regularized system:
            # `U·Uᵀ` puts 1 on every diagonal of `B + v0·v0ᵀ`, and every variable has
            # support here, so no identity row arises.
            dmax = zero(T)
            for p in 1:N
                dmax = max(dmax, diagacc[p] + oneunit(T))
            end
            ridge = (dmax > 0 ? dmax : oneunit(T)) * eps(T)
            for p in 1:N
                push!(Ci, p)
                push!(Cj, p)
                push!(Cv, ridge)
            end
            F = cholesky(Symmetric(sparse(Ci, Cj, Cv, N, N)))
            fill!(rhs, zero(T))
            for p in 1:N
                rhs[p, 1] = f[p]
            end
            for ip in 1:m
                rhs[ip, 2] = oneunit(T)
            end
            for jp in 1:n
                rhs[m+jp, 3] = oneunit(T)
            end
            # Woodbury with a 2x2 capacitance: with y = C\f and Y = C\U,
            # (C + U·Uᵀ)\f is y − Y·((I₂ + UᵀY)\(Uᵀy)).
            YU = F \ rhs
            ty = zero(T)
            tz = zero(T)
            k11 = zero(T)
            k12 = zero(T)
            k21 = zero(T)
            k22 = zero(T)
            for ip in 1:m
                ty += YU[ip, 1]
                k11 += YU[ip, 2]
                k12 += YU[ip, 3]
            end
            for jp in 1:n
                q = m + jp
                tz += YU[q, 1]
                k21 += YU[q, 2]
                k22 += YU[q, 3]
            end
            K = [oneunit(T)+k11 k12; k21 oneunit(T)+k22]
            g = K \ T[ty, tz]
            return [YU[p, 1] - g[1] * YU[p, 2] - g[2] * YU[p, 3] for p in 1:N]
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
            fnew = fκ(xnew, κ)
            while fnew > fcur && t > 500_000 * eps(T)
                t /= 2
                fnew = fκ(x .+ t .* (xnew .- x), κ)
            end
            x = x .+ t .* (xnew .- x)
            # `f_κ` is convex and the dense and Woodbury steps solve its quadratic model
            # exactly, so a whole step that leaves the violated set unchanged has reached
            # the stage's minimizer: the gradient there is the model's, which is zero.
            # The `:lsqr` solves are inexact and carry no such guarantee.
            !use_lsqr && isone(t) && _violated_matches(vpat, edges, cvals, x) && break
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
    return a, b, (; nsolves=nsolves[], lsqriters=nlsqr[],
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
