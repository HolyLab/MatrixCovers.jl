# Soft covers under `PowerMean{p}`. The objective is strictly convex in the log
# scales, so every solver here returns the unique minimizer (up to the gauge).
#
# All arithmetic is in log space: with `L[i,j] = log|A[i,j]|`, `α = log.(a)` and
# `β = log.(b)`, the exact minimizer over one row scale with the column scales held
# fixed is
#
#     F_i(β) = (logsumexp_j p*(L[i,j] - β[j]) - log(n_i)) / p,
#
# the `p`-power mean of the row's estimates `|A[i,j]|/b[j]`. `logsumexp` keeps
# `R^p` from overflowing for large `p` or wide dynamic range.

# Default sweep limit and stopping tolerance (in units of `eps` of the working type).
# Overrelaxation by `ω` amplifies roundoff in the imbalance by about `1/(2 - ω)`,
# which the tolerance must leave room for.
const POWERMEAN_MAXITER = 10_000
const POWERMEAN_TOL_EPS = 4096

# Overrelaxation: the number of plain sweeps whose imbalance contraction gives the
# first estimate of the rate; the factor by which the log of that contraction may
# grow from one sweep's estimate to the next and still count as steady; the
# largest `ω` a plain estimate may set; the window (in sweeps) over which a relaxed
# rate is measured, which lengthens as `ω → 2` because the rate per sweep is at best
# `ω - 1`, so one factor of `e` takes about `1/(2 - ω)` sweeps, and the oscillating
# imbalance shows no net decrease over shorter windows; the largest factor by
# which one window may shrink `2 - ω`; the number of consecutive windows without
# progress that halve `ω - 1`; the number of halvings a single group's relaxed
# step may try; and the largest relaxation factor used.
const POWERMEAN_NESTIMATE = 4
const POWERMEAN_STEADY = 1.25
const POWERMEAN_OMEGA_START = 1.5
const POWERMEAN_WINDOW = 16
const POWERMEAN_WINDOW_SCALE = 4.0
const POWERMEAN_RAISE = 8
const POWERMEAN_NSTALL = 4
const POWERMEAN_NBACKTRACK = 4
const POWERMEAN_OMEGA_MAX = 1.995

_powermean_window(ω) = POWERMEAN_WINDOW + round(Int, POWERMEAN_WINDOW_SCALE / (2 - ω))

_powermean_tol(::Type{T}, tol) where {T} = tol === nothing ? POWERMEAN_TOL_EPS * eps(T) : T(tol)

# Working type: at least `Float64`, since the imbalance a narrower type can resolve
# leaves the scales accurate only to about `POWERMEAN_TOL_EPS * eps`. Results are
# converted back to the element types of the scale vectors.
_powermean_type(Ts::Type...) = float(promote_type(Ts..., Float64))

# ============================================================
# Public methods
# ============================================================

# The solver keywords pass through to `_powermean_options`, which holds their defaults.
soft_cover(ρ::PowerMean, A::AbstractMatrix; kwargs...) =
    _soft_cover_powermean(ρ, A, :soft_cover; kwargs...)

soft_cover!(ρ::PowerMean, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...) =
    _soft_cover_powermean!(ρ, a, b, A, :soft_cover!; kwargs...)

soft_symcover(ρ::PowerMean, A::AbstractMatrix; kwargs...) =
    _soft_symcover_powermean(ρ, A, :soft_symcover; kwargs...)

soft_symcover!(ρ::PowerMean, a::AbstractVector, A::AbstractMatrix; kwargs...) =
    _soft_symcover_powermean!(ρ, a, A, :soft_symcover!; kwargs...)

# ============================================================
# Drivers
# ============================================================

# Input with symmetric magnitudes goes to the symmetric algorithm: the asymmetric
# minimizer then has `a[i]*b[j] == a[j]*b[i]` on the support, and the balance
# convention selects `b == a`. The test is exact, since a nearly symmetric matrix
# has a genuinely asymmetric minimizer.
# The `:unconstrained` start makes a truncated iteration scale-covariant as well.
function _soft_cover_powermean(ρ::PowerMean, A::AbstractMatrix, fname::Symbol; kwargs...)
    if _abs_symmetric_exact(A)
        a, stats = _soft_symcover_powermean(ρ, A, fname; kwargs...)
        return a, copy(a), stats
    end
    T = float(real(eltype(A)))
    a = similar(Array{T}, axes(A, 1))
    b = similar(Array{T}, axes(A, 2))
    covariant_start!(a, b, flat_support(A, T); fname)
    _balance_cover!(a, b, A)
    return _soft_cover_powermean!(ρ, a, b, A, fname; kwargs...)
end

# `tol = nothing` selects the default tolerance of `_powermean_tol`, which depends on
# the working type.
function _soft_cover_powermean!(::PowerMean{p}, a::AbstractVector, b::AbstractVector,
                                A::AbstractMatrix, fname::Symbol;
                                tol::Union{Real,Nothing}=nothing, kwargs...) where p
    opts = _powermean_options(; kwargs...)
    _prepare_soft_cover_start!(a, b, A, fname)
    T = _powermean_type(eltype(a), eltype(b), real(eltype(A)))
    rtol = _powermean_tol(T, tol)
    R, C = _log_support(_row_support(A, T)), _log_support(_col_support(A, T))
    α = map(x -> x > 0 ? log(T(x)) : zero(T), a)
    β = map(x -> x > 0 ? log(T(x)) : zero(T), b)
    stats = _powermean_solve!(α, β, R, C, T(p), rtol; opts...)
    _warn_powermean_unconverged(fname, stats, :sweeps)
    _warn_powermean_unsettled(fname, stats)
    _check_representable(α, i -> !isempty(_slots(R, i)), β, j -> !isempty(_slots(C, j)), T, fname; gauge=true)
    for i in eachindex(a, α)
        a[i] = isempty(_slots(R, i)) ? zero(eltype(a)) : exp(α[i])
    end
    for j in eachindex(b, β)
        b[j] = isempty(_slots(C, j)) ? zero(eltype(b)) : exp(β[j])
    end
    _balance_cover!(a, b, A)
    return a, b, PowerMeanStats{T}(stats)
end

function _soft_symcover_powermean(ρ::PowerMean, A::AbstractMatrix, fname::Symbol; kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("$fname requires a square matrix"))
    # Symmetry is checked by `_soft_symcover_powermean!`; the start is only read there.
    a = similar(Array{float(real(eltype(A)))}, ax)
    _initialize_symcover!(a, A, :diagonal, :none)
    return _soft_symcover_powermean!(ρ, a, A, fname; kwargs...)
end

function _soft_symcover_powermean!(::PowerMean{p}, a::AbstractVector, A::AbstractMatrix, fname::Symbol;
                                   tol::Union{Real,Nothing}=nothing, kwargs...) where p
    opts = _powermean_options(; kwargs...)
    _prepare_soft_symcover_start!(a, A, fname)
    T = _powermean_type(eltype(a), real(eltype(A)))
    rtol = _powermean_tol(T, tol)
    S = _log_support(_sym_support(A, T))
    α = map(x -> x > 0 ? log(T(x)) : zero(T), a)
    stats = _powermean_symsolve!(α, S, T(p), rtol; opts...)
    _warn_powermean_unconverged(fname, stats, :updates)
    _warn_powermean_unsettled(fname, stats)
    _balance_bipartite_sym!(α, S)
    _check_representable(α, i -> !isempty(_slots(S, i)), nothing, nothing, T, fname; gauge=false)
    for i in eachindex(a, α)
        a[i] = isempty(_slots(S, i)) ? zero(eltype(a)) : exp(α[i])
    end
    return a, PowerMeanStats{T}(stats)
end

# Whether `axes(A, 1) == axes(A, 2)` and `abs(A[i,j]) == abs(A[j,i])` for all `i, j`.
_abs_symmetric_exact(::Union{Symmetric,Hermitian,Diagonal,SymTridiagonal}) = true

function _abs_symmetric_exact(A::AbstractMatrix)
    axes(A, 1) == axes(A, 2) || return false
    sym = Ref(true)
    foreach_support(A) do i, j, v
        sym[] &= abs(A[j, i]) == v
    end
    return sym[]
end

function _abs_symmetric_exact(A::StridedMatrix)
    ax = axes(A, 1)
    axes(A, 2) == ax || return false
    for j in ax, i in first(ax):j-1
        abs(A[i, j]) == abs(A[j, i]) || return false
    end
    return true
end

function _warn_powermean_unconverged(fname::Symbol, stats, unit::Symbol)
    stats.converged && return nothing
    (; nsweeps, maxiter, imbalance, tol) = stats
    if stats.newton
        @warn "$fname: the power-mean iteration ended after $nsweeps $unit and $(stats.nnewton) of maxnewton=$(stats.maxnewton) Newton steps with imbalance $imbalance (tolerance $tol); the result may not minimize the objective. Increase `maxnewton` if the steps ran out, or `tol` if the imbalance is at the roundoff level of the entries' logarithms."
    else
        @warn "$fname: the power-mean iteration ended after $nsweeps of maxiter=$maxiter $unit with imbalance $imbalance (tolerance $tol); the result may not minimize the objective. Increase `maxiter`, or `tol` if the imbalance is at the roundoff level of the entries' logarithms."
    end
    return nothing
end

function _warn_powermean_unsettled(fname::Symbol, stats)
    (; converged, npolish, maxpolish, shift, settle) = stats
    (converged && shift > settle) || return nothing
    @warn "$fname: the cluster polish ended after $npolish of maxpolish=$maxpolish passes with largest shift $shift (settle threshold $settle); the log products across the weakest joins of the support are determined only to about that accuracy. Increase `maxpolish`, or set it to 0 to skip the polish."
    return nothing
end

# ============================================================
# Solvers: sweeps, then Newton when the sweeps are slow
# ============================================================

# The sweeps hand over to Newton when their observed rate predicts more than
# `POWERMEAN_NEWTON_PASSES` further passes over the support to reach the
# tolerance, or after `POWERMEAN_NEWTON_MAXPASSES` passes. An asymmetric sweep is
# two passes (rows, then columns); a symmetric update is one.
const POWERMEAN_NEWTON_PASSES = 200
const POWERMEAN_NEWTON_MAXPASSES = 400
const POWERMEAN_MAXNEWTON = 100
const POWERMEAN_MAXPOLISH = 100
# The polish stops when a pass shifts no cluster by more than this multiple of
# `tol/p`. On well-connected supports the Newton steps leave the log scales
# accurate only to a few `tol/p`, so smaller shifts are below the accuracy of the
# rest of the cover.
const POWERMEAN_SETTLE_FACTOR = 10

# The solver keywords of the `PowerMean` soft covers and `noise_cover`, with their
# defaults. Callers pass user keywords through this function before solving, so
# that an unsupported keyword throws a `MethodError`.
_powermean_options(; maxiter::Integer=POWERMEAN_MAXITER, newton::Bool=true,
                   maxnewton::Integer=POWERMEAN_MAXNEWTON, maxpolish::Integer=POWERMEAN_MAXPOLISH,
                   linsolve::Symbol=:auto) = (; maxiter, newton, maxnewton, maxpolish, linsolve)

# Terms of the asymmetric balance conditions beyond those of the soft covers,
# which `noise_cover` uses: a target for each row and column sum in place of its
# number of stored entries, and, for `p = 1` only, an additive rank-one term
# `κ * u * v'` that enters every row and column sum without being stored. Rows and
# columns are stacked by position, rows first, as in the Newton unknowns. The
# rank-one weights must vanish on rows and columns without stored entries, which
# the solvers skip.
struct PowerMeanTerms{T}
    lt::Vector{T}   # log target of each row and column
    lw::Vector{T}   # log rank-one weights, `log.(u)` then `log.(v)`; empty without the term
    lκ::T           # log κ
end

# One half-sweep's view of `PowerMeanTerms`: the log targets `lt` and log rank-one
# weights `lw` of the groups being updated and `lwp` of their partners, by position.
struct PowerMeanSide{T,V<:AbstractVector{T}}
    lt::V
    lw::V
    lwp::V
    lκ::T
end

function _powermean_sides(terms::PowerMeanTerms, m::Int)
    N = length(terms.lt)
    lw = terms.lw
    rw, cw = isempty(lw) ? (view(lw, 1:0), view(lw, 1:0)) : (view(lw, 1:m), view(lw, m+1:N))
    return PowerMeanSide(view(terms.lt, 1:m), rw, cw, terms.lκ),
           PowerMeanSide(view(terms.lt, m+1:N), cw, rw, terms.lκ)
end
_powermean_sides(::Nothing, ::Int) = nothing, nothing

# Asymmetric solve on row- and column-grouped `log|A|`, with the additional
# `terms` if given. Returns the statistics the drivers report: convergence, the
# sweep and Newton-step counts, the passes over the support made by Newton and
# the polish, the factorizations and linear solver, the final imbalance, and the
# polish passes and last cluster shift.
function _powermean_solve!(α, β, R::GroupedSupport{T}, C::GroupedSupport{T}, p::T, tol::T;
                           maxiter::Integer, newton::Bool, maxnewton::Integer, maxpolish::Integer,
                           linsolve::Symbol, terms::Union{Nothing,PowerMeanTerms{T}}=nothing) where {T}
    _check_powermean_linsolve(linsolve)
    nsw = newton ? min(maxiter, POWERMEAN_NEWTON_MAXPASSES ÷ 2) : maxiter
    budget = newton ? POWERMEAN_NEWTON_PASSES ÷ 2 : typemax(Int)
    converged, nsweeps, imbalance = _powermean_sinkhorn!(α, β, R, C, p, nsw, tol, budget, terms)
    nt = (; converged, nsweeps, maxiter, newton=false, nnewton=0, maxnewton, maxpolish, npasses=0,
          nfactor=0, linsolve=:none, imbalance, tol, npolish=0, shift=zero(T), settle=POWERMEAN_SETTLE_FACTOR * tol / p)
    (converged || !newton) && return nt
    sys = _powermean_system(R, eachindex(β), terms)
    x = _stack(α, β)
    converged, nnewton, npasses, nfactor, imbalance, ls, npolish, shift, settle =
        _powermean_refine!(x, sys, length(α), p, maxnewton, maxpolish, tol, linsolve)
    _unstack!(α, β, x)
    return (; nt..., converged, newton=true, nnewton, npasses, nfactor, linsolve=ls, imbalance, npolish, shift, settle)
end

# Symmetric solve on the symmetric row-grouped `log|A|`; statistics as above,
# with `nsweeps` counting updates.
function _powermean_symsolve!(α, S::GroupedSupport{T}, p::T, tol::T;
                              maxiter::Integer, newton::Bool, maxnewton::Integer, maxpolish::Integer,
                              linsolve::Symbol) where {T}
    _check_powermean_linsolve(linsolve)
    nup = newton ? min(maxiter, POWERMEAN_NEWTON_MAXPASSES) : maxiter
    budget = newton ? POWERMEAN_NEWTON_PASSES : typemax(Int)
    converged, nsweeps, imbalance = _powermean_jacobi!(α, similar(α), S, p, nup, tol, budget)
    nt = (; converged, nsweeps, maxiter, newton=false, nnewton=0, maxnewton, maxpolish, npasses=0,
          nfactor=0, linsolve=:none, imbalance, tol, npolish=0, shift=zero(T), settle=POWERMEAN_SETTLE_FACTOR * tol / p)
    (converged || !newton) && return nt
    sys = _powermean_symsystem(S)
    n = length(S.ax)
    off = first(S.ax) - 1
    x = Vector{T}(undef, n)
    for g in S.ax
        x[g-off] = α[g]
    end
    converged, nnewton, npasses, nfactor, imbalance, ls, npolish, shift, settle =
        _powermean_refine!(x, sys, nothing, p, maxnewton, maxpolish, tol, linsolve)
    for g in S.ax
        α[g] = x[g-off]
    end
    return (; nt..., converged, newton=true, nnewton, npasses, nfactor, linsolve=ls, imbalance, npolish, shift, settle)
end

_check_powermean_linsolve(linsolve::Symbol) =
    linsolve in (:auto, :dense, :cholesky, :cg) ||
        throw(ArgumentError("linsolve must be :auto, :dense, :cholesky, or :cg; got :$linsolve"))

# Row log scales in positions `1:m`, column log scales in `m+1:m+n`.
function _stack(α, β)
    x = Vector{promote_type(eltype(α), eltype(β))}(undef, length(α) + length(β))
    k = 0
    for i in eachindex(α)
        x[k+=1] = α[i]
    end
    for j in eachindex(β)
        x[k+=1] = β[j]
    end
    return x
end

function _unstack!(α, β, x)
    k = 0
    for i in eachindex(α)
        α[i] = x[k+=1]
    end
    for j in eachindex(β)
        β[j] = x[k+=1]
    end
    return α, β
end

# ============================================================
# Kernels
# ============================================================

# Replace the magnitudes of a grouped support by their logarithms.
function _log_support(S::GroupedSupport)
    map!(log, S.val, S.val)
    return S
end

# Exact minimizer of the objective over the scale of group `g` (slots `sl` of `S`,
# which holds `log|A|`), given the log scales `y` of its partners.
function _powermean_target(S::GroupedSupport{T}, sl, y, p::T) where {T}
    m = typemin(T)
    for s in sl
        m = max(m, S.val[s] - y[S.idx[s]])
    end
    t = zero(T)
    for s in sl
        t += exp(p * (S.val[s] - y[S.idx[s]] - m))
    end
    return m + (log(t) - log(T(length(sl)))) / p
end

# The same with the log target `lt` in place of `log(length(sl))` and `le`, the log
# of a term added to the group's sum `∑ exp(p (log|A| - y))` (`-Inf` for none).
function _powermean_target(S::GroupedSupport{T}, sl, y, p::T, lt::T, le::T) where {T}
    m = typemin(T)
    for s in sl
        m = max(m, S.val[s] - y[S.idx[s]])
    end
    pm = max(p * m, le)
    t = exp(le - pm)
    for s in sl
        t += exp(p * (S.val[s] - y[S.idx[s]]) - pm)
    end
    return (pm + log(t) - lt) / p
end

# Target of group `g` (position `gp`), with `lr` the log of the rank-one term's
# sum over the partners (see `_powermean_rankone_lr`).
_powermean_group_target(S, sl, y, p, ::Nothing, gp, lr) = _powermean_target(S, sl, y, p)
function _powermean_group_target(S, sl, y, p, side::PowerMeanSide, gp, lr)
    le = isempty(side.lw) ? oftype(lr, -Inf) : lr + side.lw[gp]
    return _powermean_target(S, sl, y, p, side.lt[gp], le)
end

# `log(κ ∑_k v[k] exp(-p y[k]))` over the partners `k` with weights `v`, which is
# the rank-one term of a group with weight `u` divided by `u`; `-Inf` without it.
_powermean_rankone_lr(::Nothing, y, p) = zero(p)
function _powermean_rankone_lr(side::PowerMeanSide, y, p)
    T = typeof(p)
    isempty(side.lwp) && return T(-Inf)
    m = T(-Inf)
    for (k, j) in zip(eachindex(side.lwp), eachindex(y))
        m = max(m, side.lwp[k] - p * y[j])
    end
    t = zero(T)
    for (k, j) in zip(eachindex(side.lwp), eachindex(y))
        t += exp(side.lwp[k] - p * y[j] - m)
    end
    return side.lκ + m + log(t)
end

# The objective restricted to one group scale `x`, with its partners fixed, is
# `n * ρ(x - F) + const`, where `F` is the group's target, `n` the group's target
# sum, and `ρ(z) = (exp(-p z) - 1 + p z) / p`. This gives the exact change of the
# objective under a relaxed step.
_powermean_ρ(p, z) = (expm1(-p * z) + p * z) / p

# Largest `|∑ R^p / n - 1|` over the groups of `S`, the scale-invariant imbalance
# of the balance conditions `∑_j R[i,j]^p = n_i` (with `n_i` and the sums those of
# `side` when given).
function _powermean_imbalance(x, y, S::GroupedSupport{T}, p::T, side=nothing) where {T}
    imb = zero(T)
    lr = _powermean_rankone_lr(side, y, p)
    off = first(S.ax) - 1
    for g in S.ax
        sl = _slots(S, g)
        isempty(sl) && continue
        F = _powermean_group_target(S, sl, y, p, side, g - off, lr)
        imb = max(imb, abs(expm1(p * (F - x[g]))))
    end
    return imb
end

# Update the scales `x` of every group of `S` against the partner scales `y`,
# overrelaxed by `ω`. Returns the imbalance of the groups before the update.
#
# A group's step relaxed by `w` is taken only if it reduces that group's objective
# to at most `c(w) = (1 + (w - 1)^2)/2 < 1` times its current excess over the block
# minimum. Failing that, `w - 1` is halved a few times, and then the group takes
# the exact block minimizer. Every half-sweep therefore decreases the objective by
# at least a fixed fraction of what the exact block update would, which keeps the
# iteration globally convergent for any `ω < 2`.
function _powermean_halfsweep!(x, y, S::GroupedSupport{T}, p::T, ω::T, side=nothing) where {T}
    imb = zero(T)
    lr = _powermean_rankone_lr(side, y, p)
    off = first(S.ax) - 1
    for g in S.ax
        sl = _slots(S, g)
        isempty(sl) && continue
        F = _powermean_group_target(S, sl, y, p, side, g - off, lr)
        δ = x[g] - F
        imb = max(imb, abs(expm1(-p * δ)))
        xg = F
        if ω != 1
            ρδ = _powermean_ρ(p, δ)
            w = ω
            for _ in 0:POWERMEAN_NBACKTRACK
                if _powermean_ρ(p, (1 - w) * δ) <= (1 + (w - 1)^2) / 2 * ρδ
                    xg = F + (1 - w) * δ
                    break
                end
                w = 1 + (w - 1) / 2
            end
        end
        x[g] = xg
    end
    return imb
end

# Alternating (Sinkhorn) updates of the row log scales `α` and column log scales
# `β`, with row- and column-grouped `log|A|` in `R` and `C`. Returns
# `(converged, nsweeps, imbalance)`.
#
# The linearized alternation is block Gauss-Seidel on a 2-cyclic system, whose
# plain rate per sweep is `σ²` (the second singular value squared of the
# normalized weight matrix) and whose optimal overrelaxation is
# `ω* = 2/(1 + sqrt(1 - σ²))`, with rate `ω* - 1`. Plain sweeps give a first
# estimate of `σ²` from the contraction of the imbalance.
#
# Above `ω*` the relaxed rate is `ω - 1` whatever `σ²` is, so the windows below
# read an `ω` that is too large as steady progress and never lower it. The plain
# estimate is the one that can be too large: far from the fixed point the
# imbalance can stay nearly constant for several sweeps before it contracts at a
# much faster rate. Two conditions on the estimate guard against that. It waits
# while the contraction is still improving from one sweep to the next, and it
# sets `ω` no higher than `POWERMEAN_OMEGA_START`, where a relaxed sweep contracts
# by at least half even above `ω*`. Only a window's measured rate takes `ω` higher.
#
# Each relaxed window re-estimates `σ²` from its observed rate `λ` through the SOR
# eigenvalue relation `sqrt(λ) = (ωμ + sqrt(ω²μ² - 4(ω - 1)))/2`, which raises `ω`
# when it is below `ω*`. The imbalance of a relaxed iteration is not monotone, so
# a window is represented by its smallest imbalance, and its rate is measured
# against the window before it. The imbalance is the size of the plain step, which
# a relaxed iterate overshoots by `ω - 1` times, so its level rises with `ω`:
# against the smallest imbalance of all earlier windows, windows at a raised `ω`
# could show no progress while the iteration converges. The first window after
# `ω` changes is still measured against one at the old `ω`, whose lower level
# makes it read as slow, so one window shrinks `2 - ω` by at most a factor of
# `POWERMEAN_RAISE`. After several consecutive windows without progress `ω - 1` is
# halved, down to plain sweeps and a fresh estimate.
#
# With `budget`, the iteration also returns early, unconverged, once the rate
# estimates predict that more than `budget` further sweeps are needed: from the
# plain estimate, at the rate of the relaxed iteration with the factor that
# estimate implies (which the windows raise `ω` toward), and at the end of each
# relaxed window, from that window's smallest imbalance at the observed rate.
#
# `terms`, if given, adds the targets and rank-one term of `PowerMeanTerms`.
_powermean_sinkhorn!(α, β, R::GroupedSupport{T}, C::GroupedSupport{T}, p::T,
                     maxiter::Integer, tol::T) where {T} =
    _powermean_sinkhorn!(α, β, R, C, p, maxiter, tol, typemax(Int))

function _powermean_sinkhorn!(α, β, R::GroupedSupport{T}, C::GroupedSupport{T}, p::T,
                              maxiter::Integer, tol::T, budget::Integer, terms=nothing) where {T}
    rside, cside = _powermean_sides(terms, length(α))
    ωmax = T(POWERMEAN_OMEGA_MAX)
    ω = one(T)
    nplain = 0                  # consecutive plain sweeps in the current estimate
    r1 = r2 = r3 = T(Inf)       # imbalance one, two and three plain sweeps back
    rprev = T(Inf)              # smallest imbalance of the previous relaxed window
    rwin = T(Inf)               # smallest imbalance of the current window
    nwin = 0                    # sweeps in the current window
    nstall = 0                  # consecutive windows without progress
    k = 0
    while k < maxiter
        k += 1
        r = max(_powermean_halfsweep!(α, β, R, p, ω, rside), _powermean_halfsweep!(β, α, C, p, ω, cside))
        if r <= tol
            imb = max(_powermean_imbalance(α, β, R, p, rside), _powermean_imbalance(β, α, C, p, cside))
            imb <= tol && return true, k, imb
        end
        if ω == 1
            nplain += 1
            if nplain >= POWERMEAN_NESTIMATE
                μ2, μ2prev = sqrt(r / r2), sqrt(r1 / r3)
                if zero(T) < μ2 < one(T) && zero(T) < μ2prev < one(T) &&
                        log(μ2) >= POWERMEAN_STEADY * log(μ2prev)
                    ωopt = min(ωmax, 2 / (1 + sqrt(1 - μ2)))
                    ω = min(ωopt, T(POWERMEAN_OMEGA_START))
                    rprev, rwin, nwin, nstall = r, T(Inf), 0, 0
                    _powermean_nsweeps(r, tol, _sor_rate(μ2, ωopt)) > budget && break
                end
            end
            r3, r2, r1 = r2, r1, r
        else
            nwin += 1
            rwin = min(rwin, r)
            wlen = _powermean_window(ω)
            nwin < wlen && continue
            λ = (rwin / rprev)^(one(T) / wlen)
            if !(λ < 1)
                nstall += 1
                if nstall >= POWERMEAN_NSTALL
                    nstall = 0
                    ω = 1 + (ω - 1) / 2
                    if ω - 1 < (ωmax - 1) / 16
                        ω, nplain, r1, r2, r3 = one(T), 0, T(Inf), T(Inf), T(Inf)
                    end
                end
            else
                nstall = 0
                if λ > ω - 1
                    μ2 = (λ + ω - 1)^2 / (ω^2 * λ)
                    μ2 < 1 && (ω = max(ω, min(ωmax, 2 - (2 - ω) / POWERMEAN_RAISE, 2 / (1 + sqrt(1 - μ2)))))
                end
            end
            rprev = rwin
            rwin, nwin = T(Inf), 0
            λ < 1 && _powermean_nsweeps(rprev, tol, λ) > budget && break
        end
    end
    imb = max(_powermean_imbalance(α, β, R, p, rside), _powermean_imbalance(β, α, C, p, cside))
    return imb <= tol, k, imb
end

# Sweeps needed to reduce the imbalance from `r` to `tol` at rate `λ` per sweep.
_powermean_nsweeps(r, tol, λ) = λ < 1 ? log(r / tol) / -log(λ) : oftype(λ, Inf)

# Asymptotic rate per sweep of the overrelaxed 2-cyclic iteration with factor `ω`
# when the plain rate is `μ2`: `ω - 1` at or above the optimal factor, and
# otherwise the square of the larger root of the SOR eigenvalue relation.
function _sor_rate(μ2, ω)
    ωμ = ω * sqrt(μ2)
    disc = ωμ^2 - 4 * (ω - 1)
    disc <= 0 && return ω - 1
    return ((ωμ + sqrt(disc)) / 2)^2
end

# Damped simultaneous updates `α ← (α + F(α))/2` of the symmetric log scales, with
# `S` the symmetric row-grouped `log|A|` (full-grid rows, diagonal included) and
# `F` scratch. With `g(α, β)` the asymmetric objective, `F(α)` minimizes
# `g(α, ⋅)`; joint convexity and `g(α, β) = g(β, α)` give
# `f((α + F)/2) ≤ (g(α, F) + g(F, α))/2 = g(α, F) ≤ f(α)`, so every update
# decreases the symmetric objective `f(α) = g(α, α)`. The argument requires the
# weight `1/2` exactly.
#
# `maxiter` counts updates. Returns `(converged, nupdates, imbalance)`, the
# imbalance being that of the returned `α`.
#
# With `budget`, the iteration also returns early, unconverged, once the rate
# observed over the last `POWERMEAN_RATE_WINDOW` updates predicts that more than
# `budget` further updates are needed.
_powermean_jacobi!(α, F, S::GroupedSupport{T}, p::T, maxiter::Integer, tol::T) where {T} =
    _powermean_jacobi!(α, F, S, p, maxiter, tol, typemax(Int))

const POWERMEAN_RATE_WINDOW = 8

function _powermean_jacobi!(α, F, S::GroupedSupport{T}, p::T, maxiter::Integer, tol::T,
                            budget::Integer) where {T}
    hist = fill(T(Inf), POWERMEAN_RATE_WINDOW)   # imbalances of the last updates, cyclically
    k = 0
    while true
        imb = zero(T)
        for g in S.ax
            sl = _slots(S, g)
            isempty(sl) && continue
            F[g] = _powermean_target(S, sl, α, p)
            imb = max(imb, abs(expm1(p * (F[g] - α[g]))))
        end
        imb <= tol && return true, k, imb
        k == maxiter && return false, k, imb
        h = mod1(k + 1, POWERMEAN_RATE_WINDOW)
        λ = (imb / hist[h])^(one(T) / POWERMEAN_RATE_WINDOW)
        hist[h] = imb
        λ < 1 && _powermean_nsweeps(imb, tol, λ) > budget && return false, k, imb
        for g in S.ax
            isempty(_slots(S, g)) || (α[g] = (α[g] + F[g]) / 2)
        end
        k += 1
    end
end

# A connected component of the symmetric support that is bipartite with no diagonal
# entry admits `α → α + c` on one side and `α - c` on the other without changing
# any product on the support. Fix `c` by the balance convention of the asymmetric
# covers, `∑ n_i α_i` equal on the two sides, with `n_i` the support count of row `i`.
function _balance_bipartite_sym!(α, S::GroupedSupport{T}) where {T}
    ax = S.ax
    off = first(ax) - 1
    color = zeros(Int8, length(ax))     # 0 unvisited; ±1 the side of the bipartition
    stack = eltype(ax)[]
    comp = eltype(ax)[]
    for r in ax
        (color[r-off] != 0 || isempty(_slots(S, r))) && continue
        empty!(comp)
        color[r-off] = 1
        push!(stack, r)
        bipartite = true
        while !isempty(stack)
            g = pop!(stack)
            push!(comp, g)
            for s in _slots(S, g)
                h = S.idx[s]
                if color[h-off] == 0
                    color[h-off] = -color[g-off]
                    push!(stack, h)
                elseif color[h-off] == color[g-off]
                    bipartite = false       # an odd cycle or a diagonal entry
                end
            end
        end
        bipartite || continue
        num = zero(T)
        den = 0
        for g in comp
            n = _ngroup(S, g)
            num -= color[g-off] * n * α[g]
            den += n
        end
        c = num / den
        for g in comp
            α[g] += color[g-off] * c
        end
    end
    return α
end
