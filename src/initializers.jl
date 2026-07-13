# Named starting covers, shared by every algorithm that needs a starting point:
# the soft-cover multistarts and the `*_min` solvers alike. Each strategy is a
# deterministic, scale-covariant point built from the primitives in
# `heuristic_covers.jl`; nothing here calls a solver, so the dependency runs one
# way and the start menu has a single definition.

# ============================================================
# Public interface
# ============================================================

"""
    a = initialize_symcover(A; strategy=:hardcover, feasible=true, kwargs...)

Build a starting point for the symmetric cover of `A`, as consumed by
[`symcover_min`](@ref) and by the [`soft_symcover`](@ref) multistart.

No penalty is taken: every strategy below is a property of `A` alone, so the
starting point does not depend on the objective it will be refined against.

`strategy` names the point:

- `:hardcover` — the tightened hard cover of [`symcover`](@ref). Forwards
  `maxiter` to the tightening pass. Feasible by construction.
- `:geomean` — the AbsLog{2} unconstrained minimum, the geometric mean of the
  nonzero entries of each row. This is the minimizer of the soft AbsLog{2}
  objective, and is *not* a cover.
- `:leaveout` — the geometric mean recomputed with the most-underweighted
  support entry dropped, which lands in the basin that treats that entry as
  effectively zero. Raises an `ArgumentError` when no entry can be dropped
  (empty support, or dropping it would empty a row). Not a cover.
- `:diagfeasible` — a cover grown from the diagonal by nearest-neighbor
  propagation. Feasible by construction.

`feasible=true` (the default) inflates the result bodily — by the smallest
uniform factor that makes it cover `A` — so that `a[i]*a[j] >= abs(A[i,j])` up
to the roundoff of the log-domain arithmetic. This is a no-op for the strategies
that already cover. `feasible=false` returns the strategy's own point with no
coverage guarantee, which is what the soft covers want: forcing the geometric
mean to feasibility would destroy the very property that makes it the soft
AbsLog{2} optimum.

Under either setting the result is strictly positive on every row that carries
support and exactly zero on every row that carries none.

`:hardcover` inflated and `:geomean` inflated reach the feasibility boundary by
different routes — the first raises only the rows touching violated entries, the
second moves the whole point bodily — and so land in different basins of the
non-convex `AbsLinear` objectives. That is what makes a menu of starts worth
having.

An unrecognized `strategy` raises an `ArgumentError`.

See also: [`initialize_symcover!`](@ref), [`initialize_cover`](@ref), [`symcover`](@ref), [`symcover_min`](@ref).
"""
function initialize_symcover(A::AbstractMatrix; kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("initialize_symcover requires a square matrix"))
    T = float(real(eltype(A)))
    a = similar(Array{T}, ax)
    return initialize_symcover!(a, A; kwargs...)
end

"""
    a = initialize_symcover!(a, A; strategy=:hardcover, feasible=true, kwargs...)

Mutating counterpart of [`initialize_symcover`](@ref): writes the starting cover
into `a` and returns it, rather than allocating a new vector. `eachindex(a)` must
match `axes(A, 1)` (and `A` must be square).

See also: [`initialize_symcover`](@ref).
"""
function initialize_symcover!(a::AbstractVector, A::AbstractMatrix;
                              strategy::Symbol=:hardcover, feasible::Bool=true, kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("initialize_symcover! requires a square matrix"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`, got eachindex(a)=$(eachindex(a)), axes(A, 1)=$ax"))
    if strategy === :hardcover
        symcover!(a, A; kwargs...)
    elseif strategy === :geomean
        _reject_kwargs(strategy, kwargs)
        unconstrained_min!(AbsLog{2}(), a, A)
    elseif strategy === :leaveout
        _reject_kwargs(strategy, kwargs)
        _leaveout_logmean_init!(a, A) ||
            throw(ArgumentError("strategy=:leaveout requires a support entry that can be dropped without emptying a row"))
    elseif strategy === :diagfeasible
        _reject_kwargs(strategy, kwargs)
        init_feasible_diag!(a, A)
    else
        throw(ArgumentError("unknown strategy :$strategy; expected one of :hardcover, :geomean, :leaveout, :diagfeasible"))
    end
    feasible && inflate_feasible!(a, A)
    return a
end

"""
    a, b = initialize_cover(A; strategy=:hardcover, feasible=true, kwargs...)

Build a starting point for the cover of `A`, as consumed by [`cover_min`](@ref)
and by the [`soft_cover`](@ref) multistart. This is the asymmetric analog of
[`initialize_symcover`](@ref), and takes the same `feasible` keyword, under
which the result covers `A` as `a[i]*b[j] >= abs(A[i,j])`.

Two of the strategies carry over: `:hardcover` (the tightened hard cover of
[`cover`](@ref), forwarding `maxiter`) and `:geomean` (the AbsLog{2}
unconstrained minimum). `:leaveout` and `:diagfeasible` have no asymmetric
formulation and raise an `ArgumentError`, as does any unrecognized `strategy`.

Under either `feasible` setting the result is strictly positive on every
supported row and column and exactly zero on the unsupported ones.

See also: [`initialize_cover!`](@ref), [`initialize_symcover`](@ref), [`cover`](@ref), [`cover_min`](@ref).
"""
function initialize_cover(A::AbstractMatrix; kwargs...)
    T = float(real(eltype(A)))
    a = similar(Array{T}, axes(A, 1))
    b = similar(Array{T}, axes(A, 2))
    return initialize_cover!(a, b, A; kwargs...)
end

"""
    a, b = initialize_cover!(a, b, A; strategy=:hardcover, feasible=true, kwargs...)

Mutating counterpart of [`initialize_cover`](@ref): writes the starting cover
into `a` and `b` and returns them, rather than allocating new vectors.
`eachindex(a)` must match `axes(A, 1)` and `eachindex(b)` must match
`axes(A, 2)`.

See also: [`initialize_cover`](@ref).
"""
function initialize_cover!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix;
                           strategy::Symbol=:hardcover, feasible::Bool=true, kwargs...)
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(eachindex(a)), axes(A, 1)=$(axes(A, 1))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(eachindex(b)), axes(A, 2)=$(axes(A, 2))"))
    if strategy === :hardcover
        cover!(a, b, A; kwargs...)
    elseif strategy === :geomean
        _reject_kwargs(strategy, kwargs)
        unconstrained_min!(AbsLog{2}(), a, b, A)
    elseif strategy === :leaveout || strategy === :diagfeasible
        throw(ArgumentError("strategy=:$strategy has no asymmetric formulation; expected one of :hardcover, :geomean"))
    else
        throw(ArgumentError("unknown strategy :$strategy; expected one of :hardcover, :geomean"))
    end
    feasible && inflate_feasible!(a, b, A)
    return a, b
end

# ============================================================
# Internal helpers
# ============================================================

# Strategies with no tunables of their own must reject stray keywords rather than
# discard them: a forwarded `maxiter` that silently does nothing would misreport
# which start was built.
function _reject_kwargs(strategy::Symbol, kwargs)
    isempty(kwargs) && return nothing
    throw(ArgumentError("strategy=:$strategy accepts no further keyword arguments, got $(join(keys(kwargs), ", "))"))
end

# Leave-one-out geometric mean. The geometric mean weights every nonzero entry equally, so
# an entry with |A[i,j]| far below the rest (in the scale-invariant sense of its log-residual
# z[i,j] = log|A[i,j]| - α[i] - α[j] at the unweighted minimum) skews the start into a worse
# basin than the exact-zero limit. Here the entry with the most negative residual is dropped
# from the support and the geometric mean recomputed, giving a start in the basin that treats
# that entry as effectively zero; the AbsLinear objective is finite at r = 0, so refinement
# then varies continuously as the entry vanishes.
#
# Scale-covariance: the residuals z are scale-invariant, so selecting the entry by argmin z
# is covariant, as is the reduced-support geometric mean. Residual ties are broken by
# ascending raw |A[i,j]| — NOT scale-invariant, but exact ties are precisely where
# covariance is unachievable: whenever A is scale-equivalent to a row/column permutation of
# itself (true of EVERY symmetric 2×2 with nonzero off-diagonal, via t² = A[2,2]/A[1,1]),
# the competing basins have exactly equal objectives, so no deterministic algorithm can be
# simultaneously scale-covariant, permutation-equivariant, and continuous there. The raw
# magnitude is the only continuity-relevant information left, and using it only on ties
# confines the covariance exception to that degenerate class. (Weighting all entries by raw
# |A[i,j]|² instead would carry per-entry physical units — incommensurate sums — and break
# covariance on an open set of matrices.)
#
# Returns `true` and fills `a` with the leave-one-out start, or returns `false` (leaving `a`
# unspecified) when no entry can be dropped: empty support, or dropping the selected entry
# would empty some row's support.
function _leaveout_logmean_init!(a::AbstractVector{T}, A::AbstractMatrix) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`_leaveout_logmean_init!(a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(axes(A)), axes(a)=$(axes(a)))"))
    nza = unconstrained_min!(AbsLog{2}(), a, A)
    sum(nza) == 0 && return false
    # Most negative residual over the support, with a roundoff-tolerant tie set: exact ties
    # (e.g. z[1,1] == z[2,2] for every 2×2) must not be ordered by floating-point noise.
    zmin = T(Inf)
    for j in ax, i in first(ax):j
        Aij = abs(A[i, j])
        iszero(Aij) && continue
        zmin = min(zmin, log(T(Aij)) - log(a[i]) - log(a[j]))
    end
    ztol = 64 * eps(T) * max(one(T), abs(zmin))
    ibest = jbest = first(ax) - 1
    Abest = T(Inf)
    for j in ax, i in first(ax):j
        Aij = T(abs(A[i, j]))
        iszero(Aij) && continue
        z = log(Aij) - log(a[i]) - log(a[j])
        if z <= zmin + ztol && Aij < Abest
            ibest, jbest, Abest = i, j, Aij
        end
    end
    # Dropping entry (i,j) removes one support count from row i and (if off-diagonal) row j.
    nza[ibest] > 1 || return false
    ibest == jbest || nza[jbest] > 1 || return false
    # Minimize the unconstrained AbsLog{2} objective over the reduced support by
    # Gauss-Seidel on its normal equations, starting from the full-support solution
    # already in `a`. The closed-form geometric-mean formula used by
    # `unconstrained_min!` is exactly scale-covariant only for rank-1 support
    # patterns, which the reduced support never is; a Gauss-Seidel update, by
    # contrast, is exactly covariant from any covariant iterate, for any sweep
    # count, so basin selection downstream cannot depend on the frame.
    α = similar(a)
    for i in ax
        α[i] = iszero(nza[i]) ? zero(T) : log(a[i])
    end
    for _ in 1:8
        for i in ax
            iszero(nza[i]) && continue
            num = zero(T)   # Σ_j W[i,j] (log|A[i,j]| - α[j]), α[i]-coefficient split out
            den = zero(T)
            for j in ax
                (min(i, j) == ibest && max(i, j) == jbest) && continue
                Aij = abs(A[i, j])
                iszero(Aij) && continue
                lAij = log(Aij)
                if j == i
                    num += lAij
                    den += 2
                else
                    num += lAij - α[j]
                    den += 1
                end
            end
            iszero(den) && continue   # row's only support was the dropped entry (guarded above)
            α[i] = num / den
        end
    end
    for i in ax
        a[i] = iszero(nza[i]) ? zero(T) : exp(α[i])
    end
    return true
end
