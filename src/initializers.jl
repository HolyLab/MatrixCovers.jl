# Starting covers shared by the soft-cover multistarts and `*_min` solvers.

# Default starts for nonconvex `AbsLinear` solvers.
const SYMCOVER_MIN_STRATEGIES = (:hardcover, :geomean, :leaveout)
const COVER_MIN_STRATEGIES = (:hardcover, :geomean)

# ============================================================
# Public interface
# ============================================================

"""
    a = initialize_symcover(A; strategy=:hardcover, feasible=:inflate, kwargs...)

Build a symmetric starting point for [`symcover_min`](@ref) or
[`soft_symcover`](@ref). Strategies depend only on `A`:

`strategy` names the point:

- `:geomean` — the unconstrained `AbsLog{2}` start of [`symcover`](@ref):
  geometric means of the diagonally normalized entries in each row.
- `:leaveout` — the geometric mean recomputed with the most-underweighted
  support entry omitted. It fails if removing that entry empties a row.
- `:diagfeasible` — a cover grown from the diagonal by nearest-neighbor
  propagation.
- `:hardcover` — the result of [`symcover`](@ref). It forwards `maxiter` and
  ignores `feasible`.

`feasible` controls whether and how the point is made into a cover:

- `:inflate` multiplies every scale by the smallest common factor that covers `A`.
- `:boost` raises scales that touch violated entries.
- `:none` returns the strategy's point without a coverage guarantee.

Supported rows receive positive scales; unsupported rows receive zero.

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
    a = initialize_symcover!(a, A; strategy=:hardcover, feasible=:inflate, kwargs...)

Mutating counterpart of [`initialize_symcover`](@ref): writes the starting cover
into `a` and returns it, rather than allocating a new vector. `eachindex(a)` must
match `axes(A, 1)` (and `A` must be square).

See also: [`initialize_symcover`](@ref).
"""
function initialize_symcover!(a::AbstractVector, A::AbstractMatrix;
                              strategy::Symbol=:hardcover, feasible::Symbol=:inflate, kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("initialize_symcover! requires a square matrix"))
    require_abs_symmetric(A, :initialize_symcover!)
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(ax))"))
    _initialize_symcover!(a, A, strategy, feasible; kwargs...) ||
        throw(ArgumentError("strategy=:leaveout requires a support entry that can be dropped without emptying a row"))
    return a
end

"""
    a, b = initialize_cover(A; strategy=:hardcover, feasible=:inflate, kwargs...)

Build an asymmetric starting point for [`cover_min`](@ref) or
[`soft_cover`](@ref). The `feasible` keyword matches
[`initialize_symcover`](@ref).

Supported strategies are `:hardcover` (the result of [`cover`](@ref), forwarding
`maxiter`) and `:geomean` (the unconstrained `AbsLog{2}` minimum).

Supported rows and columns receive positive scales; unsupported ones receive
zero. The factors use the balance convention of [`cover_min`](@ref).

See also: [`initialize_cover!`](@ref), [`initialize_symcover`](@ref), [`cover`](@ref), [`cover_min`](@ref).
"""
function initialize_cover(A::AbstractMatrix; kwargs...)
    T = float(real(eltype(A)))
    a = similar(Array{T}, axes(A, 1))
    b = similar(Array{T}, axes(A, 2))
    return initialize_cover!(a, b, A; kwargs...)
end

"""
    a, b = initialize_cover!(a, b, A; strategy=:hardcover, feasible=:inflate, kwargs...)

Mutating counterpart of [`initialize_cover`](@ref): writes the starting cover
into `a` and `b` and returns them, rather than allocating new vectors.
`eachindex(a)` must match `axes(A, 1)` and `eachindex(b)` must match
`axes(A, 2)`.

See also: [`initialize_cover`](@ref).
"""
function initialize_cover!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix;
                           strategy::Symbol=:hardcover, feasible::Symbol=:inflate, kwargs...)
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(string(eachindex(b))), axes(A, 2)=$(string(axes(A, 2)))"))
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
    _make_feasible!(feasible, a, b, A)
    # Restore the package's balance convention after a selective boost.
    return _balance_cover!(a, b, A)
end

# ============================================================
# Internal helpers
# ============================================================

# Build a named start. `:leaveout` returns `false` when no entry can be removed.
function _initialize_symcover!(a::AbstractVector, A::AbstractMatrix, strategy::Symbol,
                               feasible::Symbol; kwargs...)
    if strategy === :hardcover
        symcover!(a, A; kwargs...)
    elseif strategy === :geomean
        _reject_kwargs(strategy, kwargs)
        unconstrained_min!(AbsLog{2}(), a, A)
    elseif strategy === :leaveout
        _reject_kwargs(strategy, kwargs)
        _leaveout_logmean_init!(a, A) || return false
    elseif strategy === :diagfeasible
        _reject_kwargs(strategy, kwargs)
        init_feasible_diag!(a, A)
    else
        throw(ArgumentError("unknown strategy :$strategy; expected one of :hardcover, :geomean, :leaveout, :diagfeasible"))
    end
    _make_feasible!(feasible, a, A)
    return true
end

# Apply the selected feasibility step. The `Vararg` length keeps calls
# statically specialized for `juliac`.
function _make_feasible!(feasible::Symbol, scales::Vararg{Any,N}) where N
    if feasible === :inflate
        inflate_feasible!(scales...)
    elseif feasible === :boost
        boost_feasible!(scales...)
    elseif feasible !== :none
        throw(ArgumentError("unknown feasible :$feasible; expected one of :inflate, :boost, :none"))
    end
    return nothing
end

# Reject keywords unused by a strategy.
function _reject_kwargs(strategy::Symbol, kwargs)
    isempty(kwargs) && return nothing
    throw(ArgumentError("strategy=:$strategy accepts no further keyword arguments, got $(string(join(keys(kwargs), ", ")))"))
end

# Recompute the unconstrained start with the most negative log-residual entry
# removed from the support. Residual ties use raw magnitude; this is the
# strategy's only covariance exception. Return `false` if no entry can be
# removed without emptying a row.
function _leaveout_logmean_init!(a::AbstractVector{T}, A::AbstractMatrix) where T
    ax = eachindex(a)
    axes(A) == (ax, ax) || throw(DimensionMismatch("`_leaveout_logmean_init!(a, A)` requires a square matrix with matching axes to `a` (got axes(A)=$(string(axes(A))), axes(a)=$(string(axes(a))))"))
    nza = unconstrained_min!(AbsLog{2}(), a, A)
    sum(nza) == 0 && return false
    # Use a roundoff-tolerant set for tied residuals.
    zmin = Ref(T(Inf))
    foreach_support_sym(A) do i, j, v
        zmin[] = min(zmin[], log(T(v)) - log(a[i]) - log(a[j]))
    end
    ztol = 64 * eps(T) * max(one(T), abs(zmin[]))
    best = Ref((first(ax) - 1, first(ax) - 1, T(Inf)))
    foreach_support_sym(A) do i, j, v
        Aij = T(v)
        z = log(Aij) - log(a[i]) - log(a[j])
        if z <= zmin[] + ztol && Aij < best[][3]
            best[] = (i, j, Aij)
        end
    end
    ibest, jbest, _ = best[]
    # Account for both endpoints of an off-diagonal entry.
    nza[ibest] > 1 || return false
    ibest == jbest || nza[jbest] > 1 || return false
    # One sweep of the same kernel on the reduced support. Omitting the entry
    # from the iterator also omits it from the diagonal reference, so a row
    # whose diagonal entry is dropped takes its reference from its neighbors.
    foreach_entries(f) = foreach_support_sym(A) do i, j, v
        (i == ibest && j == jbest) || f(i, j, log(T(v)))
    end
    _sym_unconstrained!(a, foreach_entries)
    return true
end
