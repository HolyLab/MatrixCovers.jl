# Soft covers and their multistart/coordinate-descent implementations.

# ============================================================
# Public interface
# ============================================================

"""
    a = soft_symcover(ϕ, A; maxiter=32, starts=5, σ=2.0, rng=MersenneTwister(0))
    a = soft_symcover(A; maxiter=32, starts=5, σ=2.0, rng=MersenneTwister(0))

Approximately minimize `∑ ϕ(|A[i,j]|/(a[i]*a[j]))` for symmetric `A`, without a
hard coverage constraint.

Supported penalties are:

- `AbsLog{2}()`: the convex minimum, computed by one linear solve.
- `AbsLog{1}()`: weighted-median coordinate descent to a fixed point, which need
  not be a local minimum.
- `AbsLinear{2}()` (default): multistart coordinate descent.
- `AbsLinear{1}()`: weighted-median descent initialized from the `AbsLinear{2}`
  result.

For `AbsLinear`, `starts` controls the number of starting points and `σ` the
spread of log-normal perturbations. Pass `rng` for reproducibility. `sigma` is
an alias for `σ`.

See also: [`symcover`](@ref), [`cover_objective`](@ref), [`soft_symcover_min`](@ref).

# Examples

Round the multistart result when comparing it with exact values.

```jldoctest
julia> A = [4 -1; -1 0];

julia> round.(soft_symcover(A); digits=4)
2-element Vector{Float64}:
 2.0
 0.5

julia> round.(soft_symcover([0 1; 1 0]); digits=4)
2-element Vector{Float64}:
 1.0
 1.0
```
"""
soft_symcover(A::AbstractMatrix; kwargs...) = soft_symcover(AbsLinear{2}(), A; kwargs...)

# The soft AbsLog{2} objective is convex with one minimizer, so the heuristic and the
# minimizer coincide: both are this solve.
soft_symcover(::AbsLog{2}, A::AbstractMatrix; kwargs...) = soft_symcover_min(AbsLog{2}(), A; kwargs...)

function soft_symcover(::AbsLog{1}, A::AbstractMatrix; maxiter::Int=20)
    require_abs_symmetric(A, :soft_symcover)
    a = soft_symcover_min(AbsLog{2}(), A)
    _abslog1_iter!(a, A, maxiter)
    return a
end

# Sole owner of the starts/σ/rng defaults for the AbsLinear{2} soft-cover family;
# every other method in that family (the no-ϕ wrapper, the AbsLinear{1} method)
# forwards them via `kwargs...` rather than restating the default.
function soft_symcover(::AbsLinear{2}, A::AbstractMatrix; maxiter::Int=32, starts::Int=5,
                       σ::Union{Real,Nothing}=nothing, sigma::Union{Real,Nothing}=nothing,
                       rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED))
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    require_abs_symmetric(A, :soft_symcover)
    return _soft_symcover_abslinear2(A, maxiter, starts, _resolve_alias(σ, sigma, 2.0, :σ, :sigma), rng)
end

function soft_symcover(::AbsLinear{1}, A::AbstractMatrix; maxiter::Int=20, kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    require_abs_symmetric(A, :soft_symcover)
    # Refine the `AbsLinear{2}` multistart result by weighted-median descent.
    a = soft_symcover(AbsLinear{2}(), A; maxiter=5, kwargs...)
    _abslinear1_iter!(a, A, maxiter)
    return a
end

"""
    a = soft_symcover!(ϕ, a, A; maxiter=...)
    a = soft_symcover!(a, A; maxiter=...)

Refine one symmetric soft-cover start in place. The no-ϕ form uses
`AbsLinear{2}()`. Build a start with [`initialize_symcover`](@ref) and
`feasible=:none`.

`a` must be finite and positive on supported rows. It need not cover `A`, and
unsupported scales are set to zero.

`maxiter` bounds the descent sweeps.

See also: [`soft_symcover`](@ref), [`soft_symcover_min!`](@ref), [`initialize_symcover`](@ref), [`soft_cover!`](@ref).
"""
function soft_symcover! end
soft_symcover!(a::AbstractVector, A::AbstractMatrix; kwargs...) =
    soft_symcover!(AbsLinear{2}(), a, A; kwargs...)

function soft_symcover!(::AbsLog{2}, a::AbstractVector, A::AbstractMatrix; kwargs...)
    _prepare_soft_symcover_start!(a, A, :soft_symcover!)
    a .= soft_symcover_min(AbsLog{2}(), A; kwargs...)   # convex: the start is not read
    return a
end

function soft_symcover!(::AbsLog{1}, a::AbstractVector, A::AbstractMatrix; maxiter::Int=20)
    _prepare_soft_symcover_start!(a, A, :soft_symcover!)
    _abslog1_iter!(a, A, maxiter)
    return a
end

function soft_symcover!(::AbsLinear{2}, a::AbstractVector, A::AbstractMatrix; maxiter::Int=32)
    _prepare_soft_symcover_start!(a, A, :soft_symcover!)
    _abslinear2_iter!(a, A, maxiter)
    return a
end

function soft_symcover!(::AbsLinear{1}, a::AbstractVector, A::AbstractMatrix; maxiter::Int=20)
    _prepare_soft_symcover_start!(a, A, :soft_symcover!)
    _abslinear1_iter!(a, A, maxiter)
    return a
end

"""
    a, b = soft_cover(ϕ, A; maxiter=200, starts=4, σ=2.0, rng=MersenneTwister(0))
    a, b = soft_cover(A; maxiter=200, starts=4, σ=2.0, rng=MersenneTwister(0))

Approximately minimize `∑ ϕ(|A[i,j]|/(a[i]*b[j]))` without a hard coverage
constraint. This is the asymmetric form of [`soft_symcover`](@ref).

Supported penalties are:

- `AbsLog{2}()`: the convex minimum, computed by one linear solve.
- `AbsLog{1}()`: alternating weighted-median updates to a fixed point, which
  need not be a local minimum.
- `AbsLinear{2}()` (default): alternating least squares.
- `AbsLinear{1}()`: alternating weighted-median updates initialized from the
  `AbsLinear{2}` result.

Unsupported rows and columns receive zero scale. The factors use the balance
convention of [`cover_min`](@ref).

For `AbsLinear`, `starts` controls the number of starting points and `σ` the
spread of perturbations. Pass `rng` for reproducibility. `sigma` is an alias for
`σ`.

See also: [`cover`](@ref), [`soft_symcover`](@ref), [`cover_objective`](@ref).

# Examples

```jldoctest; filter = r"(\\d+\\.\\d{4})\\d+" => s"\\1"
julia> A = [1 2 3; 6 5 4];

julia> a, b = soft_cover(A);

julia> a * b'
2×3 Matrix{Float64}:
 1.93288  1.97239  2.50673
 4.97144  5.07307  6.44741
```
"""
soft_cover(A::AbstractMatrix; kwargs...) = soft_cover(AbsLinear{2}(), A; kwargs...)

# The soft AbsLog{2} objective is convex with one minimizer, so the heuristic and the
# minimizer coincide: both are this solve.
soft_cover(::AbsLog{2}, A::AbstractMatrix; kwargs...) = soft_cover_min(AbsLog{2}(), A; kwargs...)

function soft_cover(::AbsLog{1}, A::AbstractMatrix; maxiter::Int=20)
    a, b = soft_cover_min(AbsLog{2}(), A)
    _abslog1_iter_asym!(a, b, A, maxiter)
    return _balance_cover!(a, b, A)
end

# Sole owner of the starts/σ/rng defaults for the AbsLinear{2} soft-cover family;
# every other method in that family (the no-ϕ wrapper, the AbsLinear{1} method)
# forwards them via `kwargs...` rather than restating the default.
function soft_cover(ϕ::AbsLinear{2}, A::AbstractMatrix; maxiter::Int=200, starts::Int=4,
                    σ::Union{Real,Nothing}=nothing, sigma::Union{Real,Nothing}=nothing,
                    rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED))
    return _soft_cover_abslinear2(A, maxiter, starts, _resolve_alias(σ, sigma, 2.0, :σ, :sigma), rng)
end

function soft_cover(ϕ::AbsLinear{1}, A::AbstractMatrix; maxiter::Int=100, kwargs...)
    # Refine the `AbsLinear{2}` multistart result by weighted-median descent.
    a, b = soft_cover(AbsLinear{2}(), A; maxiter=5, kwargs...)
    _abslinear1_iter_asym!(a, b, A, maxiter)
    return _balance_cover!(a, b, A)
end

"""
    a, b = soft_cover!(ϕ, a, b, A; maxiter=...)
    a, b = soft_cover!(a, b, A; maxiter=...)

Refine the starting point `(a, b)` into a soft cover of `A` in place. Scales must
be finite and positive on supported rows and columns; unsupported scales are
zeroed. The start need not cover `A`. Build one with [`initialize_cover`](@ref)
and `feasible=:none`. The no-ϕ form uses `AbsLinear{2}()`.

The result uses the balance convention of [`cover_min`](@ref), so equivalent
rescalings `(c*a, b/c)` give the same result.

See also: [`soft_cover`](@ref), [`soft_cover_min!`](@ref), [`initialize_cover`](@ref), [`soft_symcover!`](@ref).
"""
function soft_cover! end
soft_cover!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...) =
    soft_cover!(AbsLinear{2}(), a, b, A; kwargs...)

function soft_cover!(::AbsLog{2}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...)
    _prepare_soft_cover_start!(a, b, A, :soft_cover!)
    anew, bnew = soft_cover_min(AbsLog{2}(), A; kwargs...)   # convex: the start is not read
    a .= anew
    b .= bnew
    return a, b
end

function soft_cover!(::AbsLog{1}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; maxiter::Int=20)
    _prepare_soft_cover_start!(a, b, A, :soft_cover!)
    _abslog1_iter_asym!(a, b, A, maxiter)
    return _balance_cover!(a, b, A)
end

function soft_cover!(::AbsLinear{2}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; maxiter::Int=200)
    _prepare_soft_cover_start!(a, b, A, :soft_cover!)
    _msmc_als!(a, b, A, maxiter)
    return _balance_cover!(a, b, A)
end

function soft_cover!(::AbsLinear{1}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; maxiter::Int=100)
    _prepare_soft_cover_start!(a, b, A, :soft_cover!)
    _abslinear1_iter_asym!(a, b, A, maxiter)
    return _balance_cover!(a, b, A)
end

"""
    a = soft_symcover_min(ϕ, A)
    a = soft_symcover_min(A)

Return a local minimum of `∑ ϕ(|A[i,j]|/(a[i]*a[j]))` without coverage
constraints. The no-ϕ form uses `AbsLinear{2}()`.

Supported ϕ values and required extensions:
- `AbsLog{2}()`: solved natively as linear least squares; `linsolve` has the same
  meaning as in [`symcover_min`](@ref).
- `AbsLinear{1}()`, `AbsLinear{2}()`: require JuMP and Ipopt. Each strategy in
  `strategies` is refined, and the best local minimum is returned.
- `AbsLog{1}()`: not implemented.

See also: [`soft_symcover_min!`](@ref), [`soft_symcover`](@ref), [`symcover_min`](@ref).
"""
function soft_symcover_min end
soft_symcover_min(A::AbstractMatrix; kwargs...) = soft_symcover_min(AbsLinear{2}(), A; kwargs...)

function soft_symcover_min(::AbsLog{2}, A::AbstractMatrix; kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover_min requires a square matrix"))
    a, _ = _soft_symcover_min_abslog2(A; kwargs...)
    return a
end

# Multistart driver for the symmetric Ipopt kernels. Starts need not cover `A`.
function soft_symcover_min(ϕ::AbsLinear, A::AbstractMatrix; strategies=SYMCOVER_MIN_STRATEGIES)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover_min requires a square matrix"))
    isempty(strategies) &&
        throw(ArgumentError("soft_symcover_min: `strategies` must name at least one starting cover"))
    T = float(real(eltype(A)))
    starts = [similar(Array{T}, ax) for _ in strategies]
    built = [_initialize_symcover!(a, A, strategy, :none) for (a, strategy) in zip(starts, strategies)]
    covers = [soft_symcover_min!(ϕ, a, A) for (a, ok) in zip(starts, built) if ok]
    isempty(covers) &&
        throw(ArgumentError("soft_symcover_min: no strategy in $(string(strategies)) yields a starting cover of `A`"))
    return covers[_multistart_select([cover_objective(ϕ, a, A) for a in covers])]
end

"""
    a = soft_symcover_min!(ϕ, a, A)
    a = soft_symcover_min!(a, A)

Refine `a` into a local minimum of the symmetric soft-cover objective, in place.
The no-ϕ form uses `AbsLinear{2}()`.

`a` must be positive on supported rows; unsupported scales are zeroed. It need
not cover `A`. Use `feasible=:none` with [`initialize_symcover`](@ref).

For `AbsLinear`, the result can depend on the start.

See also: [`initialize_symcover`](@ref), [`soft_symcover_min`](@ref), [`symcover_min!`](@ref).
"""
function soft_symcover_min! end
soft_symcover_min!(a::AbstractVector, A::AbstractMatrix; kwargs...) =
    soft_symcover_min!(AbsLinear{2}(), a, A; kwargs...)

function soft_symcover_min!(::AbsLog{2}, a::AbstractVector, A::AbstractMatrix; kwargs...)
    _prepare_soft_symcover_start!(a, A)
    a .= soft_symcover_min(AbsLog{2}(), A; kwargs...)   # convex: the start is not read
    return a
end

# Validate a symmetric soft-cover start and clear unsupported scales.
function _prepare_soft_symcover_start!(a::AbstractVector, A::AbstractMatrix, fname::Symbol=:soft_symcover_min!)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("$fname requires a square matrix"))
    require_abs_symmetric(A, fname)
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(ax))"))
    supp = fill!(similar(a, Bool), false)
    foreach_support_sym(A) do i, j, v
        supp[i] = true
        supp[j] = true
    end
    for i in ax
        supp[i] || (a[i] = zero(eltype(a)))
    end
    for i in ax
        supp[i] || continue
        (isfinite(a[i]) && a[i] > zero(a[i])) ||
            throw(ArgumentError("$fname requires a start with finite positive scale on every supported row, got a[$(string(i))] = $(string(a[i]))"))
    end
    return a
end

"""
    a, b = soft_cover_min(ϕ, A)
    a, b = soft_cover_min(A)

Return a local minimum of `∑ ϕ(|A[i,j]|/(a[i]*b[j]))` without coverage
constraints. The no-ϕ form uses `AbsLinear{2}()`. Factors use the balance
convention of [`cover_min`](@ref).

Supported ϕ values and required extensions:
- `AbsLog{2}()`: solved natively.
- `AbsLinear{1}()`, `AbsLinear{2}()`: require JuMP and Ipopt. Each strategy in
  `strategies` is refined, and the best local minimum is returned.
- `AbsLog{1}()`: not implemented.

See also: [`soft_cover_min!`](@ref), [`soft_symcover_min`](@ref), [`soft_cover`](@ref).
"""
function soft_cover_min end
soft_cover_min(A::AbstractMatrix; kwargs...) = soft_cover_min(AbsLinear{2}(), A; kwargs...)

function soft_cover_min(::AbsLog{2}, A::AbstractMatrix; kwargs...)
    a, b, _ = _soft_cover_min_abslog2(A; kwargs...)
    return a, b
end

# Multistart driver for the asymmetric Ipopt kernels.
function soft_cover_min(ϕ::AbsLinear, A::AbstractMatrix; strategies=COVER_MIN_STRATEGIES)
    isempty(strategies) &&
        throw(ArgumentError("soft_cover_min: `strategies` must name at least one starting cover"))
    covers = [initialize_cover(A; strategy, feasible=:none) for strategy in strategies]
    for (a, b) in covers
        soft_cover_min!(ϕ, a, b, A)
    end
    return covers[_multistart_select([cover_objective(ϕ, a, b, A) for (a, b) in covers])]
end

"""
    a, b = soft_cover_min!(ϕ, a, b, A)
    a, b = soft_cover_min!(a, b, A)

Refine `(a, b)` into a local minimum of the asymmetric soft-cover objective, in
place. The no-ϕ form uses `AbsLinear{2}()`.

`a` and `b` must be positive on supported rows and columns; unsupported scales
are zeroed. The start need not cover `A`. Build one with `feasible=:none`.

The result uses the balance convention of [`soft_cover_min`](@ref), so equivalent
rescalings `(c*a, b/c)` give the same result.

See also: [`initialize_cover`](@ref), [`soft_cover_min`](@ref), [`soft_symcover_min!`](@ref).
"""
function soft_cover_min! end
soft_cover_min!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...) =
    soft_cover_min!(AbsLinear{2}(), a, b, A; kwargs...)

function soft_cover_min!(::AbsLog{2}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...)
    _prepare_soft_cover_start!(a, b, A)
    anew, bnew = soft_cover_min(AbsLog{2}(), A; kwargs...)   # convex: the start is not read
    a .= anew
    b .= bnew
    return a, b
end

# Validate an asymmetric soft-cover start, clear unsupported scales, and balance it.
function _prepare_soft_cover_start!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix,
                                    fname::Symbol=:soft_cover_min!)
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(string(eachindex(b))), axes(A, 2)=$(string(axes(A, 2)))"))
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
    for i in eachindex(a)
        suppa[i] || continue
        (isfinite(a[i]) && a[i] > zero(a[i])) ||
            throw(ArgumentError("$fname requires a start with finite positive scale on every supported row, got a[$(string(i))] = $(string(a[i]))"))
    end
    for j in eachindex(b)
        suppb[j] || continue
        (isfinite(b[j]) && b[j] > zero(b[j])) ||
            throw(ArgumentError("$fname requires a start with finite positive scale on every supported column, got b[$(string(j))] = $(string(b[j]))"))
    end
    return _balance_cover!(a, b, A)
end

# ============================================================
# Internal helpers
# ============================================================

# Default multistart seed. Pass an explicit RNG for cross-version reproducibility.
const _MULTISTART_SEED = 0

# Resolve Unicode and ASCII keyword aliases; reject conflicting values.
function _resolve_alias(primary, alias, default, primary_name::Symbol, alias_name::Symbol)
    primary === nothing && return alias === nothing ? default : alias
    alias === nothing && return primary
    primary == alias ||
        throw(ArgumentError("both `$primary_name` and `$alias_name` were given with different values ($(string(primary)) vs $(string(alias))); specify only one"))
    return primary
end

# Margin that prevents roundoff-equivalent candidates from replacing the incumbent.
_multistart_switchtol(::Type{T}) where {T} = 5_000_000 * eps(T)

# Symmetric AbsLinear{2} starts in selection order, followed by log-normal
# perturbations of the geometric-mean point. `starts` truncates or extends the list.
function _soft_symcover_abslinear2_inits(A::AbstractMatrix, starts::Int, σ::Real, rng)
    ax = axes(A, 1)
    T = float(real(eltype(A)))
    # The soft cover imposes no coverage constraint, so the starts are taken raw
    # (`feasible=:none`); "inflate" is the one candidate that is deliberately a cover.
    ag = initialize_symcover(A; strategy=:geomean, feasible=:none)   # also the perturbation base
    labels = ["geomean"]
    inits = [copy(ag)]
    length(inits) < starts &&
        (push!(labels, "hardcover"); push!(inits, initialize_symcover(A; strategy=:hardcover, feasible=:none)))
    length(inits) < starts &&
        (push!(labels, "inflate"); push!(inits, initialize_symcover(A; strategy=:geomean, feasible=:inflate)))
    if length(inits) < starts
        # `:leaveout` is unavailable when no entry can be dropped; a multistart forfeits that
        # slot rather than fail, so it takes the start through the gated builder.
        lo = similar(ag)
        _initialize_symcover!(lo, A, :leaveout, :none) && (push!(labels, "leaveout"); push!(inits, lo))
    end
    if length(inits) < starts && _nsupport(A) < length(A)
        push!(labels, "feasible"); push!(inits, initialize_symcover(A; strategy=:diagfeasible, feasible=:none))
    end
    k = 0
    while length(inits) < starts
        p = similar(ag)
        for i in ax
            ξ = randn(rng)
            p[i] = ag[i] > 0 ? ag[i] * exp(T(σ) * T(ξ)) : zero(T)
        end
        k += 1; push!(labels, "rand$k"); push!(inits, p)
    end
    return labels, inits
end

# Return the first candidate not beaten by more than the roundoff margin.
function _multistart_select(objs)
    besti = firstindex(objs)
    Ebest = objs[besti]
    switchtol = _multistart_switchtol(float(eltype(objs)))
    for k in eachindex(objs)
        objs[k] < Ebest * (1 - switchtol) && ((besti, Ebest) = (k, objs[k]))
    end
    return besti
end

# Shared multistart driver. Optional `labels` and `objs` collect candidate data
# for tests.
function _multistart_run(inits_builder::F, iterate!::G, objective::H, A::AbstractMatrix,
                          iter::Int, starts::Int, σ::Real, rng; labels=nothing, objs=nothing) where {F,G,H}
    labs, inits = inits_builder(A, starts, σ, rng)
    for x in inits
        iterate!(x, A, iter)
    end
    E = [objective(x, A) for x in inits]
    labels === nothing || append!(labels, labs)
    objs === nothing || append!(objs, E)
    return inits[_multistart_select(E)]
end

# Scale-covariant symmetric `AbsLinear{2}` multistart.
function _soft_symcover_abslinear2(A::AbstractMatrix, iter::Int, starts::Int, σ::Real, rng;
                                   labels=nothing, objs=nothing)
    return _multistart_run(_soft_symcover_abslinear2_inits,
                            (a, A, iter) -> _abslinear2_iter!(a, A, iter),
                            (a, A) -> cover_objective(AbsLinear{2}(), a, A),
                            A, iter, starts, σ, rng; labels, objs)
end

# Coordinate descent for a symmetric `AbsLinear{2}` soft cover. Each update
# minimizes
#   ½(1 - d/x²)² + ∑_{j≠k} (1 - c_j/x)²
# where `d = |A[k,k]|` and `c_j = |A[k,j]|/a[j]`. The `d=0` case is closed form;
# otherwise safeguarded Newton solves a cubic. A scale-invariant stationarity
# residual controls early exit.
function _abslinear2_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=50_000_000 * eps(T)) where T
    ax = eachindex(a)
    ax == axes(A, 1) || throw(DimensionMismatch("row indices of `A` must match `a`, got $(string(axes(A, 1))) vs $(string(ax))"))
    S = _sym_support(A, T)
    for _ in 1:iter
        maxres = zero(T)
        for k in ax
            d  = zero(T)            # diagonal entry, absent from the support when zero
            s1 = zero(T)            # ∑_{j≠k: A[k,j]≠0} |A[k,j]| / a[j]
            s2 = zero(T)            # ∑_{j≠k: A[k,j]≠0} |A[k,j]|² / a[j]²
            for s in _slots(S, k)
                j = S.idx[s]
                Akj = S.val[s]
                if j == k
                    d = Akj
                    continue
                end
                inv_aj = one(T) / a[j]
                s1 += Akj * inv_aj
                s2 += Akj^2 * inv_aj^2
            end
            ak = a[k]
            if !iszero(ak)
                inv_ak = one(T) / ak
                res = (s1 - (s2 - d) * inv_ak) * inv_ak - d^2 * inv_ak^4
                maxres = max(maxres, abs(res))
            end
            if iszero(s1)
                x = iszero(d) ? zero(T) : sqrt(d)
            elseif iszero(d)
                x = s2 / s1
            else
                # The cubic has one positive root bracketed by `sqrt(d)` and
                # `s2/s1`; geometric bisection handles wide dynamic range.
                lo, hi = minmax(sqrt(d), s2 / s1)
                x = sqrt(lo * hi)
                while hi - lo > 2 * eps(hi)
                    gx = s1*x^3 + (d - s2)*x^2 - d^2
                    iszero(gx) && break
                    if gx > 0
                        hi = x
                    else
                        lo = x
                    end
                    gxp = 3*s1*x^2 + 2*(d - s2)*x
                    xn = x - gx/gxp
                    x = lo < xn < hi ? xn : sqrt(lo * hi)
                end
            end
            a[k] = x
        end
        maxres <= T(tol) && break
    end
    return a
end

# Weighted median using each value as its weight. Sorts in place and returns the
# lower median as a deterministic, covariant tie-break.
function _weighted_self_median!(c::AbstractVector{T}) where T
    sort!(c)
    half = sum(c) / 2
    cum  = zero(T)
    wm   = first(c)
    for ci in c
        cum += ci
        wm   = ci
        cum >= half && break
    end
    return wm
end

# Symmetric `AbsLinear{1}` coordinate objective. A top-level function avoids a
# captured, boxed variable in the inner loop.
_abslinear1_obj(x, d, c) = abs(1 - d/x^2) + 2 * sum(abs(1 - ci/x) for ci in c)

# Symmetric `AbsLinear{1}` coordinate descent. Each update chooses the better of
# the weighted median and `sqrt(d)`. Relative movement controls early exit.
function _abslinear1_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=5000 * eps(T)) where T
    ax  = eachindex(a)
    ax == axes(A, 1) || throw(DimensionMismatch("row indices of `A` must match `a`, got $(string(axes(A, 1))) vs $(string(ax))"))
    S   = _sym_support(A, T)
    buf = Vector{T}(undef, length(ax))   # reusable buffer for c_j values
    for _ in 1:iter
        maxrel = zero(T)
        for k in ax
            d  = zero(T)
            nc = 0
            for s in _slots(S, k)
                j = S.idx[s]
                Akj = S.val[s]
                if j == k
                    d = Akj
                    continue
                end
                aj = a[j]
                iszero(aj) && continue
                nc += 1
                buf[nc] = Akj / aj
            end
            if nc == 0
                x = iszero(d) ? zero(T) : sqrt(d)
            else
                c = view(buf, 1:nc)
                wm = _weighted_self_median!(c)   # sorts `c` in place
                if iszero(d)
                    x = wm
                else
                    # Compare objective at weighted median vs sqrt(d)
                    sq_d = sqrt(d)
                    x = _abslinear1_obj(wm, d, c) <= _abslinear1_obj(sq_d, d, c) ? wm : sq_d
                end
            end
            ak  = a[k]
            den = max(abs(x), abs(ak))
            iszero(den) || (maxrel = max(maxrel, abs(x - ak) / den))
            a[k] = x
        end
        maxrel <= T(tol) && break
    end
    return a
end

# Symmetric `AbsLog{1}` coordinate descent in log space. Each update is a
# weighted median; the diagonal contributes twice. Relative movement controls
# early exit.
function _abslog1_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=5000 * eps(T)) where T
    ax  = eachindex(a)
    ax == axes(A, 1) || throw(DimensionMismatch("row indices of `A` must match `a`, got $(string(axes(A, 1))) vs $(string(ax))"))
    S   = _sym_support(A, T)
    buf = Vector{T}(undef, 2 * length(ax) + 1)   # off-diagonals (×1) + diagonal (×2)
    for _ in 1:iter
        maxrel = zero(T)
        for k in ax
            iszero(a[k]) && continue    # zero rows/columns stay uncovered
            n = 0
            for s in _slots(S, k)
                j = S.idx[s]
                Akj = S.val[s]
                if j == k
                    lhalf = log(Akj) / 2
                    buf[n += 1] = lhalf
                    buf[n += 1] = lhalf
                    continue
                end
                aj = a[j]
                iszero(aj) && continue
                buf[n += 1] = log(Akj) - log(aj)
            end
            n == 0 && continue
            c = view(buf, 1:n)
            sort!(c)
            x   = exp(c[(n + 1) ÷ 2])   # lower median
            ak  = a[k]
            den = max(abs(x), abs(ak))
            iszero(den) || (maxrel = max(maxrel, abs(x - ak) / den))
            a[k] = x
        end
        maxrel <= T(tol) && break
    end
    return a
end

# Asymmetric `AbsLog{1}` alternating weighted-median descent in log space. Each
# half-sweep minimizes its block, though a fixed point need not be a local
# minimum. Relative movement controls early exit.
function _abslog1_iter_asym!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix,
                             iter::Int; tol=nothing)
    T = float(promote_type(eltype(a), eltype(b)))
    rtol = tol === nothing ? 5000 * eps(T) : T(tol)
    axr, axc = axes(A, 1), axes(A, 2)
    eachindex(a) == axr || throw(DimensionMismatch("row indices of `A` must match `a`, got $(string(axr)) vs $(string(eachindex(a)))"))
    eachindex(b) == axc || throw(DimensionMismatch("column indices of `A` must match `b`, got $(string(axc)) vs $(string(eachindex(b)))"))
    R, C = _row_support(A, T), _col_support(A, T)
    bufc = Vector{T}(undef, length(axc))   # log-points for an a-row update
    bufr = Vector{T}(undef, length(axr))   # log-points for a b-column update
    for _ in 1:iter
        maxrel = zero(T)
        for i in axr
            iszero(a[i]) && continue       # unsupported rows/columns stay at zero
            nc = 0
            for s in _slots(R, i)
                bj = b[R.idx[s]]
                iszero(bj) && continue
                bufc[nc += 1] = log(R.val[s]) - log(bj)
            end
            nc == 0 && continue
            c = view(bufc, 1:nc)
            sort!(c)
            x   = exp(c[(nc + 1) ÷ 2])     # lower median
            ai  = a[i]
            den = max(abs(x), abs(ai))
            iszero(den) || (maxrel = max(maxrel, abs(x - ai) / den))
            a[i] = x
        end
        for j in axc
            iszero(b[j]) && continue
            nr = 0
            for s in _slots(C, j)
                ai = a[C.idx[s]]
                iszero(ai) && continue
                bufr[nr += 1] = log(C.val[s]) - log(ai)
            end
            nr == 0 && continue
            c = view(bufr, 1:nr)
            sort!(c)
            x   = exp(c[(nr + 1) ÷ 2])
            bj  = b[j]
            den = max(abs(x), abs(bj))
            iszero(den) || (maxrel = max(maxrel, abs(x - bj) / den))
            b[j] = x
        end
        maxrel <= rtol && break
    end
    return a, b
end

# Asymmetric `AbsLinear{2}` starts: boosted geometric mean, tightened cover, and
# log-normal perturbations.
function _soft_cover_abslinear2_inits(A::AbstractMatrix, starts::Int, σ::Real, rng)
    T = float(real(eltype(A)))
    ag, bg = initialize_cover(A; strategy=:geomean, feasible=:boost)
    labels = ["boost"]
    inits = [(copy(ag), copy(bg))]
    # Reuse the geometric-mean and boost passes when constructing the hard start.
    length(inits) < starts && (push!(labels, "hardcover"); push!(inits, tighten_cover!(copy(ag), copy(bg), A)))
    k = 0
    while length(inits) < starts
        a = similar(ag); b = similar(bg)
        for i in axes(A, 1)
            ξ = randn(rng)
            a[i] = ag[i] > 0 ? ag[i] * exp(T(σ) * T(ξ)) : zero(T)
        end
        for j in axes(A, 2)
            η = randn(rng)
            b[j] = bg[j] > 0 ? bg[j] * exp(T(σ) * T(η)) : zero(T)
        end
        k += 1; push!(labels, "rand$k"); push!(inits, (a, b))
    end
    return labels, inits
end

# Scale-covariant asymmetric `AbsLinear{2}` multistart.
function _soft_cover_abslinear2(A::AbstractMatrix, iter::Int, starts::Int, σ::Real, rng;
                                labels=nothing, objs=nothing)
    # Explicit indexing avoids a dynamic tuple-destructuring call under `juliac`.
    a, b = _multistart_run(_soft_cover_abslinear2_inits,
                            (ab, A, iter) -> _msmc_als!(ab[1], ab[2], A, iter),
                            (ab, A) -> cover_objective(AbsLinear{2}(), ab[1], ab[2], A),
                            A, iter, starts, σ, rng; labels, objs)
    # Balance the gauge after alternating row and column updates.
    return _balance_cover!(a, b, A)
end

# Alternating least squares for `AbsLinear{2}` in inverse scales `u=1/a`,
# `v=1/b`. Each half-sweep is exact. Column-grouped support is reused throughout.
# After updating `v[j] = num[j]/den[j]`, column `j` contributes
#     ∑_i (1 - M[i,j] u[i] v[j])² = nnz[j] - 2 v[j] num[j] + v[j]² den[j]
#                                 = nnz[j] - num[j]²/den[j]
# so the objective needs no extra matrix pass.
function _msmc_als!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, iter::Int;
                    tol=nothing)
    axr, axc = axes(A, 1), axes(A, 2)
    eachindex(a) == axr || throw(DimensionMismatch("row indices of `A` must match `a`, got $(string(axr)) vs $(string(eachindex(a)))"))
    eachindex(b) == axc || throw(DimensionMismatch("column indices of `A` must match `b`, got $(string(axc)) vs $(string(eachindex(b)))"))
    T = float(promote_type(eltype(a), eltype(b), real(eltype(A))))
    # Scale the convergence floor to the precision of `T`.
    rtol = tol === nothing ? 50 * eps(T) : T(tol)
    # Invert to inverse-scale variables; empty-support rows/columns (scale 0) stay at 0.
    u = map(x -> x > 0 ? inv(T(x)) : zero(T), a)
    v = map(x -> x > 0 ? inv(T(x)) : zero(T), b)
    numu = similar(u)
    denu = similar(u)
    C = _col_support(A, T)
    E = _msmc_objective(C, u, v)
    for _ in 1:iter
        fill!(numu, zero(T))
        fill!(denu, zero(T))
        for j in axc
            vj = v[j]
            for s in _slots(C, j)
                i = C.idx[s]
                Av = C.val[s] * vj
                numu[i] += Av
                denu[i] += Av * Av
            end
        end
        for i in axr
            denu[i] > 0 && (u[i] = numu[i] / denu[i])
        end
        Enew = zero(T)
        for j in axc
            num = den = zero(T)
            for s in _slots(C, j)
                Au = C.val[s] * u[C.idx[s]]
                num += Au
                den += Au * Au
            end
            if den > 0
                v[j] = num / den
                Enew += _ngroup(C, j) - num * num / den
            end
        end
        E - Enew <= rtol * max(E, one(T)) && (E = Enew; break)
        E = Enew
    end
    for i in axr
        a[i] = iszero(u[i]) ? zero(eltype(a)) : inv(u[i])
    end
    for j in axc
        b[j] = iszero(v[j]) ? zero(eltype(b)) : inv(v[j])
    end
    return a, b
end

# Soft-cover objective in inverse scales, using column-grouped support.
function _msmc_objective(C::GroupedSupport{T}, u::AbstractVector, v::AbstractVector) where T
    E = zero(T)
    for j in C.ax
        vj = v[j]
        for s in _slots(C, j)
            r = one(T) - C.val[s] * u[C.idx[s]] * vj
            E += r * r
        end
    end
    return E
end

# Asymmetric `AbsLinear{1}` alternating weighted-median descent. Each half-sweep
# minimizes its block; relative movement controls early exit.
function _abslinear1_iter_asym!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix,
                                iter::Int; tol=nothing)
    T = float(promote_type(eltype(a), eltype(b)))
    rtol = tol === nothing ? 5000 * eps(T) : T(tol)
    axr, axc = axes(A, 1), axes(A, 2)
    eachindex(a) == axr || throw(DimensionMismatch("row indices of `A` must match `a`, got $(string(axr)) vs $(string(eachindex(a)))"))
    eachindex(b) == axc || throw(DimensionMismatch("column indices of `A` must match `b`, got $(string(axc)) vs $(string(eachindex(b)))"))
    R, C = _row_support(A, T), _col_support(A, T)
    bufc = Vector{T}(undef, length(axc))   # c_j buffer for an a-row update
    bufr = Vector{T}(undef, length(axr))   # c_i buffer for a b-column update
    for _ in 1:iter
        maxrel = zero(T)
        for i in axr
            nc = 0
            for s in _slots(R, i)
                bj = b[R.idx[s]]
                iszero(bj) && continue
                nc += 1
                bufc[nc] = R.val[s] / bj
            end
            x = nc == 0 ? zero(T) : _weighted_self_median!(view(bufc, 1:nc))
            ai  = a[i]
            den = max(abs(x), abs(ai))
            iszero(den) || (maxrel = max(maxrel, abs(x - ai) / den))
            a[i] = x
        end
        for j in axc
            nc = 0
            for s in _slots(C, j)
                ai = a[C.idx[s]]
                iszero(ai) && continue
                nc += 1
                bufr[nc] = C.val[s] / ai
            end
            x = nc == 0 ? zero(T) : _weighted_self_median!(view(bufr, 1:nc))
            bj  = b[j]
            den = max(abs(x), abs(bj))
            iszero(den) || (maxrel = max(maxrel, abs(x - bj) / den))
            b[j] = x
        end
        maxrel <= rtol && break
    end
    return a, b
end
