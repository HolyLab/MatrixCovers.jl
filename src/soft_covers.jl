# Soft covers: `symcover`/`cover` variants that penalize under-coverage instead
# of forbidding it, plus the scale-covariant multistart and coordinate-descent
# machinery they use.

# ============================================================
# Public interface
# ============================================================

"""
    a = soft_symcover(ϕ, A; maxiter=20, starts=8, σ=2.0, rng=MersenneTwister(0))
    a = soft_symcover(A; maxiter=20, starts=8, σ=2.0, rng=MersenneTwister(0))

Given a square matrix `A` assumed to be symmetric, return a vector `a` approximately
minimizing the soft-cover objective `∑_{i,j} ϕ(|A[i,j]| / (a[i]*a[j]))`.

Unlike [`symcover`](@ref), there is no hard coverage constraint: `a[i]*a[j]` may be
less than `|A[i,j]|`, with violations penalized by `ϕ`.

Supported penalty functions:
- `AbsLog{2}()`: returns the analytical unconstrained minimum (no iterations needed).
- `AbsLog{1}()`: initializes from the AbsLog{2} minimum, then refines by coordinate descent
  with a log-space weighted-median step. The AbsLog{1} objective has a flat basin of equally
  good minima; this returns the deterministic, scale-covariant representative reached by
  coordinate descent from the AbsLog{2} minimum.
- `AbsLinear{2}()` (default): non-convex; refined by coordinate descent from `starts`
  scale-covariant starting points, keeping the lowest-objective result (see below).
- `AbsLinear{1}()`: initializes from the `AbsLinear{2}()` result, coordinate descent uses a
  weighted-median step.

For the `AbsLinear` penalties the objective is non-convex, so `starts` starting points are
tried and the best kept. The first four are deterministic — the geometric-mean minimum, the
tightened hard cover, the geometric-mean minimum inflated uniformly until it covers `A`, and a
leave-one-out geometric mean that drops the support entry with the most negative log-residual
(this last start keeps the result continuous as an entry `|A[i,j]|` approaches zero) — and the
rest are multiplicative log-normal perturbations `a .* exp.(σ .* ξ)` of the geometric-mean point with
spread `σ`, `ξ` drawn from `rng`. Every start co-varies with a diagonal rescaling of `A` and
the objective is scale-invariant, so the selection is scale-covariant. The default `rng` is a
fresh `MersenneTwister(0)` per call, making repeated calls (and the two frames of a covariance
check) agree; pass your own `rng` for reproducibility you control, since default RNG streams
are not stable across Julia versions. `sigma` is accepted as an ASCII alias for `σ`.

See also: [`symcover`](@ref), [`cover_objective`](@ref), [`soft_symcover_min`](@ref).

# Examples

The multistart converges to the covariant minimizer to within its objective
tolerance; round to compare against exact values.

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

function soft_symcover(ϕ::AbsLog{2}, A::AbstractMatrix)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    T = float(real(eltype(A)))
    # Dense scale vector matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    a = similar(Array{T}, ax)
    unconstrained_min!(ϕ, a, A)   # analytical minimum; no iterations needed
    return a
end

function soft_symcover(::AbsLog{1}, A::AbstractMatrix; maxiter::Int=20)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    T = float(real(eltype(A)))
    # Dense scale vector matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    a = similar(Array{T}, ax)
    unconstrained_min!(AbsLog{2}(), a, A)   # convex AbsLog{2} minimum: a good start
    _abslog1_iter!(a, A, maxiter)
    return a
end

# Sole owner of the starts/σ/rng defaults for the AbsLinear{2} soft-cover family;
# every other method in that family (the no-ϕ wrapper, the AbsLinear{1} method)
# forwards them via `kwargs...` rather than restating the default.
function soft_symcover(::AbsLinear{2}, A::AbstractMatrix; maxiter::Int=20, starts::Int=8,
                       σ::Union{Real,Nothing}=nothing, sigma::Union{Real,Nothing}=nothing,
                       rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED))
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    return _soft_symcover_abslinear2(A, maxiter, starts, _resolve_alias(σ, sigma, 2.0, :σ, :sigma), rng)
end

function soft_symcover(::AbsLinear{1}, A::AbstractMatrix; maxiter::Int=20, kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    # Initialize from the AbsLinear{2} soft cover (a good starting point for AbsLinear{1});
    # the multistart is spent there, then the AbsLinear{1} weighted-median descent refines.
    a = soft_symcover(AbsLinear{2}(), A; maxiter=5, kwargs...)
    _abslinear1_iter!(a, A, maxiter)
    return a
end

"""
    a, b = soft_cover(ϕ, A; maxiter=100, starts=8, σ=2.0, rng=MersenneTwister(0))
    a, b = soft_cover(A; maxiter=100, starts=8, σ=2.0, rng=MersenneTwister(0))

Given a matrix `A`, return vectors `a` and `b` approximately minimizing the soft-cover
objective `∑_{i,j} ϕ(|A[i,j]| / (a[i]*b[j]))`. This is the asymmetric analog of
[`soft_symcover`](@ref).

Unlike [`cover`](@ref), there is no hard coverage constraint: `a[i]*b[j]` may be less than
`|A[i,j]|`, with violations penalized by `ϕ`.

Supported penalty functions:
- `AbsLinear{2}()` (default): in the inverse-scale variables `u = 1 ./ a`, `v = 1 ./ b`, the
  objective `∑_{i,j∈S} (1 - |A[i,j]| u[i] v[j])²` (sum over the nonzero support `S`) is
  biconvex, so alternating least squares with the closed-form half-sweeps

      u[i] = ∑_j |A[i,j]| v[j] / ∑_j (|A[i,j]| v[j])²   (dually for v[j])

  is monotone, stopping when the relative objective decrease drops below `1e-14` or after
  `maxiter` sweeps.
- `AbsLinear{1}()`: initializes from the `AbsLinear{2}()` result, then refines by alternating
  weighted-median updates — each row/column block is minimized exactly, so the descent is
  monotone. Its flat basins are broken by a deterministic lower-median tie-break, giving a
  scale-covariant representative.

Rows or columns of `A` that are entirely zero receive scale `0`.

The objective is non-convex, so `starts` starting points are tried and the lowest-objective
result kept: the geometric-mean init, the tightened hard cover [`cover`](@ref), and — for the
remaining starts — multiplicative log-normal perturbations `a .* exp.(σ .* ξ)`, with spread
`σ`, of the geometric-mean point, `ξ` drawn from `rng`. Every start co-varies with an
independent row/column rescaling of `A` and the objective is scale-invariant, so the selection
is scale-covariant. The default `rng` is a fresh `MersenneTwister(0)` per call, making repeated
calls (and the two frames of a covariance check) agree; pass your own `rng` for reproducibility
you control, since default RNG streams are not stable across Julia versions. `sigma` is accepted
as an ASCII alias for `σ`.

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

# Sole owner of the starts/σ/rng defaults for the AbsLinear{2} soft-cover family;
# every other method in that family (the no-ϕ wrapper, the AbsLinear{1} method)
# forwards them via `kwargs...` rather than restating the default.
function soft_cover(ϕ::AbsLinear{2}, A::AbstractMatrix; maxiter::Int=100, starts::Int=8,
                    σ::Union{Real,Nothing}=nothing, sigma::Union{Real,Nothing}=nothing,
                    rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED))
    return _soft_cover_abslinear2(A, maxiter, starts, _resolve_alias(σ, sigma, 2.0, :σ, :sigma), rng)
end

function soft_cover(ϕ::AbsLinear{1}, A::AbstractMatrix; maxiter::Int=100, kwargs...)
    # Spend the multistart on the AbsLinear{2} cover (a good basin selector), then refine with
    # the AbsLinear{1} weighted-median descent.
    a, b = soft_cover(AbsLinear{2}(), A; maxiter=5, kwargs...)
    _abslinear1_iter_asym!(a, b, A, maxiter)
    return a, b
end

"""
    a = soft_symcover_min(ϕ, A)
    a = soft_symcover_min(A)

Return the ϕ-minimal symmetric soft cover of `A`: minimizes `∑_{i,j} ϕ(|A[i,j]|/(a[i]*a[j]))`
with no coverage constraints. The no-ϕ form defaults to `AbsLinear{2}()`, matching
[`soft_symcover`](@ref).

Supported ϕ values and required extensions:
- `AbsLog{2}()`: requires JuMP and HiGHS.
- `AbsLinear{1}()`, `AbsLinear{2}()`: requires JuMP and Ipopt.
"""
function soft_symcover_min end
soft_symcover_min(A::AbstractMatrix; kwargs...) = soft_symcover_min(AbsLinear{2}(), A; kwargs...)

"""
    a, b = soft_cover_min(ϕ, A)
    a, b = soft_cover_min(A)

Return the ϕ-minimal asymmetric soft cover of `A`: minimizes
`∑_{i,j} ϕ(|A[i,j]|/(a[i]*b[j]))` with no coverage constraints. This is the asymmetric
analog of [`soft_symcover_min`](@ref). The no-ϕ form defaults to `AbsLinear{2}()`,
matching [`soft_cover`](@ref).

Supported ϕ values and required extensions:
- `AbsLog{2}()`: solved natively (no external solver) — the same analytic geometric-mean
  minimum [`cover`](@ref) computes as its initial point.
- `AbsLinear{1}()`, `AbsLinear{2}()`: not yet implemented.

See also: [`soft_symcover_min`](@ref), [`soft_cover`](@ref).
"""
function soft_cover_min end
soft_cover_min(A::AbstractMatrix; kwargs...) = soft_cover_min(AbsLinear{2}(), A; kwargs...)

function soft_cover_min(::AbsLog{2}, A::AbstractMatrix)
    T = float(real(eltype(A)))
    a = similar(Array{T}, axes(A, 1))
    b = similar(Array{T}, axes(A, 2))
    unconstrained_min!(AbsLog{2}(), a, b, A)
    return a, b
end

# ============================================================
# Internal helpers
# ============================================================

# Default seed for the multistart perturbation RNG. Callers wanting reproducibility across
# Julia versions (whose default RNG streams are not stable) should pass their own `rng`.
const _MULTISTART_SEED = 0

# Resolve a Unicode keyword and its ASCII alias to a single value, falling back to
# `default` when neither is given. Passing both raises an error unless they agree,
# so one can never silently override the other.
function _resolve_alias(primary, alias, default, primary_name::Symbol, alias_name::Symbol)
    primary === nothing && return alias === nothing ? default : alias
    alias === nothing && return primary
    primary == alias ||
        throw(ArgumentError("both `$primary_name` and `$alias_name` were given with different values ($primary vs $alias); specify only one"))
    return primary
end

# Relative-objective margin a later start must beat the incumbent by to replace it. Set well
# above the descent's objective tolerance (~1e-14) and well below any genuine basin gap, so
# same-basin convergence noise never flips the selection (which would break covariance) while
# real improvements always do.
const _MULTISTART_SWITCHTOL = 1e-9

# Leave-one-out geometric mean, an alternative starting point to the AbsLog{2}
# geometric-mean init for the AbsLinear soft cover. The geometric mean weights every nonzero
# entry equally, so an entry with |A[i,j]| far below the rest (in the scale-invariant sense
# of its log-residual z[i,j] = log|A[i,j]| - α[i] - α[j] at the unweighted minimum) skews
# the start into a worse basin than the exact-zero limit. Here the entry with the most
# negative residual is dropped from the support and the geometric mean recomputed, giving a
# start in the basin that treats that entry as effectively zero; the AbsLinear objective is
# finite at r = 0, so refinement then varies continuously as the entry vanishes.
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

# Labeled candidate starts for the symmetric AbsLinear{2} multistart, in selection order.
# Deterministic starts: the AbsLog{2} geometric-mean minimum (also the perturbation base),
# the tightened hard cover, the uniformly inflated geometric mean, the leave-one-out
# geometric mean (when a support entry can be dropped), and — only when `A` has a zero entry
# — the greedy feasible cover `init_feasible_diag!`. The feasible start is gated because on a
# fully dense `A` it never uniquely wins, and "which entries are zero" is invariant under a
# diagonal rescaling `D*A*D`, so the gate keeps the selection scale-covariant. Remaining
# slots, up to `starts` total, are multiplicative log-normal perturbations `a_g .* exp.(σ .* ξ)`
# of the geometric-mean point, `ξ` drawn from `rng` (drawn for every index so the stream is
# frame-independent). `starts` below the number of deterministic starts truncates the list.
function _soft_symcover_abslinear2_inits(A::AbstractMatrix, starts::Int, σ::Real, rng)
    ax = axes(A, 1)
    T = float(real(eltype(A)))
    # Dense scale vector matching cover/symcover; `similar(A, …)` is a SparseVector for sparse A.
    ag = similar(Array{T}, ax)
    unconstrained_min!(AbsLog{2}(), ag, A)   # geometric mean; also the perturbation base
    labels = ["geomean"]
    inits = [copy(ag)]
    length(inits) < starts && (push!(labels, "hardcover"); push!(inits, symcover(A)))
    if length(inits) < starts
        ce = copy(ag); inflate_feasible!(ce, A); push!(labels, "inflate"); push!(inits, ce)
    end
    if length(inits) < starts
        lo = similar(ag); _leaveout_logmean_init!(lo, A) && (push!(labels, "leaveout"); push!(inits, lo))
    end
    if length(inits) < starts && any(iszero, A)
        fe = similar(ag); init_feasible_diag!(fe, A); push!(labels, "feasible"); push!(inits, fe)
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

# Index of the multistart winner among candidate objectives `objs`: the earliest candidate not
# beaten by a strict relative improvement. Switching only on a genuine improvement is what keeps
# the selection scale-covariant — candidates landing in the same basin converge to the same
# objective only to the descent tolerance (~1e-14), and switching on that noise would forfeit
# covariance, since the incumbent (ordered geometric-mean first) is the most covariant start.
# Real basin improvements are far larger. This is the single source of the selection rule, so a
# caller that captures `objs` (below) recovers the winner exactly, without re-deriving it.
function _multistart_select(objs)
    besti = firstindex(objs)
    Ebest = objs[besti]
    for k in eachindex(objs)
        objs[k] < Ebest * (1 - _MULTISTART_SWITCHTOL) && ((besti, Ebest) = (k, objs[k]))
    end
    return besti
end

# Shared driver for the AbsLinear{2} soft-cover multistarts: build the candidate list with
# `inits_builder`, refine each candidate in place with `iterate!`, score it with `objective`,
# and return the `_multistart_select` winner. `iterate!`/`objective` take the candidate itself
# (a bare vector for the symmetric cover, an `(a, b)` tuple for the asymmetric one) so the same
# driver serves both shapes.
#
# For cheap provenance auditing (which initialization earned the selection), pass `labels`
# and/or `objs` as empty vectors: they are filled in place with every candidate's label and
# final objective, in candidate order, so the winner is `labels[_multistart_select(objs)]`.
function _multistart_run(inits_builder::F, iterate!::G, objective::H, A::AbstractMatrix,
                          iter::Int, starts::Int, σ::Real, rng; labels=nothing, objs=nothing) where {F,G,H}
    labs, inits = inits_builder(A, starts, σ, rng)
    E = map(inits) do x
        iterate!(x, A, iter)
        objective(x, A)
    end
    labels === nothing || append!(labels, labs)
    objs === nothing || append!(objs, E)
    return inits[_multistart_select(E)]
end

# Scale-covariant multistart for the symmetric AbsLinear{2} soft cover. Runs the single-start
# coordinate descent `_abslinear2_iter!` from the candidate list built by
# `_soft_symcover_abslinear2_inits` and returns the candidate `_multistart_select` picks. Every
# start co-varies with a diagonal rescaling `D*A*D` and the objective is scale-invariant, so the
# selection is scale-covariant; passing the same `rng` state across the two frames (as the
# default fresh-seeded RNG does) makes it reproducible.
function _soft_symcover_abslinear2(A::AbstractMatrix, iter::Int, starts::Int, σ::Real, rng;
                                   labels=nothing, objs=nothing)
    return _multistart_run(_soft_symcover_abslinear2_inits,
                            (a, A, iter) -> _abslinear2_iter!(a, A, iter),
                            (a, A) -> cover_objective(AbsLinear{2}(), a, A),
                            A, iter, starts, σ, rng; labels, objs)
end

# Coordinate-descent iteration for AbsLinear{2} soft cover.
# Each coordinate a[k] is updated to the exact minimizer of
#   (1 - d/x²)² + ∑_{j≠k} (1 - c_j/x)²
# where d = |A[k,k]| and c_j = |A[k,j]|/a[j].
# Closed form when d=0 (x = s2/s1); Newton on a cubic otherwise.
#
# `iter` bounds the sweeps; the descent exits early once every coordinate's
# stationarity residual r_k = ∑_j (1 - ρ)ρ (ρ = |A[k,j]|/(a[k]a[j])), the
# gradient of the objective in log a[k], has magnitude below `tol` at the start
# of a sweep. The residual is available for free from the sums already formed
# for the update (r_k = s1/a[k] - s2/a[k]² + d/a[k]² - d²/a[k]⁴), it is the exact
# quantity optimality demands be zero, and it is scale-invariant (each ρ is), so
# covariant restarts of a rescaled problem exit on the same sweep and the
# multistart selection stays covariant.
function _abslinear2_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=1e-8) where T
    ax = eachindex(a)
    for _ in 1:iter
        maxres = zero(T)
        for k in ax
            d  = T(abs(A[k, k]))   # diagonal entry
            s1 = zero(T)            # ∑_{j≠k: A[k,j]≠0} |A[k,j]| / a[j]
            s2 = zero(T)            # ∑_{j≠k: A[k,j]≠0} |A[k,j]|² / a[j]²
            for j in ax
                j == k && continue
                Akj = T(abs(A[k, j]))
                iszero(Akj) && continue
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
                # The cubic g(x) = s1*x³ + (d - s2)*x² - d² has exactly one positive root
                # (one Descartes sign change), and it is bracketed by √d and s2/s1:
                # g(√d) = d*(s1*√d - s2) and g(s2/s1) = d*(s2²/s1² - d) have opposite signs.
                # Safeguarded Newton with geometric bisection: the bracket endpoints can be
                # separated by hundreds of orders of magnitude, so fallback steps must bisect
                # in log space to converge in O(60) iterations.
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

# Weighted median of the values in `c` using the values themselves as weights: the
# point `m` with ∑_{cᵢ<m} cᵢ = ∑_{cᵢ>m} cᵢ. This minimizes ∑ᵢ |1 - cᵢ/x| over x > 0 —
# the gradient is (1/x²)(∑_{cᵢ<x} cᵢ − ∑_{cᵢ>x} cᵢ), whose positive 1/x² factor leaves
# the root at that balance point. Sorts `c` in place and returns the lower weighted
# median, a deterministic, scale-covariant tie-break on the flat basin the AbsLinear{1}
# objective admits.
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

# Coordinate-descent iteration for AbsLinear{1} soft cover.
# Each coordinate a[k] is updated to minimize ∑_j |1 - |A[k,j]|/(a[k]*a[j])|.
# For the off-diagonal sum, the minimizer is the weighted median of c_j with weights c_j,
# where c_j = |A[k,j]|/a[j].  When A[k,k] ≠ 0 we also compare against sqrt(|A[k,k]|).
#
# `iter` bounds the sweeps; the descent exits early once the largest relative
# coordinate movement in a sweep drops to `tol`. The median update reaches an
# exact fixed point (identical ordering selects the same value), so movement
# falls to zero there. Relative movement is scale-invariant, so covariant
# restarts of a rescaled problem exit on the same sweep.
function _abslinear1_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=1e-12) where T
    ax  = eachindex(a)
    buf = Vector{T}(undef, length(ax))   # reusable buffer for c_j values
    for _ in 1:iter
        maxrel = zero(T)
        for k in ax
            d  = T(abs(A[k, k]))
            nc = 0
            for j in ax
                j == k && continue
                Akj = T(abs(A[k, j]))
                iszero(Akj) && continue
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
                    obj_at(x) = abs(1 - d/x^2) + sum(abs(1 - ci/x) for ci in c)
                    x = obj_at(wm) <= obj_at(sq_d) ? wm : sq_d
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

# Coordinate-descent iteration for AbsLog{1} soft cover, working in log space (α = log a).
# Each coordinate α[k] is updated to minimize ∑_{j: A[k,j]≠0} |α[k] + α[j] - log|A[k,j]||.
# Holding the neighbours fixed, the minimizer over α[k] is the median of the points
# log|A[k,j]| - α[j], one per off-diagonal nonzero entry. The diagonal term is
# |2α[k] - log|A[k,k]|| = 2|α[k] - log|A[k,k]|/2|, i.e. the point log|A[k,k]|/2 with weight two,
# represented here by inserting it twice so a plain median carries the weighting. The AbsLog{1}
# minimum is a flat basin; the lower median is chosen for a deterministic, scale-covariant result.
#
# `iter` bounds the sweeps; the descent exits early once the largest relative
# coordinate movement in a sweep drops to `tol`. The median update reaches an
# exact fixed point, so movement falls to zero there. Relative movement is
# scale-invariant, so covariant restarts of a rescaled problem exit on the same
# sweep.
function _abslog1_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=1e-12) where T
    ax  = eachindex(a)
    buf = Vector{T}(undef, 2 * length(ax) + 1)   # off-diagonals (×1) + diagonal (×2)
    for _ in 1:iter
        maxrel = zero(T)
        for k in ax
            iszero(a[k]) && continue    # zero rows/columns stay uncovered
            n = 0
            Akk = T(abs(A[k, k]))
            if !iszero(Akk)
                lhalf = log(Akk) / 2
                buf[n += 1] = lhalf
                buf[n += 1] = lhalf
            end
            for j in ax
                j == k && continue
                Akj = T(abs(A[k, j]))
                iszero(Akj) && continue
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

# Labeled `(a, b)` candidate starts for the asymmetric AbsLinear{2} multistart, in selection
# order. Deterministic starts: the geometric-mean init `cover(A; maxiter=0)` (also the perturbation
# base) and the tightened hard cover `cover(A)`, obtained by tightening a copy of the
# geometric-mean init so the shared passes run once. Remaining slots, up to `starts` total, are
# multiplicative log-normal perturbations `a_g .* exp.(σ .* ξ)`, `b_g .* exp.(σ .* η)` of the
# geometric-mean point, `ξ`/`η` drawn from `rng` (drawn for every index so the stream is
# frame-independent).
function _soft_cover_abslinear2_inits(A::AbstractMatrix, starts::Int, σ::Real, rng)
    T = float(real(eltype(A)))
    ag, bg = cover(A; maxiter=0)   # geometric-mean init (boosted, untightened); perturbation base
    labels = ["geomean"]
    inits = [(copy(ag), copy(bg))]
    # The tightened hard cover `cover(A)` differs from `(ag, bg)` only by its tightening
    # iterations, so tighten a copy (at `tighten_cover!`'s own default `maxiter`) rather than
    # recomputing the shared geometric-mean and feasibility passes.
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

# Scale-covariant multistart for the asymmetric AbsLinear{2} soft cover. Runs the single-start
# alternating least squares `_mscm_als!` from the candidate list built by
# `_soft_cover_abslinear2_inits` and returns the pair `_multistart_select` picks. Every start
# co-varies with an independent row/column rescaling `D_r*A*D_c` and the objective is
# scale-invariant, so the selection is scale-covariant; passing the same `rng` state across the
# two frames (as the default fresh-seeded RNG does) makes it reproducible.
function _soft_cover_abslinear2(A::AbstractMatrix, iter::Int, starts::Int, σ::Real, rng;
                                labels=nothing, objs=nothing)
    return _multistart_run(_soft_cover_abslinear2_inits,
                            ((a, b), A, iter) -> _mscm_als!(a, b, A, iter),
                            ((a, b), A) -> cover_objective(AbsLinear{2}(), a, b, A),
                            A, iter, starts, σ, rng; labels, objs)
end

# Alternating least squares for the AbsLinear{2} soft cover in the inverse-scale variables
# u = 1 ./ a, v = 1 ./ b. With M = |A| restricted to its nonzero support, the objective
# E = ∑ (1 - M[i,j] u[i] v[j])² is biconvex; each half-sweep sets u[i] (resp. v[j]) to its
# exact minimizer. Rows/columns with empty support keep scale 0 and are held fixed.
# Refines `a`, `b` in place starting from their incoming values.
function _mscm_als!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, iter::Int;
                    tol=1e-14)
    axr, axc = axes(A, 1), axes(A, 2)
    eachindex(a) == axr || throw(DimensionMismatch("row indices of `A` must match `a`, got $(axr) vs $(eachindex(a))"))
    eachindex(b) == axc || throw(DimensionMismatch("column indices of `A` must match `b`, got $(axc) vs $(eachindex(b))"))
    T = float(promote_type(eltype(a), eltype(b), real(eltype(A))))
    # Invert to inverse-scale variables; empty-support rows/columns (scale 0) stay at 0.
    u = map(x -> x > 0 ? inv(T(x)) : zero(T), a)
    v = map(x -> x > 0 ? inv(T(x)) : zero(T), b)
    E = _mscm_objective(A, u, v)
    for _ in 1:iter
        for i in axr
            num = den = zero(T)
            for j in axc
                Aij = T(abs(A[i, j]))
                iszero(Aij) && continue
                Av = Aij * v[j]
                num += Av
                den += Av * Av
            end
            den > 0 && (u[i] = num / den)
        end
        for j in axc
            num = den = zero(T)
            for i in axr
                Aij = T(abs(A[i, j]))
                iszero(Aij) && continue
                Au = Aij * u[i]
                num += Au
                den += Au * Au
            end
            den > 0 && (v[j] = num / den)
        end
        Enew = _mscm_objective(A, u, v)
        E - Enew <= tol * max(E, one(T)) && (E = Enew; break)
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

# Soft-cover objective in inverse-scale variables: ∑_{i,j: A[i,j]≠0} (1 - |A[i,j]| u[i] v[j])².
function _mscm_objective(A::AbstractMatrix, u::AbstractVector, v::AbstractVector)
    T = float(promote_type(real(eltype(A)), eltype(u), eltype(v)))
    E = zero(T)
    for j in axes(A, 2)
        vj = v[j]
        for i in axes(A, 1)
            Aij = T(abs(A[i, j]))
            iszero(Aij) && continue
            r = one(T) - Aij * u[i] * vj
            E += r * r
        end
    end
    return E
end

# Alternating weighted-median descent for the asymmetric AbsLinear{1} soft cover.
# Updating a[i] with b fixed minimizes ∑_j |1 - |A[i,j]|/(a[i] b[j])| over a[i] > 0, whose
# minimizer is the weighted median of c_j = |A[i,j]|/b[j] weighted by the same c_j (see
# `_weighted_self_median!`); the b-update is dual. There is no self-coupled diagonal term (the
# (i,i) entry enters the row update through b[i] like any other column), so each full sweep is
# an exact block minimization and the objective decreases monotonically. Refines `a`, `b` in
# place from their incoming values; exits early once the largest relative coordinate movement
# in a sweep drops to `tol` (scale-invariant, so covariant restarts of a rescaled problem exit
# on the same sweep). Rows/columns with empty support keep scale 0.
function _abslinear1_iter_asym!(a::AbstractVector{T}, b::AbstractVector{T}, A::AbstractMatrix,
                                iter::Int; tol::Real=1e-12) where T
    axr, axc = axes(A, 1), axes(A, 2)
    eachindex(a) == axr || throw(DimensionMismatch("row indices of `A` must match `a`, got $(axr) vs $(eachindex(a))"))
    eachindex(b) == axc || throw(DimensionMismatch("column indices of `A` must match `b`, got $(axc) vs $(eachindex(b))"))
    bufc = Vector{T}(undef, length(axc))   # c_j buffer for an a-row update
    bufr = Vector{T}(undef, length(axr))   # c_i buffer for a b-column update
    for _ in 1:iter
        maxrel = zero(T)
        for i in axr
            nc = 0
            for j in axc
                Aij = T(abs(A[i, j]))
                iszero(Aij) && continue
                bj = b[j]
                iszero(bj) && continue
                nc += 1
                bufc[nc] = Aij / bj
            end
            x = nc == 0 ? zero(T) : _weighted_self_median!(view(bufc, 1:nc))
            ai  = a[i]
            den = max(abs(x), abs(ai))
            iszero(den) || (maxrel = max(maxrel, abs(x - ai) / den))
            a[i] = x
        end
        for j in axc
            nc = 0
            for i in axr
                Aij = T(abs(A[i, j]))
                iszero(Aij) && continue
                ai = a[i]
                iszero(ai) && continue
                nc += 1
                bufr[nc] = Aij / ai
            end
            x = nc == 0 ? zero(T) : _weighted_self_median!(view(bufr, 1:nc))
            bj  = b[j]
            den = max(abs(x), abs(bj))
            iszero(den) || (maxrel = max(maxrel, abs(x - bj) / den))
            b[j] = x
        end
        maxrel <= T(tol) && break
    end
    return a, b
end
