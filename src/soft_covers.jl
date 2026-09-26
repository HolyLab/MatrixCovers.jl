# Soft covers: public entry points and the `AbsLinear` multistart driver.

# ============================================================
# Public interface
# ============================================================

"""
    a = soft_symcover(ϕ, A; kwargs...)
    a = soft_symcover(A; kwargs...)

Minimize `∑ ϕ(|A[i,j]|/(a[i]*a[j]))` for symmetric `A`, without a hard coverage
constraint. The default penalty is `PowerMean{2}()`.

Supported penalties and their keywords are:

- `PowerMean{p}()` (default): the convex minimum, computed by damped
  simultaneous power-mean updates `a[k] ← sqrt(a[k] * M_p(|A[k,j]|/a[j]))`, where
  `M_p` is the `p`-power mean over the nonzeros of row `k`, followed when they
  are slow by Newton steps with backtracking on the objective. Every update
  decreases the objective. The iteration stops when, in every nonzero row, the mean of `r^p`
  over the ratios `r = |A[k,j]|/(a[k]*a[j])` of the nonzero entries is within
  `tol` of one. It computes in at least `Float64`, and `tol` defaults to
  `4096*eps` of that type, and a warning reports an iteration that ends without
  reaching it. The keywords `maxiter`, `newton`, `maxnewton`, and `linsolve` are
  described in the extended help.
- `AbsLog{2}()`: the convex minimum, computed by one linear solve; `linsolve` has
  the same meaning as in [`symcover_min`](@ref).
- `AbsLog{1}()`: the convex minimum, computed as a linear program; requires JuMP
  and HiGHS. The minimizer need not be unique, and among the minimizers the one
  with the least `AbsLog{2}` objective is returned.
- `AbsLinear{1}()`, `AbsLinear{2}()`: the objective is not convex. Several starts
  are each refined to a local minimum, and the one with the least objective is
  returned; requires JuMP and Ipopt. `starts` (default 5) is the number of
  starts: a few structured starts, then log-normal perturbations of the
  geometric-mean start with spread `σ` (default 2.0; `sigma` is an alias),
  drawn from `rng` (default `MersenneTwister(0)`).

Unsupported rows receive zero scale. When a connected component of the support
is bipartite with no diagonal entry, the products on the support do not
determine the scales; `PowerMean` and `AbsLog` then apply the balance convention
of [`cover_min`](@ref) between the two sides of the bipartition.

See also: [`symcover`](@ref), [`cover_objective`](@ref), [`soft_symcover!`](@ref), [`PowerMean`](@ref).

# Examples

```jldoctest
julia> A = [4 -1; -1 1];

julia> a = soft_symcover(A);

julia> R = abs.(A) ./ (a .* a');

julia> round.(sum(R .^ 2; dims=2); digits=8)   # each row's mean square ratio is one
2×1 Matrix{Float64}:
 2.0
 2.0

julia> round.(soft_symcover([0 1; 1 0]); digits=4)
2-element Vector{Float64}:
 1.0
 1.0
```

# Extended help

## `PowerMean` keywords

- `maxiter` (default `10_000`): the largest number of power-mean updates.
- `newton` (default `true`): whether Newton steps finish the solve. The updates
  hand over to Newton when the rate of decrease of the imbalance over recent
  updates predicts more than 200 further updates, and after at most
  `min(maxiter, 400)` updates. With `newton=false` the updates continue until
  convergence or `maxiter`.
- `maxnewton` (default `100`): the largest number of Newton steps. Each step
  solves the Newton equations and backtracks on the objective.
- `linsolve` (default `:auto`): the solver for the Newton equations. `:dense`
  factorizes the dense Hessian; `:cholesky` factorizes the sparse Hessian with
  CHOLMOD (`Float64` only); `:cg` uses conjugate gradients with a diagonal
  preconditioner, at one pass over the support per iteration, for problems too
  large to factor. `:auto` chooses `:dense` for support that fills at least a
  quarter of the grid and otherwise `:cholesky`, each when its predicted flop
  count is at most `8e3` per stored entry (and, for `:cholesky`, its predicted
  storage at most `2^30` bytes), and `:cg` when neither fits. Poorly
  conditioned problems can need many conjugate-gradient iterations.
"""
soft_symcover(A::AbstractMatrix; kwargs...) = soft_symcover(PowerMean{2}(), A; kwargs...)

function soft_symcover(::AbsLog{2}, A::AbstractMatrix; kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    a, _ = _soft_symcover_abslog2(A; kwargs...)
    return a
end

# Sole owner of the starts/σ/rng defaults for the symmetric `AbsLinear` multistart.
function soft_symcover(ϕ::AbsLinear, A::AbstractMatrix; starts::Int=5,
                       σ::Union{Real,Nothing}=nothing, sigma::Union{Real,Nothing}=nothing,
                       rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED))
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover requires a square matrix"))
    require_abs_symmetric(A, :soft_symcover)
    return _soft_symcover_abslinear(ϕ, A, starts, _resolve_alias(σ, sigma, 2.0, :σ, :sigma), rng)
end

"""
    a = soft_symcover!(ϕ, a, A; kwargs...)
    a = soft_symcover!(a, A; kwargs...)

Refine one symmetric soft-cover start in place. The no-ϕ form uses
`PowerMean{2}()`. Build a start with [`initialize_symcover`](@ref) and
`feasible=:none`.

`a` must be finite and positive on supported rows. It need not cover `A`, and
unsupported scales are set to zero.

For the convex penalties (`PowerMean`, `AbsLog`) the result is the minimizer
returned by [`soft_symcover`](@ref), with the same keywords, and does not depend
on the start. For `AbsLinear` it is the local minimum reached from the start, and
there are no keywords.

See also: [`soft_symcover`](@ref), [`initialize_symcover`](@ref), [`soft_cover!`](@ref).
"""
function soft_symcover! end
soft_symcover!(a::AbstractVector, A::AbstractMatrix; kwargs...) =
    soft_symcover!(PowerMean{2}(), a, A; kwargs...)

function soft_symcover!(::AbsLog{2}, a::AbstractVector, A::AbstractMatrix; kwargs...)
    _prepare_soft_symcover_start!(a, A)
    a .= soft_symcover(AbsLog{2}(), A; kwargs...)   # convex: the start is not read
    return a
end

"""
    a, b = soft_cover(ϕ, A; kwargs...)
    a, b = soft_cover(A; kwargs...)

Minimize `∑ ϕ(|A[i,j]|/(a[i]*b[j]))` without a hard coverage constraint. This is
the asymmetric form of [`soft_symcover`](@ref). The default penalty is
`PowerMean{2}()`.

Supported penalties and their keywords are:

- `PowerMean{p}()` (default): the convex minimum, computed by alternating
  power-mean updates of the row and column scales (Sinkhorn scaling of
  `abs.(A).^p` to row sums `n_i` and column sums `m_j`, the nonzero counts),
  with adaptive overrelaxation, followed when the sweeps are slow by Newton
  steps with backtracking on the objective. The iteration stops when, in every
  nonzero row and column, the mean of `r^p` over the ratios `r` of the nonzero
  entries is within `tol` of one. It computes in at least `Float64`, and `tol`
  defaults to `4096*eps` of that type; a warning reports an iteration that ends
  without reaching it. `maxiter` (default `10_000`) bounds the sweeps, each an
  update of all rows and then all columns. The sweeps hand over to Newton when
  their rate predicts more than 100 further sweeps, and after at most
  `min(maxiter, 200)` sweeps. `newton`, `maxnewton`, and `linsolve` have the
  meanings given in the extended help of [`soft_symcover`](@ref). When
  `abs.(A)` is exactly symmetric, the minimizer has `b == a` under the balance
  convention, and `soft_cover` instead computes `a` by the algorithm of
  [`soft_symcover`](@ref) (with the same keywords; `maxiter` then counts its
  updates) and returns `(a, copy(a))`. The in-place [`soft_cover!`](@ref) always
  uses the alternating iteration.
- `AbsLog{2}()`: the convex minimum, computed by one linear solve; `linsolve` has
  the same meaning as in [`cover_min`](@ref).
- `AbsLog{1}()`: the convex minimum, computed as a linear program; requires JuMP
  and HiGHS. Among the minimizers, the one with the least `AbsLog{2}` objective
  is returned.
- `AbsLinear{1}()`, `AbsLinear{2}()`: the objective is not convex. Several starts
  are each refined to a local minimum, and the one with the least objective is
  returned; requires JuMP and Ipopt. `starts` (default 4) is the number of
  starts: the geometric-mean and hard-cover starts, then log-normal
  perturbations of the first with spread `σ` (default 2.0; `sigma` is an alias),
  drawn from `rng` (default `MersenneTwister(0)`).

Unsupported rows and columns receive zero scale. The factors use the balance
convention of [`cover_min`](@ref).

See also: [`cover`](@ref), [`soft_symcover`](@ref), [`soft_cover!`](@ref), [`cover_objective`](@ref), [`PowerMean`](@ref).

# Examples

```jldoctest; filter = r"(\\d+\\.\\d{4})\\d+" => s"\\1"
julia> A = [1 2 3; 6 5 4];

julia> a, b = soft_cover(A);

julia> a * b'
2×3 Matrix{Float64}:
 1.73729  1.93617  2.37048
 4.64478  5.17648  6.33766
```
"""
soft_cover(A::AbstractMatrix; kwargs...) = soft_cover(PowerMean{2}(), A; kwargs...)

function soft_cover(::AbsLog{2}, A::AbstractMatrix; kwargs...)
    a, b, _ = _soft_cover_abslog2(A; kwargs...)
    return a, b
end

# Sole owner of the starts/σ/rng defaults for the asymmetric `AbsLinear` multistart.
function soft_cover(ϕ::AbsLinear, A::AbstractMatrix; starts::Int=4,
                    σ::Union{Real,Nothing}=nothing, sigma::Union{Real,Nothing}=nothing,
                    rng::AbstractRNG=MersenneTwister(_MULTISTART_SEED))
    return _soft_cover_abslinear(ϕ, A, starts, _resolve_alias(σ, sigma, 2.0, :σ, :sigma), rng)
end

"""
    a, b = soft_cover!(ϕ, a, b, A; kwargs...)
    a, b = soft_cover!(a, b, A; kwargs...)

Refine the starting point `(a, b)` into a soft cover of `A` in place. Scales must
be finite and positive on supported rows and columns; unsupported scales are
zeroed. The start need not cover `A`. Build one with [`initialize_cover`](@ref)
and `feasible=:none`. The no-ϕ form uses `PowerMean{2}()`.

For the convex penalties (`PowerMean`, `AbsLog`) the result is the minimizer
returned by [`soft_cover`](@ref), with the same keywords, and does not depend on
the start. For `AbsLinear` it is the local minimum reached from the start, and
there are no keywords. The result uses the balance convention of
[`cover_min`](@ref), so equivalent rescalings `(c*a, b/c)` give the same result.

See also: [`soft_cover`](@ref), [`initialize_cover`](@ref), [`soft_symcover!`](@ref).
"""
function soft_cover! end
soft_cover!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...) =
    soft_cover!(PowerMean{2}(), a, b, A; kwargs...)

function soft_cover!(::AbsLog{2}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...)
    _prepare_soft_cover_start!(a, b, A)
    anew, bnew = soft_cover(AbsLog{2}(), A; kwargs...)   # convex: the start is not read
    a .= anew
    b .= bnew
    return a, b
end

# Validate a symmetric soft-cover start and clear unsupported scales.
function _prepare_soft_symcover_start!(a::AbstractVector, A::AbstractMatrix, fname::Symbol=:soft_symcover!)
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

# Validate an asymmetric soft-cover start, clear unsupported scales, and balance it.
function _prepare_soft_cover_start!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix,
                                    fname::Symbol=:soft_cover!)
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
# AbsLinear multistart
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

# Shared multistart driver: refine every start, then select. Optional `labels` and
# `objs` collect candidate data for tests.
function _multistart_run(inits_builder::F, refine!::G, objective::H, A::AbstractMatrix,
                         starts::Int, σ::Real, rng; labels=nothing, objs=nothing) where {F,G,H}
    labs, inits = inits_builder(A, starts, σ, rng)
    for x in inits
        refine!(x, A)
    end
    E = [objective(x, A) for x in inits]
    labels === nothing || append!(labels, labs)
    objs === nothing || append!(objs, E)
    return inits[_multistart_select(E)]
end

# Symmetric `AbsLinear` starts in selection order, followed by log-normal
# perturbations of the geometric-mean point. `starts` truncates or extends the list.
function _soft_symcover_abslinear_inits(A::AbstractMatrix, starts::Int, σ::Real, rng)
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

# Scale-covariant symmetric `AbsLinear` multistart.
function _soft_symcover_abslinear(ϕ::AbsLinear, A::AbstractMatrix, starts::Int, σ::Real, rng;
                                  labels=nothing, objs=nothing)
    return _multistart_run(_soft_symcover_abslinear_inits,
                           (a, A) -> soft_symcover!(ϕ, a, A),
                           (a, A) -> cover_objective(ϕ, a, A),
                           A, starts, σ, rng; labels, objs)
end

# Asymmetric `AbsLinear` starts: boosted geometric mean, tightened cover, and
# log-normal perturbations.
function _soft_cover_abslinear_inits(A::AbstractMatrix, starts::Int, σ::Real, rng)
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

# Scale-covariant asymmetric `AbsLinear` multistart.
function _soft_cover_abslinear(ϕ::AbsLinear, A::AbstractMatrix, starts::Int, σ::Real, rng;
                               labels=nothing, objs=nothing)
    return _multistart_run(_soft_cover_abslinear_inits,
                           (ab, A) -> soft_cover!(ϕ, ab[1], ab[2], A),
                           (ab, A) -> cover_objective(ϕ, ab[1], ab[2], A),
                           A, starts, σ, rng; labels, objs)
end
