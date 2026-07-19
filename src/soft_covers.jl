# Soft covers: `symcover`/`cover` variants that penalize under-coverage instead
# of forbidding it, plus the scale-covariant multistart and coordinate-descent
# machinery they use.

# ============================================================
# Public interface
# ============================================================

"""
    a = soft_symcover(ϕ, A; maxiter=32, starts=5, σ=2.0, rng=MersenneTwister(0))
    a = soft_symcover(A; maxiter=32, starts=5, σ=2.0, rng=MersenneTwister(0))

Given a square matrix `A` assumed to be symmetric, return a vector `a` approximately
minimizing the soft-cover objective `∑_{i,j} ϕ(|A[i,j]| / (a[i]*a[j]))`.

Unlike [`symcover`](@ref), there is no hard coverage constraint: `a[i]*a[j]` may be
less than `|A[i,j]|`, with violations penalized by `ϕ`.

Supported penalty functions:
- `AbsLog{2}()`: convex, and returns its exact unconstrained minimum from a single linear
  solve. Identical to [`soft_symcover_min`](@ref)`(AbsLog{2}(), A)` — with one minimizer
  there is nothing for a heuristic and a minimizer to disagree about.
- `AbsLog{1}()`: initializes from the AbsLog{2} minimum, then refines by coordinate descent
  with a log-space weighted-median step, reaching a deterministic and scale-covariant fixed
  point. That point is not in general a minimizer: each step minimizes exactly over one
  coordinate, but the objective's nonsmoothness couples `a[i]` with `a[j]`, so the descent
  can settle where no single-coordinate move improves and the objective still sits
  materially above its minimum. [`soft_symcover_min`](@ref) does not yet offer an exact
  `AbsLog{1}` alternative.
- `AbsLinear{2}()` (default): non-convex; refined by coordinate descent from `starts`
  scale-covariant starting points, keeping the lowest-objective result (see below).
- `AbsLinear{1}()`: initializes from the `AbsLinear{2}()` result, coordinate descent uses a
  weighted-median step.

For the `AbsLinear` penalties the objective is non-convex, so `starts` starting points are
tried and the best kept, taken in this order: the geometric-mean minimum, the tightened hard
cover, the geometric-mean minimum inflated uniformly until it covers `A`, a leave-one-out
geometric mean that drops the support entry with the most negative log-residual (this start
keeps the result continuous as an entry `|A[i,j]|` approaches zero), and — only when `A` has a
zero entry — a greedy feasible cover. Any slots left over are multiplicative log-normal
perturbations `a .* exp.(σ .* ξ)` of the geometric-mean point with spread `σ`, `ξ` drawn from
`rng`; at the default `starts=5` there is at most one such perturbation. Every start co-varies
with a diagonal rescaling of `A` and the objective is scale-invariant, so the selection is
scale-covariant. The default `rng` is a fresh `MersenneTwister(0)` per call, making repeated
calls (and the two frames of a covariance check) agree; pass your own `rng` for reproducibility
you control, since default RNG streams are not stable across Julia versions. `sigma` is
accepted as an ASCII alias for `σ`.

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

# The soft AbsLog{2} objective is convex with one minimizer, so the heuristic and the
# minimizer coincide: both are this solve.
soft_symcover(::AbsLog{2}, A::AbstractMatrix; kwargs...) = soft_symcover_min(AbsLog{2}(), A; kwargs...)

function soft_symcover(::AbsLog{1}, A::AbstractMatrix; maxiter::Int=20)
    require_abs_symmetric(A, :soft_symcover)
    a = soft_symcover_min(AbsLog{2}(), A)   # convex AbsLog{2} minimum: a good start
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
    # Initialize from the AbsLinear{2} soft cover (a good starting point for AbsLinear{1});
    # the multistart is spent there, then the AbsLinear{1} weighted-median descent refines.
    a = soft_symcover(AbsLinear{2}(), A; maxiter=5, kwargs...)
    _abslinear1_iter!(a, A, maxiter)
    return a
end

"""
    a = soft_symcover!(ϕ, a, A; maxiter=...)
    a = soft_symcover!(a, A; maxiter=...)

Refine the starting point `a` into a symmetric soft cover of `A`, in place, and return it.
The no-ϕ form defaults to `AbsLinear{2}()`, matching [`soft_symcover`](@ref), whose
supported ϕ values these methods share.

This is the refiner half of [`soft_symcover`](@ref): the non-mutating form owns a
multistart menu, so its result is a property of `A`, while this one descends from the
single start you hand it, so its result is a property of `A` *and* that start. Building
the start is the caller's job — see [`initialize_symcover`](@ref), and pass
`feasible=:none`, since a soft cover is under no obligation to cover `A`.

`a` must be finite and strictly positive on every row of `A` that carries support; scales
on rows carrying no support are inert, and are zero on output. Unlike [`symcover_min!`](@ref),
`a` need *not* cover `A` — the soft objective imposes no coverage constraint.

`maxiter` bounds the descent sweeps; its default matches the corresponding
[`soft_symcover`](@ref) method. Under `AbsLog{2}` the objective is convex with a unique
minimizer, so the start is honored but not visible in the result.

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

Given a matrix `A`, return vectors `a` and `b` approximately minimizing the soft-cover
objective `∑_{i,j} ϕ(|A[i,j]| / (a[i]*b[j]))`. This is the asymmetric analog of
[`soft_symcover`](@ref).

Unlike [`cover`](@ref), there is no hard coverage constraint: `a[i]*b[j]` may be less than
`|A[i,j]|`, with violations penalized by `ϕ`.

Supported penalty functions:
- `AbsLog{2}()`: convex, and returns its exact unconstrained minimum from a single linear
  solve. Identical to [`soft_cover_min`](@ref)`(AbsLog{2}(), A)` — with one minimizer
  there is nothing for a heuristic and a minimizer to disagree about.
- `AbsLog{1}()`: initializes from the `AbsLog{2}()` minimum, then refines by alternating
  weighted-median row and column updates, reaching a deterministic and scale-covariant fixed
  point. As in [`soft_symcover`](@ref), that point is not in general a minimizer: each
  half-sweep minimizes exactly, but the objective's nonsmoothness couples `a[i]` with `b[j]`,
  so the descent can settle where no such sweep improves and the objective still sits
  materially above its minimum. [`soft_cover_min`](@ref) does not yet offer an exact
  `AbsLog{1}` alternative.
- `AbsLinear{2}()` (default): in the inverse-scale variables `u = 1 ./ a`, `v = 1 ./ b`, the
  objective `∑_{i,j∈S} (1 - |A[i,j]| u[i] v[j])²` (sum over the nonzero support `S`) is
  biconvex, so alternating least squares with the closed-form half-sweeps

      u[i] = ∑_j |A[i,j]| v[j] / ∑_j (|A[i,j]| v[j])²   (dually for v[j])

  is monotone, stopping when the relative objective decrease falls to rounding level
  for the element type, or after `maxiter` sweeps.
- `AbsLinear{1}()`: initializes from the `AbsLinear{2}()` result, then refines by alternating
  weighted-median updates — each row/column block is minimized exactly, so the descent is
  monotone. Its flat basins are broken by a deterministic lower-median tie-break, giving a
  scale-covariant representative.

Rows or columns of `A` that are entirely zero receive scale `0`. As with [`cover`](@ref),
only the products `a[i] * b[j]` are determined by the problem; the split is fixed by the
balance convention `∑ nzaᵢ log a[i] = ∑ nzbⱼ log b[j]`, imposed within each connected
component of the support (the gauge acts independently on each).

The objective is non-convex, so `starts` starting points are tried and the lowest-objective
result kept: the geometric mean boosted until it covers `A`, the tightened hard cover
[`cover`](@ref), and — for the remaining starts — multiplicative log-normal perturbations
`a .* exp.(σ .* ξ)`, with spread `σ`, of that boosted point, `ξ` drawn from `rng`. Every start co-varies with an
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

# The soft AbsLog{2} objective is convex with one minimizer, so the heuristic and the
# minimizer coincide: both are this solve.
soft_cover(::AbsLog{2}, A::AbstractMatrix; kwargs...) = soft_cover_min(AbsLog{2}(), A; kwargs...)

function soft_cover(::AbsLog{1}, A::AbstractMatrix; maxiter::Int=20)
    a, b = soft_cover_min(AbsLog{2}(), A)   # convex AbsLog{2} minimum: a good start
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
    # Spend the multistart on the AbsLinear{2} cover (a good basin selector), then refine with
    # the AbsLinear{1} weighted-median descent.
    a, b = soft_cover(AbsLinear{2}(), A; maxiter=5, kwargs...)
    _abslinear1_iter_asym!(a, b, A, maxiter)
    return _balance_cover!(a, b, A)
end

"""
    a, b = soft_cover!(ϕ, a, b, A; maxiter=...)
    a, b = soft_cover!(a, b, A; maxiter=...)

Refine the starting point `(a, b)` into a soft cover of `A`, in place, and return it. This
is the asymmetric counterpart of [`soft_symcover!`](@ref) and the refiner half of
[`soft_cover`](@ref), carrying the same contract on the start: finite and strictly
positive on every supported row and column, inert (and zero on output) elsewhere, and
under no obligation to cover `A`. Build one with [`initialize_cover`](@ref) and
`feasible=:none`. The no-ϕ form defaults to `AbsLinear{2}()`, matching
[`soft_cover`](@ref).

The product `a[i]*b[j]` is unchanged by `a -> c*a`, `b -> b/c`, so the start is read only
up to that gauge: `(a, b)` and `(2a, b/2)` give the same result. The result itself is
pinned to the balance convention of [`cover_min`](@ref), as every asymmetric cover in this
package is.

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

Return the ϕ-minimal symmetric soft cover of `A`: minimizes `∑_{i,j} ϕ(|A[i,j]|/(a[i]*a[j]))`
with no coverage constraints. The no-ϕ form defaults to `AbsLinear{2}()`, matching
[`soft_symcover`](@ref).

Supported ϕ values and required extensions:
- `AbsLog{2}()`: solved natively (no external solver). In log space the objective is a
  linear least-squares, so one solve settles it, and being convex it has a unique
  minimizer that no start can influence. `linsolve` selects the inner solve, exactly as
  in [`symcover_min`](@ref).
- `AbsLinear{1}()`, `AbsLinear{2}()`: requires JuMP and Ipopt. These objectives are
  non-convex, so the solver returns the minimum of the basin it starts in. Rather than
  commit to one start, these methods refine each of `strategies` — the
  [`initialize_symcover`](@ref) menu, by default `$(SYMCOVER_MIN_STRATEGIES)`, without
  forcing feasibility — and return the best cover found, at a cost of one solve per start.
- `AbsLog{1}()`: not yet implemented. The objective is an LP in log space, but its optimum
  is a face, and the lexicographic AbsLog{2} selection that [`symcover_min`](@ref) uses to
  pin one member of the corresponding hard face does not carry over: the hard face is bounded
  by the coverage constraints, while this one is a level set of an unconstrained piecewise-
  linear objective, across which the quadratic pulls far enough to cost most of the exactly
  tight residuals that make `AbsLog{1}` worth choosing.

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

# Multistart driver for the non-convex soft AbsLinear covers, the unconstrained counterpart
# of `symcover_min(::AbsLinear)`. The kernels (`soft_symcover_min!`) live in MatrixCoversIpoptExt; the
# menu and the selection are native. Starts are taken raw: a soft cover is under no
# obligation to cover `A`.
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
        throw(ArgumentError("soft_symcover_min: no strategy in $strategies yields a starting cover of `A`"))
    return covers[_multistart_select([cover_objective(ϕ, a, A) for a in covers])]
end

"""
    a = soft_symcover_min!(ϕ, a, A)
    a = soft_symcover_min!(a, A)

Refine the starting point `a` into the ϕ-minimal symmetric soft cover of `A`, in place.
This is the soft counterpart of [`symcover_min!`](@ref), and the second half of the
initialize/refine pair whose first half is [`initialize_symcover`](@ref). The no-ϕ form
defaults to `AbsLinear{2}()`, matching [`soft_symcover_min`](@ref), whose supported ϕ
values these methods share.

`a` must be strictly positive on every row of `A` that carries support; scales on rows
carrying no support are inert, and are zero on output. Unlike [`symcover_min!`](@ref), `a`
need *not* cover `A` — the soft objective imposes no coverage constraint, and the natural
starts do not satisfy one. Pass `feasible=:none` when building a start with
[`initialize_symcover`](@ref).

The `AbsLinear` penalties are non-convex, so the start selects the local minimum the solver
descends into; that is why [`soft_symcover_min`](@ref) tries several rather than committing
to one. Under `AbsLog{2}` the objective is convex with a unique minimizer, so the start is
honored but not visible in the result.

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

# Shared prologue of the symmetric soft refiners (`soft_symcover!`, `soft_symcover_min!`).
# The soft objective constrains nothing, so — unlike `_prepare_symcover_start!` — this
# checks positivity only, and moves the start nowhere. `fname` names the caller so the
# error reports the function the user actually called.
function _prepare_soft_symcover_start!(a::AbstractVector, A::AbstractMatrix, fname::Symbol=:soft_symcover_min!)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("$fname requires a square matrix"))
    require_abs_symmetric(A, fname)
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`, got eachindex(a)=$(eachindex(a)), axes(A, 1)=$ax"))
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
            throw(ArgumentError("$fname requires a start with finite positive scale on every supported row, got a[$i] = $(a[i])"))
    end
    return a
end

"""
    a, b = soft_cover_min(ϕ, A)
    a, b = soft_cover_min(A)

Return the ϕ-minimal asymmetric soft cover of `A`: minimizes
`∑_{i,j} ϕ(|A[i,j]|/(a[i]*b[j]))` with no coverage constraints. This is the asymmetric
analog of [`soft_symcover_min`](@ref). The no-ϕ form defaults to `AbsLinear{2}()`,
matching [`soft_cover`](@ref).

The row/column scales are pinned to the balance convention
`∑ nzaᵢ log a[i] = ∑ nzbⱼ log b[j]` (`nzaᵢ`, `nzbⱼ` = nonzero counts of row `i`, column
`j`), imposed within each connected component of the support, as in [`cover_min`](@ref):
the objective depends on `a` and `b` only through the products `a[i]*b[j]`, so without a
convention the split between them would be arbitrary.

Supported ϕ values and required extensions:
- `AbsLog{2}()`: solved natively (no external solver) — the same analytic geometric-mean
  minimum [`cover`](@ref) computes as its initial point. Convex, so the minimizer is unique.
- `AbsLinear{1}()`, `AbsLinear{2}()`: requires JuMP and Ipopt. These objectives are
  non-convex, so the solver returns the minimum of the basin it starts in. Rather than
  commit to one start, these methods refine each of `strategies` — the
  [`initialize_cover`](@ref) menu, by default `$(COVER_MIN_STRATEGIES)`, taken raw
  (`feasible=:none`, since the soft objective constrains nothing) — and return the best
  cover found, at a cost of one solve per start. The result is the best *local* minimum on
  that menu: the multistart is a hedge against a poor basin, not a certificate of global
  optimality.
- `AbsLog{1}()`: not yet implemented. The objective is an LP in log space, but its optimum
  is a face, and the lexicographic AbsLog{2} selection that [`symcover_min`](@ref) uses to
  pin one member of the corresponding hard face does not carry over: the hard face is bounded
  by the coverage constraints, while this one is a level set of an unconstrained piecewise-
  linear objective, across which the quadratic pulls far enough to cost most of the exactly
  tight residuals that make `AbsLog{1}` worth choosing.

Every start on the menu co-varies with a rescaling of `A` and the objective is
scale-invariant, so the selection — and hence the result — is scale-covariant.

See also: [`soft_cover_min!`](@ref), [`soft_symcover_min`](@ref), [`soft_cover`](@ref).
"""
function soft_cover_min end
soft_cover_min(A::AbstractMatrix; kwargs...) = soft_cover_min(AbsLinear{2}(), A; kwargs...)

function soft_cover_min(::AbsLog{2}, A::AbstractMatrix; kwargs...)
    a, b, _ = _soft_cover_min_abslog2(A; kwargs...)
    return a, b
end

# Multistart driver for the non-convex asymmetric soft covers, the unconstrained
# counterpart of `cover_min(::AbsLinear)`. The kernels (`soft_cover_min!`) live in MatrixCoversIpoptExt.
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

Refine the starting point `(a, b)` into the ϕ-minimal asymmetric soft cover of `A`, in
place. This is the asymmetric counterpart of [`soft_symcover_min!`](@ref). The no-ϕ form
defaults to `AbsLinear{2}()`, matching [`soft_cover_min`](@ref), whose supported ϕ values
these methods share.

`a` and `b` must be strictly positive on every supported row and column; scales on
unsupported rows and columns are inert, and are zero on output. As with
[`soft_symcover_min!`](@ref) — and unlike [`cover_min!`](@ref) — the start need *not* cover
`A`. Build one with `feasible=:none`.

The product `a[i]*b[j]` is unchanged by `a -> c*a`, `b -> b/c`, so the start is read only
up to that gauge, and the result is pinned to the balance convention of
[`soft_cover_min`](@ref).

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

# Shared prologue of the asymmetric soft refiners (`soft_cover!`, `soft_cover_min!`); the
# counterpart of `_prepare_soft_symcover_start!`. Positivity only — the soft objective
# constrains nothing — plus the balance pin, so the refiners read the start only up to the
# row/column gauge. `fname` names the caller so the error reports the function the user
# actually called.
function _prepare_soft_cover_start!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix,
                                    fname::Symbol=:soft_cover_min!)
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(eachindex(a)), axes(A, 1)=$(axes(A, 1))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(eachindex(b)), axes(A, 2)=$(axes(A, 2))"))
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
            throw(ArgumentError("$fname requires a start with finite positive scale on every supported row, got a[$i] = $(a[i])"))
    end
    for j in eachindex(b)
        suppb[j] || continue
        (isfinite(b[j]) && b[j] > zero(b[j])) ||
            throw(ArgumentError("$fname requires a start with finite positive scale on every supported column, got b[$j] = $(b[j])"))
    end
    return _balance_cover!(a, b, A)
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

# Relative-objective margin a later start must beat the incumbent by to replace it.
#
# Two candidates that reach the same point differ, between one frame and a rescaled one, only
# by roundoff in evaluating the objective — a sum of O(n²) terms, so of relative size O(n·eps).
# The margin must exceed that, or the selection could flip with the frame and forfeit
# covariance; it must also stay below any genuine basin gap, which is orders of magnitude
# larger. Candidates stopped short of convergence may differ by far more than roundoff, but
# such differences are deterministic and co-vary with the frame, so switching on them is safe.
_multistart_switchtol(::Type{T}) where {T} = 5_000_000 * eps(T)

# Labeled candidate starts for the symmetric AbsLinear{2} multistart, in selection order.
# The deterministic starts are the `initialize_symcover` menu: the geometric mean (raw — it is
# the exact soft AbsLog{2} optimum, so forcing it to feasibility would spoil it — and also the
# perturbation base), the tightened hard cover, the uniformly inflated geometric mean, the
# leave-one-out geometric mean (when a support entry can be dropped), and — only when `A` has
# a zero entry — the greedy feasible cover. The feasible start is gated because on a fully
# dense `A` it never uniquely wins, and "which entries are zero" is invariant under a diagonal
# rescaling `D*A*D`, so the gate keeps the selection scale-covariant. Remaining slots, up to
# `starts` total, are multiplicative log-normal perturbations `a_g .* exp.(σ .* ξ)` of the
# geometric-mean point, `ξ` drawn from `rng` (drawn for every index so the stream is
# frame-independent). `starts` below the number of deterministic starts truncates the list.
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
    switchtol = _multistart_switchtol(float(eltype(objs)))
    for k in eachindex(objs)
        objs[k] < Ebest * (1 - switchtol) && ((besti, Ebest) = (k, objs[k]))
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
function _abslinear2_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=50_000_000 * eps(T)) where T
    ax = eachindex(a)
    ax == axes(A, 1) || throw(DimensionMismatch("row indices of `A` must match `a`, got $(axes(A, 1)) vs $(ax)"))
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
function _abslinear1_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=5000 * eps(T)) where T
    ax  = eachindex(a)
    ax == axes(A, 1) || throw(DimensionMismatch("row indices of `A` must match `a`, got $(axes(A, 1)) vs $(ax)"))
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
function _abslog1_iter!(a::AbstractVector{T}, A::AbstractMatrix, iter::Int; tol::Real=5000 * eps(T)) where T
    ax  = eachindex(a)
    ax == axes(A, 1) || throw(DimensionMismatch("row indices of `A` must match `a`, got $(axes(A, 1)) vs $(ax)"))
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

# Alternating weighted-median descent for the asymmetric AbsLog{1} soft cover, working in
# log space (α = log a, β = log b). Updating α[i] with β fixed minimizes
# ∑_{j: A[i,j]≠0} |α[i] + β[j] - log|A[i,j]||, whose minimizer is the median of the points
# log|A[i,j]| - β[j], one per nonzero entry of row i; the β-update is dual. Row and column
# scales are distinct variables, so no term is self-coupled — the symmetric solver's
# double-weighted diagonal has no counterpart here — and each half-sweep is an exact block
# minimization. The AbsLog{1} minimum is a flat basin; the lower median is chosen for a
# deterministic, scale-covariant result.
#
# Each half-sweep minimizes exactly, but the objective's nonsmoothness couples α[i] with
# β[j], so a fixed point of the sweeps need not minimize it. See `soft_cover`.
#
# `iter` bounds the sweeps; the descent exits early once the largest relative coordinate
# movement in a sweep drops to `tol` (scale-invariant, so covariant restarts of a rescaled
# problem exit on the same sweep). Rows/columns with empty support keep scale 0.
function _abslog1_iter_asym!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix,
                             iter::Int; tol=nothing)
    T = float(promote_type(eltype(a), eltype(b)))
    rtol = tol === nothing ? 5000 * eps(T) : T(tol)
    axr, axc = axes(A, 1), axes(A, 2)
    eachindex(a) == axr || throw(DimensionMismatch("row indices of `A` must match `a`, got $(axr) vs $(eachindex(a))"))
    eachindex(b) == axc || throw(DimensionMismatch("column indices of `A` must match `b`, got $(axc) vs $(eachindex(b))"))
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

# Labeled `(a, b)` candidate starts for the asymmetric AbsLinear{2} multistart, in selection
# order. Deterministic starts: the boosted geometric mean (also the perturbation base) and the
# tightened hard cover, obtained by tightening a copy of the former so the shared passes run
# once. Remaining slots, up to `starts` total, are multiplicative log-normal perturbations
# `a_g .* exp.(σ .* ξ)`, `b_g .* exp.(σ .* η)` of that base, `ξ`/`η` drawn from `rng` (drawn
# for every index so the stream is frame-independent).
function _soft_cover_abslinear2_inits(A::AbstractMatrix, starts::Int, σ::Real, rng)
    T = float(real(eltype(A)))
    ag, bg = initialize_cover(A; strategy=:geomean, feasible=:boost)
    labels = ["boost"]
    inits = [(copy(ag), copy(bg))]
    # The tightened hard cover `cover(A)` is exactly this point tightened, so tighten a copy
    # (at `tighten_cover!`'s own default `maxiter`) rather than recomputing the shared
    # geometric-mean and boost passes.
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
# alternating least squares `_msmc_als!` from the candidate list built by
# `_soft_cover_abslinear2_inits` and returns the pair `_multistart_select` picks. Every start
# co-varies with an independent row/column rescaling `D_r*A*D_c` and the objective is
# scale-invariant, so the selection is scale-covariant; passing the same `rng` state across the
# two frames (as the default fresh-seeded RNG does) makes it reproducible.
function _soft_cover_abslinear2(A::AbstractMatrix, iter::Int, starts::Int, σ::Real, rng;
                                labels=nothing, objs=nothing)
    a, b = _multistart_run(_soft_cover_abslinear2_inits,
                            ((a, b), A, iter) -> _msmc_als!(a, b, A, iter),
                            ((a, b), A) -> cover_objective(AbsLinear{2}(), a, b, A),
                            A, iter, starts, σ, rng; labels, objs)
    # The alternating half-sweeps rescale rows and columns independently, so they leave the
    # gauge where it falls; pin it to the package's convention. The objective cannot see the
    # gauge, so this changes no product a[i]*b[j] and no candidate's score.
    return _balance_cover!(a, b, A)
end

# Alternating least squares for the AbsLinear{2} soft cover in the inverse-scale variables
# u = 1 ./ a, v = 1 ./ b. With M = |A| restricted to its nonzero support, the objective
# E = ∑ (1 - M[i,j] u[i] v[j])² is biconvex; each half-sweep sets u[i] (resp. v[j]) to its
# exact minimizer. Rows/columns with empty support keep scale 0 and are held fixed.
# Refines `a`, `b` in place starting from their incoming values.
#
# Both half-sweeps accumulate over `i` with `j` held fixed, so the support is gathered
# by column once up front and each sweep walks that gather rather than the full grid.
#
# The post-sweep objective costs no extra pass over `A`: once the v-half-sweep has set
# v[j] = num[j]/den[j] from num[j] = ∑_i M[i,j] u[i] and den[j] = ∑_i (M[i,j] u[i])²,
# column j contributes
#     ∑_i (1 - M[i,j] u[i] v[j])² = nnz[j] - 2 v[j] num[j] + v[j]² den[j]
#                                 = nnz[j] - num[j]²/den[j]
# with nnz[j] the number of nonzeros in column j, which is that column's size in the
# gather. An empty column has den[j] = nnz[j] = 0 and contributes nothing.
function _msmc_als!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix, iter::Int;
                    tol=nothing)
    axr, axc = axes(A, 1), axes(A, 2)
    eachindex(a) == axr || throw(DimensionMismatch("row indices of `A` must match `a`, got $(axr) vs $(eachindex(a))"))
    eachindex(b) == axc || throw(DimensionMismatch("column indices of `A` must match `b`, got $(axc) vs $(eachindex(b))"))
    T = float(promote_type(eltype(a), eltype(b), real(eltype(A))))
    # The convergence test is on a relative movement, so its floor is set by the
    # precision of `T`: a fixed Float64-scaled literal can never be reached in
    # Float32 (every call would run to `iter`) and stops far short of what a wider
    # type can resolve.
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

# Soft-cover objective in inverse-scale variables: ∑_{i,j: A[i,j]≠0} (1 - |A[i,j]| u[i] v[j])².
# Takes the column-grouped support rather than the matrix, so the sweeps and the
# objective share one gather.
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

# Alternating weighted-median descent for the asymmetric AbsLinear{1} soft cover.
# Updating a[i] with b fixed minimizes ∑_j |1 - |A[i,j]|/(a[i] b[j])| over a[i] > 0, whose
# minimizer is the weighted median of c_j = |A[i,j]|/b[j] weighted by the same c_j (see
# `_weighted_self_median!`); the b-update is dual. There is no self-coupled diagonal term (the
# (i,i) entry enters the row update through b[i] like any other column), so each full sweep is
# an exact block minimization and the objective decreases monotonically. Refines `a`, `b` in
# place from their incoming values; exits early once the largest relative coordinate movement
# in a sweep drops to `tol` (scale-invariant, so covariant restarts of a rescaled problem exit
# on the same sweep). Rows/columns with empty support keep scale 0.
function _abslinear1_iter_asym!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix,
                                iter::Int; tol=nothing)
    T = float(promote_type(eltype(a), eltype(b)))
    rtol = tol === nothing ? 5000 * eps(T) : T(tol)
    axr, axc = axes(A, 1), axes(A, 2)
    eachindex(a) == axr || throw(DimensionMismatch("row indices of `A` must match `a`, got $(axr) vs $(eachindex(a))"))
    eachindex(b) == axc || throw(DimensionMismatch("column indices of `A` must match `b`, got $(axc) vs $(eachindex(b))"))
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
