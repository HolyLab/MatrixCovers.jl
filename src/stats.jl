# Statistics returned as the last output of the optimizing cover solvers.

"""
    AbstractCoverStats

Supertype of the statistics that the optimizing cover solvers return as their
last output, e.g. `a, stats = symcover_min(A)` or `a, b, stats = cover_min(A)`.
Each solver family has its own subtype, documented with its fields.

A subtype implements those of [`converged`](@ref MatrixCovers.converged),
[`iterations`](@ref MatrixCovers.iterations),
[`residual`](@ref MatrixCovers.residual), and
[`tolerance`](@ref MatrixCovers.tolerance) that apply to its solver. A solver
without a stopping criterion implements none of them, so calling one throws a
`MethodError`.
"""
abstract type AbstractCoverStats end

"""
    converged(stats::AbstractCoverStats) -> Bool

Report whether the returned cover meets the solver's stopping criterion.
"""
function converged end

"""
    iterations(stats::AbstractCoverStats) -> Int

Return the number of iterations of the solver's main loop; each stats type
states which loop that is.
"""
function iterations end

"""
    residual(stats::AbstractCoverStats)

Return the residual of the returned cover that the stopping criterion compares
with [`tolerance(stats)`](@ref MatrixCovers.tolerance).
"""
function residual end

"""
    tolerance(stats::AbstractCoverStats)

Return the tolerance that [`residual(stats)`](@ref MatrixCovers.residual) is
compared with.
"""
function tolerance end

"""
    ActiveSetStats{T}

Result of the active-set method that finishes an `AbsLog{2}` minimal-cover
solve whose multiplier iteration stopped before converging.

# Fields

- `certified::Bool`: the KKT conditions hold to the tolerances below, so the
  result is the minimizer to rounding error. When `false`, the solver returns
  the multiplier iterate instead.
- `nsteps::Int`: active-set steps, each a graph-Laplacian solve.
- `primal::T`, `ptol::T`: the largest constraint violation of the result, in
  the log domain, and its tolerance.
- `dual::T`, `dtol::T`: the magnitude of the most negative multiplier of an
  active constraint, and its tolerance.

`primal` and `dual` are `NaN` when `certified` is `false`.
"""
struct ActiveSetStats{T}
    certified::Bool
    nsteps::Int
    primal::T
    ptol::T
    dual::T
    dtol::T
end

"""
    AugmentedLagrangianStats{T} <: AbstractCoverStats

Statistics of the native `AbsLog{2}` minimal-cover solver, returned by
[`symcover_min`](@ref), [`cover_min`](@ref), [`cover_transversal`](@ref), and
their mutating forms.

The multiplier iteration stops when its KKT residual reaches `tol`. If it stops
earlier, the active-set method described by `polish` finishes the solve. The
accessors describe the returned cover: `converged` is `true` when either stage
met its criterion, and `residual` and `tolerance` are those of the stage that
produced the result (`polish.primal` and `polish.ptol` after a certified finish,
otherwise `last(kkt)` and `tol`). `iterations` counts multiplier updates.

# Fields

- `converged::Bool`, `tol::T`: as above.
- `nouter::Int`: multiplier updates.
- `kkt::Vector{T}`, `κs::Vector{T}`, `exits::Vector{Symbol}`, `drops::Vector{T}`:
  per update, the KKT residual after the update, the penalty weight, how the
  inner minimization ended, and its last relative decrease of the objective.
- `polish::Union{Nothing,ActiveSetStats{T}}`: the active-set finish, or
  `nothing` if it did not run.
- `linsolve::Symbol`: the inner solver (`:dense`, `:woodbury`, or `:lsqr`).
- The remaining fields count the inner linear-algebra work and describe the
  preconditioner; their names and meanings may change between releases:
  `nsolves`, `lsqriters`, `lsqrtrace`, `cgiters`, `cholsolves`, `precond`,
  `nrefactor`, `nforest`, `ndiagonal`, `fill_entries`, `factor_flops`,
  `nzeroed`.
"""
struct AugmentedLagrangianStats{T} <: AbstractCoverStats
    converged::Bool
    tol::T
    nouter::Int
    kkt::Vector{T}
    κs::Vector{T}
    exits::Vector{Symbol}
    drops::Vector{T}
    polish::Union{Nothing,ActiveSetStats{T}}
    linsolve::Symbol
    nsolves::Int
    lsqriters::Int
    lsqrtrace::Vector{Int}
    cgiters::Int
    cholsolves::Int
    precond::Symbol
    nrefactor::Int
    nforest::Int
    ndiagonal::Int
    fill_entries::Int
    factor_flops::Float64
    nzeroed::Int
end

function _certified(s::AugmentedLagrangianStats)
    p = s.polish
    return p !== nothing && p.certified
end

converged(s::AugmentedLagrangianStats) = s.converged
iterations(s::AugmentedLagrangianStats) = s.nouter
residual(s::AugmentedLagrangianStats{T}) where {T} =
    _certified(s) ? s.polish.primal : isempty(s.kkt) ? T(NaN) : last(s.kkt)
tolerance(s::AugmentedLagrangianStats) = _certified(s) ? s.polish.ptol : s.tol

"""
    LeastSquaresStats <: AbstractCoverStats

Statistics of the soft `AbsLog{2}` covers ([`soft_symcover`](@ref) and
[`soft_cover`](@ref) with `AbsLog{2}()`), which are computed by one linear
least-squares solve. There is no stopping criterion, so none of the accessors
of [`AbstractCoverStats`](@ref MatrixCovers.AbstractCoverStats) apply.

# Fields

- `linsolve::Symbol`: the solver (`:dense`, `:woodbury`, or `:lsqr`).
- The remaining fields count the linear-algebra work and describe the
  preconditioner, as for
  [`AugmentedLagrangianStats`](@ref MatrixCovers.AugmentedLagrangianStats); their
  names and meanings may change between releases: `lsqriters`, `cgiters`,
  `cholsolves`, `precond`, `fill_entries`, `factor_flops`.
"""
struct LeastSquaresStats <: AbstractCoverStats
    linsolve::Symbol
    lsqriters::Int
    cgiters::Int
    cholsolves::Int
    precond::Symbol
    fill_entries::Int
    factor_flops::Float64
end

# Conversions from the statistics tuple of `_abslog2_auglag`, as amended by the
# minimal-cover workers. The workers compute in at least `Float64`.
_auglag_stats_type(A::AbstractMatrix) = promote_type(float(real(eltype(A))), Float64)

AugmentedLagrangianStats(A::AbstractMatrix, nt::NamedTuple) =
    AugmentedLagrangianStats{_auglag_stats_type(A)}(nt)

function AugmentedLagrangianStats{T}(nt::NamedTuple) where {T}
    p = nt.polish
    polish = p === nothing ? nothing :
             ActiveSetStats{T}(p.certified, p.nsteps, p.primal, p.ptol, p.dual, p.dtol)
    return AugmentedLagrangianStats{T}(nt.converged, nt.tol, nt.nouter, collect(T, nt.kkt),
                                       collect(T, nt.κs), collect(Symbol, nt.exits),
                                       collect(T, nt.drops), polish, nt.linsolve, nt.nsolves,
                                       nt.lsqriters, collect(Int, nt.lsqrtrace), nt.cgiters,
                                       nt.cholsolves, nt.precond, nt.nrefactor, nt.nforest,
                                       nt.ndiagonal, nt.fill_entries, nt.factor_flops, nt.nzeroed)
end

LeastSquaresStats(nt::NamedTuple) =
    LeastSquaresStats(nt.linsolve, nt.lsqriters, nt.cgiters, nt.cholsolves, nt.precond,
                      nt.fill_entries, nt.factor_flops)
