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

"""
    PowerMeanStats{T} <: AbstractCoverStats

Statistics of the `PowerMean` soft covers ([`soft_symcover`](@ref),
[`soft_cover`](@ref), and their mutating forms with a `PowerMean` penalty or no
penalty).

The imbalance of a nonzero row (or column) is `|mean(r^p) - 1|`, the mean taken
over the ratios `r` of its nonzero entries. The solve stops when the largest
imbalance is at most `tol`. Power-mean updates run first; when they are slow,
Newton steps finish the solve. `converged`, `residual`, and `tolerance` return
the fields of the same names (`residual` returns `imbalance`); `iterations`
returns `nsweeps + nnewton`.

# Fields

- `converged::Bool`: `imbalance <= tol`.
- `tol::T`: the stopping tolerance.
- `imbalance::T`: the largest imbalance of the returned cover.
- `nsweeps::Int`: power-mean updates. For the asymmetric solver each counts an
  update of all rows and then all columns (a sweep); for the symmetric solver,
  one simultaneous update of all scales.
- `newton::Bool`: whether Newton steps ran.
- `nnewton::Int`: Newton steps.
- The remaining fields count the work of the Newton stage; their names and
  meanings may change between releases: `npasses` (passes over the support),
  `nfactor` (factorizations), and `linsolve` (the solver for the Newton
  equations: `:dense`, `:cholesky`, or `:cg`, and `:none` when Newton did not
  run).
"""
struct PowerMeanStats{T} <: AbstractCoverStats
    converged::Bool
    tol::T
    imbalance::T
    nsweeps::Int
    newton::Bool
    nnewton::Int
    npasses::Int
    nfactor::Int
    linsolve::Symbol
end

converged(s::PowerMeanStats) = s.converged
iterations(s::PowerMeanStats) = s.nsweeps + s.nnewton
residual(s::PowerMeanStats) = s.imbalance
tolerance(s::PowerMeanStats) = s.tol

# Conversion from the statistics tuple of `_powermean_solve!` and
# `_powermean_symsolve!`; `T` is their working type.
PowerMeanStats{T}(nt::NamedTuple) where {T} =
    PowerMeanStats{T}(nt.converged, nt.tol, nt.imbalance, nt.nsweeps, nt.newton, nt.nnewton,
                      nt.npasses, nt.nfactor, nt.linsolve)

"""
    ExternalSolverStats <: AbstractCoverStats

Statistics of the covers computed by an external optimizer through JuMP: HiGHS
for `AbsLog{1}` ([`symcover_min`](@ref), [`cover_min`](@ref),
[`soft_symcover`](@ref), [`soft_cover`](@ref), and their mutating forms), and
Ipopt for one local refinement under `AbsLinear` ([`symcover_min!`](@ref),
[`cover_min!`](@ref), [`soft_symcover!`](@ref), and [`soft_cover!`](@ref)).

A solve may consist of several stages, each an optimization of one model. Ipopt
runs one. HiGHS runs two unless `A` has no nonzero entries: a linear program
for the `AbsLog{1}` objective, then a quadratic program that selects the point
of least `AbsLog{2}` objective among its minimizers. The fields hold one entry
per stage, in order.

A stage that does not end with a solved status throws a
[`SolverFailure`](@ref), so every status recorded here is a solved one and
`converged` returns `true`. The accessors `iterations`, `residual`, and
`tolerance` are not implemented.

# Fields

- `solver::Symbol`: `:HiGHS` or `:Ipopt`.
- `status::Vector{Symbol}`: the termination status of each stage, as the name of
  the `MathOptInterface.TerminationStatusCode` (`:OPTIMAL` or `:LOCALLY_SOLVED`).
- `objective::Vector{Float64}`: the optimal value of each stage's model. The
  models work in log scales and may include slack variables or constant terms,
  so this need not equal [`cover_objective`](@ref) of the returned cover.
- `niters::Vector{Int}`: the iterations of each stage as the solver reports them
  (for HiGHS, its simplex, interior-point, and QP iterations; for Ipopt, its
  interior-point iterations).
- `solvetime::Vector{Float64}`: the solver's run time of each stage, in seconds.

`niters` and `solvetime` may change meaning between releases.
"""
struct ExternalSolverStats <: AbstractCoverStats
    solver::Symbol
    status::Vector{Symbol}
    objective::Vector{Float64}
    niters::Vector{Int}
    solvetime::Vector{Float64}
end

# Termination statuses accepted as solved, by name, so that the core package
# does not depend on MathOptInterface.
const SOLVED_STATUSES = (:OPTIMAL, :LOCALLY_SOLVED)

converged(s::ExternalSolverStats) = all(in(SOLVED_STATUSES), s.status)

"""
    MultistartStats{S,T} <: AbstractCoverStats

Statistics of a cover selected from several local refinements: the `AbsLinear`
forms of [`symcover_min`](@ref) and [`cover_min`](@ref), which refine one start
per entry of `strategies`, and of [`soft_symcover`](@ref) and
[`soft_cover`](@ref), which refine deterministic and random starts. The cover
with the least objective is returned.

A failed refinement of a random start (one that throws
[`SolverFailure`](@ref)) is left out of the selection and recorded in `failed`;
any other failure throws. The accessors [`converged`](@ref MatrixCovers.converged),
[`iterations`](@ref MatrixCovers.iterations),
[`residual`](@ref MatrixCovers.residual), and
[`tolerance`](@ref MatrixCovers.tolerance) return those of the selected start,
`stats[selected]`, where implemented for `S`.

# Fields

- `labels::Vector{Symbol}`: the starts whose refinement succeeded, in the order
  they ran: the strategy names, or for the soft covers names such as `:geomean`
  and `:rand1`.
- `stats::Vector{S}`: the statistics of each of these refinements.
- `objectives::Vector{T}`: the [`cover_objective`](@ref) of each refined cover.
- `selected::Int`: the index (into the vectors above) of the returned cover.
- `failed::Vector{Symbol}`: the labels of the random starts whose refinement
  failed.
"""
struct MultistartStats{S<:AbstractCoverStats,T} <: AbstractCoverStats
    labels::Vector{Symbol}
    stats::Vector{S}
    objectives::Vector{T}
    selected::Int
    failed::Vector{Symbol}

    function MultistartStats{S,T}(labels, stats, objectives, selected, failed) where {S,T}
        length(labels) == length(stats) == length(objectives) ||
            throw(DimensionMismatch("labels, stats, and objectives must have equal lengths; got $(length(labels)), $(length(stats)), and $(length(objectives))"))
        selected in eachindex(labels) ||
            throw(ArgumentError("selected=$selected is not an index of the $(length(labels)) refined starts"))
        return new{S,T}(labels, stats, objectives, selected, failed)
    end
end

_selected(s::MultistartStats) = s.stats[s.selected]
converged(s::MultistartStats) = converged(_selected(s))
iterations(s::MultistartStats) = iterations(_selected(s))
residual(s::MultistartStats) = residual(_selected(s))
tolerance(s::MultistartStats) = tolerance(_selected(s))
