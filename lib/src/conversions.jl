# Conversions between the C-facing arguments of the entrypoints and the
# arguments of `MatrixCovers`.
#
# Omitted options are replaced by the defaults `MatrixCovers` uses for
# `Float64` input, read from the package's own constants so the two stay in
# step. Every keyword is then passed with a concrete value on every call.
#
# The helpers that take optional (`Union{T,Nothing}`) values are `@inline`: a
# call that is not inlined must be compiled for every combination of `nothing`
# and non-`nothing` arguments, and `juliac --trim=safe` rejects a call with
# more combinations than inference splits.

# The default of `MatrixCovers.symcover`'s `maxiter`.
const SYMCOVER_MAXITER = 3

# The convergence statistics returned alongside a solver's result:
# `converged`, `iterations`, `residual`, and `tolerance`. All four are
# `nothing` for a solver without a stopping criterion.
const SymcoverStatsResult = Tuple{Vector{Float64}, Union{Bool, Nothing}, Union{Int64, Nothing},
                                  Union{Float64, Nothing}, Union{Float64, Nothing}}
const CoverStatsResult = Tuple{Vector{Float64}, Vector{Float64}, Union{Bool, Nothing}, Union{Int64, Nothing},
                               Union{Float64, Nothing}, Union{Float64, Nothing}}

_stats(s::MatrixCovers.AbstractCoverStats) =
    (MatrixCovers.converged(s)::Bool, Int64(MatrixCovers.iterations(s)),
     Float64(MatrixCovers.residual(s)), Float64(MatrixCovers.tolerance(s)))
_stats(::MatrixCovers.LeastSquaresStats) = (nothing, nothing, nothing, nothing)

# --- Enums to Symbols ---------------------------------------------------------

function _symbol(x::Linsolve)
    x === Linsolves.auto && return :auto
    x === Linsolves.dense && return :dense
    x === Linsolves.woodbury && return :woodbury
    x === Linsolves.lsqr && return :lsqr
    x === Linsolves.cholesky && return :cholesky
    return :cg
end

function _symbol(x::Start)
    x === Starts.auto && return :auto
    x === Starts.transversal && return :transversal
    x === Starts.unconstrained && return :unconstrained
    return :geomean
end

function _symbol(x::SymStrategy)
    x === SymStrategies.hardcover && return :hardcover
    x === SymStrategies.diagonal && return :diagonal
    x === SymStrategies.leaveout && return :leaveout
    return :diagfeasible
end

function _symbol(x::Strategy)
    x === Strategies.hardcover && return :hardcover
    x === Strategies.geomean && return :geomean
    return :unconstrained
end

function _symbol(x::Feasible)
    x === Feasibles.inflate && return :inflate
    x === Feasibles.boost && return :boost
    return :none
end

# --- Penalties ------------------------------------------------------------------

const _NEEDS_HIGHS = "requires JuMP and HiGHS (the MatrixCoversJuMPExt extension), which are not linked into this library"
const _NEEDS_IPOPT = "require JuMP and Ipopt (the MatrixCoversIpoptExt extension), which are not linked into this library"

function _unsupported(penalty::Penalty, fname::String)
    penalty === Penalties.abslog1 && return ArgumentError(fname * ": penalty abslog1 " * _NEEDS_HIGHS)
    (penalty === Penalties.abslinear1 || penalty === Penalties.abslinear2) &&
        return ArgumentError(fname * ": the abslinear penalties " * _NEEDS_IPOPT)
    penalty === Penalties.abslog2 && return ArgumentError(fname * ": penalty abslog2 is not supported")
    penalty === Penalties.powermean1 && return ArgumentError(fname * ": penalty powermean1 is not supported")
    return ArgumentError(fname * ": penalty powermean2 is not supported")
end

_require_abslog2(penalty::Penalty, fname::String) =
    penalty === Penalties.abslog2 || throw(_unsupported(penalty, fname))

_require_powermean(penalty::Penalty, fname::String) =
    (penalty === Penalties.powermean1 || penalty === Penalties.powermean2) || throw(_unsupported(penalty, fname))

# Whether any of the options was given. Options that apply to one penalty
# only are rejected for the others, as `MatrixCovers` itself does.
@inline _given(xs::Vararg{Any, N}) where {N} = any(!isnothing, xs)

# --- Solver options -------------------------------------------------------------

@inline _linsolve_options(linsolve::Linsolve, fillbudget::Union{Float64, Nothing}, flopbudget::Union{Float64, Nothing}) =
    (; linsolve = _symbol(linsolve),
     fillbudget = something(fillbudget, Float64(MatrixCovers.LSQR_FILL_BUDGET)),
     flopbudget = something(flopbudget, Float64(MatrixCovers.LSQR_FLOP_BUDGET)))

@inline _auglag(theta, maxouter, maxiter, tol, linsolve, fillbudget, flopbudget) =
    (; θ = something(theta, Float64(MatrixCovers.AL_PENALTY)),
     maxouter = something(maxouter, MatrixCovers.AL_MAXOUTER),
     maxiter = something(maxiter, MatrixCovers.AL_MAXITER),
     tol = something(tol, MatrixCovers.AL_TOL_EPS * eps(Float64)),
     _linsolve_options(linsolve, fillbudget, flopbudget)...)

@inline _powermean(maxiter, newton, maxnewton, maxpolish, linsolve) =
    (; maxiter = something(maxiter, MatrixCovers.POWERMEAN_MAXITER),
     newton = something(newton, true),
     maxnewton = something(maxnewton, MatrixCovers.POWERMEAN_MAXNEWTON),
     maxpolish = something(maxpolish, MatrixCovers.POWERMEAN_MAXPOLISH),
     linsolve = _symbol(linsolve))

@inline _powermean_tol(tol::Union{Float64, Nothing}) = something(tol, MatrixCovers.POWERMEAN_TOL_EPS * eps(Float64))

# --- Objectives -----------------------------------------------------------------

function _cover_objective(penalty::Penalty, args::Vararg{Any, N}) where {N}
    penalty === Penalties.abslog1 && return Float64(MatrixCovers.cover_objective(AbsLog{1}(), args...))
    penalty === Penalties.abslog2 && return Float64(MatrixCovers.cover_objective(AbsLog{2}(), args...))
    penalty === Penalties.abslinear1 && return Float64(MatrixCovers.cover_objective(AbsLinear{1}(), args...))
    penalty === Penalties.abslinear2 && return Float64(MatrixCovers.cover_objective(AbsLinear{2}(), args...))
    penalty === Penalties.powermean1 && return Float64(MatrixCovers.cover_objective(PowerMean{1}(), args...))
    return Float64(MatrixCovers.cover_objective(PowerMean{2}(), args...))
end
