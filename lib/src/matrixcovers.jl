# The binding layer: one `@api` entrypoint per exposed `MatrixCovers` call,
# each with the docstring the generated Python, MATLAB, and C interfaces carry.
#
# Optional keywords arrive as `nothing` when the caller omits them. Each
# entrypoint replaces `nothing` by the default `MatrixCovers` itself uses and
# passes every keyword on every call, so that each call site has one fixed set
# of keywords: `juliac --trim=safe` cannot resolve a call whose keyword set
# depends on runtime values. An entrypoint with several optional keywords is an
# `@inline` front, which resolves them into concrete values, and a `@noinline`
# worker that takes only those (see `conversions.jl` for why).

"""
    matrixcovers

The binding layer for `MatrixCovers`. `@api` declares which of the package's
cover solvers, objectives, and predicates a foreign caller may reach. The enums
[`Penalty`](@ref), [`Linsolve`](@ref), [`Start`](@ref), [`SymStrategy`](@ref),
[`Strategy`](@ref), and [`Feasible`](@ref) stand in for the penalty objects and
`Symbol` options of the Julia API.

Every `Matrix{Float64}`/`Vector{Float64}` argument arrives as a zero-copy view
of the caller's memory. None of the wrapped functions writes to its arguments,
and no entrypoint returns an argument, so every returned array is owned by the
caller.
"""
module matrixcovers

using JLWInterop
using MatrixCovers: MatrixCovers, AbsLog, AbsLinear, PowerMean

@export_release_entrypoints

# Each enum lives in its own module so that member names (`auto`, `geomean`,
# `unconstrained`, `hardcover`) may repeat across enums.

module Penalties
"""
    Penalty

The penalty an entrypoint scores or minimizes: `abslog1` (`AbsLog{1}`),
`abslog2` (`AbsLog{2}`), `abslinear1` (`AbsLinear{1}`), `abslinear2`
(`AbsLinear{2}`), `powermean1` (`PowerMean{1}`), or `powermean2`
(`PowerMean{2}`). `cover_objective` evaluates all six; each solver accepts the
subset listed in its documentation.
"""
@enum Penalty::Int32 abslog1 = 1 abslog2 = 2 abslinear1 = 3 abslinear2 = 4 powermean1 = 5 powermean2 = 6
end
using .Penalties: Penalties, Penalty

module Linsolves
"""
    Linsolve

The linear solver of a Newton or least-squares step. The minimal covers,
`cover_transversal`, and the `abslog2` soft covers accept `auto`, `dense`,
`woodbury`, and `lsqr`; the `powermean` soft covers and `noise_cover` accept
`auto`, `dense`, `cholesky`, and `cg`.
"""
@enum Linsolve::Int32 auto = 1 dense = 2 woodbury = 3 lsqr = 4 cholesky = 5 cg = 6
end
using .Linsolves: Linsolves, Linsolve

module Starts
"""
    Start

The starting point of the heuristic `cover`: `auto`, `transversal`,
`unconstrained`, or `geomean`.
"""
@enum Start::Int32 auto = 1 transversal = 2 unconstrained = 3 geomean = 4
end
using .Starts: Starts, Start

module SymStrategies
"""
    SymStrategy

The starting point built by `initialize_symcover`: `hardcover`, `diagonal`,
`leaveout`, or `diagfeasible`.
"""
@enum SymStrategy::Int32 hardcover = 1 diagonal = 2 leaveout = 3 diagfeasible = 4
end
using .SymStrategies: SymStrategies, SymStrategy

module Strategies
"""
    Strategy

The starting point built by `initialize_cover`: `hardcover`, `geomean`, or
`unconstrained`.
"""
@enum Strategy::Int32 hardcover = 1 geomean = 2 unconstrained = 3
end
using .Strategies: Strategies, Strategy

module Feasibles
"""
    Feasible

How `initialize_symcover`/`initialize_cover` make their point a cover:
`inflate` multiplies every scale by the smallest common factor that covers `A`,
`boost` raises only the scales of violated entries, and `none` leaves the point
as it is.
"""
@enum Feasible::Int32 inflate = 1 boost = 2 none = 3
end
using .Feasibles: Feasibles, Feasible

include("conversions.jl")

# --- Heuristic hard covers ----------------------------------------------------

"""
Heuristic symmetric hard cover of the symmetric matrix `A`: `a` with
`a[i]*a[j] >= abs(A[i, j])`. `maxiter` (default 3) is the number of tightening
iterations.
"""
symcover(A::Matrix{Float64}; maxiter::Union{Int64, Nothing} = nothing) =
    MatrixCovers.symcover(A; maxiter = something(maxiter, SYMCOVER_MAXITER))

@api symcover(A::Matrix{Float64}; maxiter::Union{Int64, Nothing} = nothing)::Vector{Float64}

"""
Heuristic hard cover of `A`: `(a, b)` with `a[i]*b[j] >= abs(A[i, j])`.
`start` selects the kind of cover (`auto`, `transversal`, `unconstrained`, or
`geomean`); `maxiter` and `cgiter` default to values that depend on `start`.
"""
cover(A::Matrix{Float64}; maxiter::Union{Int64, Nothing} = nothing,
      cgiter::Union{Int64, Nothing} = nothing, start::Start = Starts.auto) =
    MatrixCovers.cover(A; maxiter, cgiter, start = _symbol(start))

@api cover(A::Matrix{Float64}; maxiter::Union{Int64, Nothing} = nothing,
           cgiter::Union{Int64, Nothing} = nothing,
           start::Start = Starts.auto)::Tuple{Vector{Float64}, Vector{Float64}}

"""
The column scale `b` of the tightest cover of `A` with row scale `a` held
fixed: `b[j] = max_i abs(A[i, j]) / a[i]`.
"""
cover_given_rows(A::Matrix{Float64}, a::Vector{Float64}) = MatrixCovers.cover(A, a)[2]

@api cover_given_rows(A::Matrix{Float64}, a::Vector{Float64})::Vector{Float64}

# --- Minimal hard covers -------------------------------------------------------

"""
The symmetric hard cover `a` of `A` that minimizes `penalty`, followed by the
solver's convergence statistics `converged`, `iterations`, `residual`, and
`tolerance`. Only `abslog2` is solved natively; `abslog1` needs JuMP/HiGHS and
the `abslinear` penalties need JuMP/Ipopt, neither linked into this library.
"""
function symcover_min(A::Matrix{Float64}; penalty::Penalty = Penalties.abslog2,
                      theta::Union{Float64, Nothing} = nothing,
                      maxouter::Union{Int64, Nothing} = nothing,
                      maxiter::Union{Int64, Nothing} = nothing,
                      tol::Union{Float64, Nothing} = nothing,
                      linsolve::Linsolve = Linsolves.auto,
                      fillbudget::Union{Float64, Nothing} = nothing,
                      flopbudget::Union{Float64, Nothing} = nothing)
    @inline
    return _symcover_min(A, penalty, _auglag(theta, maxouter, maxiter, tol, linsolve, fillbudget, flopbudget))
end
@noinline function _symcover_min(A::Matrix{Float64}, penalty::Penalty, opts::NamedTuple)
    _require_abslog2(penalty, "symcover_min")
    a, stats = MatrixCovers.symcover_min(AbsLog{2}(), A; opts...)
    return (a, _stats(stats)...)
end

@api symcover_min(A::Matrix{Float64}; penalty::Penalty = Penalties.abslog2,
                  theta::Union{Float64, Nothing} = nothing,
                  maxouter::Union{Int64, Nothing} = nothing,
                  maxiter::Union{Int64, Nothing} = nothing,
                  tol::Union{Float64, Nothing} = nothing,
                  linsolve::Linsolve = Linsolves.auto,
                  fillbudget::Union{Float64, Nothing} = nothing,
                  flopbudget::Union{Float64, Nothing} = nothing)::SymcoverStatsResult

"""
The hard cover `(a, b)` of `A` that minimizes `penalty`, followed by the
solver's convergence statistics `converged`, `iterations`, `residual`, and
`tolerance`. Only `abslog2` is solved natively; `abslog1` needs JuMP/HiGHS and
the `abslinear` penalties need JuMP/Ipopt, neither linked into this library.
"""
function cover_min(A::Matrix{Float64}; penalty::Penalty = Penalties.abslog2,
                   theta::Union{Float64, Nothing} = nothing,
                   maxouter::Union{Int64, Nothing} = nothing,
                   maxiter::Union{Int64, Nothing} = nothing,
                   tol::Union{Float64, Nothing} = nothing,
                   linsolve::Linsolve = Linsolves.auto,
                   fillbudget::Union{Float64, Nothing} = nothing,
                   flopbudget::Union{Float64, Nothing} = nothing)
    @inline
    return _cover_min(A, penalty, _auglag(theta, maxouter, maxiter, tol, linsolve, fillbudget, flopbudget))
end
@noinline function _cover_min(A::Matrix{Float64}, penalty::Penalty, opts::NamedTuple)
    _require_abslog2(penalty, "cover_min")
    a, b, stats = MatrixCovers.cover_min(AbsLog{2}(), A; opts...)
    return (a, b, _stats(stats)...)
end

@api cover_min(A::Matrix{Float64}; penalty::Penalty = Penalties.abslog2,
               theta::Union{Float64, Nothing} = nothing,
               maxouter::Union{Int64, Nothing} = nothing,
               maxiter::Union{Int64, Nothing} = nothing,
               tol::Union{Float64, Nothing} = nothing,
               linsolve::Linsolve = Linsolves.auto,
               fillbudget::Union{Float64, Nothing} = nothing,
               flopbudget::Union{Float64, Nothing} = nothing)::CoverStatsResult

"""
The `abslog2`-minimal hard cover `(a, b)` of the square matrix `A` among those
tight on its maximum-product transversals, followed by the solver's convergence
statistics `converged`, `iterations`, `residual`, and `tolerance`.
"""
function cover_transversal(A::Matrix{Float64};
                           theta::Union{Float64, Nothing} = nothing,
                           maxouter::Union{Int64, Nothing} = nothing,
                           maxiter::Union{Int64, Nothing} = nothing,
                           tol::Union{Float64, Nothing} = nothing,
                           linsolve::Linsolve = Linsolves.auto,
                           fillbudget::Union{Float64, Nothing} = nothing,
                           flopbudget::Union{Float64, Nothing} = nothing)
    @inline
    return _cover_transversal(A, _auglag(theta, maxouter, maxiter, tol, linsolve, fillbudget, flopbudget))
end
@noinline function _cover_transversal(A::Matrix{Float64}, opts::NamedTuple)
    a, b, stats = MatrixCovers.cover_transversal(A; opts...)
    return (a, b, _stats(stats)...)
end

@api cover_transversal(A::Matrix{Float64};
                       theta::Union{Float64, Nothing} = nothing,
                       maxouter::Union{Int64, Nothing} = nothing,
                       maxiter::Union{Int64, Nothing} = nothing,
                       tol::Union{Float64, Nothing} = nothing,
                       linsolve::Linsolve = Linsolves.auto,
                       fillbudget::Union{Float64, Nothing} = nothing,
                       flopbudget::Union{Float64, Nothing} = nothing)::CoverStatsResult

"""
A hard cover `(a, b)` of the square matrix `A`, tight on its maximum-product
transversals, that pushes the coverage ratios of the other nonzero entries
toward `target` (default 0.01) under the penalty `powermean1` or `powermean2`
(default), followed by the solver's convergence statistics `converged`,
`iterations`, `residual`, and `tolerance`. `maxbalance` (default false)
max-balances the scales within each fully indecomposable block.
"""
function cover_transversal_powermean(A::Matrix{Float64}; penalty::Penalty = Penalties.powermean2,
                                     target::Float64 = 0.01,
                                     maxbalance::Bool = false,
                                     tol::Union{Float64, Nothing} = nothing,
                                     maxiter::Union{Int64, Nothing} = nothing,
                                     fillbudget::Union{Float64, Nothing} = nothing,
                                     flopbudget::Union{Float64, Nothing} = nothing)
    @inline
    # `maxiter = nothing` selects a limit that grows with the size of `A`; the
    # sentinel `-1` carries that choice as an `Int`.
    (isnothing(maxiter) || maxiter >= 0) ||
        throw(ArgumentError("cover_transversal_powermean: maxiter must be nonnegative"))
    return _cover_transversal_powermean(A, penalty, target, maxbalance,
                                        something(tol, MatrixCovers.BALANCED_TOL), something(maxiter, -1),
                                        something(fillbudget, Float64(MatrixCovers.LSQR_FILL_BUDGET)),
                                        something(flopbudget, MatrixCovers.LSQR_FLOP_BUDGET))
end
@noinline function _cover_transversal_powermean(A::Matrix{Float64}, penalty::Penalty, target::Float64,
                                                maxbalance::Bool, tol::Float64, maxiter::Int64,
                                                fillbudget::Float64, flopbudget::Float64)
    if penalty === Penalties.powermean1
        a, b, stats = maxiter < 0 ?
            MatrixCovers.cover_transversal(PowerMean{1}(), A; target, maxbalance, tol, fillbudget, flopbudget) :
            MatrixCovers.cover_transversal(PowerMean{1}(), A; target, maxbalance, tol, maxiter, fillbudget, flopbudget)
        return (a, b, _stats(stats)...)
    end
    penalty === Penalties.powermean2 ||
        throw(ArgumentError("cover_transversal_powermean: penalty must be powermean1 or powermean2"))
    a, b, stats = maxiter < 0 ?
        MatrixCovers.cover_transversal(PowerMean{2}(), A; target, maxbalance, tol, fillbudget, flopbudget) :
        MatrixCovers.cover_transversal(PowerMean{2}(), A; target, maxbalance, tol, maxiter, fillbudget, flopbudget)
    return (a, b, _stats(stats)...)
end

@api cover_transversal_powermean(A::Matrix{Float64}; penalty::Penalty = Penalties.powermean2,
                                 target::Float64 = 0.01,
                                 maxbalance::Bool = false,
                                 tol::Union{Float64, Nothing} = nothing,
                                 maxiter::Union{Int64, Nothing} = nothing,
                                 fillbudget::Union{Float64, Nothing} = nothing,
                                 flopbudget::Union{Float64, Nothing} = nothing)::CoverStatsResult

# --- Soft covers ---------------------------------------------------------------

"""
The symmetric soft cover `a` of `A` that minimizes `penalty` without a coverage
constraint, followed by the solver's convergence statistics `converged`,
`iterations`, `residual`, and `tolerance`. `abslog2` is a single linear solve
and reports no statistics (`None` in Python, `[]` in MATLAB). `powermean1`,
`powermean2` (default), and `abslog2` are solved natively; `abslog1` needs
JuMP/HiGHS and the `abslinear` penalties need JuMP/Ipopt, neither linked into
this library. `maxiter`, `tol`, `newton`, `maxnewton`, and `maxpolish` apply to
the `powermean` penalties, and `fillbudget` and `flopbudget` to `abslog2`.
"""
function soft_symcover(A::Matrix{Float64}; penalty::Penalty = Penalties.powermean2,
                       maxiter::Union{Int64, Nothing} = nothing,
                       tol::Union{Float64, Nothing} = nothing,
                       newton::Union{Bool, Nothing} = nothing,
                       maxnewton::Union{Int64, Nothing} = nothing,
                       maxpolish::Union{Int64, Nothing} = nothing,
                       linsolve::Linsolve = Linsolves.auto,
                       fillbudget::Union{Float64, Nothing} = nothing,
                       flopbudget::Union{Float64, Nothing} = nothing)
    @inline
    return _soft_symcover(A, penalty, _given(maxiter, tol, newton, maxnewton, maxpolish), _given(fillbudget, flopbudget),
                   _powermean(maxiter, newton, maxnewton, maxpolish, linsolve), _powermean_tol(tol),
                   _linsolve_options(linsolve, fillbudget, flopbudget))
end
@noinline function _soft_symcover(A::Matrix{Float64}, penalty::Penalty, powermean_given::Bool, budgets_given::Bool,
                          pmopts::NamedTuple, tol::Float64, lsopts::NamedTuple)
    if penalty === Penalties.abslog2
        powermean_given && throw(ArgumentError("soft_symcover: maxiter, tol, newton, maxnewton, and maxpolish apply only to the powermean penalties"))
        a, stats = MatrixCovers.soft_symcover(AbsLog{2}(), A; lsopts...)
        return (a, _stats(stats)...)
    end
    _require_powermean(penalty, "soft_symcover")
    budgets_given && throw(ArgumentError("soft_symcover: fillbudget and flopbudget apply only to penalty abslog2"))
    if penalty === Penalties.powermean1
        a, stats = MatrixCovers.soft_symcover(PowerMean{1}(), A; tol, pmopts...)
        return (a, _stats(stats)...)
    end
    a, stats = MatrixCovers.soft_symcover(PowerMean{2}(), A; tol, pmopts...)
    return (a, _stats(stats)...)
end

@api soft_symcover(A::Matrix{Float64}; penalty::Penalty = Penalties.powermean2,
                   maxiter::Union{Int64, Nothing} = nothing,
                   tol::Union{Float64, Nothing} = nothing,
                   newton::Union{Bool, Nothing} = nothing,
                   maxnewton::Union{Int64, Nothing} = nothing,
                   maxpolish::Union{Int64, Nothing} = nothing,
                   linsolve::Linsolve = Linsolves.auto,
                   fillbudget::Union{Float64, Nothing} = nothing,
                   flopbudget::Union{Float64, Nothing} = nothing)::SymcoverStatsResult

"""
The soft cover `(a, b)` of `A` that minimizes `penalty` without a coverage
constraint, followed by the solver's convergence statistics `converged`,
`iterations`, `residual`, and `tolerance`. `abslog2` is a single linear solve
and reports no statistics (`None` in Python, `[]` in MATLAB). Penalties and
keywords are as for `soft_symcover`.
"""
function soft_cover(A::Matrix{Float64}; penalty::Penalty = Penalties.powermean2,
                    maxiter::Union{Int64, Nothing} = nothing,
                    tol::Union{Float64, Nothing} = nothing,
                    newton::Union{Bool, Nothing} = nothing,
                    maxnewton::Union{Int64, Nothing} = nothing,
                    maxpolish::Union{Int64, Nothing} = nothing,
                    linsolve::Linsolve = Linsolves.auto,
                    fillbudget::Union{Float64, Nothing} = nothing,
                    flopbudget::Union{Float64, Nothing} = nothing)
    @inline
    return _soft_cover(A, penalty, _given(maxiter, tol, newton, maxnewton, maxpolish), _given(fillbudget, flopbudget),
                   _powermean(maxiter, newton, maxnewton, maxpolish, linsolve), _powermean_tol(tol),
                   _linsolve_options(linsolve, fillbudget, flopbudget))
end
@noinline function _soft_cover(A::Matrix{Float64}, penalty::Penalty, powermean_given::Bool, budgets_given::Bool,
                          pmopts::NamedTuple, tol::Float64, lsopts::NamedTuple)
    if penalty === Penalties.abslog2
        powermean_given && throw(ArgumentError("soft_cover: maxiter, tol, newton, maxnewton, and maxpolish apply only to the powermean penalties"))
        a, b, stats = MatrixCovers.soft_cover(AbsLog{2}(), A; lsopts...)
        return (a, b, _stats(stats)...)
    end
    _require_powermean(penalty, "soft_cover")
    budgets_given && throw(ArgumentError("soft_cover: fillbudget and flopbudget apply only to penalty abslog2"))
    if penalty === Penalties.powermean1
        a, b, stats = MatrixCovers.soft_cover(PowerMean{1}(), A; tol, pmopts...)
        return (a, b, _stats(stats)...)
    end
    a, b, stats = MatrixCovers.soft_cover(PowerMean{2}(), A; tol, pmopts...)
    return (a, b, _stats(stats)...)
end

@api soft_cover(A::Matrix{Float64}; penalty::Penalty = Penalties.powermean2,
                maxiter::Union{Int64, Nothing} = nothing,
                tol::Union{Float64, Nothing} = nothing,
                newton::Union{Bool, Nothing} = nothing,
                maxnewton::Union{Int64, Nothing} = nothing,
                maxpolish::Union{Int64, Nothing} = nothing,
                linsolve::Linsolve = Linsolves.auto,
                fillbudget::Union{Float64, Nothing} = nothing,
                flopbudget::Union{Float64, Nothing} = nothing)::CoverStatsResult

"""
Scales `(a, b)` for a data matrix `Y` whose entries have noise variances `V`:
every nonzero row and column of `Y ./ (a .* b')` has average noise variance one.
The scales are followed by the solver's convergence statistics `converged`,
`iterations`, `residual`, and `tolerance`. `shrink` (default 0) blends `V` with
its rank-one fit, which guarantees that the scales exist; with `shrink = 0`,
some zero patterns of `V` admit no scales and raise an error that describes the
obstruction. The remaining keywords are the `powermean` keywords of
`soft_cover`.
"""
function noise_cover(V::Matrix{Float64}; shrink::Float64 = 0.0,
                     maxiter::Union{Int64, Nothing} = nothing,
                     tol::Union{Float64, Nothing} = nothing,
                     newton::Union{Bool, Nothing} = nothing,
                     maxnewton::Union{Int64, Nothing} = nothing,
                     maxpolish::Union{Int64, Nothing} = nothing,
                     linsolve::Linsolve = Linsolves.auto)
    @inline
    return _noise_cover(V, shrink, _powermean_tol(tol), _powermean(maxiter, newton, maxnewton, maxpolish, linsolve))
end
@noinline function _noise_cover(V::Matrix{Float64}, shrink::Float64, tol::Float64, opts::NamedTuple)
    a, b, stats = try
        MatrixCovers.noise_cover(V; shrink, tol, opts...)
    catch err
        # `@api` reports an exception type it does not recognize by its name
        # alone; an `ErrorException` carries the explanation to the caller.
        err isa MatrixCovers.SolverFailure && throw(ErrorException(err.msg))
        rethrow()
    end
    return (a, b, _stats(stats)...)
end

@api noise_cover(V::Matrix{Float64}; shrink::Float64 = 0.0,
                 maxiter::Union{Int64, Nothing} = nothing,
                 tol::Union{Float64, Nothing} = nothing,
                 newton::Union{Bool, Nothing} = nothing,
                 maxnewton::Union{Int64, Nothing} = nothing,
                 maxpolish::Union{Int64, Nothing} = nothing,
                 linsolve::Linsolve = Linsolves.auto)::CoverStatsResult

# --- Starting points -----------------------------------------------------------

"""
A symmetric starting point `a` for `symcover_min` or `soft_symcover`. `strategy`
selects the point: `hardcover` (default) is the result of `symcover`; `diagonal`
is the starting point of `symcover`, built from the diagonally normalized
entries; `leaveout` is the `diagonal` point recomputed without the entry it
overestimates most; and `diagfeasible` grows a cover from the diagonal by
nearest-neighbor propagation. `feasible` says how a point other than
`hardcover`, which is already a cover, is made into one: `inflate` (default)
multiplies every scale by the smallest common factor that covers `A`; `boost`
splits each violated entry's shortfall, in log magnitude, between its two scales
and raises each scale by the largest share it is asked for; `none` returns the
point without a coverage guarantee. `maxiter` applies to `hardcover` only.
"""
function initialize_symcover(A::Matrix{Float64}; strategy::SymStrategy = SymStrategies.hardcover,
                             feasible::Feasible = Feasibles.inflate,
                             maxiter::Union{Int64, Nothing} = nothing)
    fs = _symbol(feasible)
    if strategy === SymStrategies.hardcover
        return MatrixCovers.initialize_symcover(A; strategy = :hardcover, feasible = fs,
                                                maxiter = something(maxiter, SYMCOVER_MAXITER))
    end
    isnothing(maxiter) || throw(ArgumentError("initialize_symcover: maxiter applies only to strategy hardcover"))
    return MatrixCovers.initialize_symcover(A; strategy = _symbol(strategy), feasible = fs)
end

@api initialize_symcover(A::Matrix{Float64}; strategy::SymStrategy = SymStrategies.hardcover,
                         feasible::Feasible = Feasibles.inflate,
                         maxiter::Union{Int64, Nothing} = nothing)::Vector{Float64}

"""
An asymmetric starting point `(a, b)` for `cover_min` or `soft_cover`.
`strategy` selects the point: `hardcover` (default) is the result of `cover`;
`geomean` uses the geometric means of the nonzero entries in each row and
column; and `unconstrained` is the scale-covariant start of `cover` that
approximates the unconstrained least-squares fit. `feasible` is as for
`initialize_symcover`. `maxiter`, `cgiter`, and `start` apply to `hardcover`
only and have the meanings they have for `cover`.
"""
function initialize_cover(A::Matrix{Float64}; strategy::Strategy = Strategies.hardcover,
                          feasible::Feasible = Feasibles.inflate,
                          maxiter::Union{Int64, Nothing} = nothing,
                          cgiter::Union{Int64, Nothing} = nothing,
                          start::Start = Starts.auto)
    fs = _symbol(feasible)
    if strategy === Strategies.hardcover
        return MatrixCovers.initialize_cover(A; strategy = :hardcover, feasible = fs,
                                             maxiter, cgiter, start = _symbol(start))
    end
    (isnothing(maxiter) && isnothing(cgiter) && start === Starts.auto) ||
        throw(ArgumentError("initialize_cover: maxiter, cgiter, and start apply only to strategy hardcover"))
    return MatrixCovers.initialize_cover(A; strategy = _symbol(strategy), feasible = fs)
end

@api initialize_cover(A::Matrix{Float64}; strategy::Strategy = Strategies.hardcover,
                      feasible::Feasible = Feasibles.inflate,
                      maxiter::Union{Int64, Nothing} = nothing,
                      cgiter::Union{Int64, Nothing} = nothing,
                      start::Start = Starts.auto)::Tuple{Vector{Float64}, Vector{Float64}}

# --- Gram covers -----------------------------------------------------------------

"Symmetric cover `s` of the Gram matrix `A'*A`."
gramcover(A::Matrix{Float64}) = MatrixCovers.gramcover(A)

@api gramcover(A::Matrix{Float64})::Vector{Float64}

"Symmetric cover `s` of the weighted Gram matrix `A'*diag(w)*A`."
gramcover_weighted(A::Matrix{Float64}, w::Vector{Float64}) = MatrixCovers.gramcover(A, w)

@api gramcover_weighted(A::Matrix{Float64}, w::Vector{Float64})::Vector{Float64}

"""
Symmetric cover `s` of the weighted Gram matrix `A'*W*A` for a general
(possibly indefinite or asymmetric) `W`, using the default symmetric cover `v`
of `abs(W)`.
"""
gramcover_matrix(A::Matrix{Float64}, W::Matrix{Float64}) = MatrixCovers.gramcover(A, W)

@api gramcover_matrix(A::Matrix{Float64}, W::Matrix{Float64})::Vector{Float64}

"""
Symmetric cover `s` of the weighted Gram matrix `A'*W*A` for a general `W`,
given `v` with `abs(W[i, k]) <= v[i]*v[k]`.
"""
gramcover_matrix_v(A::Matrix{Float64}, W::Matrix{Float64}, v::Vector{Float64}) = MatrixCovers.gramcover(A, W; v)

@api gramcover_matrix_v(A::Matrix{Float64}, W::Matrix{Float64}, v::Vector{Float64})::Vector{Float64}

# --- Predicates and objectives ----------------------------------------------------

"Whether `a*a'` covers `A`: `a[i]*a[j] >= abs(A[i, j])*(1 - rtol) - atol`."
iscover_sym(a::Vector{Float64}, A::Matrix{Float64}; rtol::Float64 = 0.0, atol::Float64 = 0.0) =
    MatrixCovers.iscover(a, A; rtol, atol)

@api iscover_sym(a::Vector{Float64}, A::Matrix{Float64}; rtol::Float64 = 0.0, atol::Float64 = 0.0)::Bool

"Whether `a*b'` covers `A`: `a[i]*b[j] >= abs(A[i, j])*(1 - rtol) - atol`."
iscover_ab(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64}; rtol::Float64 = 0.0, atol::Float64 = 0.0) =
    MatrixCovers.iscover(a, b, A; rtol, atol)

@api iscover_ab(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64};
                rtol::Float64 = 0.0, atol::Float64 = 0.0)::Bool

"The largest coverage ratio `abs(A[i,j])/(a[i]*a[j])` over the nonzero entries of `A`."
max_coverage_ratio_sym(a::Vector{Float64}, A::Matrix{Float64}) = Float64(MatrixCovers.max_coverage_ratio(a, A))

@api max_coverage_ratio_sym(a::Vector{Float64}, A::Matrix{Float64})::Float64

"The largest coverage ratio `abs(A[i,j])/(a[i]*b[j])` over the nonzero entries of `A`."
max_coverage_ratio_ab(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64}) =
    Float64(MatrixCovers.max_coverage_ratio(a, b, A))

@api max_coverage_ratio_ab(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64})::Float64

"""
`(level, dispersion)`: the mean and standard deviation of
`log((a[i]*a[j]) / (aref[i]*aref[j]))` over the nonzero entries of `A`.
"""
function level_dispersion_sym(a::Vector{Float64}, aref::Vector{Float64}, A::Matrix{Float64})
    ld = MatrixCovers.level_dispersion(a, aref, A)
    return (Float64(ld.level), Float64(ld.dispersion))
end

@api level_dispersion_sym(a::Vector{Float64}, aref::Vector{Float64}, A::Matrix{Float64})::Tuple{Float64, Float64}

"""
`(level, dispersion)`: the mean and standard deviation of
`log((a[i]*b[j]) / (aref[i]*bref[j]))` over the nonzero entries of `A`.
"""
function level_dispersion_ab(a::Vector{Float64}, b::Vector{Float64}, aref::Vector{Float64},
                             bref::Vector{Float64}, A::Matrix{Float64})
    ld = MatrixCovers.level_dispersion((a, b), (aref, bref), A)
    return (Float64(ld.level), Float64(ld.dispersion))
end

@api level_dispersion_ab(a::Vector{Float64}, b::Vector{Float64}, aref::Vector{Float64},
                         bref::Vector{Float64}, A::Matrix{Float64})::Tuple{Float64, Float64}

"`sum(penalty(log(a[i]*a[j]) - log(abs(A[i,j]))))` over the nonzero entries of `A`."
cover_objective_sym(a::Vector{Float64}, A::Matrix{Float64}; penalty::Penalty = Penalties.abslog2) =
    _cover_objective(penalty, a, A)

@api cover_objective_sym(a::Vector{Float64}, A::Matrix{Float64}; penalty::Penalty = Penalties.abslog2)::Float64

"`sum(penalty(log(a[i]*b[j]) - log(abs(A[i,j]))))` over the nonzero entries of `A`."
cover_objective_ab(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64}; penalty::Penalty = Penalties.abslog2) =
    _cover_objective(penalty, a, b, A)

@api cover_objective_ab(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64};
                        penalty::Penalty = Penalties.abslog2)::Float64

end # module
