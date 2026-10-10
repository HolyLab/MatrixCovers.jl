# The binding layer: one `@api` entrypoint per exposed `MatrixCovers` call,
# each with the docstring the generated Python and C interfaces carry. Calls
# `juliac --trim=safe` cannot resolve in their natural form are reshaped in
# `trimmability.jl`; entrypoints that need such a call delegate to it.

"""
    matrixcovers

The binding layer for [`MatrixCovers`](@ref). `@api` declares which of the
package's cover solvers, objective, and predicate a foreign caller may reach;
[`Penalty`](@ref) and [`Linsolve`](@ref) select the penalty and inner linear
solve for the entrypoints that need them.

Every `Matrix{Float64}`/`Vector{Float64}` argument arrives as a zero-copy view
of the caller's memory. None of the wrapped `MatrixCovers` functions mutate a
caller-supplied array, so no entrypoint copies its input; a hard-cover
entrypoint that needs scratch storage for its result allocates that storage
itself.
"""
module matrixcovers

using JLWInterop
using MatrixCovers: MatrixCovers, AbsLog, AbsLinear
using Random: MersenneTwister

@export_release_entrypoints

"""
    Penalty

Which cover penalty an entrypoint scores or minimizes against: `abslog1`
(`AbsLog{1}`), `abslog2` (`AbsLog{2}`), `abslinear1` (`AbsLinear{1}`), or
`abslinear2` (`AbsLinear{2}`).
"""
@enum Penalty::Int32 abslog1 = 1 abslog2 = 2 abslinear1 = 3 abslinear2 = 4

"""
    Linsolve

The inner linear solve [`symcover_min`](@ref)/[`cover_min`](@ref) uses for
`AbsLog{2}`: `auto` (dense, for a dense matrix), `dense`, or `lsqr`
(matrix-free, intended for large sparse supports).
"""
@enum Linsolve::Int32 auto = 1 dense = 2 lsqr = 3

const _EXT_JUMP = "penalty AbsLog{1} requires the MatrixCoversJuMPExt extension (JuMP and HiGHS)"
const _EXT_IPOPT = "penalty AbsLinear requires the MatrixCoversIpoptExt extension (JuMP and Ipopt)"

_linsolve_symbol(ls::Linsolve) = ls === auto ? :auto : ls === dense ? :dense : :lsqr

include("trimmability.jl")

# --- Hard covers ------------------------------------------------------------

"Heuristic symmetric hard cover: `a` with `a[i]*a[j] >= abs(A[i, j])`."
symcover(A::Matrix{Float64}; maxiter::Union{Int64, Nothing} = nothing) = _symcover(A, maxiter)

@api symcover(A::Matrix{Float64}; maxiter::Union{Int64, Nothing} = nothing)::Vector{Float64}

"Heuristic hard cover of `A`, as `vcat(a, b)`; `a` has length `size(A, 1)`."
cover_ab(A::Matrix{Float64}; maxiter::Union{Int64, Nothing} = nothing) = _cover_ab(A, maxiter)

@api cover_ab(A::Matrix{Float64}; maxiter::Union{Int64, Nothing} = nothing)::Vector{Float64}

"""
ϕ-minimal symmetric hard cover of `A`. Only `abslog2` is solved natively;
`abslog1` needs the JuMP/HiGHS extension and the `abslinear` penalties need the
JuMP/Ipopt extension, neither linked into this library.
"""
function symcover_min(A::Matrix{Float64}; penalty::Penalty = abslog2,
                      maxiter::Union{Int64, Nothing} = nothing, linsolve::Linsolve = auto)
    penalty === abslog2 || throw(ArgumentError(penalty === abslog1 ? _EXT_JUMP : _EXT_IPOPT))
    ls = _linsolve_symbol(linsolve)
    return isnothing(maxiter) ? MatrixCovers.symcover_min(AbsLog{2}(), A; linsolve = ls) :
        MatrixCovers.symcover_min(AbsLog{2}(), A; maxiter = Int(maxiter), linsolve = ls)
end

@api symcover_min(A::Matrix{Float64}; penalty::Penalty = abslog2,
                  maxiter::Union{Int64, Nothing} = nothing, linsolve::Linsolve = auto)::Vector{Float64}

"""
ϕ-minimal hard cover of `A`, as `vcat(a, b)`; `a` has length `size(A, 1)`. Only
`abslog2` is solved natively; `abslog1` needs the JuMP/HiGHS extension and the
`abslinear` penalties need the JuMP/Ipopt extension, neither linked into this
library.
"""
function cover_min_ab(A::Matrix{Float64}; penalty::Penalty = abslog2,
                      maxiter::Union{Int64, Nothing} = nothing, linsolve::Linsolve = auto)
    penalty === abslog2 || throw(ArgumentError(penalty === abslog1 ? _EXT_JUMP : _EXT_IPOPT))
    ls = _linsolve_symbol(linsolve)
    a, b = isnothing(maxiter) ? MatrixCovers.cover_min(AbsLog{2}(), A; linsolve = ls) :
        MatrixCovers.cover_min(AbsLog{2}(), A; maxiter = Int(maxiter), linsolve = ls)
    return vcat(a, b)
end

@api cover_min_ab(A::Matrix{Float64}; penalty::Penalty = abslog2,
                  maxiter::Union{Int64, Nothing} = nothing, linsolve::Linsolve = auto)::Vector{Float64}

# --- Soft covers -------------------------------------------------------------
#
# The soft-cover entrypoints resolve each optional keyword to a concrete
# sentinel (`-1`/`NaN` for "omitted") and do nothing else: the branch on
# `penalty` and the sentinels lives in the `@noinline` dispatchers in
# `trimmability.jl`.

"""
Symmetric soft cover of `A` minimizing the penalty, with no coverage
constraint. `abslog1` and `abslog2` accept only `maxiter`; `abslinear1` and
`abslinear2` additionally accept `starts`, `sigma`, and `seed` (which seeds the
multistart perturbation stream).
"""
function soft_symcover(A::Matrix{Float64}; penalty::Penalty = abslinear2,
                       maxiter::Union{Int64, Nothing} = nothing,
                       starts::Union{Int64, Nothing} = nothing,
                       sigma::Union{Float64, Nothing} = nothing, seed::Int64 = 0)
    mi = isnothing(maxiter) ? -1 : Int(maxiter)
    st = isnothing(starts) ? -1 : Int(starts)
    sg = isnothing(sigma) ? NaN : Float64(sigma)
    return _soft_symcover_dispatch(A, penalty, mi, st, sg, seed)
end

@api soft_symcover(A::Matrix{Float64}; penalty::Penalty = abslinear2,
                   maxiter::Union{Int64, Nothing} = nothing,
                   starts::Union{Int64, Nothing} = nothing,
                   sigma::Union{Float64, Nothing} = nothing, seed::Int64 = 0)::Vector{Float64}

"""
Asymmetric soft cover of `A` minimizing the penalty, with no coverage
constraint, as `vcat(a, b)`; `a` has length `size(A, 1)`. `abslog1` and
`abslog2` accept only `maxiter`; `abslinear1` and `abslinear2` additionally
accept `starts`, `sigma`, and `seed`.
"""
function soft_cover_ab(A::Matrix{Float64}; penalty::Penalty = abslinear2,
                       maxiter::Union{Int64, Nothing} = nothing,
                       starts::Union{Int64, Nothing} = nothing,
                       sigma::Union{Float64, Nothing} = nothing, seed::Int64 = 0)
    mi = isnothing(maxiter) ? -1 : Int(maxiter)
    st = isnothing(starts) ? -1 : Int(starts)
    sg = isnothing(sigma) ? NaN : Float64(sigma)
    return _soft_cover_dispatch(A, penalty, mi, st, sg, seed)
end

@api soft_cover_ab(A::Matrix{Float64}; penalty::Penalty = abslinear2,
                   maxiter::Union{Int64, Nothing} = nothing,
                   starts::Union{Int64, Nothing} = nothing,
                   sigma::Union{Float64, Nothing} = nothing, seed::Int64 = 0)::Vector{Float64}

"""
ϕ-minimal symmetric soft cover of `A`, with no coverage constraint. Only
`abslog2` is solved natively; `MatrixCovers` does not implement `abslog1` for
this entrypoint, and the `abslinear` penalties need the JuMP/Ipopt extension.
"""
function soft_symcover_min(A::Matrix{Float64}; penalty::Penalty = abslog2,
                           maxiter::Union{Int64, Nothing} = nothing)
    mi = isnothing(maxiter) ? -1 : Int(maxiter)
    return _soft_symcover_min_dispatch(A, penalty, mi)
end

@api soft_symcover_min(A::Matrix{Float64}; penalty::Penalty = abslog2,
                       maxiter::Union{Int64, Nothing} = nothing)::Vector{Float64}

"""
ϕ-minimal asymmetric soft cover of `A`, with no coverage constraint, as
`vcat(a, b)`; `a` has length `size(A, 1)`. Only `abslog2` is solved natively;
`MatrixCovers` does not implement `abslog1` for this entrypoint, and the
`abslinear` penalties need the JuMP/Ipopt extension.
"""
function soft_cover_min_ab(A::Matrix{Float64}; penalty::Penalty = abslog2,
                          maxiter::Union{Int64, Nothing} = nothing)
    mi = isnothing(maxiter) ? -1 : Int(maxiter)
    return _soft_cover_min_dispatch(A, penalty, mi)
end

@api soft_cover_min_ab(A::Matrix{Float64}; penalty::Penalty = abslog2,
                       maxiter::Union{Int64, Nothing} = nothing)::Vector{Float64}

# --- Predicates and objectives -----------------------------------------------

"Whether `a` (playing the role of both `a` and `b`) covers `A`: `a[i]*a[j] >= abs(A[i, j])`."
iscover_sym(a::Vector{Float64}, A::Matrix{Float64}; rtol::Float64 = 0.0, atol::Float64 = 0.0) =
    MatrixCovers.iscover(a, A; rtol, atol)

@api iscover_sym(a::Vector{Float64}, A::Matrix{Float64}; rtol::Float64 = 0.0, atol::Float64 = 0.0)::Bool

"Whether `a`, `b` cover `A`: `a[i]*b[j] >= abs(A[i, j])`."
iscover_ab(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64}; rtol::Float64 = 0.0, atol::Float64 = 0.0) =
    MatrixCovers.iscover(a, b, A; rtol, atol)

@api iscover_ab(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64};
                rtol::Float64 = 0.0, atol::Float64 = 0.0)::Bool

_cover_objective(::Type{P}, a::Vector{Float64}, A::Matrix{Float64}) where {P <: MatrixCovers.AbstractCoverPenalty} =
    MatrixCovers.cover_objective(P(), a, A)
_cover_objective(::Type{P}, a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64}) where {P <: MatrixCovers.AbstractCoverPenalty} =
    MatrixCovers.cover_objective(P(), a, b, A)

"`sum(penalty(abs(A[i,j]) / (a[i]*a[j])))`."
function cover_objective_sym(a::Vector{Float64}, A::Matrix{Float64}; penalty::Penalty = abslog2)
    penalty === abslog1 && return _cover_objective(AbsLog{1}, a, A)
    penalty === abslog2 && return _cover_objective(AbsLog{2}, a, A)
    penalty === abslinear1 && return _cover_objective(AbsLinear{1}, a, A)
    return _cover_objective(AbsLinear{2}, a, A)
end

@api cover_objective_sym(a::Vector{Float64}, A::Matrix{Float64}; penalty::Penalty = abslog2)::Float64

"`sum(penalty(abs(A[i,j]) / (a[i]*b[j])))`."
function cover_objective_ab(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64}; penalty::Penalty = abslog2)
    penalty === abslog1 && return _cover_objective(AbsLog{1}, a, b, A)
    penalty === abslog2 && return _cover_objective(AbsLog{2}, a, b, A)
    penalty === abslinear1 && return _cover_objective(AbsLinear{1}, a, b, A)
    return _cover_objective(AbsLinear{2}, a, b, A)
end

@api cover_objective_ab(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64};
                        penalty::Penalty = abslog2)::Float64

# --- Gram covers --------------------------------------------------------------

"Symmetric cover of the Gram matrix `A'*A`, from a cover `(a, b)` of `A`."
function gramcover(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64})
    s = Vector{Float64}(undef, size(A, 2))
    MatrixCovers.gramcover!(s, a, b, A)
    return s
end

@api gramcover(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64})::Vector{Float64}

"Symmetric cover of the weighted Gram matrix `A'*Diagonal(w)*A`, from a cover `(a, b)` of `A`."
function gramcover_weighted(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64}, w::Vector{Float64})
    s = Vector{Float64}(undef, size(A, 2))
    MatrixCovers.gramcover!(s, a, b, A, w)
    return s
end

@api gramcover_weighted(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64}, w::Vector{Float64})::Vector{Float64}

"Symmetric cover of the weighted Gram matrix `A'*W*A`, from a cover `(a, b)` of `A`."
function gramcover_matrix(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64}, W::Matrix{Float64})
    s = Vector{Float64}(undef, size(A, 2))
    MatrixCovers.gramcover!(s, a, b, A, W)
    return s
end

@api gramcover_matrix(a::Vector{Float64}, b::Vector{Float64}, A::Matrix{Float64}, W::Matrix{Float64})::Vector{Float64}

end # module
