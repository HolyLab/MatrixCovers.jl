module ScaleInvariantAnalysis

using LinearAlgebra
using PrecompileTools
using Random

export AbsLog, AbsLinear
export cover_objective
export cover, cover!, symcover, symcover!, soft_symcover, soft_cover
export symcover_min, cover_min, soft_symcover_min, soft_cover_min
export dotabs, divmag

include("penalties.jl")
include("support.jl")
include("heuristic_covers.jl")
include("soft_covers.jl")
include("minimal_covers.jl")


"""
    dotabs(x, y)

Compute the sum of absolute values of elementwise products of `x` and `y`:

    ∑_i |x[i] * y[i]|
"""
function dotabs(x::AbstractVector, y::AbstractVector)
    s = zero(eltype(x)) * zero(eltype(y))
    for i in eachindex(x, y)
        s += abs(x[i] * y[i])
    end
    return s
end

"""
    a, mag = divmag(A, b; use_cond::Bool=false)

Given a symmetric matrix `A` and vector `b`, for `x = A \\ b` return a pair
where `mag` is a naive estimate of the magnitude of `sum(abs.(x .* a))`. `a` and
`mag` are scale-covariant in circumstances where `A \\ b` is contravariant. With
`use_cond=false`, the estimate is based only on the magnitudes of the numbers
in `A` and `b`, and does not account for the conditioning of `A` or
cancellation in the solution process.

This can be used to form scale-invariant estimates of roundoff errors in
computations involving `A`, `b`, and `x`.
"""
function divmag(A, b; use_cond::Bool=false)
    a = symcover(A)
    κ = use_cond ? LinearAlgebra.cond(A ./ (a .* a')) : 1
    return a, κ * sum(abs ∘ splat(ratio_nz), zip(b, a))
end
ratio_nz(n, d) = iszero(d) ? zero(n) / oneunit(d) : n / d


# True only when a MethodError's argument types are consistent with the
# `(ϕ::AbstractCoverPenalty, A::AbstractMatrix; kwargs...)` calling convention
# of the `*_min` solvers, i.e. the failure could plausibly be fixed by loading
# an extension rather than by passing arguments of the right kind.
_looks_like_missing_extension(argtypes) = length(argtypes) >= 2 && argtypes[1] <: AbstractCoverPenalty && argtypes[2] <: AbstractMatrix

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        _looks_like_missing_extension(argtypes) || return
        if exc.f === symcover_min || exc.f === cover_min
            printstyled(io, "\nAbsLog{2} is solved natively; other penalties require loading JuMP plus HiGHS (for AbsLog{1}) or Ipopt (for AbsLinear)."; color=:yellow)
            return true
        end
        if exc.f === soft_symcover_min
            printstyled(io, "\nThis method requires loading JuMP plus HiGHS (for AbsLog{2}) or Ipopt (for AbsLinear)."; color=:yellow)
            return true
        end
        if exc.f === soft_cover_min
            printstyled(io, "\nOnly AbsLog{2}() is currently implemented (solved natively); AbsLinear penalties are not yet supported."; color=:yellow)
            return true
        end
    end
end

@compile_workload begin
    symcover([1.0 0.1; 0.1 4.0])
    soft_symcover([1.0 0.1; 0.1 4.0])
    cover([1.0 0.5; 0.1 4.0])
end

end # module
