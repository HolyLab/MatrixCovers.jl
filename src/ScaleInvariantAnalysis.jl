module ScaleInvariantAnalysis

using LinearAlgebra: LinearAlgebra, Adjoint, Bidiagonal, Diagonal, SymTridiagonal,
                     Symmetric, Transpose, Tridiagonal, dot, norm
using PrecompileTools: PrecompileTools, @compile_workload
using Random: Random, AbstractRNG, MersenneTwister

export AbsLog, AbsLinear
export cover_objective
export cover, cover!, symcover, symcover!, soft_symcover, soft_cover
export initialize_cover, initialize_cover!, initialize_symcover, initialize_symcover!
export symcover_min, symcover_min!, cover_min, cover_min!
export soft_symcover_min, soft_symcover_min!, soft_cover_min, soft_cover_min!
export dotabs, divmag

include("penalties.jl")
include("support.jl")
include("heuristic_covers.jl")
include("initializers.jl")   # the start menu; consumed by both solver families below
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


# True only when a MethodError's argument types are consistent with the calling
# convention of the `*_min` solvers — a penalty, the scale vectors the mutating
# forms refine in place, and the matrix — i.e. the failure could plausibly be fixed
# by loading an extension rather than by passing arguments of the right kind.
function _looks_like_missing_extension(argtypes)
    length(argtypes) >= 2 || return false
    argtypes[1] <: AbstractCoverPenalty || return false
    argtypes[end] <: AbstractMatrix || return false
    return all(T -> T <: AbstractVector, argtypes[2:end-1])
end

function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        _looks_like_missing_extension(argtypes) || return
        if exc.f === symcover_min || exc.f === symcover_min! ||
           exc.f === cover_min || exc.f === cover_min!
            printstyled(io, "\nAbsLog{2} is solved natively; other penalties require loading JuMP plus HiGHS (for AbsLog{1}) or Ipopt (for AbsLinear)."; color=:yellow)
            return true
        end
        if exc.f === soft_symcover_min || exc.f === soft_symcover_min! ||
           exc.f === soft_cover_min || exc.f === soft_cover_min!
            printstyled(io, "\nAbsLog{2} is solved natively; AbsLinear penalties require loading JuMP plus Ipopt. AbsLog{1} is not yet supported."; color=:yellow)
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
