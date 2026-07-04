module ScaleInvariantAnalysis

using LinearAlgebra
using PrecompileTools
using Random

export AbsLog, AbsLinear
export cover_objective
export cover, symcover, soft_symcover, soft_cover
export symcover_min, cover_min, soft_symcover_min
export dotabs

include("penalties.jl")
include("heuristic_covers.jl")
include("soft_covers.jl")
include("minimal_covers.jl")
include("structured.jl")


"""
    dotabs(x, y)

Compute the sum of absolute values of elementwise products of `x` and `y`:

    ∑_i |x[i] * y[i]|
"""
function dotabs(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    s = zero(eltype(x)) * zero(eltype(y))
    for i in eachindex(x, y)
        s += abs(x[i] * y[i])
    end
    return s
end

"""
    a, mag = divmag(A, b; cond::Bool=false)

Given a symmetric matrix `A` and vector `b`, for `x = A \\ b` return a pair
where `mag` is a naive estimate of the magnitude of `sum(abs.(x .* a))`. `a` and
`mag` are scale-covariant in circumstances where `A \\ b` is contravariant. With
`cond=false`, the estimate is based only on the magnitudes of the numbers in `A`
and `b`, and does not account for the conditioning of `A` or cancellation in the
solution process.

This can be used to form scale-invariant estimates of roundoff errors in
computations involving `A`, `b`, and `x`.
"""
function divmag(A, b; cond::Bool=false)
    a = symcover(A)
    κ = cond ? LinearAlgebra.cond(A ./ (a .* a')) : 1
    return a, κ * sum(abs ∘ splat(ratio_nz), zip(b, a))
end
ratio_nz(n, d) = iszero(d) ? zero(n) / oneunit(d) : n / d


function __init__()
    Base.Experimental.register_error_hint(MethodError) do io, exc, argtypes, kwargs
        if exc.f === symcover_min || exc.f === cover_min
            printstyled(io, "\nAbsLog{2} is solved natively; other penalties require loading JuMP plus HiGHS (for AbsLog{1}) or Ipopt (for AbsLinear)."; color=:yellow)
            return true
        end
        if exc.f === soft_symcover_min
            printstyled(io, "\nThis method requires loading JuMP plus HiGHS (for AbsLog{2}) or Ipopt (for AbsLinear)."; color=:yellow)
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
