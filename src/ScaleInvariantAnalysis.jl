module ScaleInvariantAnalysis

using LinearAlgebra: LinearAlgebra, Adjoint, Bidiagonal, Diagonal, SymTridiagonal,
                     Symmetric, Transpose, Tridiagonal, dot, norm
using PrecompileTools: PrecompileTools, @compile_workload
using Random: Random, AbstractRNG, MersenneTwister

export AbsLog, AbsLinear
export cover_objective, iscover
export cover, cover!, symcover, symcover!
export soft_symcover, soft_symcover!, soft_cover, soft_cover!
export initialize_cover, initialize_cover!, initialize_symcover, initialize_symcover!
export symcover_min, symcover_min!, cover_min, cover_min!
export soft_symcover_min, soft_symcover_min!, soft_cover_min, soft_cover_min!

# `public` is parsed as a keyword only from Julia 1.11; this package supports 1.10.
@static if VERSION >= v"1.11"
    eval(Meta.parse("public AbstractCoverPenalty, foreach_support, foreach_support_sym"))
end

include("penalties.jl")
include("support.jl")
include("iscover.jl")
include("heuristic_covers.jl")
include("initializers.jl")   # the start menu; consumed by both solver families below
include("soft_covers.jl")
include("minimal_covers.jl")


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
