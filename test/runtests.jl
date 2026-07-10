using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: divmag, dotabs, foreach_support, foreach_support_sym, unconstrained_min!, tighten_cover!
using JuMP, HiGHS, Ipopt   # triggers SIAJuMP and SIAIpopt extensions
using SparseArrays  # triggers SIASparseArrays extension
using LinearAlgebra
using OffsetArrays
using Statistics: median
using Random: MersenneTwister
using Test

# Cover feasibility up to an additive slack: a[i]*b[j] >= |A[i,j]| - atol on every entry.
iscover(a, b, A; atol) = all(a[i] * b[j] >= abs(A[i, j]) - atol for i in axes(A, 1), j in axes(A, 2))
iscover(a, A; atol) = iscover(a, a, A; atol)

@testset "ScaleInvariantAnalysis.jl" begin

    include("penalties.jl")         # cover_objective, dotabs, divmag
    include("support.jl")           # foreach_support(_sym) traversal
    include("heuristic_covers.jl")  # symcover/cover and their internals
    include("soft_covers.jl")       # soft_symcover/soft_cover multistart descent
    include("minimal_covers.jl")    # the *_min family (native solvers)
    include("storage_types.jl")     # sparse/structured/wrapped storage vs dense reference
    include("extensions.jl")        # JuMP/HiGHS and Ipopt solvers, missing-extension hints

    @testset "method ambiguities" begin
        @test isempty(detect_ambiguities(ScaleInvariantAnalysis; recursive=true))
        ext = Base.get_extension(ScaleInvariantAnalysis, :SIASparseArrays)
        @test isempty(detect_ambiguities(ScaleInvariantAnalysis, ext; recursive=true))
    end

end
