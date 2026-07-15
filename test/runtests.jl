using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: divmag, dotabs, foreach_support, foreach_support_sym, unconstrained_min!, tighten_cover!
using JuMP, HiGHS, Ipopt   # triggers SIAJuMP and SIAIpopt extensions
using SparseArrays  # triggers SIASparseArrays extension
using LinearAlgebra
using OffsetArrays
using Statistics: median
using Random: MersenneTwister
using StableRNGs: StableRNG
using Test

include("helpers.jl")               # iscover, covaries, PENALTIES

@testset "ScaleInvariantAnalysis.jl" begin

    include("penalties.jl")         # cover_objective, dotabs, divmag
    include("support.jl")           # foreach_support(_sym) traversal
    include("heuristic_covers.jl")  # symcover/cover and their internals
    include("soft_covers.jl")       # soft_symcover/soft_cover multistart descent
    include("initializers.jl")      # initialize_symcover/initialize_cover strategies
    include("minimal_covers.jl")    # the *_min family (native solvers)
    include("storage_types.jl")     # sparse/structured/wrapped storage vs dense reference
    include("extensions.jl")        # JuMP/HiGHS and Ipopt solvers, missing-extension hints
    include("invariants.jl")        # shared conventions checked across every notion

    @testset "method ambiguities" begin
        @test isempty(detect_ambiguities(ScaleInvariantAnalysis; recursive=true))
        for extname in (:SIASparseArrays, :SIAJuMP, :SIAIpopt)
            ext = Base.get_extension(ScaleInvariantAnalysis, extname)
            @test ext !== nothing
            @test isempty(detect_ambiguities(ScaleInvariantAnalysis, ext; recursive=true))
        end
    end

end
