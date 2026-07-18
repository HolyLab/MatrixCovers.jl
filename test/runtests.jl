using MatrixCovers
using MatrixCovers: foreach_support, foreach_support_sym, unconstrained_min!, tighten_cover!
using JuMP, HiGHS, Ipopt   # triggers MatrixCoversJuMPExt and MatrixCoversIpoptExt extensions
using SparseArrays  # triggers MatrixCoversSparseArraysExt extension
using Unitful       # triggers MatrixCoversUnitfulExt extension
using LinearAlgebra
using OffsetArrays
using Statistics: median
using Random: MersenneTwister
using StableRNGs: StableRNG
using Aqua
using ExplicitImports
using Test

include("helpers.jl")               # isbalanced, covaries, PENALTIES

@testset "MatrixCovers.jl" begin

    include("penalties.jl")         # cover_objective
    include("support.jl")           # foreach_support(_sym) traversal
    include("iscover.jl")           # the cover predicate
    include("heuristic_covers.jl")  # symcover/cover and their internals
    include("soft_covers.jl")       # soft_symcover/soft_cover multistart descent
    include("initializers.jl")      # initialize_symcover/initialize_cover strategies
    include("minimal_covers.jl")    # the *_min family (native solvers)
    include("storage_types.jl")     # sparse/structured/wrapped storage vs dense reference
    include("extensions.jl")        # JuMP/HiGHS and Ipopt solvers, missing-extension hints
    include("unitful.jl")           # dimensional covers via the Unitful extension
    include("invariants.jl")        # shared conventions checked across every notion

    Aqua.test_all(MatrixCovers)

    @testset "ExplicitImports" begin
        # The public-ness checks consult `Base.ispublic` only on 1.11+; before that they
        # fall back to `isexported` and flag every `public`-but-unexported binding, so
        # they are meaningful only on 1.11+. The other five checks run on every version.
        #
        # These are this package's own internals, which its extensions legitimately
        # extend and call: extension and package ship from one repo at one version, so
        # there is no cross-package promise to break.
        internals = (:_cover_min_abslog2, :_symcover_min_abslog2,
                     :_prepare_cover_start!, :_prepare_symcover_start!,
                     :_prepare_soft_cover_start!, :_prepare_soft_symcover_start!,
                     :foreach_support, :foreach_support_sym,
                     :cover_min_jump, :symcover_min_jump, :check_solved)
        # Non-public names owned by other packages, each with no public equivalent:
        # `FreeUnits`/`Unit` are Unitful's unit representation, `Optimizer` is the
        # solver handle JuMP's own documented `Model(HiGHS.Optimizer)` entry point
        # names, and `register_error_hint` is Base-internal.
        foreign = (:FreeUnits, :Unit, :Optimizer, :Experimental, :register_error_hint)
        test_explicit_imports(
            MatrixCovers;
            all_explicit_imports_are_public = VERSION >= v"1.11" ?
                (; ignore = (internals..., foreign...)) : false,
            all_qualified_accesses_are_public = VERSION >= v"1.11" ?
                (; ignore = (internals..., foreign...)) : false,
        )
    end

    # Aqua checks the package alone; the extensions need their own sweep.
    @testset "method ambiguities" begin
        @test isempty(detect_ambiguities(MatrixCovers; recursive=true))
        for extname in (:MatrixCoversSparseArraysExt, :MatrixCoversJuMPExt, :MatrixCoversIpoptExt, :MatrixCoversUnitfulExt, :MatrixCoversSparseArraysUnitfulExt)
            ext = Base.get_extension(MatrixCovers, extname)
            @test ext !== nothing
            @test isempty(detect_ambiguities(MatrixCovers, ext; recursive=true))
        end
    end

end
