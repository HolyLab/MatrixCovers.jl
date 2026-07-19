# Element types other than Float64. The internal convergence tolerances are
# multiples of `eps(T)`, so both a narrower and a wider type must behave: a
# Float64-scaled literal is unreachable in Float32 (every descent would run to
# `maxiter`) and stops far short of what BigFloat can resolve.

@testset "element types" begin

    @testset "Float32 flows through the family" begin
        A = Float32[4 1.5; 1.5 1]
        for a in (symcover(A), soft_symcover(A), soft_symcover(AbsLog{1}(), A),
                  soft_symcover(AbsLinear{1}(), A), symcover_min(AbsLog{2}(), A),
                  soft_symcover_min(AbsLog{2}(), A))
            @test eltype(a) === Float32
            @test all(isfinite, a)
        end
        @test iscover(symcover(A), A; rtol=8eps(Float32))
        @test iscover(symcover_min(AbsLog{2}(), A), A; rtol=8eps(Float32))

        B = Float32[4 1.5 0.5; 1.5 1 2]
        a, b = cover(B)
        @test eltype(a) === eltype(b) === Float32
        @test iscover(a, b, B; rtol=8eps(Float32))
    end

    @testset "BigFloat flows through the family" begin
        A = BigFloat[4 1.5; 1.5 1]
        for a in (symcover(A), soft_symcover(A), symcover_min(AbsLog{2}(), A),
                  soft_symcover_min(AbsLog{2}(), A))
            @test eltype(a) === BigFloat
            @test all(isfinite, a)
        end
        @test iscover(symcover(A), A; rtol=8eps(BigFloat))
    end

    @testset "row and column scales may differ in element type" begin
        # The public entry points take plain `AbstractVector`s, so a mismatch must
        # not surface as a MethodError from an unexported internal. Each vector
        # keeps its own element type; the arithmetic promotes.
        A = [4.0 1.5 0.5; 1.5 1.0 2.0]
        a0, b0 = Float64[3.0, 3.0], Float32[3.0, 3.0, 3.0]

        a, b = cover!(copy(a0), copy(b0), A)
        @test eltype(a) === Float64
        @test eltype(b) === Float32
        @test iscover(a, b, A; rtol=1e-6)

        for ϕ in (AbsLog{1}(), AbsLinear{1}(), AbsLinear{2}())
            x, y = soft_cover!(ϕ, copy(a0), copy(b0), A)
            @test eltype(x) === Float64
            @test eltype(y) === Float32
            @test all(isfinite, x) && all(isfinite, y)
        end

        # The internals the public path routes through, reached directly.
        @test MatrixCovers.tighten_cover!(copy(a0), copy(b0), A) isa Tuple
        @test MatrixCovers.tighten_cover!(copy(b0), copy(a0), A') isa Tuple
        @test MatrixCovers.unconstrained_min!(AbsLog{2}(), copy(a0), copy(b0), A) isa Tuple
    end

    @testset "tolerances follow the element type" begin
        # Each convergence threshold must sit above the type's resolution, or the
        # test it guards can never fire.
        for T in (Float32, Float64, BigFloat)
            @test MatrixCovers._multistart_switchtol(T) > eps(T)
        end

        # And must sit below it for a wider type, so the extra precision is used.
        # Running the ALS kernel from one start under the eltype-scaled tolerance
        # and under the Float64-scaled literal it replaced separates the two.
        A = BigFloat[4 1.5 0.3; 1.5 1 0.7; 0.3 0.7 2.0]
        start = initialize_symcover(A; feasible=:none)
        u1, v1 = copy(start), copy(start)
        u2, v2 = copy(start), copy(start)
        MatrixCovers._msmc_als!(u1, v1, A, 500)
        MatrixCovers._msmc_als!(u2, v2, A, 500; tol=1e-14)
        @test cover_objective(AbsLinear{2}(), u1, v1, A) <
              cover_objective(AbsLinear{2}(), u2, v2, A)
    end

end
