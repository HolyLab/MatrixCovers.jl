@testset "iscover" begin
    A = [1.0 2.0; 3.0 4.0]

    @testset "the exact inequality, with no default slack" begin
        @test iscover([1.0, 3.0], [1.0, 2.0], A)         # a*b' == A exactly
        @test !iscover([1.0, 1.0], [1.0, 1.0], A)        # ones cannot reach A[2,2]
        # Exactly tight is covered; a hair under is not.
        @test iscover([1.0], [1.0], fill(1.0, 1, 1))
        @test !iscover([1.0], [prevfloat(1.0)], fill(1.0, 1, 1))
        @test iscover([1.0], [prevfloat(1.0)], fill(1.0, 1, 1); rtol=1e-15)
    end

    @testset "slack" begin
        Aone = fill(1.0, 1, 1)
        @test !iscover([0.9], [1.0], Aone)
        @test iscover([0.9], [1.0], Aone; rtol=0.2)
        @test iscover([0.9], [1.0], Aone; atol=0.2)
        @test !iscover([0.9], [1.0], Aone; rtol=0.05)
        @test !iscover([0.9], [1.0], Aone; atol=0.05)
    end

    @testset "symmetric form" begin
        Asym = [4.0 1.0; 1.0 3.0]
        @test iscover([2.0, 2.0], Asym)
        @test !iscover([1.0, 1.0], Asym)
        @test iscover(symcover(Asym), Asym; rtol=8eps())
        # The two-argument form is the symmetric cover a*a'.
        a = [2.0, 2.0]
        @test iscover(a, Asym) == iscover(a, a, Asym)
        @test_throws DimensionMismatch iscover([1.0, 1.0], [1.0 2.0 3.0; 4.0 5.0 6.0])
    end

    @testset "zero entries constrain nothing" begin
        # A zero entry is satisfied by any nonnegative scales, including zero ones.
        Az = [1.0 0.0; 0.0 0.0]
        @test iscover([1.0, 0.0], [1.0, 0.0], Az)
        @test iscover([1.0, 0.0], Az)
        # ... but a nonzero entry backed by a zero scale is not covered.
        @test !iscover([1.0, 0.0], [1.0, 0.0], [1.0 0.0; 0.0 1.0])
    end

    @testset "negative scales are rejected" begin
        @test_throws "iscover requires nonnegative scales" iscover([-1.0, 3.0], [1.0, 2.0], A)
        @test_throws "iscover requires nonnegative scales" iscover([1.0, 3.0], [-1.0, 2.0], A)
        @test_throws "iscover requires nonnegative scales" iscover([-1.0, 1.0], [4.0 1.0; 1.0 3.0])
        # NaN fails every comparison, so it is rejected too rather than silently passing.
        @test_throws "iscover requires nonnegative scales" iscover([NaN, 3.0], [1.0, 2.0], A)
        # The check runs before the traversal, so it fires even when no entry is violated.
        @test_throws "iscover requires nonnegative scales" iscover([-1.0, -1.0], [-1.0, -1.0], zeros(2, 2))
    end

    @testset "axes must match" begin
        @test_throws DimensionMismatch iscover([1.0], [1.0, 2.0], A)
        @test_throws DimensionMismatch iscover([1.0, 3.0], [1.0], A)
        Ao = OffsetArray(A, -1:0, -1:0)
        @test_throws DimensionMismatch iscover([1.0, 3.0], [1.0, 2.0], Ao)
        @test iscover(OffsetArray([1.0, 3.0], -1:0), OffsetArray([1.0, 2.0], -1:0), Ao)
    end

    @testset "storage types agree with the dense reference: $(typeof(S))" for S in (
            sparse([1.0 2.0; 3.0 4.0]),
            Diagonal([2.0, 3.0]),
            SymTridiagonal([2.0, 3.0], [1.0]),
            Tridiagonal([1.0], [2.0, 3.0], [0.5]),
            Bidiagonal([2.0, 3.0], [1.0], :U),
            Symmetric(sparse([4.0 1.0; 1.0 3.0])))
        D = Matrix(S)
        for a in ([1.0, 1.0], [2.0, 2.0], [3.0, 3.0])
            @test iscover(a, a, S) == iscover(a, a, D)
        end
    end

    @testset "dimensional scales" begin
        # A scale carries its units in the value, not the type: `zero(Quantity{Float64})`
        # is undefined, so the nonnegativity check must ask the element, not the eltype.
        L = u"m"
        Au = [1.0/L^2 2.0/L^2; 2.0/L^2 4.0/L^2]
        au = [2.0/L, 2.0/L]
        @test iscover(au, Au)
        @test !iscover([1.0/L, 1.0/L], Au)
        @test iscover(symcover(Au), Au; rtol=8eps())
        @test_throws "iscover requires nonnegative scales" iscover([-1.0/L, 2.0/L], Au)
    end

    @testset "a soft cover is not guaranteed to cover" begin
        # The predicate's reason for existing: soft covers penalize under-coverage
        # rather than forbidding it, so whether one covers is a real question.
        Asoft = [1.0 4.0; 4.0 1.0]
        @test iscover(symcover(Asoft), Asoft; rtol=8eps())
        @test !iscover(soft_symcover(AbsLinear{2}(), Asoft), Asoft)
    end
end
