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
        @test !iscover(first(soft_symcover(AbsLinear{2}(), Asoft)), Asoft)
    end
end

@testset "max_coverage_ratio" begin
    A = [1.0 2.0; 3.0 4.0]

    @testset "the worst entry, and agreement with iscover" begin
        @test max_coverage_ratio([1.0, 3.0], [1.0, 2.0], A) == 1     # a*b' == A exactly
        @test max_coverage_ratio([1.0, 1.0], [1.0, 1.0], A) == 4
        @test max_coverage_ratio([8.0, 8.0], [1.0, 1.0], A) == 0.5
        for (a, b) in (([1.0, 3.0], [1.0, 2.0]), ([1.0, 1.0], [1.0, 1.0]), ([8.0, 8.0], [1.0, 1.0]))
            @test iscover(a, b, A) == (max_coverage_ratio(a, b, A) <= 1)
            # Scaling the product by the ratio gives a cover that is tight at the worst entry.
            r = max_coverage_ratio(a, b, A)
            @test max_coverage_ratio(r .* a, b, A) ≈ 1
        end
    end

    @testset "symmetric form" begin
        Asym = [4.0 1.0; 1.0 3.0]
        a = [1.0, 1.0]
        @test max_coverage_ratio(a, Asym) == max_coverage_ratio(a, a, Asym) == 4
        r = max_coverage_ratio(a, Asym)
        @test max_coverage_ratio(sqrt(r) .* a, Asym) ≈ 1
        @test_throws DimensionMismatch max_coverage_ratio([1.0, 1.0], [1.0 2.0 3.0; 4.0 5.0 6.0])
    end

    @testset "zero entries and zero scales" begin
        @test max_coverage_ratio([1.0, 0.0], [1.0, 0.0], [1.0 0.0; 0.0 0.0]) == 1
        @test max_coverage_ratio([1.0, 0.0], [1.0, 0.0], [1.0 0.0; 0.0 1.0]) == Inf
        @test max_coverage_ratio([1.0, 1.0], [1.0, 1.0], zeros(2, 2)) === 0.0
        @test max_coverage_ratio(zeros(2), zeros(2), zeros(2, 2)) === 0.0
    end

    @testset "argument checks" begin
        @test_throws "max_coverage_ratio requires nonnegative scales" max_coverage_ratio([-1.0, 3.0], [1.0, 2.0], A)
        @test_throws "max_coverage_ratio requires nonnegative scales" max_coverage_ratio([1.0, 3.0], [NaN, 2.0], A)
        @test_throws DimensionMismatch max_coverage_ratio([1.0], [1.0, 2.0], A)
        @test_throws DimensionMismatch max_coverage_ratio([1.0, 3.0], [1.0], A)
    end

    @testset "offset axes" begin
        Ao = OffsetArray(A, -1:0, 3:4)
        ao, bo = OffsetArray([1.0, 1.0], -1:0), OffsetArray([1.0, 2.0], 3:4)
        @test max_coverage_ratio(ao, bo, Ao) == max_coverage_ratio([1.0, 1.0], [1.0, 2.0], A) == 3
        @test_throws DimensionMismatch max_coverage_ratio([1.0, 1.0], [1.0, 2.0], Ao)
    end

    @testset "element types" begin
        @test max_coverage_ratio(Float32[1, 1], Float32[1, 1], Float32[1 2; 3 4]) isa Float32
        @test max_coverage_ratio([1, 1], [1, 1], [1 2; 3 4]) === 4.0
    end

    @testset "storage types agree with the dense reference: $(typeof(S))" for S in (
            sparse([1.0 2.0; 3.0 4.0]),
            Diagonal([2.0, 3.0]),
            SymTridiagonal([2.0, 3.0], [1.0]),
            Tridiagonal([1.0], [2.0, 3.0], [0.5]),
            Bidiagonal([2.0, 3.0], [1.0], :U),
            Symmetric(sparse([4.0 1.0; 1.0 3.0])))
        D = Matrix(S)
        for a in ([1.0, 1.0], [2.0, 0.5], [3.0, 3.0])
            @test max_coverage_ratio(a, a, S) == max_coverage_ratio(a, a, D)
        end
    end

    @testset "units cancel" begin
        L = u"m"
        Au = [1.0/L^2 2.0/L^2; 2.0/L^2 4.0/L^2]
        @test max_coverage_ratio([1.0/L, 1.0/L], Au) === 4.0
        # A change of units on each coordinate leaves the ratio unchanged.
        d = [2.0, 0.25]
        @test max_coverage_ratio([1.0, 1.0] .* d, A .* (d * d')) ≈ max_coverage_ratio([1.0, 1.0], A)
    end
end

@testset "level_profile" begin
    A = [1.0 2.0 0.0; 3.0 0.0 5.0]

    @testset "agreement with the definition" begin
        a, b, aref, bref = [1.0, 3.0], [0.5, 2.0, 4.0], [2.0, 1.0], [1.0, 1.0, 3.0]
        ℓ = [log(a[i] * b[j] / (aref[i] * bref[j])) for (i, j) in Tuple.(findall(!iszero, A))]
        lp = level_profile((a, b), (aref, bref), A)
        @test lp.level ≈ sum(ℓ) / length(ℓ)
        @test lp.profile ≈ sqrt(sum(abs2, ℓ .- lp.level) / length(ℓ))
        # Exchanging the products negates the level.
        lpr = level_profile((aref, bref), (a, b), A)
        @test lpr.level ≈ -lp.level
        @test lpr.profile ≈ lp.profile
    end

    @testset "level and profile separate" begin
        a, b = cover(A)
        @test level_profile((3a, b), (a, b), A) == (level=log(3.0), profile=0.0)
        @test level_profile((a, b), (a, b), A) == (level=0.0, profile=0.0)
        # A gauge that differs between components leaves the products on the support unchanged.
        Ab = [1.0 2.0 0.0 0.0; 3.0 4.0 0.0 0.0; 0.0 0.0 5.0 6.0]
        a, b = cover(Ab)
        g = [10.0, 10.0, 0.01]
        h = [0.1, 0.1, 100.0, 100.0]
        lp = level_profile((a .* g, b .* h), (a, b), Ab)
        @test abs(lp.level) < 10eps()
        @test lp.profile < 10eps()
    end

    @testset "symmetric form" begin
        Asym = [4.0 1.0; 1.0 0.0]
        a, aref = [2.0, 1.0], [1.0, 3.0]
        @test level_profile(a, aref, Asym) == level_profile((a, a), (aref, aref), Asym)
        @test_throws DimensionMismatch level_profile([1.0, 1.0], [1.0, 1.0], A)
    end

    @testset "zero entries are skipped" begin
        # b[2] and bref[2] multiply only zeros of A2, so they do not matter.
        A2 = [1.0 0.0; 3.0 0.0]
        @test level_profile(([1.0, 2.0], [1.0, 0.0]), ([1.0, 1.0], [1.0, 7.0]), A2) ==
              level_profile(([1.0, 2.0], [1.0, 5.0]), ([1.0, 1.0], [1.0, 1.0]), A2)
    end

    @testset "argument checks" begin
        o = ([1.0, 1.0], [1.0, 1.0, 1.0])
        @test_throws "level_profile requires products that are positive and finite" level_profile(([1.0, 0.0], o[2]), o, A)
        @test_throws "level_profile requires products that are positive and finite" level_profile(o, ([1.0, 1.0], [1.0, 1.0, Inf]), A)
        @test_throws "level_profile requires `A` to have at least one nonzero entry" level_profile(o, o, zeros(2, 3))
        @test_throws "level_profile requires nonnegative scales" level_profile(([-1.0, 1.0], o[2]), o, A)
        @test_throws "level_profile requires nonnegative scales" level_profile(o, (o[1], [1.0, NaN, 1.0]), A)
        @test_throws "indices of `aref` must match" level_profile(o, ([1.0], o[2]), A)
        @test_throws "indices of `b` must match" level_profile(([1.0, 1.0], [1.0, 1.0]), o, A)
    end

    @testset "offset axes" begin
        Ao = OffsetArray(A, -1:0, 3:5)
        a, b, aref, bref = [1.0, 3.0], [0.5, 2.0, 4.0], [2.0, 1.0], [1.0, 1.0, 3.0]
        oa(x) = OffsetArray(x, -1:0)
        ob(x) = OffsetArray(x, 3:5)
        @test level_profile((oa(a), ob(b)), (oa(aref), ob(bref)), Ao) ==
              level_profile((a, b), (aref, bref), A)
        @test_throws DimensionMismatch level_profile((a, b), (aref, bref), Ao)
    end

    @testset "element types" begin
        o = (Float32[1, 1], Float32[1, 1, 1])
        @test level_profile((Float32[2, 1], o[2]), o, Float32.(A)).level isa Float32
        @test level_profile(([2, 1], [1, 1, 1]), ([1, 1], [1, 1, 1]), [1 2 0; 3 0 5]).level isa Float64
    end

    @testset "storage types agree with the dense reference: $(typeof(S))" for S in (
            sparse([1.0 2.0; 3.0 4.0]),
            Diagonal([2.0, 3.0]),
            SymTridiagonal([2.0, 3.0], [1.0]),
            Tridiagonal([1.0], [2.0, 3.0], [0.5]),
            Bidiagonal([2.0, 3.0], [1.0], :U),
            Symmetric(sparse([4.0 1.0; 1.0 3.0])))
        D = Matrix(S)
        a, aref = [2.0, 0.5], [1.0, 3.0]
        @test all(values(level_profile(a, aref, S)) .≈ values(level_profile(a, aref, D)))
    end

    @testset "units cancel" begin
        L = u"m"
        Au = [1.0/L^2 2.0/L^2; 2.0/L^2 0.0/L^2]
        lp = level_profile([2.0/L, 1.0/L], [1.0/L, 1.0/L], Au)
        @test lp == level_profile([2.0, 1.0], [1.0, 1.0], ustrip.(Au))
        # A change of units on each coordinate leaves level and profile unchanged.
        d = [2.0, 0.25]
        Ad = [1.0 2.0; 2.0 0.0]
        @test all(values(level_profile([2.0, 1.0] .* d, d, Ad .* (d * d'))) .≈ values(level_profile([2.0, 1.0], [1.0, 1.0], Ad)))
    end
end
