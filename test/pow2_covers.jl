# Heuristic symmetric covers with power-of-two factors.

const symcover_exponents = MatrixCovers.symcover_exponents

ispow2orzero(x) = iszero(x) || significand(x) == 1

# Every supported row is tight on some entry: no single exponent can decrease.
function exponents_maximal(α, A)
    ok = true
    for i in axes(A, 1)
        α[i] == typemin(Int) && continue
        tight = false
        for j in axes(A, 2)
            v = abs(A[i, j])
            iszero(v) && continue
            e = MatrixCovers._ceil_log2(v)
            tight |= i == j ? 2α[i] - e <= 1 : α[i] + α[j] == e
        end
        ok &= tight
    end
    return ok
end

@testset "symcover_exponents" begin
    @test symcover_exponents([4.0 1; 1 4]) == [1, 1]
    @test symcover([4.0 1; 1 4]; pow2=true) == [2.0, 2.0]
    # An odd diagonal exponent is covered with a factor of two to spare.
    @test symcover([2.0 1; 1 3]; pow2=true) == [2.0, 2.0]
    # A row with no nonzero entry has exponent typemin(Int), factor zero.
    α = symcover_exponents([1.0 0 0; 0 0 0; 0 0 4])
    @test α == [0, typemin(Int), 1]
    @test symcover([1.0 0 0; 0 0 0; 0 0 4]; pow2=true) == [1.0, 0.0, 2.0]
    # Components without a nonzero diagonal entry.
    @test symcover_exponents([0.0 2; 2 0]) == [0, 1]
    @test symcover_exponents([0.0 2 0; 2 0 3; 0 3 0]) == [0, 1, 1]

    rng = StableRNG(11)
    gens = (n -> (B = rand(rng, n, n); B + B'),
            n -> (B = Float64.(rand(rng, 0:4, n, n)); B + B' + 8I),
            n -> (B = exp.(10 .* randn(rng, n, n)); B + B'),
            # Entries on and just above powers of two.
            n -> (B = ldexp.(1.0, rand(rng, -3:3, n, n)) .* (1 .+ eps() .* rand(rng, 0:1, n, n)); B + B'),
            n -> (B = sprandn(rng, n, n, 0.05); B + B' + Diagonal(rand(rng, n))))
    for gen in gens, n in (5, 30, 120), _ in 1:5
        A = gen(n)
        α = symcover_exponents(A)
        a = symcover(A; pow2=true)
        @test a == ldexp.(1.0, α)
        @test all(ispow2orzero, a)
        @test iscover(a, A)
        @test exponents_maximal(α, A)
        # Exact covariance under power-of-two rescaling.
        k = rand(rng, -30:30, n)
        d = ldexp.(1.0, k)
        @test symcover_exponents(d .* A .* d') == α .+ k
        @test symcover(d .* A .* d'; pow2=true) == d .* a
    end

    # Exact covariance when only some rows have a nonzero diagonal entry.
    A = [0.0 3 0 0; 3 5 1 0; 0 1 0 7; 0 0 7 0]
    α = symcover_exponents(A)
    k = [3, -2, 5, 1]
    d = ldexp.(1.0, k)
    @test symcover_exponents(d .* A .* d') == α .+ k
end

@testset "symcover_exponents storage and element types" begin
    A = [4.0 1 0.25; 1 9 2; 0.25 2 0.5]
    α = symcover_exponents(A)
    @test symcover_exponents(sparse(A)) == α
    @test symcover_exponents(Symmetric(A)) == α
    @test symcover_exponents(Symmetric(triu(A))) == α
    @test symcover_exponents(view(A, :, :)) == α
    @test symcover_exponents(SMatrix{3,3}(A)) == α
    @test symcover_exponents(SymTridiagonal([4.0, 9, 0.5], [1.0, 2])) == symcover_exponents([4.0 1 0; 1 9 2; 0 2 0.5])
    @test symcover_exponents(Float32.(A)) == α
    @test symcover_exponents(big.(A)) == α
    @test symcover_exponents(-A) == α
    @test symcover_exponents(4 .* A) == α .+ 1
    # Integer entries, with exact exponents at and around powers of two.
    @test symcover_exponents([4 1; 1 4]) == [1, 1]
    @test symcover_exponents([2 5; 5 2]) == [1, 2]
    @test symcover_exponents([big(2)^200 0; 0 1]) == [100, 0]
    # Subnormal entries have exact exponents.
    @test symcover_exponents([nextfloat(0.0) 0; 0 1]) == [-537, 0]

    Ao = OffsetArray(A, 0:2, 0:2)
    αo = symcover_exponents(Ao)
    @test axes(αo) == (axes(Ao, 1),)
    @test parent(αo) == α
    ao = symcover(Ao; pow2=true)
    @test axes(ao) == (axes(Ao, 1),)
    @test parent(ao) == ldexp.(1.0, α)
    abuf = OffsetVector(zeros(3), 0:2)
    @test symcover!(abuf, Ao; pow2=true) === abuf
    @test abuf == ao
    @test symcover(AbsLog{2}(), A; pow2=true) == ldexp.(1.0, α)
    @test symcover!(zeros(Float32, 3), A; pow2=true) == Float32.(ldexp.(1.0, α))
end

@testset "symcover_exponents errors" begin
    @test_throws "symcover_exponents requires finite entries" symcover_exponents([Inf 1; 1 1.0])
    @test_throws "symcover_exponents requires finite entries" symcover_exponents([NaN 1; 1 1.0])
    @test_throws "symcover_exponents requires a square matrix" symcover_exponents([1.0 2; 3 4; 5 6])
    @test_throws "requires `abs.(A)` to be symmetric" symcover_exponents([1.0 2; 3 4])
    @test_throws "requires floating-point or integer entries" symcover_exponents([1//2 1; 1 1])
    @test_throws "maxiter must be nonnegative" symcover_exponents([1.0 0; 0 1]; maxiter=-1)
    @test_throws "the factor 2^167 for index 1 is not a normal Float32" symcover!(zeros(Float32, 2), [1e100 0; 0 1]; pow2=true)
    @test_throws "the factor 2^-1500 for index 1 is not a normal Float64" symcover!(zeros(2), [big(2)^-3000 0; 0 1]; pow2=true)
end
