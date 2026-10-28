@testset "noise_cover" begin
    # Row and column sums of `V ./ (a .* b').^2`, with 0/0 = 0.
    function noise_sums(V, a, b)
        W = [iszero(V[i, j]) ? 0.0 : V[i, j] / (a[i] * b[j])^2 for i in axes(V, 1), j in axes(V, 2)]
        return vec(sum(W; dims=2)), vec(sum(W; dims=1))
    end
    rankone(V) = sum(V; dims=2) * sum(V; dims=1) / sum(V)
    # The matrix that `shrink = δ` scales; it has the same total as `V`.
    shrunk(V, δ) = (V .+ δ .* rankone(V)) ./ (1 + δ)

    @testset "positive V matches the PowerMean{1} soft cover" begin
        rng = StableRNG(1)
        V = rand(rng, 12, 20) .+ 0.1
        for δ in (0.0, 0.5)
            a, b, stats = noise_cover(V; shrink=δ)
            @test MatrixCovers.converged(stats)
            @test MatrixCovers.residual(stats) <= MatrixCovers.tolerance(stats)
            ap, bp, _ = soft_cover(PowerMean{1}(), shrunk(V, δ))
            @test (a .* b') .^ 2 ≈ ap .* bp' rtol = 1e-10
            rs, cs = noise_sums(shrunk(V, δ), a, b)
            @test rs ≈ fill(20, 12)
            @test cs ≈ fill(12, 20)
            @test isbalanced(a, b, V)
        end
        # Large `shrink` tends to the rank-one fit itself: (a[i]*b[j])^2 = r[i]*c[j]/s.
        a, b, _ = noise_cover(V; shrink=1e8)
        @test (a .* b') .^ 2 ≈ rankone(V) rtol = 1e-6
    end

    @testset "sparse and dense agree" begin
        rng = StableRNG(2)
        S = sprand(rng, 40, 90, 0.2)
        S = S[vec(count(!iszero, S; dims=2)) .> 2, vec(count(!iszero, S; dims=1)) .> 2]
        for δ in (0.0, 0.3)
            a, b, stats = noise_cover(S; shrink=δ)
            ad, bd, _ = noise_cover(Matrix(S); shrink=δ)
            @test MatrixCovers.converged(stats)
            @test a ≈ ad
            @test b ≈ bd
        end
        # A pattern with zeros for which the scaling exists balances to the targets.
        a, b, _ = noise_cover(S)
        rs, cs = noise_sums(Matrix(S), a, b)
        @test rs ≈ fill(size(S, 2), size(S, 1))
        @test cs ≈ fill(size(S, 1), size(S, 2))
    end

    @testset "zero rows and columns" begin
        V = [0 0 0 0 0; 1 2 0 3 0; 0 0 0 0 0; 4 0 5 6 0.0]
        for δ in (0.0, 0.5)
            a, b, stats = noise_cover(V; shrink=δ)
            @test MatrixCovers.converged(stats)
            @test a[1] == a[3] == b[5] == 0
            @test all(>(0), a[[2, 4]]) && all(>(0), b[1:4])
            # Targets count the nonzero columns (4) and rows (2).
            rs, cs = noise_sums(shrunk(V, δ), a, b)
            @test rs ≈ [0, 4, 0, 4]
            @test cs ≈ [2, 2, 2, 2, 0]
        end
        a, b, stats = noise_cover(zeros(3, 2))
        @test iszero(a) && iszero(b) && MatrixCovers.converged(stats)
    end

    @testset "zero patterns without a scaling" begin
        # Rows 1 and 2 have entries only in column 1: 2*3 > 1*3.
        V = [1.0 0 0; 1 0 0; 1 1 1]
        @test_throws MatrixCovers.SolverFailure noise_cover(V)
        @test_throws "the 2 row(s) 1, 2 have their nonzero entries in only 1 column(s)" noise_cover(V)
        @test_throws "pass `shrink > 0`" noise_cover(sparse(V))
        # The sums are attainable only if V[1,2] vanishes.
        V = [1.0 1; 0 1]
        @test_throws "only in a limit in which some scale factors diverge" noise_cover(V)
        @test_throws "including V[1, 2]" noise_cover(sparse(V))
        # Rescue by shrinkage.
        for V in ([1.0 0 0; 1 0 0; 1 1 1], [1.0 1; 0 1])
            a, b, stats = noise_cover(V; shrink=0.1)
            @test MatrixCovers.converged(stats)
            rs, cs = noise_sums(shrunk(V, 0.1), a, b)
            @test rs ≈ fill(size(V, 2), size(V, 1))
            @test cs ≈ fill(size(V, 1), size(V, 2))
        end
        # Separate blocks whose sizes match the targets admit a scaling.
        a, b, stats = noise_cover([1.0 0; 0 2])
        @test MatrixCovers.converged(stats)
        @test (a[1] * b[1])^2 ≈ 0.5
        @test (a[2] * b[2])^2 ≈ 1.0
        @test isbalanced(a, b, [1.0 0; 0 2])
    end

    @testset "zero pattern near the existence boundary" begin
        # Rows 1:20 have entries only in columns 1:101, and 20*1000 is just below
        # 101*200: the scaling exists, but the sweeps converge slowly and Newton
        # steps finish the solve.
        rng = StableRNG(3)
        m, n = 200, 1000
        V = zeros(m, n)
        for j in 1:n, i in 1:m
            (i <= 20 && j > 101) && continue
            rand(rng) < 0.3 && (V[i, j] = 1 + rand(rng))
        end
        for W in (V, sparse(V))
            a, b, stats = noise_cover(W)
            @test MatrixCovers.converged(stats)
            @test stats.newton
            rs, cs = noise_sums(V, a, b)
            @test rs ≈ fill(n, m)
            @test cs ≈ fill(m, n)
        end
        # A tiny shrink leaves the problem nearly as hard; the rank-one term
        # reaches every Newton solver.
        for linsolve in (:dense, :cg, :cholesky)
            a, b, stats = noise_cover(V; shrink=1e-6, linsolve)
            @test MatrixCovers.converged(stats)
            @test stats.newton
            rs, cs = noise_sums(shrunk(V, 1e-6), a, b)
            @test rs ≈ fill(n, m)
            @test cs ≈ fill(m, n)
        end
    end

    @testset "generic axes" begin
        rng = StableRNG(3)
        V = rand(rng, 6, 9) .+ 0.05
        V[2, 3] = 0
        a, b, _ = noise_cover(V)
        OV = OffsetArray(V, -2:3, 4:12)
        oa, ob, _ = noise_cover(OV)
        @test axes(oa) == (-2:3,) && axes(ob) == (4:12,)
        @test parent(oa) ≈ a && parent(ob) ≈ b
        W = rand(rng, 10, 12) .+ 0.05
        va, vb, _ = noise_cover(view(W, 2:7, 3:11))
        wa, wb, _ = noise_cover(W[2:7, 3:11])
        @test va ≈ wa && vb ≈ wb
    end

    @testset "element types" begin
        V = Float32[3 0 1; 0 2 5; 4 1 0]
        a, b, stats = noise_cover(V)
        @test eltype(a) == eltype(b) == Float32
        @test stats isa MatrixCovers.PowerMeanStats{Float64}
        a64, b64, _ = noise_cover(Float64.(V))
        @test a .* b' ≈ a64 .* b64' rtol = 1e-6
        ai, bi, _ = noise_cover([3 0 1; 0 2 5; 4 1 0])
        @test eltype(ai) == Float64
        @test ai .* bi' ≈ a64 .* b64'
    end

    @testset "input validation" begin
        @test_throws "must be finite and nonnegative, but `V` has an entry -1.0" noise_cover([1.0 -1; 1 1])
        @test_throws "has an entry NaN" noise_cover([1.0 NaN; 1 1])
        @test_throws "has an entry Inf" noise_cover(sparse([1.0 Inf; 1 1]))
        @test_throws "`shrink` must be finite and nonnegative" noise_cover([1.0 1; 1 1]; shrink=-1)
        @test_throws "`shrink` must be finite and nonnegative" noise_cover([1.0 1; 1 1]; shrink=Inf)
        @test_throws MethodError noise_cover([1.0 1; 1 1]; maxiters=3)
        @test_throws MethodError noise_cover(zeros(2, 2); terms=nothing)
    end

    @testset "non-convergence warns" begin
        rng = StableRNG(4)
        V = rand(rng, 8, 8) .+ 0.01
        a, b, stats = @test_logs (:warn, r"noise_cover: the power-mean iteration ended after 1 of maxiter=1 sweeps") noise_cover(V; maxiter=1, newton=false)
        @test !MatrixCovers.converged(stats)
        @test MatrixCovers.iterations(stats) == 1
        a, b, stats = @test_logs (:warn, r"0 of maxnewton=0 Newton steps") noise_cover(V; maxiter=1, maxnewton=0)
        @test !MatrixCovers.converged(stats)
    end
end
