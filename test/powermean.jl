# PowerMean soft covers: closed forms, balance identities, descent, covariance,
# the overrelaxation safeguard, and generic indexing.

# Largest |∑_j R[i,j]^p / n_i - 1| over supported rows and |∑_i R[i,j]^p / m_j - 1|
# over supported columns.
function pm_imbalance(p, a, b, A)
    R = zeros(axes(A)...)
    n = zeros(Int, axes(A, 1)); m = zeros(Int, axes(A, 2))
    MatrixCovers.foreach_support(A) do i, j, v
        R[i, j] = (v / (a[i] * b[j]))^p
        n[i] += 1; m[j] += 1
    end
    rs = [abs(sum(R[i, :]) / n[i] - 1) for i in axes(A, 1) if n[i] > 0]
    cs = [abs(sum(R[:, j]) / m[j] - 1) for j in axes(A, 2) if m[j] > 0]
    return max(maximum(rs; init=0.0), maximum(cs; init=0.0))
end

# `A` with entries spanning `σ` natural-log units, a fraction `z` of them zeroed,
# and the listed rows and columns emptied.
function pm_testmatrix(rng, m, n; σ=3.0, z=0.3, zrows=(), zcols=())
    A = exp.(σ .* randn(rng, m, n)) .* sign.(randn(rng, m, n))
    A[rand(rng, m, n) .< z] .= 0
    for i in zrows; A[i, :] .= 0; end
    for j in zcols; A[:, j] .= 0; end
    return A
end

@testset "PowerMean soft covers" begin

    @testset "symmetric 2×2 closed form" begin
        for κ in (1/4, 4.0, 32.0), p in (1, 2, 3)
            m = sqrt(κ)
            A = [1.0 -m; -m 1.0]
            R11 = (2 / (1 + κ^(p/2)))^(1/p)
            R12 = (2 * κ^(p/2) / (1 + κ^(p/2)))^(1/p)
            E = (4 / p) * log(cosh((p / 4) * log(κ)))
            a = soft_symcover(PowerMean{p}(), A)
            R = abs.(A) ./ (a .* a')
            @test R[1, 1] ≈ R11 rtol=1e-13
            @test R[2, 2] ≈ R11 rtol=1e-13
            @test R[1, 2] ≈ R12 rtol=1e-13
            @test cover_objective(PowerMean{p}(), a, A) ≈ E rtol=1e-12
            # The asymmetric solver returns symmetric products on symmetric input.
            x, y = soft_cover(PowerMean{p}(), A)
            Rxy = abs.(A) ./ (x .* y')
            @test Rxy ≈ R rtol=1e-12
            @test Rxy ≈ Rxy' rtol=1e-12
            @test cover_objective(PowerMean{p}(), x, y, A) ≈ E rtol=1e-12
        end
    end

    @testset "balance identities" begin
        rng = StableRNG(314)
        # Larger `p` on sparse support with wide dynamic range can leave the weights
        # `R.^p` spanning tens of decades, where the first-order iteration crawls;
        # dense support is exercised at larger `p` below.
        for p in (1, 2)
            ϕ = PowerMean{p}()
            dense = pm_testmatrix(rng, 30, 30; σ=2.0, z=0.0)
            rect = pm_testmatrix(rng, 40, 17; σ=4.0, zrows=(3, 11), zcols=(5,))
            wide = pm_testmatrix(rng, 12, 45; σ=6.0, z=0.5, zrows=(1,), zcols=(2, 44))
            sp = sprand(rng, 80, 60, 0.08)
            sp.nzval .= exp.(3 .* randn(rng, nnz(sp)))
            sp[7, :] .= 0; sp[:, 9] .= 0; dropzeros!(sp)
            for A in (dense, rect, wide, sp)
                a, b = @test_logs soft_cover(ϕ, A)
                @test pm_imbalance(p, a, b, A) < 1e-11
                @test isbalanced(a, b, A)
                @test all(iszero(a[i]) == all(iszero, A[i, :]) for i in axes(A, 1))
                @test all(iszero(b[j]) == all(iszero, A[:, j]) for j in axes(A, 2))
            end
            @test soft_cover(ϕ, sp)[1] isa Vector{Float64}

            S = pm_testmatrix(rng, 35, 35; σ=3.0, zrows=(4,), zcols=(4,))
            S = S + S'
            Ssp = sparse(S)
            for M in (S, Ssp, Symmetric(Ssp))
                a = soft_symcover(ϕ, M)
                @test pm_imbalance(p, a, a, M) < 1e-11
                @test a[4] == 0
                # The unique minimizer is symmetric, so the asymmetric solver agrees.
                x, y = soft_cover(ϕ, M)
                supp = findall(!iszero, S)
                @test (x .* y')[supp] ≈ (a .* a')[supp] rtol=1e-10
                @test soft_symcover_min(ϕ, M) == a
            end
        end
        for p in (3, 3.5, 8)
            A = pm_testmatrix(rng, 30, 20; σ=3.0, z=0.0)
            a, b = @test_logs soft_cover(PowerMean{p}(), A)
            @test pm_imbalance(p, a, b, A) < 1e-11
        end
        S = pm_testmatrix(rng, 20, 20; σ=2.0, z=0.0)
        S = S + S'
        s = @test_logs soft_symcover(PowerMean{3}(), S)
        @test pm_imbalance(3, s, s, S) < 1e-11
    end

    @testset "symmetric iteration decreases the objective" begin
        rng = StableRNG(2)
        X = pm_testmatrix(rng, 30, 30; σ=3.0, z=0.4)
        A = X + X'
        a0 = initialize_symcover(A; strategy=:geomean, feasible=:none)
        E = Float64[]
        for k in 0:12
            ak = @test_logs (:warn, r"soft_symcover!: the power-mean iteration ended after") match_mode=:any soft_symcover!(PowerMean{2}(), copy(a0), A; maxiter=k)
            push!(E, cover_objective(PowerMean{2}(), ak, A))
        end
        @test all(<(0), diff(E))
        @test cover_objective(PowerMean{2}(), soft_symcover(A), A) < E[end]
    end

    @testset "non-convergence warns" begin
        rng = StableRNG(3)
        A = exp.(3 .* randn(rng, 30, 30))
        @test_logs (:warn, r"^soft_cover: the power-mean iteration ended after 3 of maxiter=3 sweeps") soft_cover(A; maxiter=3)
        @test_logs (:warn, r"^soft_cover_min: .*Increase `maxiter`") soft_cover_min(A; maxiter=3)
        @test_logs (:warn, r"^soft_symcover: the power-mean iteration") soft_symcover(A + A'; maxiter=2)
        # A looser tolerance is honored.
        a, b = @test_logs soft_cover(A; tol=1e-4)
        @test 1e-11 < pm_imbalance(2, a, b, A) <= 1e-4
    end

    @testset "scale covariance" begin
        rng = StableRNG(17)
        A = pm_testmatrix(rng, 25, 18; σ=2.0, z=0.3)
        d1 = exp.(4 .* randn(rng, 25)); d2 = exp.(4 .* randn(rng, 18))
        supp = findall(!iszero, A)
        for p in (1, 2, 3)
            a, b = soft_cover(PowerMean{p}(), A)
            aD, bD = soft_cover(PowerMean{p}(), d1 .* A .* d2')
            @test (aD .* bD')[supp] ≈ (d1 .* (a .* b') .* d2')[supp] rtol=1e-11
        end
        S = pm_testmatrix(rng, 25, 25; σ=2.0, z=0.3)
        S = S + S'
        for p in (1, 2, 3)
            a = soft_symcover(PowerMean{p}(), S)
            @test soft_symcover(PowerMean{p}(), d1 .* S .* d1') ≈ d1 .* a rtol=1e-11
        end
    end

    @testset "violation bound" begin
        rng = StableRNG(23)
        for (p, A) in ((1, pm_testmatrix(rng, 20, 30; σ=3.0, z=0.6)),
                       (2, pm_testmatrix(rng, 20, 30; σ=3.0, z=0.6)),
                       (3, pm_testmatrix(rng, 20, 30; σ=2.0, z=0.2)),
                       (1, sparse(pm_testmatrix(rng, 50, 50; σ=3.0, z=0.9))),
                       (2, sparse(pm_testmatrix(rng, 50, 50; σ=3.0, z=0.9))))
            a, b = @test_logs soft_cover(PowerMean{p}(), A)
            n = vec(count(!iszero, A; dims=2)); m = vec(count(!iszero, A; dims=1))
            for I in findall(!iszero, A)
                i, j = Tuple(I)
                @test abs(A[i, j]) / (a[i] * b[j]) <= min(n[i], m[j])^(1/p) * (1 + 1e-12)
            end
        end
    end

    @testset "overrelaxation reaches slow problems" begin
        rng = StableRNG(5)
        W = exp.(3 .* randn(rng, 50, 50))
        B = Matrix(Bidiagonal(fill(10.0, 100), ones(99), :U))
        for A in (W, B, sparse(B))
            a, b = @test_logs soft_cover(A)
            @test pm_imbalance(2, a, b, A) < 1e-11
        end
        # From a flat start the bidiagonal problem takes over 10^5 plain sweeps; the
        # relaxed iteration must finish within the default limit.
        a, b = @test_logs soft_cover!(PowerMean{2}(), ones(100), ones(100), B)
        @test pm_imbalance(2, a, b, B) < 1e-11
        nsweeps = MatrixCovers._powermean_sinkhorn!(zeros(100), zeros(100),
            MatrixCovers._log_support(MatrixCovers._row_support(B, Float64)),
            MatrixCovers._log_support(MatrixCovers._col_support(B, Float64)),
            2.0, 10^5, 4096 * eps())[2]
        @test nsweeps < 10_000
        # Wide dynamic range with sparse support.
        Ssp = sprand(rng, 200, 200, 0.03) + I
        Ssp.nzval .= exp.(4.6 .* randn(rng, nnz(Ssp)))
        a, b = @test_logs soft_cover(Ssp)
        @test pm_imbalance(2, a, b, Ssp) < 1e-11
        # The symmetric iteration handles diagonally dominant banded support directly.
        T = SymTridiagonal(fill(10.0, 100), ones(99))
        a = @test_logs soft_symcover(T)
        @test pm_imbalance(2, a, a, T) < 1e-11
    end

    @testset "start independence and the bipartite gauge" begin
        A = [1.0 2.0 0.5; 0.25 3.0 1.0]
        ref = soft_cover(A)
        a, b = soft_cover!(PowerMean{2}(), [5.0, 0.1], [1.0, 20.0, 0.3], A)
        @test a .* b' ≈ ref[1] .* ref[2]' rtol=1e-12
        @test isbalanced(a, b, A)
        # Loopless bipartite symmetric support: products fixed, sides balanced.
        S = [0 3.0 0; 3.0 0 5.0; 0 5.0 0]
        s = soft_symcover(S)
        @test s[1] * s[2] ≈ 3 && s[2] * s[3] ≈ 5
        @test 2 * log(s[2]) ≈ log(s[1]) + log(s[3])
        @test soft_symcover!(PowerMean{2}(), [7.0, 0.01, 2.0], S) ≈ s
        @test soft_symcover([0 1; 1 0]) ≈ [1.0, 1.0]
    end

    @testset "element types" begin
        A = [4.0 1.5 0.5; 1.5 1.0 2.0; 0.5 2.0 3.0]
        a64 = soft_symcover(A)
        a32 = soft_symcover(Float32.(A))
        @test a32 isa Vector{Float32}
        @test a32 ≈ a64 rtol=2eps(Float32)
        abig = soft_symcover(big.(A))
        @test abig isa Vector{BigFloat}
        Rbig = abs.(big.(A)) ./ (abig .* abig')
        @test maximum(abs, sum(Rbig .^ 2; dims=2) ./ 3 .- 1) < 1e-70
        x, y = soft_cover!(PowerMean{2}(), [1.0, 1.0, 1.0], Float32[1, 1, 1], A)
        @test eltype(x) === Float64 && eltype(y) === Float32
    end

    @testset "generic indexing" begin
        rng = StableRNG(8)
        A = pm_testmatrix(rng, 6, 4; σ=2.0, z=0.2, zrows=(2,))
        a, b = soft_cover(A)
        Ao = OffsetArray(A, -2:3, 5:8)
        ao, bo = soft_cover(Ao)
        @test axes(ao, 1) == axes(Ao, 1) && axes(bo, 1) == axes(Ao, 2)
        @test collect(ao) == a && collect(bo) == b
        Av = view(hcat(A, ones(6)), :, 1:4)
        @test soft_cover(Av) == (a, b)
        ao2, bo2 = soft_cover!(PowerMean{2}(), OffsetArray(ones(6), -2:3), OffsetArray(ones(4), 5:8), Ao)
        @test collect(ao2) .* collect(bo2)' ≈ a .* b' rtol=1e-12

        S = A[1:4, :] + A[1:4, :]'
        s = soft_symcover(S)
        So = OffsetArray(S, 0:3, 0:3)
        so = soft_symcover(So)
        @test axes(so, 1) == axes(So, 1)
        @test collect(so) == s
        @test soft_symcover(view(S, 1:4, 1:4)) == s
        @test_throws DimensionMismatch soft_symcover!(PowerMean{2}(), ones(3), S)
        @test_throws DimensionMismatch soft_cover!(PowerMean{2}(), ones(5), ones(4), A)
        @test_throws "soft_symcover requires a square matrix" soft_symcover(A)
    end

    @testset "default dispatch" begin
        A = [1.0 2.0 3.0; 6.0 5.0 4.0]
        S = [4.0 1.0; 1.0 2.0]
        @test soft_cover(A) == soft_cover(PowerMean{2}(), A) == soft_cover_min(A)
        @test soft_symcover(S) == soft_symcover(PowerMean{2}(), S) == soft_symcover_min(S)
        a, b = soft_cover(A)
        for refine! in (soft_cover!, soft_cover_min!)
            x, y = refine!(ones(2), ones(3), A)
            @test x .* y' ≈ a .* b' rtol=1e-12
        end
        @test soft_symcover!(ones(2), S) ≈ soft_symcover(S) rtol=1e-12
        @test soft_symcover_min!(ones(2), S) ≈ soft_symcover(S) rtol=1e-12
    end
end
