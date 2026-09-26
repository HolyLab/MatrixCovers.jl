# gramcover/gramcover!: symmetric covers of A'*W*A computed from A and W, and
# cover(A, a): the cover of A for a fixed row scale.

@testset "gramcover" begin

    # `s*s'` covers `abs.(G)`, compared in BigFloat so that `G` is (nearly) exact.
    function covers_gram(s, A, W)
        Ab = BigFloat.(A)
        G = Ab' * BigFloat.(W) * Ab
        sb = BigFloat.(s)
        return all(sb * sb' .>= abs.(G))
    end

    # Random sparse pattern with lognormal magnitudes spanning many decades.
    function lognormal_matrix(rng, m, n; density=0.4, σ=4.0)
        L = zeros(m, n)
        for j in 1:n, i in 1:m
            rand(rng) < density || continue
            L[i, j] = (rand(rng, Bool) ? 1 : -1) * exp(σ * randn(rng))
        end
        return L
    end

    rng = StableRNG(17)
    A = [4.0 1.0 0.0 0.0
         1.0 3.0 2.0 0.0
         0.0 2.0 5.0 0.0
         1.0 0.0 1.0 0.0
         0.0 0.0 0.5 0.0]   # column 4 has no support
    m, n = size(A)

    @testset "unweighted and diagonal weights" begin
        for AA in (A, sparse(A))
            s = gramcover(AA)
            @test covers_gram(s, A, I(m))
            @test s[4] == 0
            @test s .^ 2 ≈ diag(A' * A)
            # Mixed signs: a cover, but not generally minimal.
            w = [1.0, -2.0, 0.5, 3.0, -0.25]
            sw = gramcover(AA, w)
            @test covers_gram(sw, A, Diagonal(w))
            @test gramcover(AA, Diagonal(w)) == sw
            @test sw .^ 2 ≈ diag(A' * Diagonal(abs.(w)) * A)
            @test sw[4] == 0
            # Zero weights drop their rows.
            wz = [1.0, 0.0, 2.0, 0.0, 1.0]
            sz = gramcover(AA, wz)
            @test covers_gram(sz, A, Diagonal(wz))
            @test sz ≈ gramcover(A[[1, 3, 5], :], wz[[1, 3, 5]])
        end
    end

    @testset "minimality for nonnegative weights" begin
        for _ in 1:5
            B = randn(rng, 7, 4)
            w = rand(rng, 7)
            s = gramcover(B, w)
            @test s .^ 2 ≈ diag(B' * Diagonal(w) * B)
            @test covers_gram(s, B, Diagonal(w))
        end
    end

    @testset "Cholesky weight" begin
        for _ in 1:5
            X = randn(rng, m, m)
            W = X * X' + 0.1I
            for AA in (A, sparse(A))
                s = gramcover(AA, cholesky(W))
                @test s .^ 2 ≈ diag(A' * W * A)
                @test covers_gram(s, A, W)
                @test s[4] == 0
            end
            # Lower-triangular storage gives the same scales.
            @test gramcover(A, cholesky(Symmetric(W, :L))) ≈ gramcover(A, cholesky(W))
        end
    end

    @testset "general W" begin
        # Indefinite and asymmetric.
        W = randn(rng, m, m)
        @test !issymmetric(W)
        for AA in (A, sparse(A))
            s = gramcover(AA, W)
            @test covers_gram(s, A, W)
            @test s[4] == 0
        end
        # A user-supplied `v`.
        v = fill(nextfloat(sqrt(maximum(abs, W))), m)
        s = gramcover(A, W; v)
        @test s ≈ v[1] .* vec(sum(abs, A; dims=1))
        @test covers_gram(s, A, W)
        # Sparse W.
        Ws = sprandn(rng, m, m, 0.4) + I
        @test covers_gram(gramcover(A, Ws), A, Ws)
        # For diagonal W, the general bound is looser than the Diagonal method
        # by at most sqrt(n_j).
        w = rand(rng, m) .+ 0.1
        sg = gramcover(A, Matrix(Diagonal(w)))
        sd = gramcover(A, w)
        nj = vec(count(!iszero, A; dims=1))
        @test all(sd .<= sg)
        @test all(sg .<= sqrt.(nj) .* sd .* (1 + 1e-12))
    end

    @testset "sandwich against the implied cover of A" begin
        for _ in 1:5
            B = lognormal_matrix(rng, 9, 6; density=0.5)
            w = exp.(2 .* randn(rng, 9))
            _, b = cover(B, 1 ./ sqrt.(w))
            s = gramcover(B, w)
            nj = vec(count(!iszero, B; dims=1))
            @test all(b .<= s)
            @test all(s .<= sqrt.(nj) .* b .* (1 + 1e-12))
        end
    end

    @testset "scale covariance" begin
        B = randn(rng, 6, 4)
        w = randn(rng, 6)
        d1 = exp.(3 .* randn(rng, 6))
        d2 = exp.(3 .* randn(rng, 4))
        B2 = Diagonal(d1) * B * Diagonal(d2)
        w2 = w ./ d1 .^ 2
        @test gramcover(B2, w2) ≈ d2 .* gramcover(B, w) rtol = 1e-12
        X = randn(rng, 6, 6)
        W = X * X' + I
        W2 = Diagonal(1 ./ d1) * W * Diagonal(1 ./ d1)
        @test gramcover(B2, cholesky(Symmetric(W2))) ≈ d2 .* gramcover(B, cholesky(W)) rtol = 1e-10
        Wg = randn(rng, 6, 6)
        Wg2 = Diagonal(1 ./ d1) * Wg * Diagonal(1 ./ d1)
        @test gramcover(B2, Wg2) ≈ d2 .* gramcover(B, Wg) rtol = 1e-8
    end

    @testset "covering in floating point, wide dynamic range" begin
        for _ in 1:20
            B = lognormal_matrix(rng, 12, 8)
            w = (rand(rng, 12) .- 0.3) .* exp.(4 .* randn(rng, 12))
            @test covers_gram(gramcover(B), B, I(12))
            @test covers_gram(gramcover(B, w), B, Diagonal(w))
            # Against the Gram matrix formed in Float64.
            s = gramcover(B, abs.(w))
            @test all(s * s' .>= abs.(B' * Diagonal(abs.(w)) * B))
            Wg = lognormal_matrix(rng, 12, 12; density=0.5)
            @test covers_gram(gramcover(B, Wg), B, Wg)
        end
    end

    @testset "generic axes" begin
        w = [1.0, -2.0, 0.5, 3.0, -0.25]
        W = randn(rng, m, m)
        Ao = OffsetArray(A, -3, 5)
        wo = OffsetArray(w, -3)
        Wo = OffsetArray(W, -3, -3)
        for (s, so) in ((gramcover(A), gramcover(Ao)),
                        (gramcover(A, w), gramcover(Ao, wo)),
                        (gramcover(A, W), gramcover(Ao, Wo)))
            @test axes(so) == (axes(Ao, 2),)
            @test parent(so) == s
        end
        # A factorization has one-based axes; only the columns of `A` may be offset.
        X = randn(rng, m, m)
        C = cholesky(X * X' + I)
        Ac = OffsetArray(A, 0, 5)
        sc = gramcover(Ac, C)
        @test axes(sc) == (axes(Ac, 2),)
        @test parent(sc) == gramcover(A, C)
        @test_throws "axes(A, 1) must be" gramcover(Ao, C)
        # Views.
        Av = view(A, 1:4, 2:4)
        @test gramcover(Av, w[1:4]) == gramcover(A[1:4, 2:4], w[1:4])
        @test gramcover(Av, C.U[1:4, 1:4]' * C.U[1:4, 1:4] |> cholesky) ≈
              gramcover(A[1:4, 2:4], C.U[1:4, 1:4]' * C.U[1:4, 1:4] |> cholesky)
        @test gramcover(Av, W[1:4, 1:4]) == gramcover(A[1:4, 2:4], W[1:4, 1:4])
        # cover(A, a)
        a = rand(rng, m) .+ 0.5
        ao, bo = cover(Ao, OffsetArray(a, -3))
        @test axes(bo) == (axes(Ao, 2),)
        @test parent(bo) == cover(A, a)[2]
    end

    @testset "gramcover!" begin
        w = rand(rng, m)
        W = randn(rng, m, m)
        C = cholesky(W * W' + I)
        s = zeros(n)
        @test gramcover!(s, A) === s
        @test s == gramcover(A)
        @test gramcover!(s, A, w) === s
        @test s == gramcover(A, w)
        @test gramcover!(s, A, Diagonal(w)) === s
        @test s == gramcover(A, w)
        @test gramcover!(s, A, C) === s
        @test s == gramcover(A, C)
        @test gramcover!(s, A, W) === s
        @test s == gramcover(A, W)
        v = fill(nextfloat(sqrt(maximum(abs, W))), m)
        @test gramcover!(s, A, W; v) === s
        @test s == gramcover(A, W; v)
    end

    @testset "argument errors" begin
        W = randn(rng, m, m)
        @test_throws DimensionMismatch gramcover!(zeros(3), A)
        @test_throws "eachindex(s) must be" gramcover!(zeros(3), A)
        @test_throws "eachindex(s) must be" gramcover!(zeros(3), A, ones(m))
        @test_throws "eachindex(s) must be" gramcover!(zeros(3), A, cholesky(W * W' + I))
        @test_throws "eachindex(s) must be" gramcover!(zeros(3), A, W)
        @test_throws DimensionMismatch gramcover(A, ones(3))
        @test_throws "eachindex(w) must be" gramcover(A, ones(3))
        @test_throws "eachindex(w) must be" gramcover(A, Diagonal(ones(3)))
        @test_throws "axes(W) must be" gramcover(A, zeros(3, 3))
        @test_throws "axes(W) must be" gramcover(A, zeros(m, m + 1))
        @test_throws "axes(W) must be" gramcover(A, zeros(3, 3); v=ones(3))
        @test_throws "axes(A, 1) must be" gramcover(A, cholesky(Matrix(1.0I, 3, 3)))
        @test_throws "eachindex(v) must be" gramcover(A, W; v=ones(3))
        vneg = fill(10.0, m); vneg[2] = -1
        @test_throws "`v` must be nonnegative" gramcover(A, W; v=vneg)
        @test_throws "abs(W[i,k]) <= v[i]*v[k]" gramcover(A, W; v=fill(1e-3, m))
    end
end

@testset "cover(A, a)" begin
    rng = StableRNG(5)
    A = [4.0 1.0 0.0 0.0
         1.0 3.0 2.0 0.0
         0.0 2.0 5.0 0.0
         1.0 0.0 1.0 0.0]
    a = [1.0, 2.0, 0.5, 4.0]
    a2, b = cover(A, a)
    @test a2 === a
    @test b == [4.0, 4.0, 10.0, 0.0]
    @test iscover(a, b, A)
    @test cover(sparse(A), a)[2] == b

    for _ in 1:20
        B = sprandn(rng, 10, 7, 0.4)
        B = B .* exp.(4 .* randn(rng, 10))
        a = exp.(3 .* randn(rng, 10))
        _, b = cover(B, a)
        @test iscover(a, b, B)
        # Tight in every supported column.
        for j in axes(B, 2)
            col = abs.(B[:, j])
            if iszero(col)
                @test b[j] == 0
            else
                @test maximum(col ./ (a .* b[j])) ≈ 1 rtol = 4eps()
            end
        end
        # With `a` from a full cover, `b` is no looser than that cover's.
        a0, b0 = cover(B)
        _, b1 = cover(B, a0)
        @test iscover(a0, b1, B)
        @test all(b1 .<= b0)
    end

    # Unsupported rows may have a zero scale; supported rows may not.
    A0 = [1.0 2.0; 0.0 0.0]
    @test cover(A0, [1.0, 0.0])[2] == [1.0, 2.0]
    @test_throws ArgumentError cover(A, [1.0, 0.0, 1.0, 1.0])
    @test_throws "positive on every row that has a stored nonzero" cover(A, [1.0, 0.0, 1.0, 1.0])
    @test_throws "must be nonnegative" cover(A0, [1.0, -1.0])
    @test_throws "must be nonnegative" cover(A, [1.0, NaN, 1.0, 1.0])
    @test_throws DimensionMismatch cover(A, ones(3))
end
