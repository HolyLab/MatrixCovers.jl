# gramcover/gramcover!: symmetric covers of A'*W*A built directly from an
# asymmetric cover of A, without ever forming the Gram matrix.

@testset "gramcover" begin

    @testset "random dense J: exact coverage of A'A" begin
        rng = StableRNG(3)
        J = randn(rng, 8, 5)
        for coverfn in (cover, cover_min)
            a, b = coverfn(J)
            s = gramcover(a, b, J)
            @test all(s * s' .>= abs.(J' * J))
            # Cross-check the bound against a BigFloat computation of J'J.
            Jb = BigFloat.(J)
            Gb = Jb' * Jb
            @test all(BigFloat.(s) * BigFloat.(s)' .>= abs.(Gb))
        end
    end

    @testset "block-diagonal: per-component structure" begin
        rng = StableRNG(1)
        B = randn(rng, 4, 3)
        C = randn(rng, 3, 2)
        J = [B zeros(4, 2); zeros(3, 3) C]
        a, b = cover(J)
        s = gramcover(a, b, J)

        # gramcover on the joint (a, b) restricted to a block equals gramcover on
        # that block alone with the corresponding sub-vectors: the computation is
        # a function of the connected component, and block-diagonal J has disjoint
        # components per block.
        sB = gramcover(a[1:4], b[1:3], B)
        sC = gramcover(a[5:7], b[4:5], C)
        @test isapprox(s, vcat(sB, sC); rtol=1e-9)

        # Entrywise tighter than the naive global bound, strictly so since a
        # second component carries weight.
        @test all(s .<= norm(a) .* b)
        @test any(s .< norm(a) .* b .- 1e-12)
    end

    @testset "gauge invariance" begin
        rng = StableRNG(1)
        B = randn(rng, 4, 3)
        C = randn(rng, 3, 2)
        J = [B zeros(4, 2); zeros(3, 3) C]
        a, b = cover(J)
        s = gramcover(a, b, J)

        a2, b2 = copy(a), copy(b)
        γ = 10.0
        a2[1:4] .*= γ;   b2[1:3] ./= γ
        a2[5:7] .*= 0.3; b2[4:5] ./= 0.3
        s2 = gramcover(a2, b2, J)
        @test isapprox(s, s2; rtol=1e-12)
    end

    @testset "diagonal weights: positive, zero, and negative" begin
        rng = StableRNG(5)
        J = randn(rng, 6, 4)
        a, b = cover(J)
        w = [1.5, 0.0, -2.0, 3.0, 0.0, -0.5]
        s = gramcover(a, b, J, w)
        @test all(s * s' .>= abs.(J' * Diagonal(w) * J))
        @test gramcover(a, b, J, Diagonal(w)) == gramcover(a, b, J, w)
    end

    @testset "dense W: symmetric PSD, nonsymmetric, and component-coupling" begin
        rng = StableRNG(1)
        B = randn(rng, 4, 3)
        C = randn(rng, 3, 2)
        J = [B zeros(4, 2); zeros(3, 3) C]
        a, b = cover(J)
        m = size(J, 1)

        R = randn(rng, m, m)
        W = R'R
        s = gramcover(a, b, J, W)
        G = J' * W * J
        @test all(s * s' .>= abs.(G) .- 1e-9 * maximum(abs, G))

        Wn = randn(rng, m, m)
        sn = gramcover(a, b, J, Wn)
        Gn = J' * Wn * J
        @test all(sn * sn' .>= abs.(Gn) .- 1e-9 * maximum(abs, Gn))

        # A nonzero off-diagonal coupling rows from the two different blocks:
        # component merging must still yield a valid bound.
        Wc = Matrix{Float64}(I, m, m)
        Wc[1, 5] = Wc[5, 1] = 2.0
        sc = gramcover(a, b, J, Wc)
        Gc = J' * Wc * J
        @test all(sc * sc' .>= abs.(Gc) .- 1e-9 * maximum(abs, Gc))
    end

    @testset "sparse J" begin
        Js = sparse([1, 2, 3, 3], [1, 2, 1, 3], [2.0, 3.0, 1.0, 4.0], 3, 3)
        a, b = cover(Js)
        s = gramcover(a, b, Js)
        @test all(s * s' .>= abs.(Matrix(Js)' * Matrix(Js)))
        w = [1.0, -1.0, 2.0]
        sw = gramcover(a, b, Js, w)
        @test all(sw * sw' .>= abs.(Matrix(Js)' * Diagonal(w) * Matrix(Js)))
    end

    @testset "empty column" begin
        J = [1.0 0.0; 2.0 0.0; 0.0 0.0]
        a, b = cover(J)
        s = gramcover(a, b, J)
        @test s[2] == 0
        @test all(s * s' .>= abs.(J' * J))
    end

    @testset "OffsetArray" begin
        J = [4.0 1.0; 1.0 3.0; 2.0 0.5]
        Jo = OffsetArray(J, 0:2, 0:1)
        ao, bo = cover(Jo)
        so = gramcover(ao, bo, Jo)
        @test axes(so, 1) == axes(Jo, 2)

        a, b = cover(J)
        s = gramcover(a, b, J)
        @test collect(so) ≈ s

        # And with an offset weight vector.
        wo = OffsetArray([1.0, 2.0, -0.5], 0:2)
        sow = gramcover(ao, bo, Jo, wo)
        w = collect(wo)
        sw = gramcover(a, b, J, w)
        @test collect(sow) ≈ sw
    end

    @testset "gramcover!" begin
        J = [4.0 1.0; 1.0 3.0; 2.0 0.5]
        a, b = cover(J)
        s = gramcover(a, b, J)

        sbuf = similar(b)
        r = gramcover!(sbuf, a, b, J)
        @test r === sbuf
        @test sbuf == s

        w = [1.0, -2.0, 0.5]
        swbuf = similar(b)
        rw = gramcover!(swbuf, a, b, J, w)
        @test rw === swbuf
        @test swbuf == gramcover(a, b, J, w)

        W = Matrix{Float64}(I, 3, 3)
        sWbuf = similar(b)
        rW = gramcover!(sWbuf, a, b, J, W)
        @test rW === sWbuf
        @test sWbuf == gramcover(a, b, J, W)

        @test_throws "indices of `s` must match column-indexing of `A`" gramcover!(zeros(3), a, b, J)
        @test_throws "indices of `a` must match row-indexing of `A`" gramcover!(similar(b), zeros(4), b, J)
        @test_throws "indices of `b` must match column-indexing of `A`" gramcover!(similar(b), a, zeros(4), J)
        @test_throws "indices of `w` must match row-indexing of `A`" gramcover(a, b, J, zeros(4))
        @test_throws "axes of `W` must equal" gramcover(a, b, J, zeros(4, 4))
    end

end
