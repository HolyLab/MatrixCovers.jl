# gramcover/gramcover!: symmetric covers of A'*W*A built directly from an
# asymmetric cover of A, without ever forming the Gram matrix.

@testset "gramcover" begin

    # Explicit symmetrized block sums covered by `gramcover`.
    function blocksums(a, J, W)
        sc = MatrixCovers.support_components(J)
        k = MatrixCovers.ncomponents(sc)
        M = zeros(k, k)
        for i in axes(W, 1), ip in axes(W, 2)
            ci, cip = MatrixCovers.rowcomponent(sc, i), MatrixCovers.rowcomponent(sc, ip)
            (iszero(ci) || iszero(cip)) && continue
            M[ci, cip] += a[i] * abs(W[i, ip]) * a[ip]
        end
        return [max(M[p, q], M[q, p]) for p in 1:k, q in 1:k]
    end

    # Recover `σ[p]` from any supported column in component `p`.
    function groupscales(s, b, J)
        sc = MatrixCovers.support_components(J)
        return [(j = findfirst(==(p), sc.colcomp); s[j] / b[j])
                for p in 1:MatrixCovers.ncomponents(sc)]
    end

    # Three square blocks on the diagonal, one support component each.
    function threeblocks(rng)
        b1, b2, b3 = randn(rng, 2, 2), randn(rng, 2, 2), randn(rng, 2, 2)
        return [b1 zeros(2, 2) zeros(2, 2)
                zeros(2, 2) b2 zeros(2, 2)
                zeros(2, 2) zeros(2, 2) b3]
    end

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

    @testset "one component: the global bound is attained" begin
        # A single component matches `norm(a)*b` within the documented roundoff
        # inflation.
        for (seed, m, k) in ((3, 5, 4), (7, 4, 4), (11, 12, 3))
            rng = StableRNG(seed)
            J = randn(rng, m, k)
            a, b = cover(J)
            s = gramcover(a, b, J)
            @test ncomponents(support_components(J)) == 1
            @test s ≈ norm(a) .* b rtol = (2 * length(a) + 3) * eps(Float64)
            @test !all(s .<= norm(a) .* b)
            # Coverage, which the margin exists to deliver, is unconditional.
            @test all(s * s' .>= abs.(J' * J))
        end
    end

    @testset "block-diagonal: per-component structure" begin
        rng = StableRNG(1)
        B = randn(rng, 4, 3)
        C = randn(rng, 3, 2)
        J = [B zeros(4, 2); zeros(3, 3) C]
        a, b = cover(J)
        s = gramcover(a, b, J)

        # Each block-diagonal support component is independent.
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

        # Coupled components remain invariant under independent input gauges.
        m = size(J, 1)
        for W in (Matrix(2.0I, m, m) + [i == 1 && ip == 5 for i in 1:m, ip in 1:m],
                  fill(0.5, m, m) + Diagonal(1:m))
            @test isapprox(gramcover(a, b, J, W), gramcover(a2, b2, J, W); rtol=1e-12)
        end

        # A coupling fixes the scale of a component whose diagonal block vanishes.
        Wz = zeros(m, m)
        Wz[1:4, 1:4] .= 1.0
        Wz[1, 5] = Wz[5, 1] = 2.0
        @test isapprox(gramcover(a, b, J, Wz), gramcover(a2, b2, J, Wz); rtol=1e-12)
        sz = gramcover(a, b, J, Wz)
        Gz = J' * Wz * J
        @test all(sz * sz' .>= abs.(Gz) .- 1e-9 * maximum(abs, Gz))
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

        # A single loopless edge admits no gauge-invariant cover.
        Wo = zeros(m, m)
        Wo[1, 5] = Wo[5, 1] = 2.0
        @test_throws "no gauge-invariant cover exists" gramcover(a, b, J, Wo)
        so = gramcover(a, b, J, Wo; degenerate=:uniform)
        Go = J' * Wo * J
        @test all(so * so' .>= abs.(Go) .- 1e-9 * maximum(abs, Go))
        a2, b2 = copy(a), copy(b)
        a2[1:4] .*= 8; b2[1:3] ./= 8
        @test !isapprox(so, gramcover(a2, b2, J, Wo; degenerate=:uniform); rtol=1e-6)
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

    @testset "sparse W matches its dense reading" begin
        rng = StableRNG(21)
        B = randn(rng, 4, 3); C = randn(rng, 3, 2)
        J = [B zeros(4, 2); zeros(3, 3) C]   # two support components
        a, b = cover(J)
        m = size(J, 1)
        Wsp = sparse(1.0I, m, m)
        Wsp[1, 5] = Wsp[5, 1] = 0.5
        s = gramcover(a, b, J, Wsp)
        @test s == gramcover(a, b, J, Matrix(Wsp))
        @test all(s * s' .>= abs.(J' * Matrix(Wsp) * J))
        # Uncoupled weights.
        Dsp = sparse(2.0I, m, m)
        @test gramcover(a, b, J, Dsp) == gramcover(a, b, J, Matrix(Dsp))
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

        @test_throws "`s` holds one Gram scale per support column" gramcover!(zeros(3), a, b, J)
        @test_throws "`a` holds one scale per support row" gramcover!(similar(b), zeros(4), b, J)
        @test_throws "`b` holds one scale per support column" gramcover!(similar(b), a, zeros(4), J)
        @test_throws "`w` holds one weight per support row" gramcover(a, b, J, zeros(4))
        @test_throws "`W` couples support rows" gramcover(a, b, J, zeros(4, 4))
    end

    @testset "SupportComponents form and Diagonal weights" begin
        B = randn(StableRNG(2), 4, 3)
        C = randn(StableRNG(3), 3, 2)
        J = [B zeros(4, 2); zeros(3, 3) C]
        a, b = cover(J)
        w = [1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25]
        sc = MatrixCovers.support_components(J)

        # Passing the components in place of the matrix matches the matrix forms.
        @test gramcover(a, b, sc) == gramcover(a, b, J)
        @test gramcover(a, b, sc, w) == gramcover(a, b, J, w)

        # `W::Diagonal` is the vector-weighted form, for every entry point.
        @test gramcover(a, b, J, Diagonal(w)) == gramcover(a, b, J, w)
        @test gramcover(a, b, sc, Diagonal(w)) == gramcover(a, b, sc, w)

        sbuf = similar(b)
        @test gramcover!(sbuf, a, b, sc) === sbuf
        @test sbuf == gramcover(a, b, J)
        @test gramcover!(sbuf, a, b, J, Diagonal(w)) === sbuf
        @test sbuf == gramcover(a, b, J, w)
        @test gramcover!(sbuf, a, b, sc, Diagonal(w)) === sbuf
        @test sbuf == gramcover(a, b, sc, w)

        # Matrix-weight methods accept and validate `degenerate` consistently.
        @test gramcover(a, b, J, Diagonal(w); degenerate=:uniform) == gramcover(a, b, J, w)
        @test gramcover(a, b, sc, Diagonal(w); degenerate=:uniform) == gramcover(a, b, sc, w)
        @test gramcover!(sbuf, a, b, J, Diagonal(w); degenerate=:uniform) === sbuf
        @test sbuf == gramcover(a, b, J, w)
        @test_throws "`degenerate` must be :error or :uniform" gramcover(a, b, J, Diagonal(w); degenerate=:nonsense)
        @test_throws "`degenerate` must be :error or :uniform" gramcover(a, b, sc, Diagonal(w); degenerate=:nonsense)
        @test_throws "`degenerate` must be :error or :uniform" gramcover!(sbuf, a, b, J, Diagonal(w); degenerate=:nonsense)
        @test_throws "`degenerate` must be :error or :uniform" gramcover!(sbuf, a, b, sc, Diagonal(w); degenerate=:nonsense)
    end

    @testset "general W leaves an uncoupled component as its own group" begin
        # Three support components; `W` couples the first two and leaves the third
        # alone, so that third component forms a singleton group of its own.
        rng = StableRNG(5)
        B = randn(rng, 2, 2); C = randn(rng, 2, 2); D = randn(rng, 2, 2)
        J = [B zeros(2, 4); zeros(2, 2) C zeros(2, 2); zeros(2, 4) D]
        a, b = cover(J)
        @test MatrixCovers.ncomponents(MatrixCovers.support_components(J)) == 3

        W = Matrix{Float64}(I, 6, 6)
        W[1, 3] = W[3, 1] = 1.5      # couples component 1 (rows 1-2) to component 2 (rows 3-4)
        s = gramcover(a, b, J, W)
        G = J' * W * J
        @test all(s * s' .>= abs.(G) .- 1e-9 * maximum(abs, G))
        # The uncoupled third component (W is the identity there) reproduces the
        # unweighted cover, up to the roundoff inflation.
        @test s[5:6] ≈ gramcover(a, b, J)[5:6]
    end

    @testset "two coupled components, both diagonal blocks nonzero" begin
        # Closed form for a 2×2 `Ms` with positive diagonal.
        rng = StableRNG(9)
        B = randn(rng, 4, 3); C = randn(rng, 3, 2)
        J = [B zeros(4, 2); zeros(3, 3) C]
        a, b = cover(J)
        m = size(J, 1)
        κs = Float64[]
        for c in (0.05, 40.0)
            W = Matrix{Float64}(I, m, m)
            W[1, 5] = W[5, 1] = c
            s = gramcover(a, b, J, W)
            Ms = blocksums(a, J, W)
            κ = Ms[1, 2] / sqrt(Ms[1, 1] * Ms[2, 2])
            push!(κs, κ)
            σ = groupscales(s, b, J)
            @test σ ≈ sqrt.([Ms[1, 1], Ms[2, 2]]) .* sqrt(max(1, κ)) rtol = 1e-6
            G = J' * W * J
            @test all(s * s' .>= abs.(G))
        end
        @test κs[1] < 1 < κs[2]
    end

    @testset "two coupled components, one diagonal block zero" begin
        # For `Ms = [d e; e 0]`, the minimal cover saturates its two constraints.
        rng = StableRNG(11)
        B = randn(rng, 4, 3); C = randn(rng, 3, 2)
        J = [B zeros(4, 2); zeros(3, 3) C]
        a, b = cover(J)
        m = size(J, 1)
        W = zeros(m, m)
        W[1:4, 1:4] .= 1.0
        W[1, 5] = W[5, 1] = 2.0
        s = gramcover(a, b, J, W)
        Ms = blocksums(a, J, W)
        @test iszero(Ms[2, 2])
        σ = groupscales(s, b, J)
        @test σ[1] ≈ sqrt(Ms[1, 1]) rtol = 1e-6
        @test σ[2] ≈ Ms[1, 2] / sqrt(Ms[1, 1]) rtol = 1e-6
        G = J' * W * J
        @test all(s * s' .>= abs.(G))
    end

    @testset "three coupled components in a triangle, every diagonal block zero" begin
        # An odd cycle fixes the gauge without a diagonal block.
        rng = StableRNG(13)
        J = threeblocks(rng)
        a, b = cover(J)
        m = size(J, 1)
        W = zeros(m, m)
        W[1, 3] = W[3, 1] = 1.5
        W[1, 5] = W[5, 1] = 0.75
        W[3, 5] = W[5, 3] = 2.25
        s = gramcover(a, b, J, W)
        Ms = blocksums(a, J, W)
        @test all(iszero, [Ms[p, p] for p in 1:3])
        σ = groupscales(s, b, J)
        @test σ ≈ [sqrt(Ms[1, 2] * Ms[1, 3] / Ms[2, 3]),
                   sqrt(Ms[1, 2] * Ms[2, 3] / Ms[1, 3]),
                   sqrt(Ms[1, 3] * Ms[2, 3] / Ms[1, 2])] rtol = 1e-6
        G = J' * W * J
        @test all(s * s' .>= abs.(G))

        # The result remains gauge-invariant.
        a2, b2 = copy(a), copy(b)
        for (p, (rows, cols)) in enumerate(((1:2, 1:2), (3:4, 3:4), (5:6, 5:6)))
            γ = (2.0, 0.3, 7.0)[p]
            a2[rows] .*= γ; b2[cols] ./= γ
        end
        @test isapprox(s, gramcover(a2, b2, J, W); rtol=1e-6)
    end

    @testset "loopless bipartite coupling graphs are refused" begin
        # An even cycle without loops is bipartite.
        rng = StableRNG(17)
        blocks = [randn(rng, 2, 2) for _ in 1:4]
        J = zeros(8, 8)
        for p in 1:4
            J[2p-1:2p, 2p-1:2p] .= blocks[p]
        end
        a, b = cover(J)
        @test MatrixCovers.ncomponents(MatrixCovers.support_components(J)) == 4
        W = zeros(8, 8)
        for (i, ip) in ((1, 3), (3, 5), (5, 7), (7, 1))
            W[i, ip] = W[ip, i] = 1.0 + 0.5 * i
        end
        @test_throws "no gauge-invariant cover exists" gramcover(a, b, J, W)
        s = gramcover(a, b, J, W; degenerate=:uniform)
        G = J' * W * J
        @test all(s * s' .>= abs.(G))

        @test_throws "`degenerate` must be :error or :uniform" gramcover(a, b, J, W; degenerate=:nonsense)

        # One loop removes the obstruction.
        W[1, 1] = 1.0
        s1 = gramcover(a, b, J, W)
        G1 = J' * W * J
        @test all(s1 * s1' .>= abs.(G1))
    end

    @testset "generic W over three coupled components" begin
        # Compare the minimal cover with a valid diagonal-normalized cover.
        rng = StableRNG(19)
        J = threeblocks(rng)
        a, b = cover(J)
        m = size(J, 1)
        W = abs.(randn(rng, m, m)) .+ 0.1
        s = gramcover(a, b, J, W)
        G = J' * W * J
        @test all(s * s' .>= abs.(G))

        Ms = blocksums(a, J, W)
        σ = groupscales(s, b, J)
        @test σ ≈ symcover_min(AbsLog{2}(), Ms) rtol = 1e-6

        # This diagonal-normalized formula covers `Ms` but need not be minimal.
        d = [Ms[p, p] for p in 1:3]
        σh = [sqrt(sum(Ms[p, q] * sqrt(d[p] / d[q]) for q in 1:3)) for p in 1:3]
        @test all(σh * σh' .>= Ms)
        @test cover_objective(AbsLog{2}(), σ, Ms) <= cover_objective(AbsLog{2}(), σh, Ms)

        a2, b2 = copy(a), copy(b)
        for (p, γ) in enumerate((5.0, 0.2, 1.7))
            a2[2p-1:2p] .*= γ; b2[2p-1:2p] ./= γ
        end
        @test isapprox(s, gramcover(a2, b2, J, W); rtol=1e-6)
    end

end
