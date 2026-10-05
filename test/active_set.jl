using MatrixCovers: _cover_min_abslog2, _polish_difference_qp

@testset "active-set solution of stalled minimal covers" begin
    # `maxouter=1` stops the augmented Lagrangian far from convergence, so the
    # result rests on the active-set solve.
    idx_sub = Set(round.(Int, range(1, length(general_matrices), length=100)))
    npolish = 0
    for (k, (_, A)) in enumerate(general_matrices)
        k in idx_sub || continue
        Af = Float64.(A)
        aj, bj = MatrixCovers.cover_min_jump(AbsLog{2}(), Af)
        for linsolve in (:dense, :lsqr)
            a, b, s = _cover_min_abslog2(Af; maxouter=1, linsolve)
            haskey(s, :polish) && (npolish += 1)
            @test s.converged
            @test iscover(a, b, Af)
            @test logproduct_gap(a, b, aj, bj, Af) <= 1e-6
        end
        if all(!iszero, Af)
            a, b, s = _cover_min_abslog2(Af; maxouter=1, linsolve=:woodbury)
            @test logproduct_gap(a, b, aj, bj, Af) <= 1e-6
        end
        size(Af, 1) == size(Af, 2) || continue
        (atj, btj), _ = MatrixCovers.cover_transversal_jump(Af)
        a, b, s = _cover_min_abslog2(Af; transversal=true, maxouter=1, fname=:cover_transversal)
        @test iscover(a, b, Af)
        @test logproduct_gap(a, b, atj, btj, Af) <= 1e-6
    end
    @test npolish >= 100

    # A sparse problem large enough for the iterative path.
    for seed in 1:3
        rng = StableRNG(seed)
        n = 60
        A = sprandn(rng, n, n, 0.08) + spdiagm(0 => exp.(3 .* randn(rng, n)))
        ar, br, sr = _cover_min_abslog2(A)
        @test sr.converged
        a, b, s = _cover_min_abslog2(A; maxouter=2)
        @test s.polish.certified
        @test logproduct_gap(a, b, ar, br, A) <= 1e-8
        at, bt, _ = _cover_min_abslog2(A; transversal=true, fname=:cover_transversal)
        a, b, s = _cover_min_abslog2(A; transversal=true, maxouter=1, fname=:cover_transversal)
        @test s.polish.certified
        @test logproduct_gap(a, b, at, bt, A) <= 1e-8
    end

    # Wider element types use a dense factorization.
    Af = Float64.(general_matrices[3][2])
    a, b, s = _cover_min_abslog2(Double64.(Af); maxouter=1)
    @test s.polish.certified
    @test logproduct_gap(Float64.(a), Float64.(b), MatrixCovers.cover_min_jump(AbsLog{2}(), Af)..., Af) <= 1e-6

    # Without the active-set solve, the stalled iteration warns.
    @test_logs (:warn, r"multiplier iteration ended") _cover_min_abslog2(Af; maxouter=1, polish=false)
end

@testset "difference-constraint active-set solver" begin
    # From a start far from the solution, the seeds include constraints that must
    # be dropped and tree paths that must be exchanged; the minimizer is unique,
    # so every start must reach it.
    rng = StableRNG(7)
    for trial in 1:20
        nV = 12
        edges = Tuple{Int,Int}[]
        for p in 1:nV, q in 1:nV
            p != q && rand(rng) < 0.3 && push!(edges, (p, q))
        end
        # Costs from a reference potential keep the constraints feasible.
        φ = randn(rng, nV)
        cvals = [φ[p] - φ[q] - abs(randn(rng)) * (rand(rng) < 0.5) for (p, q) in edges]
        u1, c1, _ = _polish_difference_qp(edges, cvals, φ)
        u2, c2, _ = _polish_difference_qp(edges, cvals, randn(rng, nV); τ=10.0)
        @test c1 && c2
        d1 = [u1[p] - u1[q] - cvals[e] for (e, (p, q)) in enumerate(edges)]
        d2 = [u2[p] - u2[q] - cvals[e] for (e, (p, q)) in enumerate(edges)]
        @test all(>=(-1e-12), d1)
        @test d1 ≈ d2 atol=1e-10
    end
    # Infeasible constraints (a cycle with positive total cost) are not certified.
    edges = [(1, 2), (2, 3), (3, 1)]
    u, certified, _ = _polish_difference_qp(edges, [1.0, 1.0, 1.0], zeros(3))
    @test !certified
    @test u == zeros(3)
end
