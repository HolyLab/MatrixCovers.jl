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
        # Arbitrary nonnegative seed weights order the seed but not the result.
        seed = rand(rng, length(edges)) .* (rand(rng, length(edges)) .< 0.5)
        u3, c3, _ = _polish_difference_qp(edges, cvals, randn(rng, nV); seed)
        @test c1 && c2 && c3
        d1 = [u1[p] - u1[q] - cvals[e] for (e, (p, q)) in enumerate(edges)]
        d2 = [u2[p] - u2[q] - cvals[e] for (e, (p, q)) in enumerate(edges)]
        d3 = [u3[p] - u3[q] - cvals[e] for (e, (p, q)) in enumerate(edges)]
        @test all(>=(-1e-12), d1)
        @test d1 ≈ d2 atol=1e-10
        @test d1 ≈ d3 atol=1e-10
    end
    @test_throws "seed must have axes 1:3" _polish_difference_qp([(1, 2), (2, 3), (1, 3)], [0.0, 0.0, 0.0], zeros(3); seed=ones(2))
    # Infeasible constraints (a cycle with positive total cost) are not certified.
    edges = [(1, 2), (2, 3), (3, 1)]
    u, certified, _ = _polish_difference_qp(edges, [1.0, 1.0, 1.0], zeros(3))
    @test !certified
    @test u == zeros(3)
end

@testset "tree Laplacian assembly" begin
    # The dense and triplet assemblies of the tree-contracted Laplacian must
    # agree, including the choice of pinned tree in each connected component.
    rng = StableRNG(11)
    for trial in 1:10
        # Two blocks of nodes with edges only within a block, so the contracted
        # graph has two components; a random forest `W` joins some nodes.
        nb = 8
        edges = Tuple{Int,Int}[]
        for blk in 0:1, p in 1:nb, q in 1:nb
            p != q && rand(rng) < 0.6 && push!(edges, (blk * nb + p, blk * nb + q))
        end
        nV = 2nb
        cvals = randn(rng, length(edges))
        inW = falses(length(edges))
        uf = collect(1:nV)
        find(x) = (while uf[x] != x; x = uf[x]; end; x)
        for e in randperm(rng, length(edges))
            rand(rng) < 0.5 || continue
            rp, rq = find(edges[e][1]), find(edges[e][2])
            rp == rq && continue
            uf[rp] = rq
            inW[e] = true
        end
        supp = MatrixCovers.EdgeList{Float64}(edges, cvals, -1.0)
        F = MatrixCovers._Forest(nV, supp, findall(inW))
        idx1, L1 = MatrixCovers._tree_laplacian_triplets(F, supp)
        idx2, L2 = MatrixCovers._tree_laplacian_dense(F, supp)
        @test idx1 == idx2
        @test Matrix(L1) == L2
        @test count(iszero, idx1) >= 2   # at least one pinned tree per component
    end

    # A complete directed graph has more edges than squared trees as soon as
    # `W` is nonempty, so the polish runs on the dense assembly throughout.
    for trial in 1:5
        nV = 30
        edges = [(p, q) for p in 1:nV for q in 1:nV if p != q]
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

    # Over the fill or flop budget, the pinned Laplacian is solved by CG, to the
    # accuracy of the factorization.
    nV = 200
    edges = unique!([(min(p, q), max(p, q)) for (p, q) in zip(rand(rng, 1:nV, 6nV), rand(rng, 1:nV, 6nV)) if p != q])
    rows = [first.(edges); last.(edges); 1:nV-1]
    cols = [last.(edges); first.(edges); 1:nV-1]
    vals = [fill(-1.0, 2 * length(edges)); zeros(nV - 1)]
    S = sparse(rows, cols, vals, nV, nV)[2:nV, 2:nV]   # pin node 1
    for k in 1:nV-1
        S[k, k] = count(e -> k + 1 in e, edges)
    end
    L = sparse(triu(S))
    Fch = MatrixCovers._laplacian_factor(L)
    Fcg = MatrixCovers._laplacian_factor(L; flopbudget=0)
    @test Fch isa MatrixCovers.SparseCholesky
    @test Fcg isa MatrixCovers._LaplacianCG
    b = randn(rng, nV - 1)
    xch = MatrixCovers._laplacian_solve!(zeros(nV - 1), Fch, b)
    xcg = MatrixCovers._laplacian_solve!(zeros(nV - 1), Fcg, b)
    @test !Fcg.factored[]
    @test xcg ≈ xch rtol=1e-13
    @test norm(S * xcg - b) <= 8 * eps() * (opnorm(Matrix(S)) * norm(xcg) + norm(b))
end

@testset "difference-grid active-set solver" begin
    # The grid must reproduce the edge list of its supported entries, with rows
    # and columns sharing unknowns and some entries outside the support.
    rng = StableRNG(13)
    ncert = 0
    for trial in 1:20
        nV, m, n = 10, 14, 12
        rowidx = rand(rng, 1:nV, m)
        colidx = rand(rng, 1:nV, n)
        φ = randn(rng, nV)
        C = [φ[rowidx[i]] - φ[colidx[j]] - abs(randn(rng)) * (rand(rng) < 0.5) for i in 1:m, j in 1:n]
        C[randperm(rng, m * n)[1:10]] .= -Inf
        for j in 1:n, i in 1:m
            rowidx[i] == colidx[j] && (C[i, j] = -Inf)
        end
        grid = MatrixCovers.DiffGrid{Float64}(C, rowidx, colidx)
        edges = [(rowidx[i], colidx[j]) for j in 1:n for i in 1:m if isfinite(C[i, j])]
        cvals = [C[i, j] for j in 1:n for i in 1:m if isfinite(C[i, j])]
        for (u0, τ) in ((φ, 1e-4), (randn(rng, nV), 10.0))
            ug, cg, _ = _polish_difference_qp(grid, u0; τ)
            ue, ce, _ = _polish_difference_qp(edges, cvals, u0; τ)
            @test cg == ce
            ncert += cg
            dg = [ug[p] - ug[q] - cvals[e] for (e, (p, q)) in enumerate(edges)]
            de = [ue[p] - ue[q] - cvals[e] for (e, (p, q)) in enumerate(edges)]
            @test dg ≈ de atol=1e-10
            # Seed weights index `C` linearly; weights outside the support are ignored.
            us, cs, _ = _polish_difference_qp(grid, u0; seed=vec(rand(rng, m, n)))
            @test cs == cg
            ds = [us[p] - us[q] - cvals[e] for (e, (p, q)) in enumerate(edges)]
            cg && @test ds ≈ dg atol=1e-10
        end

        # Dense and triplet tree Laplacians of the grid agree.
        s = MatrixCovers._qp_support(grid)
        inW = fill(false, m, n)
        uf = collect(1:nV)
        find(x) = (while uf[x] != x; x = uf[x]; end; x)
        for e in randperm(rng, m * n)
            (isfinite(C[e]) && rand(rng) < 0.5) || continue
            p, q = MatrixCovers._edge(s, e)
            rp, rq = find(p), find(q)
            rp == rq && continue
            uf[rp] = rq
            inW[e] = true
        end
        F = MatrixCovers._Forest(nV, s, findall(vec(inW)))
        idx1, L1 = MatrixCovers._tree_laplacian_triplets(F, s)
        idx2, L2 = MatrixCovers._tree_laplacian_dense(F, s)
        @test idx1 == idx2
        @test Matrix(L1) == L2
    end
    @test ncert >= 30

    # Only difference constraints are accepted.
    supp = MatrixCovers.EdgeList{Float64}([(1, 2)], [0.0])
    @test_throws "requires difference constraints" _polish_difference_qp(supp, zeros(2))
end
