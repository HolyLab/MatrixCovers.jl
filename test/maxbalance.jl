# Transversal-tight covers max-balanced within the fully indecomposable blocks.
# Uses `max_product_transversals` and `logproduct_gap` from transversal.jl, and
# `minimax_ratio`, `offtransversal_ratios`, and `check_balanced_cover` from
# balanced.jl.

using MatrixCovers: _cover_transversal_balanced, _transversal_system, _maxbalance

# Whether the square `A`, whose diagonal is nonzero, is fully indecomposable:
# whether every row reaches every other through the nonzero entries.
function reach(A)
    n = size(A, 1)
    R = (A .!= 0) .| Matrix(I, n, n)
    for k in 1:n, i in 1:n, j in 1:n
        R[i, j] |= R[i, k] & R[k, j]
    end
    return R
end
fully_indecomposable(A) = all(reach(A))
# The fully indecomposable blocks of `A` (nonzero diagonal), as lists of rows.
function indecomposable_blocks(A)
    R = reach(A)
    return unique([findall(k -> R[i, k] && R[k, i], axes(A, 1)) for i in axes(A, 1)])
end

# Largest discrepancy, over every partition of `rows` into two nonempty sets, of
# the log of the largest ratio of an entry leading from one set to the other
# and of the largest leading back. Entry `A[i,j]` off the transversal `σ` leads
# from row `i` to the row `k` with `σ[k] == j`; only entries between `rows`
# count.
function cut_imbalance(a, b, A, σ, rows)
    τ = invperm(σ)
    E = [(i, τ[j], log(abs(A[i, j]) / (a[i] * b[j]))) for i in rows, j in axes(A, 2)
         if !iszero(A[i, j]) && j != σ[i] && τ[j] in rows]
    m = length(rows)
    worst = 0.0
    for mask in 1:(2^m-2)
        S = Set(rows[k] for k in 1:m if isodd(mask >> (k - 1)))
        out = maximum((w for (u, v, w) in E if u in S && !(v in S)); init=-Inf)
        inn = maximum((w for (u, v, w) in E if !(u in S) && v in S); init=-Inf)
        worst = max(worst, out == inn ? 0.0 : abs(out - inn))
    end
    return worst
end

# A random fully indecomposable `n×n` matrix with a nonzero diagonal.
function random_indecomposable(rng, n)
    while true
        A = exp.(2 .* randn(rng, n, n)) .* (rand(rng, n, n) .< 0.5)
        for i in 1:n
            A[i, i] = exp(2 * randn(rng))
        end
        fully_indecomposable(A) && return A
    end
end

# Whether the shift `x` max-balances the strongly connected digraph with arcs
# `src[e] → dst[e]` of weight `w[e]`: for every partition of its vertices into
# two nonempty sets, the largest shifted weight leaving one set equals the
# largest entering it.
function maxbalanced(nv, src, dst, w, x; tol=1e-10)
    wx = [w[e] - x[src[e]] + x[dst[e]] for e in eachindex(w)]
    for mask in 1:(2^nv-2)
        inset(v) = isodd(mask >> (v - 1))
        out = maximum((wx[e] for e in eachindex(w) if inset(src[e]) && !inset(dst[e])); init=-Inf)
        inn = maximum((wx[e] for e in eachindex(w) if !inset(src[e]) && inset(dst[e])); init=-Inf)
        abs(out - inn) <= tol || return false
    end
    return true
end

@testset "_maxbalance" begin
    # Random strongly connected digraphs with parallel arcs, self loops, and
    # (on even trials) many tied weights.
    rng = StableRNG(3)
    for trial in 1:200
        nv = rand(rng, 2:9)
        perm = randperm(rng, nv)
        extra = rand(rng, 0:3nv)
        src = vcat(perm, rand(rng, 1:nv, extra))
        dst = vcat(perm[mod1.(2:nv+1, nv)], rand(rng, 1:nv, extra))
        w = isodd(trial) ? randn(rng, length(src)) : Float64.(rand(rng, -2:2, length(src)))
        x = _maxbalance(nv, src, dst, w)
        @test x[1] == 0
        @test maxbalanced(nv, src, dst, w, x)
    end
    # A banded digraph, whose shortest-path tree is deep and moves as a whole at
    # every contraction. It has too many vertices to check every partition, so
    # only the cuts around single vertices are checked.
    nv = 300
    src = vcat(1:nv-1, 2:nv, 1:nv-2)
    dst = vcat(2:nv, 1:nv-1, 3:nv)
    w = randn(rng, length(src))
    x = _maxbalance(nv, src, dst, w)
    wx = [w[e] - x[src[e]] + x[dst[e]] for e in eachindex(w)]
    for v in 1:nv
        out = maximum(wx[e] for e in eachindex(w) if src[e] == v)
        inn = maximum(wx[e] for e in eachindex(w) if dst[e] == v)
        @test out ≈ inn atol = 1e-9
    end
end

@testset "cover_transversal(::PowerMean; maxbalance=true)" begin
    # Example 3.11 of Hook, Pestana, Tisseur & Hogg (SIAM J. Matrix Anal. Appl.
    # 40:320–346, 2019): the max-balanced Hungarian scaling of `H`.
    H = [1 1 1; exp(-1) 1 exp(-2); 0 exp(-4) 1]
    M = exp.([0 -0.5 -2.25; -0.5 0 -3.75; -Inf -2.25 0])
    a, b, st = cover_transversal(PowerMean{2}(), H; maxbalance=true)
    @test MatrixCovers.converged(st)
    @test st.maxbalanced && st.nblocks == 1 && st.nnewton == 0
    @test H ./ (a .* b') ≈ M rtol = 1e-14
    # Covariance: the same ratios in other units.
    D1, D2 = Diagonal([1e3, 2.0, 1e-5]), Diagonal([7.0, 1e-4, 3.0])
    a, b = cover_transversal(PowerMean{2}(), D1 * H * D2; maxbalance=true)
    @test (D1 * H * D2) ./ (a .* b') ≈ M rtol = 1e-12

    # Fully indecomposable matrices: the ratios balance across every cut, the
    # largest ratio is the minimax ratio `θ`, and no cover pushed toward a
    # target has a smaller largest ratio.
    rng = StableRNG(11)
    for _ in 1:100
        n = rand(rng, 2:8)
        A = random_indecomposable(rng, n)
        a, b, st = cover_transversal(PowerMean{2}(), A; maxbalance=true)
        @test MatrixCovers.converged(st)
        @test st.maxbalanced && st.nblocks == 1
        @test check_balanced_cover(a, b, A)
        θ, σ = minimax_ratio(A)
        @test cut_imbalance(a, b, A, σ, 1:n) <= 1e-10
        @test maximum(last, offtransversal_ratios(a, b, A, σ)) ≈ θ rtol = 1e-10
        # The products do not depend on `ϕ` or `target`.
        a1, b1 = cover_transversal(PowerMean{1}(), A; target=0.5, maxbalance=true)
        @test logproduct_gap(a, b, a1, b1, A) <= 1e-12
        a2, b2 = cover_transversal(PowerMean{2}(), A)
        @test maximum(last, offtransversal_ratios(a2, b2, A, σ)) >= θ * (1 - 1e-12)
    end

    # Agreement with the max-balanced Hungarian scaling of Hook et al. as
    # computed by an independent implementation of their Algorithm 3.10
    # (`maxbalanced_hungarian` in `src/maxbalance.jl` of the paper "Numerical
    # methods for scale-invariant analysis"); these are its products `a .* b'`.
    refs = [([4.0 1 0 2; 0 3 5 0; 1 0 2 7; 6 1 0 1],
             [4.898979485566356 1.0 1.1665046746253358 2.857539627437798;
              20.998542021016053 4.286309441156698 4.999999999999999 12.248299083566028;
              12.000833188694234 2.4496598167132055 2.8575396274377978 6.999999999999999;
              6.0 1.2247448713915892 1.4286706177016946 3.4997570035026757]),
            ([1.0 8 0 0 2; 0 1 3 0 0; 0 0 2 9 0; 5 0 0 1 4; 0 3 0 1 6],
             [2.714417616594907 7.999999999999998 5.009957159878615 4.7061882644660615 5.894450397824619;
              1.625413669202318 4.790460124529586 3.000000000000001 2.818100902431721 3.52964119834955;
              5.190986245452685 15.299005469805346 9.580920249059176 9.000000000000002 11.272403609726895;
              5.0 14.73612599456154 9.228419991915873 8.66887290241235 10.857670466379625;
              2.7630236239802906 8.143252849784718 5.0996684899351115 4.790460124529582 6.0]),
            ([2.0 1e3 0; 1e-3 5 7; 3 1e-2 1],
             [25.61158329974988 999.9999999999998 109.32553318667111;
              1.6398829978000684 64.02895824937472 6.999999999999997;
              3.000000000000001 117.13449984286204 12.805791649874939])]
    for (A, P) in refs
        a, b = cover_transversal(PowerMean{2}(), A; maxbalance=true)
        @test a .* b' ≈ P rtol = 1e-10
    end

    # Partly decomposable: three blocks in block upper-triangular form.
    A11 = [4.0 1 0.5; 1 4 1; 0.5 1 4]
    A22 = [3.0 0.5; 1 5]
    A33 = [2.0 0 1; 1 3 0; 0 0.25 1]
    A12 = [1e-3 5.0; 2.0 0; 0 30.0]
    A13 = [0 1e2 0; 0 0 0; 4.0 0 0.1]
    A23 = [0.5 0 0; 0 1e-2 2.0]
    A = [A11 A12 A13; zeros(2, 3) A22 A23; zeros(3, 5) A33]
    blocks = (1:3, 4:5, 6:8)
    for (p, t) in ((2, 1e-2), (1, 0.1), (3, 0.5))
        a, b, st = cover_transversal(PowerMean{p}(), A; target=t, maxbalance=true)
        @test MatrixCovers.converged(st)
        @test st.maxbalanced && st.nblocks == 3
        @test check_balanced_cover(a, b, A)
        R = A ./ (a .* b')
        # The KKT conditions across the cuts after the first and second blocks.
        for k in (3, 5)
            rcut = [R[i, j] for i in 1:k, j in k+1:8 if A[i, j] != 0]
            @test maximum(rcut) < 1
            @test sum(r -> r^p, rcut) / length(rcut) ≈ t^p rtol = 1e-8
        end
        # Within each block, the products are those of the block alone.
        for r in blocks
            ab, bb = cover_transversal(PowerMean{p}(), A[r, r]; target=t, maxbalance=true)
            @test logproduct_gap(a[r], b[r], ab, bb, A[r, r]) <= 1e-12
            @test cut_imbalance(a[r], b[r], A[r, r], collect(1:length(r)), 1:length(r)) <= 1e-10
        end
        # Covariance under power-of-two changes of units, to roundoff with a
        # stopping tolerance near it.
        rng = StableRNG(p)
        d, e = 2.0 .^ rand(rng, -30:30, 8), 2.0 .^ rand(rng, -30:30, 8)
        a12, b12 = cover_transversal(PowerMean{p}(), A; target=t, maxbalance=true, tol=1e-12)
        ã, b̃ = cover_transversal(PowerMean{p}(), Diagonal(d) * A * Diagonal(e); target=t, maxbalance=true,
                                 tol=1e-12)
        @test logproduct_gap(ã ./ d, b̃ ./ e, a12, b12, A) <= 1e-12
        # Permutation equivariance.
        pr, pc = randperm(rng, 8), randperm(rng, 8)
        ap, bp = cover_transversal(PowerMean{p}(), A[pr, pc]; target=t, maxbalance=true)
        @test (ap .* bp') ≈ (a .* b')[pr, pc] rtol = 1e-10
    end

    # A cut whose largest entry would exceed its cover at the unconstrained
    # minimizer: it is held at ratio one, and the p-power mean across the cut
    # is below the target, so that raising the ratios further would decrease
    # the objective.
    Bk = [2.0 1; 1 2]
    A = [Bk [1.0 0; 1e-6 1e-6]; zeros(2, 2) Bk]
    for (p, t) in ((2, 0.9), (1, 0.5))
        a, b, st = cover_transversal(PowerMean{p}(), A; target=t, maxbalance=true)
        @test MatrixCovers.converged(st)
        @test st.nblocks == 2 && st.nactive == 1
        @test check_balanced_cover(a, b, A)
        rcut = [A[i, j] / (a[i] * b[j]) for i in 1:2, j in 3:4 if A[i, j] != 0]
        @test maximum(rcut) ≈ 1 rtol = 1e-12
        @test sum(r -> r^p, rcut) / length(rcut) < t^p
    end

    # Ties: 0/1 matrices, whose maximum-product transversals are many and
    # whose rows are joined into tight components.
    rng = StableRNG(5)
    ntie = ntight = 0
    for _ in 1:40
        n = rand(rng, 3:7)
        A = Float64.(rand(rng, n, n) .< 0.4)
        for i in 1:n
            A[i, i] = 1
        end
        a, b, st = cover_transversal(PowerMean{2}(), A; maxbalance=true)
        @test MatrixCovers.converged(st)
        @test check_balanced_cover(a, b, A)
        blks = indecomposable_blocks(A)
        @test st.nblocks == length(blks)
        for r in blks
            @test cut_imbalance(a, b, A, collect(1:n), r) <= 1e-10
        end
        σs, _ = max_product_transversals(A)
        length(σs) > 1 && (ntie += 1)
        _transversal_system(_row_support(A, Float64), axes(A, 2), Float64, :test, true).nc < n && (ntight += 1)
        for σ in σs[1:min(end, 4)]
            aσ, bσ, _ = _cover_transversal_balanced(A, 2, 1e-2; transversal=σ, maxbalance=true)
            @test logproduct_gap(a, b, aσ, bσ, A) <= 1e-12
        end
    end
    @test ntie > 10 && ntight > 10
    # Generic magnitudes with tied transversals.
    ntie = 0
    for (_, A) in general_matrices
        ntie >= 10 && break
        Af = Float64.(A)
        σs, _ = max_product_transversals(Af)
        length(σs) > 1 || continue
        ntie += 1
        a, b, st = cover_transversal(PowerMean{2}(), Af; maxbalance=true)
        @test MatrixCovers.converged(st)
        @test check_balanced_cover(a, b, Af)
        for σ in σs
            aσ, bσ, _ = _cover_transversal_balanced(Af, 2, 1e-2; transversal=σ, maxbalance=true)
            @test logproduct_gap(a, b, aσ, bσ, Af) <= 1e-10
        end
    end
    @test ntie == 10

    # Small cases: a 1×1 matrix, a diagonal matrix, and a triangular one.
    a, b, st = cover_transversal(PowerMean{2}(), fill(3.0, 1, 1); maxbalance=true)
    @test a[1] * b[1] ≈ 3
    @test st.nblocks == 1 && st.nnewton == 0 && MatrixCovers.converged(st)
    D = [2.0 0 0; 0 3 0; 0 0 5]
    a, b, st = cover_transversal(PowerMean{2}(), D; maxbalance=true)
    @test a .* b' .* (D .!= 0) ≈ D
    @test st.nblocks == 3 && st.nactive == 0 && MatrixCovers.converged(st)
    A = [1.0 1e6; 0 1.0]
    a, b, st = cover_transversal(PowerMean{2}(), A; target=1e-2, maxbalance=true)
    @test A[1, 2] / (a[1] * b[2]) ≈ 1e-2
    @test st.nblocks == 2

    # Sparse matrices: sparse and dense storage, changes of units, and the
    # balance of each row of the blocks between its largest ratio leaving and
    # entering.
    for seed in 1:2
        rng = StableRNG(seed)
        n = 200
        A = sprandn(rng, n, n, 0.01) + spdiagm(0 => exp.(3 .* randn(rng, n)))
        a, b, st = cover_transversal(PowerMean{2}(), A; maxbalance=true)
        @test MatrixCovers.converged(st)
        @test st.nblocks > 1
        @test check_balanced_cover(a, b, A; atol=1e-8)
        @test isbalanced(a, b, A)
        ad, bd = cover_transversal(PowerMean{2}(), Matrix(A); maxbalance=true)
        @test logproduct_gap(a, b, ad, bd, A) <= 1e-10
        d, e = 2.0 .^ rand(rng, -30:30, n), 2.0 .^ rand(rng, -30:30, n)
        ã, b̃ = cover_transversal(PowerMean{2}(), Diagonal(d) * A * Diagonal(e); maxbalance=true)
        @test logproduct_gap(ã ./ d, b̃ ./ e, a, b, A) <= 1e-10
        σ, _ = MatrixCovers._max_product_transversal(_row_support(A, Float64), axes(A, 2), :test)
        τ = invperm(σ)
        R = reach(Matrix(A[:, σ]) .!= 0)
        lout, lin = fill(-Inf, n), fill(-Inf, n)
        for (i, j, v) in zip(findnz(A)...)
            k = τ[j]
            (k != i && R[i, k] && R[k, i]) || continue
            w = log(abs(v) / (a[i] * b[j]))
            lout[i] = max(lout[i], w)
            lin[k] = max(lin[k], w)
        end
        @test all(k -> lout[k] == lin[k] || abs(lout[k] - lin[k]) <= 1e-10, 1:n)
        @test count(isfinite, lout) > n ÷ 2
    end

    # Generic indexing, element types, and the mutating form.
    A = [1.0 0.5 0 0.3; 0.2 1.0 0 0; 0 0 1.0 0.5; 0 0 0.4 1.0]
    a, b, st = cover_transversal(PowerMean{2}(), A; maxbalance=true)
    @test st isa MatrixCovers.BalancedCoverStats{Float64}
    @test st.maxbalanced && st.nblocks == 2
    @test !cover_transversal(PowerMean{2}(), A)[3].maxbalanced
    @test cover_transversal(PowerMean{2}(), A)[3].nblocks == 2
    Ao = OffsetArray(A, -1:2, 5:8)
    ao, bo = cover_transversal(PowerMean{2}(), Ao; maxbalance=true)
    @test axes(ao, 1) == axes(Ao, 1) && axes(bo, 1) == axes(Ao, 2)
    @test parent(ao) .* parent(bo)' ≈ a .* b' rtol = 1e-12
    B = [9.0 9 9 9 9; 9 1 0.5 0 0.3; 9 0.2 1 0 0; 9 0 0 1 0.5; 9 0 0 0.4 1]
    av, bv = cover_transversal(PowerMean{2}(), view(B, 2:5, 2:5); maxbalance=true)
    @test av .* bv' ≈ a .* b' rtol = 1e-12
    as, bs = cover_transversal(PowerMean{2}(), sparse(A); maxbalance=true)
    @test as .* bs' ≈ a .* b' rtol = 1e-12
    ao, bo = cover_transversal!(PowerMean{2}(), OffsetArray(zeros(4), -1:2), OffsetArray(zeros(4), 5:8), Ao;
                                maxbalance=true)
    @test parent(ao) .* parent(bo)' ≈ a .* b' rtol = 1e-12
    a32, b32, st32 = cover_transversal(PowerMean{2}(), Float32.(A); maxbalance=true)
    @test eltype(a32) == eltype(b32) == Float32
    @test st32.maxbalanced
    @test iscover(a32, b32, Float32.(A))
    @test a32 .* b32' ≈ a .* b' rtol = 1e-6
    aB, bB, stB = cover_transversal(PowerMean{2}(), BigFloat.(A); maxbalance=true)
    @test eltype(aB) == BigFloat
    @test aB .* bB' ≈ a .* b' rtol = 1e-12
    a2, b2 = zeros(4), zeros(4)
    _, _, st2 = cover_transversal!(PowerMean{2}(), a2, b2, A; maxbalance=true)
    @test (a2, b2) == (a, b)
    @test st2.maxbalanced

    # An iteration limit on the problem between blocks: a warning, and still a
    # hard cover tight on the transversal.
    A = [Bk [1.0 0.3; 0.2 0.1]; zeros(2, 2) Bk]
    a, b, st = @test_logs (:warn, r"may not minimize the objective") cover_transversal(PowerMean{2}(), A;
                                                                                         maxbalance=true, maxiter=0)
    @test !MatrixCovers.converged(st)
    @test check_balanced_cover(a, b, A)

    # Errors.
    @test_throws "requires a finite exponent" cover_transversal(PowerMean{Inf}(), A; maxbalance=true)
    @test_throws "target must satisfy 0 < target <= 1, got 0" cover_transversal(PowerMean{2}(), A; target=0,
                                                                                 maxbalance=true)
    @test_throws "structurally nonsingular" cover_transversal(PowerMean{2}(), [1.0 1 1; 1 0 0; 1 0 0];
                                                              maxbalance=true)
    @test_throws "requires a square matrix" cover_transversal(PowerMean{2}(), ones(2, 3); maxbalance=true)
end
