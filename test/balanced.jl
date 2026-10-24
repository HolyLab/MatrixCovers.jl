# Transversal-tight covers with a PowerMean penalty and a target ratio.
# Uses `max_product_transversals`, `logproduct_gap`, `_max_product_transversal`, and
# `_row_support` from transversal.jl.

using MatrixCovers: _flow_deficit, _cover_transversal_balanced, _balanced_step!

if !isdefined(@__MODULE__, :symmetric_matrices)
    include("testmatrices.jl")
end

# Coverage ratios of the entries of `A` off the transversal `σ` (by row), as
# `(i, j) => r` pairs.
function offtransversal_ratios(a, b, A, σ)
    return [(i, j) => abs(A[i, j]) / (a[i] * b[j]) for i in axes(A, 1), j in axes(A, 2)
            if !iszero(A[i, j]) && j != σ[i]]
end

# The objective of `cover_transversal(PowerMean{p}(), A; target=t)` over the
# entries off `σ`.
function balanced_objective(a, b, A, σ, p, t)
    return sum(((r / t)^p - 1 - p * log(r / t)) * t^p / p for (_, r) in offtransversal_ratios(a, b, A, σ);
               init=0.0)
end

# Largest geometric-mean coverage ratio of a cycle of entries off the unique
# maximum-product transversal: the least possible largest ratio of a cover tight
# on the transversal. Entry `(i, j)` off the transversal joins row `i` to the row
# `k` matched to column `j` with weight `log|A[i,j]| - log|A[k,j]|`; Karp's
# algorithm gives the largest mean weight of a cycle.
function minimax_ratio(A)
    σs, _ = max_product_transversals(A)
    length(σs) == 1 || error("the maximum-product transversal is not unique")
    σ = only(σs)
    n = size(A, 1)
    τ = invperm(σ)
    E = [(i, τ[j], log(abs(A[i, j])) - log(abs(A[τ[j], j]))) for i in 1:n, j in 1:n
         if !iszero(A[i, j]) && j != σ[i]]
    D = fill(-Inf, n + 1, n)   # D[k+1, v]: heaviest walk of k edges ending at v
    D[1, :] .= 0
    for k in 1:n, (u, v, c) in E
        D[k+1, v] = max(D[k+1, v], D[k, u] + c)
    end
    λ = -Inf
    for v in 1:n
        isfinite(D[n+1, v]) || continue
        λ = max(λ, minimum((D[n+1, v] - D[k+1, v]) / (n - k) for k in 0:n-1))
    end
    return exp(λ), σ
end

# Checks shared by every solve: a certified hard cover whose log product is the
# largest log product of a transversal, so that it is tight on every
# maximum-product transversal. For small `A`, the transversals are enumerated
# and the tightness is checked entry by entry.
function check_balanced_cover(a, b, A; atol=1e-9)
    iscover(a, b, A) || return false
    if size(A, 1) <= 7
        σs, lπ = max_product_transversals(A)
        all(σ -> all(i -> abs(log(a[i] * b[σ[i]] / abs(A[i, σ[i]]))) <= atol, axes(A, 1)), σs) ||
            return false
    else
        _, _, _, lπ = _max_product_transversal(_row_support(A, Float64), axes(A, 2), :test)
    end
    return abs(sum(log, a) + sum(log, b) - lπ) <= atol * max(1, abs(lπ))
end

@testset "maximum-flow multiplier test" begin
    # Opposite supplies on a 2-cycle: either orientation routes.
    for g in ([1.0, -1.0], [-1.0, 1.0])
        deficit, _ = _flow_deficit(2, [1, 2], [2, 1], g)
        @test deficit == 0
    end
    # One arc: feasible along it, infeasible against it, with `S` the stranded supply.
    deficit, _ = _flow_deficit(2, [1], [2], [1.0, -1.0])
    @test deficit == 0
    deficit, S = _flow_deficit(2, [1], [2], [-1.0, 1.0])
    @test deficit == 1
    @test S == [false, true]
    # Redundant arcs forming an undirected cycle: two routes from 1 to 3.
    deficit, _ = _flow_deficit(3, [1, 1, 2], [2, 3, 3], [2.0, 0.0, -2.0])
    @test deficit == 0
    deficit, S = _flow_deficit(3, [1, 1, 2], [2, 3, 3], [-2.0, 0.0, 2.0])
    @test deficit == 2
    @test S == [false, false, true]
    # Partly routable: the chain 1 → 2 → 3 absorbs only 0.5 of the supply.
    g = [1.0, 0.5, -0.5, -1.0]
    deficit, S = _flow_deficit(4, [1, 2], [2, 3], g)
    @test deficit ≈ 1
    @test S == [true, true, true, false]
    @test deficit ≈ sum(g[S])
    # No arcs and no supply.
    deficit, _ = _flow_deficit(2, Int[], Int[], [0.0, 0.0])
    @test deficit == 0
    @test_throws DimensionMismatch _flow_deficit(3, [1], [2], [1.0, -1.0])
    # On random digraphs, no arc leaves `S` and the deficit is the supply in `S`.
    rng = StableRNG(4)
    for _ in 1:50
        nv = rand(rng, 2:8)
        na = rand(rng, 0:12)
        tails, heads = rand(rng, 1:nv, na), rand(rng, 1:nv, na)
        g = randn(rng, nv)
        g .-= sum(g) / nv
        deficit, S = _flow_deficit(nv, tails, heads, g)
        @test deficit >= 0
        @test !any(k -> S[tails[k]] && !S[heads[k]], eachindex(tails))
        @test deficit ≈ max(sum(g[S]; init=0.0), 0) atol = 1e-12
    end
end

@testset "line search with cancelling slack changes" begin
    # Row 2 has one entry entering and one leaving, so moving it changes the sum
    # of the slacks by exactly zero. The descent comes from the change `ε` of
    # the slack of the first entry, which is far below the resolution of the
    # change `δ` of row 2: summed by entry, `ε` is lost and the slope is
    # positive.
    p, tp = 2.0, 1e-4
    ε, δ = -1e-17, 0.25
    P, Q = [4, 1, 2], [3, 2, 3]
    d = [30.0, -log(2e-21) / p, -log(1e-21) / p]
    y, C = zeros(4), -d
    δv = [0, δ, 0, ε]
    status = _balanced_step!(y, δv, d, zeros(3), falses(3), 1:4, P, Q, C, p, tp, 1.0)
    @test status == 0
    @test y == δv
    @test_throws "not constant on a label" _balanced_step!(zeros(4), δv, d, zeros(3), falses(3), [1, 1, 2, 3],
                                                          P, Q, C, p, tp, 1.0)
end

@testset "line search at the resolution of the arithmetic" begin
    # Moving row 2, which has one entry entering and one leaving, is stationary
    # to first order when their slacks are equal, and uphill when the entering
    # one has the smaller slack.
    p, tp = 2.0, 1e-4
    P, Q = [1, 2], [2, 3]
    for (d, expected) in (([5.0, 5.0], :floor), ([1.0, 2.0], :stalled))
        y = zeros(3)
        @test _balanced_step!(y, [0, 0.5, 0], d, zeros(2), falses(2), 1:3, P, Q, -d, p, tp, 1.0) === expected
        @test iszero(y)
    end
    # An inactive entry at zero slack that the step would make negative is
    # made active, and the scales do not move.
    y, active = zeros(3), falses(2)
    @test _balanced_step!(y, [0, 0.5, 0], [0.0, 1.0], zeros(2), active, 1:3, P, Q, [0.0, -1.0], p, tp, 1.0) == 1
    @test active == [true, false] && iszero(y)
end

@testset "groups whose terms differ greatly in size" begin
    # Entries spanning about 1e±11, in units spanning a further 2^±30: some
    # groups' gradients have terms many orders of magnitude smaller than
    # others', and the solve must still reach the tight tolerance.
    rng = StableRNG(16)
    A = sprandn(rng, 12, 12, 0.15)
    A = A .* exp.(6 .* sprandn(rng, 12, 12, 1.0) .* (A .!= 0))
    A = dropzeros(A + spdiagm(0 => exp.(6 .* randn(rng, 12))))
    a, b, st = cover_transversal(PowerMean{2}(), A)
    @test MatrixCovers.converged(st)
    @test check_balanced_cover(a, b, A)
    d, e = 2.0 .^ rand(rng, -30:30, 12), 2.0 .^ rand(rng, -30:30, 12)
    a12, b12 = cover_transversal(PowerMean{2}(), A; tol=1e-12)
    ã, b̃ = cover_transversal(PowerMean{2}(), Diagonal(d) * A * Diagonal(e); tol=1e-12)
    @test logproduct_gap(ã ./ d, b̃ ./ e, a12, b12, A) <= 1e-9
end

@testset "cover_transversal(::PowerMean)" begin
    # Agreement with the Ipopt reference on the committed matrices, whose many
    # repeated magnitudes leave degenerate active sets.
    idx = round.(Int, range(1, length(general_matrices), length=40))
    nreleases = 0
    for k in idx
        Af = Float64.(general_matrices[k][2])
        for (p, t) in ((1, 1e-2), (2, 1e-2), (2, 0.3), (4, 0.3), (1, 1.0), (2, 1.0), (4, 1.0))
            a, b, st = cover_transversal(PowerMean{p}(), Af; target=t)
            @test MatrixCovers.converged(st)
            @test check_balanced_cover(a, b, Af)
            (aj, bj), lπ = MatrixCovers.cover_transversal_jump(PowerMean{p}(), Af; target=t)
            @test logproduct_gap(a, b, aj, bj, Af) <= 1e-6
            nreleases += st.nreleases
        end
    end
    @test nreleases > 0

    # Generic magnitudes with sparse patterns.
    rng = StableRNG(3)
    for _ in 1:30
        n = rand(rng, 3:7)
        A = exp.(2 .* randn(rng, n, n)) .* (rand(rng, n, n) .< 0.6)
        for i in 1:n
            A[i, i] = exp(2 * randn(rng)) + 0.1
        end
        for (p, t, rtol) in ((1, 1e-2, 1e-6), (2, 1e-2, 1e-6), (2, 0.3, 1e-6), (2, 1.0, 1e-5))
            a, b, st = cover_transversal(PowerMean{p}(), A; target=t)
            @test MatrixCovers.converged(st)
            @test check_balanced_cover(a, b, A)
            (aj, bj), _ = MatrixCovers.cover_transversal_jump(PowerMean{p}(), A; target=t)
            @test logproduct_gap(a, b, aj, bj, A) <= rtol
        end
        # For p = 4 and a small target the objective is too flat for Ipopt to
        # resolve the products; compare with an extended-precision solve.
        a, b, st = cover_transversal(PowerMean{4}(), A; target=1e-2)
        @test MatrixCovers.converged(st)
        ab, bb, sb = _cover_transversal_balanced(BigFloat.(A), 4, BigFloat(1e-2); tol=1e-30)
        @test sb.converged
        @test logproduct_gap(a, b, ab, bb, A) <= 1e-8
    end

    # KKT conditions across a cut: with every entry of `A12` covered loosely,
    # shifting the second block's scales is unconstrained, so the p-power mean
    # of the ratios of the entries crossing the cut equals the target.
    A11 = [4.0 1 0.5; 1 4 1; 0.5 1 4]
    A22 = [3.0 0.5 1; 1 5 0.5; 2 1 3]
    for (p, t, A12) in ((2, 1e-2, [1e-3 5.0 0; 2.0 0 1e3; 0 30.0 0.1]),
                        (1, 0.1, [1.0 0 0; 0 0 1e-4; 2.0 0 0]),
                        (3, 0.5, [7.0 1 1; 1 1 1; 1 1 0.01]))
        A = [A11 A12; zeros(3, 3) A22]
        a, b, st = cover_transversal(PowerMean{p}(), A; target=t)
        @test MatrixCovers.converged(st)
        rcut = [A[i, j] / (a[i] * b[j]) for i in 1:3, j in 4:6 if A[i, j] != 0]
        @test maximum(rcut) < 1
        @test sum(r -> r^p, rcut) / length(rcut) ≈ t^p rtol = 1e-8
    end

    # A group with an inactive entry inside it: rows 4:end are tied to row 3
    # (all of the transversals through them are maximal), and the entries
    # `A[1, 4:end]`, far below their targets, pull row 1 down against row 3
    # until the cycle 1 → 2 → 3 → 1 has two tight entries; its third entry keeps
    # the cycle's slack.
    function pulled_cycle(npull)
        n = 3 + npull
        A = Matrix{Float64}(I, n, n)
        A[1, 2] = A[2, 3] = A[3, 1] = 0.5
        for k in 4:n
            A[3, k] = A[k, 3] = 1.0
            A[1, k] = 1e-3
        end
        return A
    end
    for (npull, p, t) in ((2, 1, 0.5), (4, 2, 0.5))
        A = pulled_cycle(npull)
        a, b, st = cover_transversal(PowerMean{p}(), A; target=t)
        @test MatrixCovers.converged(st)
        R = A ./ (a .* b')
        @test R[1, 2] ≈ 1 && R[2, 3] ≈ 1
        @test R[3, 1] ≈ 0.125
        @test st.nactive == 2
        (aj, bj), _ = MatrixCovers.cover_transversal_jump(PowerMean{p}(), A; target=t)
        @test logproduct_gap(a, b, aj, bj, A) <= 1e-6
    end
    # With fewer pulling entries, no entry of the cycle is tight.
    A = pulled_cycle(1)
    a, b, st = cover_transversal(PowerMean{1}(), A; target=0.5)
    @test st.nactive == 0
    @test maximum(A[i, j] / (a[i] * b[j]) for (i, j) in ((1, 2), (2, 3), (3, 1))) < 0.99

    # 0/1 matrices: every off-transversal ratio is tied at the start.
    rng = StableRNG(5)
    for _ in 1:30
        n = rand(rng, 3:7)
        A = Float64.(rand(rng, n, n) .< 0.5)
        for i in 1:n
            A[i, i] = 1
        end
        for (p, t, rtol) in ((1, 0.3, 1e-6), (2, 0.3, 1e-6), (2, 1.0, 1e-5))
            a, b, st = cover_transversal(PowerMean{p}(), A; target=t)
            @test MatrixCovers.converged(st)
            @test check_balanced_cover(a, b, A)
            (aj, bj), _ = MatrixCovers.cover_transversal_jump(PowerMean{p}(), A; target=t)
            @test logproduct_gap(a, b, aj, bj, A) <= rtol
        end
        a, b, st = cover_transversal(PowerMean{2}(), A; target=1e-2)
        @test MatrixCovers.converged(st)
        @test check_balanced_cover(a, b, A)
    end
    # Every transversal is maximal: one tight component and nothing to solve.
    a, b, st = cover_transversal(PowerMean{2}(), ones(4, 4))
    @test a .* b' ≈ ones(4, 4)
    @test MatrixCovers.converged(st) && st.nnewton == 0

    # Large p approaches the cover with the least largest ratio off the
    # transversal, which equals the largest geometric-mean ratio of a cycle.
    rng = StableRNG(21)
    for _ in 1:8
        A = exp.(randn(rng, 5, 5))
        θ, σ = minimax_ratio(A)
        a, b, st = cover_transversal(PowerMean{16}(), A; target=1e-2)
        @test MatrixCovers.converged(st)
        rmax = maximum(last, offtransversal_ratios(a, b, A, σ))
        @test θ * (1 - 1e-12) <= rmax <= 1.02 * θ
    end

    # A 2×2 triangular matrix: the off-diagonal ratio is the target, so the
    # scaled matrix approaches the identity as the target shrinks.
    A = [1.0 1e6; 0 1.0]
    for t in (1e-1, 1e-2, 1e-4)
        a, b, st = cover_transversal(PowerMean{2}(), A; target=t)
        @test A[1, 2] / (a[1] * b[2]) ≈ t
        @test cond(A ./ (a .* b')) <= 1 + 2t
    end

    # The minimizer depends on `target`, and the objective is minimal among
    # nearby covers tight on the transversal.
    A = Float64.(general_matrices[17][2])
    σ = first(max_product_transversals(A)[1])
    M = exp.(randn(StableRNG(8), 5, 5)) .* (rand(StableRNG(9), 5, 5) .< 0.5) + 4I
    a1, b1 = cover_transversal(PowerMean{2}(), M; target=0.5)
    a2, b2 = cover_transversal(PowerMean{2}(), M; target=0.05)
    @test logproduct_gap(a1, b1, a2, b2, M) > 1e-3
    a, b = cover_transversal(PowerMean{2}(), A; target=0.1)
    F = balanced_objective(a, b, A, σ, 2, 0.1)
    rng = StableRNG(9)
    for _ in 1:20
        # Perturb the row scales, keep the transversal tight, and restore coverage.
        x = 1e-3 .* randn(rng, 5)
        ã = a .* exp.(x)
        b̃ = copy(b)
        for i in 1:5
            b̃[σ[i]] = b[σ[i]] * exp(-x[i])
        end
        iscover(ã, b̃, A) || continue
        @test balanced_objective(ã, b̃, A, σ, 2, 0.1) >= F - 1e-12
    end

    # Independence from the maximum-product transversal used.
    ntie = 0
    for (_, A) in general_matrices
        ntie >= 10 && break
        Af = Float64.(A)
        σs, _ = max_product_transversals(Af)
        length(σs) > 1 || continue
        ntie += 1
        a, b = cover_transversal(PowerMean{2}(), Af)
        for σ in σs
            aσ, bσ, _ = _cover_transversal_balanced(Af, 2, 1e-2; transversal=σ)
            @test logproduct_gap(a, b, aσ, bσ, Af) <= 1e-8
        end
    end
    @test ntie == 10

    # Scale covariance and permutation equivariance.
    rng = StableRNG(1)
    for (_, A) in general_matrices[1:20]
        Af = Float64.(A)
        dr = exp.(2 .* randn(rng, 5)); dc = exp.(2 .* randn(rng, 5))
        @test covaries(M -> scales(cover_transversal(PowerMean{2}(), M)), Af, dr, dc; rtol=1e-7)
        pr, pc = randperm(rng, 5), randperm(rng, 5)
        a, b = cover_transversal(PowerMean{2}(), Af)
        ap, bp = cover_transversal(PowerMean{2}(), Af[pr, pc])
        @test (ap .* bp') ≈ (a .* b')[pr, pc] rtol = 1e-8
    end

    # Sparse matrices: power-of-two rescaling, sparse and dense storage, and the
    # sparse and conjugate-gradient Newton solves.
    for seed in 1:3
        rng = StableRNG(seed)
        n = 200
        A = sprandn(rng, n, n, 0.02) + spdiagm(0 => exp.(3 .* randn(rng, n)))
        a, b, st = cover_transversal(PowerMean{2}(), A)
        @test MatrixCovers.converged(st)
        @test st.linsolve === :cholesky
        @test check_balanced_cover(a, b, A; atol=1e-8)
        @test isbalanced(a, b, A)
        d = 2.0 .^ rand(rng, -30:30, n)
        e = 2.0 .^ rand(rng, -30:30, n)
        ã, b̃ = scales(cover_transversal(PowerMean{2}(), Diagonal(d) * A * Diagonal(e)))
        @test logproduct_gap(ã ./ d, b̃ ./ e, a, b, A) <= 1e-8
        ad, bd, sd = cover_transversal(PowerMean{2}(), Matrix(A))
        @test logproduct_gap(a, b, ad, bd, A) <= 1e-8
        ac, bc, sc = cover_transversal(PowerMean{2}(), A; flopbudget=0)
        @test sc.linsolve === :cg
        @test MatrixCovers.converged(sc)
        @test logproduct_gap(a, b, ac, bc, A) <= 1e-8
    end

    # Several support components, each balanced separately.
    Ablk = [1.0 2.0 0 0; 0.25 3.0 0 0; 0 0 1.5 2.5; 0 0 0.75 0]
    a, b = cover_transversal(PowerMean{2}(), Ablk)
    @test check_balanced_cover(a, b, Ablk)
    @test isbalanced(a, b, Ablk)

    # The AbsLog{2} penalty is the default solver.
    A = [1.0 1 0; 0.25 1 1; 1 0 0]
    @test scales(cover_transversal(AbsLog{2}(), A)) == scales(cover_transversal(A))
    a2, b2 = zeros(3), zeros(3)
    @test scales(cover_transversal!(AbsLog{2}(), a2, b2, A)) === (a2, b2)
    @test (a2, b2) == scales(cover_transversal(A))

    # Generic indexing, element types, and the mutating form.
    A = [1.0 0.5 0.2; 0.3 1.0 0.5; 0.4 0.1 1.0]
    a, b, st = cover_transversal(PowerMean{2}(), A)
    @test st isa MatrixCovers.BalancedCoverStats{Float64}
    @test MatrixCovers.iterations(st) == st.nnewton
    @test MatrixCovers.residual(st) == st.gradnorm
    @test MatrixCovers.tolerance(st) == st.tol == 1e-9
    Ao = OffsetArray(A, -1:1, 5:7)
    ao, bo = cover_transversal(PowerMean{2}(), Ao)
    @test axes(ao, 1) == axes(Ao, 1) && axes(bo, 1) == axes(Ao, 2)
    @test parent(ao) .* parent(bo)' ≈ a .* b'
    B = [9.0 9 9 9; 9 1 0.5 0.2; 9 0.3 1 0.5; 9 0.4 0.1 1]
    av, bv = cover_transversal(PowerMean{2}(), view(B, 2:4, 2:4))
    @test av .* bv' ≈ a .* b'
    ao, bo = cover_transversal!(PowerMean{2}(), OffsetArray(zeros(3), -1:1), OffsetArray(zeros(3), 5:7), Ao)
    @test parent(ao) .* parent(bo)' ≈ a .* b'
    a32, b32, st32 = cover_transversal(PowerMean{2}(), Float32.(A))
    @test eltype(a32) == eltype(b32) == Float32
    @test st32 isa MatrixCovers.BalancedCoverStats{Float64}
    @test iscover(a32, b32, Float32.(A))
    @test a32 .* b32' ≈ a .* b' rtol = 1e-6
    aC, bC = cover_transversal(PowerMean{2}(), A .* cis.(reshape(1:9, 3, 3)))
    @test aC .* bC' ≈ a .* b'
    aB, bB, stB = cover_transversal(PowerMean{2}(), BigFloat.(A))
    @test eltype(aB) == BigFloat
    @test stB isa MatrixCovers.BalancedCoverStats{BigFloat}
    @test aB .* bB' ≈ a .* b' rtol = 1e-8
    a2, b2 = zeros(3), zeros(3)
    @test scales(cover_transversal!(PowerMean{2}(), a2, b2, A)) === (a2, b2)
    @test (a2, b2) == (a, b)
    @test_throws DimensionMismatch cover_transversal!(PowerMean{2}(), zeros(2), zeros(3), A)

    # An iteration limit: a warning, and still a hard cover tight on the transversal.
    M = exp.(randn(StableRNG(2), 6, 6))
    (a, b, st), logs = capture_logs(() -> cover_transversal(PowerMean{2}(), M; maxiter=1))
    @test !MatrixCovers.converged(st)
    @test !st.stalled
    @test any(l -> l.level == Base.CoreLogging.Warn && occursin("maxiter=1", l.message), logs)
    @test check_balanced_cover(a, b, M)
    a, b, st = @test_logs (:warn, r"may not minimize the objective") cover_transversal(PowerMean{2}(), M; maxiter=0)
    @test check_balanced_cover(a, b, M)
    @test isbalanced(a, b, M)

    # Errors.
    @test_throws "requires a square matrix" cover_transversal(PowerMean{2}(), [1.0 2.0 3.0; 4.0 5.0 6.0])
    @test_throws "structurally nonsingular" cover_transversal(PowerMean{2}(), [1.0 1 1; 1 0 0; 1 0 0])
    @test_throws "structurally nonsingular" cover_transversal(PowerMean{2}(), sparse([1.0 1 1; 1 0 0; 1 0 0]))
    @test_throws "finite entries" cover_transversal(PowerMean{2}(), [1.0 Inf; 1 1])
    @test_throws "target must satisfy 0 < target <= 1, got 0" cover_transversal(PowerMean{2}(), A; target=0)
    @test_throws "target must satisfy 0 < target <= 1, got 1.5" cover_transversal(PowerMean{2}(), A; target=1.5)
    @test_throws "requires a finite exponent" cover_transversal(PowerMean{Inf}(), A)
    @test_throws "underflows" cover_transversal(PowerMean{2}(), A; target=1e-200)
    @test_throws "tol must be positive" cover_transversal(PowerMean{2}(), A; tol=0)
    @test_throws "maxiter must be nonnegative" cover_transversal(PowerMean{2}(), A; maxiter=-1)
end
