# Unconstrained AbsLinear soft covers: descent, multistart, and start provenance.

# Reuse the committed 5×5 matrix libraries when already loaded.
if !isdefined(@__MODULE__, :symmetric_matrices)
    include("testmatrices.jl")
end

@testset "soft_symcover" begin
    # At the minimum, gradients should be near zero
    for A in ([2.0 1.0; 1.0 3.0], [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0])
        # At the minimum, ∑_j (log a[k] + log a[j] - log|A[k,j]|) ≈ 0 for each k
        a = soft_symcover(AbsLog{2}(), A)
        for k in axes(A, 1)
            residual = sum(abs(A[k,j]) != 0 ? log(a[k]) + log(a[j]) - log(abs(A[k,j])) : 0.0
                           for j in axes(A, 2))
            @test abs(residual) < 1e-10
        end
        # AbsLinear{2}: at the minimum, ∑_j (1 - r_{kj}) r_{kj} ≈ 0 for each k
        a = soft_symcover(AbsLinear{2}(), A)
        for k in axes(A, 1)
            residual = sum((rj = abs(A[k,j])/(a[k]*a[j]); (1 - rj) * rj) for j in axes(A, 2))
            @test abs(residual) < 1e-10
        end
    end

    # Non-square rejected (default dispatch and all ϕ)
    @test_throws "requires a square matrix" soft_symcover([1.0 2.0; 3.0 4.0; 5.0 6.0])
    for ϕ in PENALTIES
        @test_throws "requires a square matrix" soft_symcover(ϕ, [1.0 2.0; 3.0 4.0; 5.0 6.0])
    end

    # Rank-1 is exact: A = [4 2; 2 1] = [2;1]*[2 1], so exact cover has all ratios = 1
    A = [4.0 2.0; 2.0 1.0]
    for ϕ in PENALTIES
        a = soft_symcover(ϕ, A)
        @test cover_objective(ϕ, a, A) ≈ 0.0 atol=1e-8
    end

    # Diagonal-scaling covariance: soft_symcover(ϕ, ·) co-varies with a diagonal rescaling of A.
    A = [2.0 1.0; 1.0 3.0]
    d = [2.0, 0.5]
    for ϕ in PENALTIES
        @test covaries(a -> soft_symcover(ϕ, a), A, d; rtol=1e-5)
    end

    # Lower objective than naive uniform scaling perturbations
    A = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]
    for ϕ in PENALTIES
        a = soft_symcover(ϕ, A)
        E = cover_objective(ϕ, a, A)
        for scale in [0.5, 0.9, 1.1, 2.0]
            @test E <= cover_objective(ϕ, scale .* a, A) + 1e-8
        end
    end

    # `AbsLinear` remains continuous as an entry vanishes.
    γ = 0.5
    A_zero  = [γ 1.0; 1.0 0.0]
    A_small = [γ 1.0; 1.0 1e-10]
    for ϕ in (AbsLinear{1}(), AbsLinear{2}())
        a_zero  = soft_symcover(ϕ, A_zero;  maxiter=50)
        a_small = soft_symcover(ϕ, A_small; maxiter=50)
        @test a_small ≈ a_zero atol=1e-5
    end

    # The leave-one-out basin selection is covariant away from its documented tie.
    for ϕ in (AbsLinear{1}(), AbsLinear{2}())
        for (B, d) in ((A_small, [50.0, 0.02]),
                       ([3.0 7.6e-10; 7.6e-10 80.0], [35.0, 3400.0]))
            a0 = soft_symcover(ϕ, B; maxiter=100)
            as = soft_symcover(ϕ, B .* d .* d'; maxiter=100)
            @test as .* as' ≈ (d .* d') .* (a0 .* a0') rtol=1e-6
        end
    end

    # Default dispatch uses AbsLinear{2}
    A = [2.0 1.0; 1.0 3.0]
    @test soft_symcover(A) ≈ soft_symcover(AbsLinear{2}(), A)
end

@testset "soft_cover" begin
    # Closed-form uniform-basin critical point for `Aε = [1 ε; ε 1]`.
    for ε in (0.5, 0.1, 1e-3)
        Aε = [1.0 ε; ε 1.0]
        target = (1 + ε^2) / (1 + ε)
        a, b = soft_cover(Aε; starts=1, maxiter=200)
        @test all(≈(target; atol=1e-10), a * b')
        # Multistart cannot be worse than the included single start.
        am, bm = soft_cover(Aε; maxiter=200)
        @test cover_objective(AbsLinear{2}(), am, bm, Aε) <=
              cover_objective(AbsLinear{2}(), a, b, Aε) + 1e-12
    end

    # Default dispatch uses AbsLinear{2}
    A = [1.0 2.0 3.0; 6.0 5.0 4.0]
    @test soft_cover(A) == soft_cover(AbsLinear{2}(), A)

    # Monotone descent: the returned objective never exceeds the geometric-mean init's.
    for A in ([1.0 2.0 3.0; 6.0 5.0 4.0],
              [2.0 1.0 0.5; 0.1 4.0 3.0; 1.0 2.0 0.2; 5.0 0.3 1.0])
        a0, b0 = cover(A; maxiter=0)
        Einit = cover_objective(AbsLinear{2}(), a0, b0, A)
        a, b = soft_cover(A)
        @test cover_objective(AbsLinear{2}(), a, b, A) <= Einit + 1e-12
    end

    # Scale-covariance of the product: independent row/column rescalings D_r*A*D_c leave
    # the product a*b' scaled by D_r * D_c.
    A = [2.0 1.0 0.5; 0.1 4.0 3.0; 1.0 2.0 0.2; 5.0 0.3 1.0]
    dr = [3.0, 0.5, 2.0, 0.25]
    dc = [4.0, 0.1, 1.5]
    @test covaries(soft_cover, A, dr, dc; rtol=1e-8)
    @test covaries_objective(AbsLinear{2}(), soft_cover, A, dr, dc; rtol=1e-12)

    # Zeros: an entirely-zero row/column gets scale 0; scattered zeros are handled.
    Az = [0.0 0.0 0.0; 1.0 2.0 3.0; 4.0 0.0 5.0]
    az, bz = soft_cover(Az)
    @test az[1] == 0
    @test all(isfinite, az) && all(isfinite, bz)
    @test isfinite(cover_objective(AbsLinear{2}(), az, bz, Az))

    # AbsLinear{1}: alternating weighted-median descent, warm-started from AbsLinear{2}.
    @testset "AbsLinear{1}" begin
        # A rank-1 matrix A = a*b' is exactly coverable, so the L1 objective is 0.
        A1 = [2.0, 0.5, 3.0] * [1.0, 4.0, 0.25, 2.0]'
        a, b = soft_cover(AbsLinear{1}(), A1)
        @test cover_objective(AbsLinear{1}(), a, b, A1) ≈ 0 atol=1e-12
        @test a isa Vector{Float64} && b isa Vector{Float64}

        # Determinism (the default rng is evaluated fresh per call).
        @test soft_cover(AbsLinear{1}(), A1) == soft_cover(AbsLinear{1}(), A1)

        # The weighted-median refinement never worsens the L1 objective of its
        # AbsLinear{2} warm start (each block update is an exact minimization).
        for M in ([1.0 2.0 3.0; 6.0 5.0 4.0],
                  [2.0 1.0 0.5; 0.1 4.0 3.0; 1.0 2.0 0.2; 5.0 0.3 1.0])
            a0, b0 = soft_cover(AbsLinear{2}(), M; maxiter=5, starts=8, rng=StableRNG(0))
            Einit = cover_objective(AbsLinear{1}(), a0, b0, M)
            ar, br = soft_cover(AbsLinear{1}(), M; rng=StableRNG(0))
            @test cover_objective(AbsLinear{1}(), ar, br, M) <= Einit + 1e-12
        end

        # Scale-covariance of the product under independent row/column rescalings.
        Ac1 = [2.0 1.0 0.5; 0.1 4.0 3.0; 1.0 2.0 0.2; 5.0 0.3 1.0]
        dr = [3.0, 0.5, 2.0, 0.25]; dc = [4.0, 0.1, 1.5]
        @test covaries(A -> soft_cover(AbsLinear{1}(), A; rng=StableRNG(7)), Ac1, dr, dc; rtol=1e-8)

        # Zeros: an entirely-zero row gets scale 0; results stay finite.
        Az1 = [0.0 0.0 0.0; 1.0 2.0 3.0; 4.0 0.0 5.0]
        az, bz = soft_cover(AbsLinear{1}(), Az1)
        @test az[1] == 0
        @test all(isfinite, az) && all(isfinite, bz)
        @test isfinite(cover_objective(AbsLinear{1}(), az, bz, Az1))
    end

    # AbsLog{2} is the convex soft cover, and identical to its minimizer.
    @test soft_cover(AbsLog{2}(), A) == soft_cover_min(AbsLog{2}(), A)
end

@testset "soft_cover AbsLog{1}" begin
    # Each half-sweep is an exact block minimization, so the descent never worsens the
    # AbsLog{2} start it refines.
    for M in ([1.0 2.0 3.0; 6.0 5.0 4.0],
              [2.0 1.0 0.5; 0.1 4.0 3.0; 1.0 2.0 0.2; 5.0 0.3 1.0],
              [1.0 2.0 0.0 4.0; 0.0 5.0 6.0 1.0; 3.0 0.0 2.0 8.0])
        a0, b0 = soft_cover_min(AbsLog{2}(), M)
        a, b = soft_cover(AbsLog{1}(), M)
        @test cover_objective(AbsLog{1}(), a, b, M) <= cover_objective(AbsLog{1}(), a0, b0, M) + 1e-12
        @test isbalanced(a, b, M)
        @test a isa Vector{Float64} && b isa Vector{Float64}
    end

    # A rank-1 matrix is exactly coverable, so the L1 objective reaches 0.
    A1 = [2.0, 0.5, 3.0] * [1.0, 4.0, 0.25, 2.0]'
    a, b = soft_cover(AbsLog{1}(), A1)
    @test cover_objective(AbsLog{1}(), a, b, A1) ≈ 0 atol=1e-10

    # Deterministic, and scale-covariant in the product under row/column rescaling.
    @test soft_cover(AbsLog{1}(), A1) == soft_cover(AbsLog{1}(), A1)
    Ac = [2.0 1.0 0.5; 0.1 4.0 3.0; 1.0 2.0 0.2; 5.0 0.3 1.0]
    dr = [3.0, 0.5, 2.0, 0.25]; dc = [4.0, 0.1, 1.5]
    @test covaries(A -> soft_cover(AbsLog{1}(), A), Ac, dr, dc; rtol=1e-8)

    # An entirely-zero row keeps scale 0 and the rest stays finite.
    Az = [0.0 0.0 0.0; 1.0 2.0 3.0; 4.0 0.0 5.0]
    az, bz = soft_cover(AbsLog{1}(), Az)
    @test az[1] == 0
    @test all(isfinite, az) && all(isfinite, bz)

    # Free row and column factors need not match the symmetric solution.
    S = [4.0 1.0 2.0; 1.0 9.0 3.0; 2.0 3.0 16.0]
    S1 = (v = [2.0, 0.5, 3.0]; v * v')
    a, b = soft_cover(AbsLog{1}(), S1)
    @test a .* b' ≈ soft_symcover(AbsLog{1}(), S1) .* soft_symcover(AbsLog{1}(), S1)' rtol=1e-8

    # The minimizers stay unimplemented; the heuristic is not a stand-in for one.
    @test_throws MethodError soft_cover_min(AbsLog{1}(), A1)
    @test_throws MethodError soft_symcover_min(AbsLog{1}(), S)
end

@testset "AbsLinear soft-cover multistart" begin
    A = [1.0 2.0 3.0; 6.0 5.0 4.0]
    As = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]

    # A caller-supplied `rng` is threaded through: identical RNG state ⇒ identical result.
    @test soft_cover(A; rng=StableRNG(7)) == soft_cover(A; rng=StableRNG(7))
    @test soft_symcover(As; rng=StableRNG(7)) == soft_symcover(As; rng=StableRNG(7))

    # `starts` and `σ` are accepted; `starts=1` is a plain single start.
    @test soft_cover(A; starts=1, σ=1.5) isa Tuple
    @test soft_symcover(As; starts=1, σ=1.5) isa Vector

    # `sigma` is an ASCII alias for `σ`; passing both is fine when they agree and an
    # error when they don't.
    @test soft_cover(A; starts=1, sigma=1.5) == soft_cover(A; starts=1, σ=1.5)
    @test soft_symcover(As; starts=1, sigma=1.5) == soft_symcover(As; starts=1, σ=1.5)
    @test soft_cover(A; starts=1, σ=1.5, sigma=1.5) == soft_cover(A; starts=1, σ=1.5)
    @test_throws "specify only one" soft_cover(A; σ=1.5, sigma=2.0)
    @test_throws "specify only one" soft_symcover(As; σ=1.5, sigma=2.0)

    # Best-of-eight cannot exceed the included single-start objective.
    for (_, M) in general_matrices
        Mf = float.(M)
        a1, b1 = soft_cover(Mf; starts=1)
        a8, b8 = soft_cover(Mf; starts=8)
        @test cover_objective(AbsLinear{2}(), a8, b8, Mf) <=
              cover_objective(AbsLinear{2}(), a1, b1, Mf) + 1e-12
    end
    for (_, M) in symmetric_matrices
        Mf = float.(M)
        a1 = soft_symcover(Mf; starts=1)
        a8 = soft_symcover(Mf; starts=8)
        @test cover_objective(AbsLinear{2}(), a8, Mf) <=
              cover_objective(AbsLinear{2}(), a1, Mf) + 1e-12
    end

    # Scale-covariance of the returned product: every start co-varies with a rescaling
    # of A and the objective is scale-invariant, so the selected product co-varies too.
    Ac = [2.0 1.0 0.5; 0.1 4.0 3.0; 1.0 2.0 0.2; 5.0 0.3 1.0]
    dr = [3.0, 0.5, 2.0, 0.25]; dc = [4.0, 0.1, 1.5]
    @test covaries(soft_cover, Ac, dr, dc; rtol=1e-7)
    d = [2.0, 0.5, 3.0]
    @test covaries(soft_symcover, As, d; rtol=1e-7)

    # Objectives match across rescaled frames to roundoff.
    @test covaries_objective(AbsLinear{2}(), soft_cover, Ac, dr, dc; rtol=1e-12)
    @test covaries_objective(AbsLinear{2}(), soft_symcover, As, d; rtol=1e-12)

    # Stable RNGs make the fixed-corpus multistart comparison reproducible.
    rng = StableRNG(2024)
    imp_sym = 0; imp_gen = 0
    for k in 1:40
        S = exp.(5 .* randn(rng, 6, 6)); S = (S + S') / 2
        e1 = cover_objective(AbsLinear{2}(), soft_symcover(S; starts=1, rng=StableRNG(k)), S)
        e8 = cover_objective(AbsLinear{2}(), soft_symcover(S; starts=8, rng=StableRNG(k)), S)
        e8 < e1 - 1e-9 && (imp_sym += 1)
        G = exp.(5 .* randn(rng, 5, 7))
        g1a, g1b = soft_cover(G; starts=1, rng=StableRNG(k)); g8a, g8b = soft_cover(G; starts=8, rng=StableRNG(k))
        cover_objective(AbsLinear{2}(), g8a, g8b, G) <
            cover_objective(AbsLinear{2}(), g1a, g1b, G) - 1e-9 && (imp_gen += 1)
    end
    # Require improvement on a substantial fraction of the corpus.
    @test imp_sym >= 24
    @test imp_gen >= 19
end

@testset "feasible start and provenance" begin
    # Instrumented multistart exposes candidate labels and objectives.
    function provenance(A; rng=StableRNG(0))
        labels = String[]; objs = Float64[]
        a = MatrixCovers._soft_symcover_abslinear2(A, 32, 5, 2.0, rng; labels, objs)
        return a, labels[MatrixCovers._multistart_select(objs)], labels, objs
    end

    # The feasible start reaches the better basin for this matrix with zeros.
    Afe = Float64[0 11 18 0 12; 11 0 1 0 20; 18 1 0 3 18; 0 0 3 18 0; 12 20 18 0 17]
    a, winner, labels, objs = provenance(Afe)
    @test winner == "feasible"
    # The instrumented call returns exactly what the public entry point selects.
    @test a == soft_symcover(Afe; rng=StableRNG(0))
    # The winning basin is separated beyond descent tolerance.
    fi = findfirst(==("feasible"), labels)
    @test objs[fi] <= minimum(objs) * (1 + 1e-6)
    other_basin = minimum(o for o in objs if o > objs[fi] * (1 + 1e-6))
    @test objs[fi] < other_basin * (1 - 1e-3)

    # Selecting the `feasible` start preserves scale-covariance of the returned cover.
    d = [1.5, 0.3, 4.0, 0.7, 2.2]
    a_scaled = soft_symcover(Afe .* d .* d'; rng=StableRNG(0))
    @test a_scaled * a_scaled' ≈ (d .* d') .* (a * a') rtol=1e-7

    # Gate off: a fully dense matrix (no zeros) never offers the `feasible` start.
    _, _, dense_labels, _ = provenance(Float64[4 1 2; 1 5 3; 2 3 6])
    @test "feasible" ∉ dense_labels

    # The asymmetric multistart exposes provenance the same way (no `feasible` start there).
    # As above, the positional arguments mirror `soft_cover`'s defaults.
    Ag = [1.0 2.0 3.0; 6.0 5.0 4.0]
    albs = String[]; aobjs = Float64[]
    ab = MatrixCovers._soft_cover_abslinear2(Ag, 200, 4, 2.0, StableRNG(0);
                                                       labels=albs, objs=aobjs)
    @test ab == soft_cover(Ag; rng=StableRNG(0))
    @test albs[MatrixCovers._multistart_select(aobjs)] in albs
    @test "feasible" ∉ albs
end

@testset "converged cover covariance" begin
    # A low-curvature direction permits O(sqrt(eps)) factor differences while
    # objectives agree to O(eps).
    rng = StableRNG(1)
    B = exp.(2 .* randn(rng, 40, 40)) .* randn(rng, 40, 40)
    A = (B + B') / 2
    dr = exp.(randn(rng, 40)); dc = exp.(randn(rng, 40))

    @test covaries_objective(AbsLinear{2}(), soft_cover, A, dr, dc; rtol=1e-12)
    # Gauge balancing makes the rescaled factors agree to roundoff.
    @test covaries(soft_cover, A, dr, dc; rtol=1e-9)
end

@testset "soft AbsLog{2} is the exact unconstrained minimum" begin
    # Direct log-space least squares supplies the oracle; compare gauge-invariant
    # objectives because `M` may be singular.
    function exact_sym(A)
        n = size(A, 1)
        S = [(i, j) for i in 1:n, j in 1:n if !iszero(A[i, j])]
        M = zeros(length(S), n)
        z = zeros(length(S))
        for (k, (i, j)) in enumerate(S)
            M[k, i] += 1
            M[k, j] += 1
            z[k] = log(abs(A[i, j]))
        end
        return exp.(pinv(M) * z)
    end
    function exact_asym(A)
        m, n = size(A)
        S = [(i, j) for i in 1:m, j in 1:n if !iszero(A[i, j])]
        M = zeros(length(S), m + n)
        z = zeros(length(S))
        for (k, (i, j)) in enumerate(S)
            M[k, i] = 1
            M[k, m+j] = 1
            z[k] = log(abs(A[i, j]))
        end
        x = pinv(M) * z
        return exp.(x[1:m]), exp.(x[m+1:end])
    end

    # A zero entry is what separates the exact minimum from the geometric mean; a
    # fully supported `A` cannot tell them apart.
    sym_zeros = [4.0 1.0 0.0 2.0; 1.0 9.0 3.0 0.0; 0.0 3.0 1.0 5.0; 2.0 0.0 5.0 16.0]
    asym_zeros = [1.0 2.0 0.0 4.0; 0.0 5.0 6.0 1.0; 3.0 0.0 2.0 8.0]
    rng = StableRNG(11)
    sym_dense = (M = rand(rng, 5, 5); (M + M') ./ 2)
    asym_dense = rand(rng, 4, 6)

    for A in (sym_zeros, sym_dense, float.(last(symmetric_matrices[1])))
        a = soft_symcover_min(AbsLog{2}(), A)
        @test cover_objective(AbsLog{2}(), a, A) ≈ cover_objective(AbsLog{2}(), exact_sym(A), A) rtol=1e-8
        # Stationarity of the convex objective: ∂/∂α_k ∑ (α_i + α_j − log|A_ij|)² = 0.
        α = log.(a)
        g = [sum(2 * (α[i] + α[j] - log(abs(A[i, j]))) * ((i == k) + (j == k))
                 for i in axes(A, 1), j in axes(A, 2) if !iszero(A[i, j])) for k in axes(A, 1)]
        @test maximum(abs, g) < 1e-8 * max(1, maximum(abs, α))
    end
    for A in (asym_zeros, asym_dense, float.(last(general_matrices[1])))
        a, b = soft_cover_min(AbsLog{2}(), A)
        ae, be = exact_asym(A)
        @test cover_objective(AbsLog{2}(), a, b, A) ≈ cover_objective(AbsLog{2}(), ae, be, A) rtol=1e-8
        @test isbalanced(a, b, A)
    end

    # Sparse support distinguishes the exact minimum from the geometric mean.
    @test cover_objective(AbsLog{2}(), initialize_symcover(sym_dense; strategy=:geomean, feasible=:none), sym_dense) ≈
          cover_objective(AbsLog{2}(), soft_symcover_min(AbsLog{2}(), sym_dense), sym_dense) rtol=1e-8
    @test cover_objective(AbsLog{2}(), initialize_symcover(sym_zeros; strategy=:geomean, feasible=:none), sym_zeros) >
          cover_objective(AbsLog{2}(), soft_symcover_min(AbsLog{2}(), sym_zeros), sym_zeros) * (1 + 1e-6)

    # Convex with one minimizer, so the heuristic and the minimizer are the same
    # function, and a refiner cannot be steered by its start.
    @test soft_symcover(AbsLog{2}(), sym_zeros) == soft_symcover_min(AbsLog{2}(), sym_zeros)
    @test soft_cover(AbsLog{2}(), asym_zeros) == soft_cover_min(AbsLog{2}(), asym_zeros)
    for strategy in (:geomean, :leaveout, :diagfeasible, :hardcover)
        a0 = initialize_symcover(sym_zeros; strategy, feasible=:none)
        @test soft_symcover_min!(AbsLog{2}(), a0, sym_zeros) ≈ soft_symcover_min(AbsLog{2}(), sym_zeros)
    end
    a0, b0 = initialize_cover(asym_zeros; strategy=:geomean, feasible=:none)
    ar, br = soft_cover_min!(AbsLog{2}(), a0, b0, asym_zeros)
    @test (ar, br) == soft_cover_min(AbsLog{2}(), asym_zeros)

    # A bipartite support graph makes the signless Laplacian singular; the balanced
    # representative is the one reported.
    @test soft_symcover_min(AbsLog{2}(), [0 1; 1 0]) ≈ [1.0, 1.0]

    # The `:lsqr` and dense paths solve the same problem.
    @test soft_symcover_min(AbsLog{2}(), sym_zeros; linsolve=:lsqr) ≈
          soft_symcover_min(AbsLog{2}(), sym_zeros; linsolve=:dense) rtol=1e-6

    # The soft cover is the hard worker's cold solve, so it reaches the Woodbury path
    # too: one sparse Cholesky in place of the dense factorization, same answer.
    rng = StableRNG(77)
    X = exp.(randn(rng, 40, 40))
    Adense = (X .+ X') ./ 2
    @test soft_symcover_min(AbsLog{2}(), Adense; linsolve=:woodbury) ≈
          soft_symcover_min(AbsLog{2}(), Adense; linsolve=:dense) rtol=1e-8
    @test MatrixCovers._soft_symcover_min_abslog2(Adense)[2].linsolve === :woodbury
    Y = exp.(randn(rng, 40, 30))
    aw, bw = soft_cover_min(AbsLog{2}(), Y; linsolve=:woodbury)
    ad, bd = soft_cover_min(AbsLog{2}(), Y; linsolve=:dense)
    @test aw .* bw' ≈ ad .* bd' rtol=1e-8
    @test MatrixCovers._soft_cover_min_abslog2(Y)[3].linsolve === :woodbury
end

@testset "sparse soft AbsLog{2} defaults to matrix-free LSQR" begin
    rng = StableRNG(19)
    S0 = sprand(rng, 60, 60, 0.1)
    Ssp = S0 + S0' + I     # supported diagonal keeps every component non-bipartite
    a = soft_symcover_min(AbsLog{2}(), Ssp)
    @test a isa Vector{Float64}
    @test a == soft_symcover_min(AbsLog{2}(), Ssp; linsolve=:lsqr)
    @test a ≈ soft_symcover_min(AbsLog{2}(), Matrix(Ssp); linsolve=:dense) rtol=1e-6
    U = Symmetric(sparse(triu(Ssp)))
    @test soft_symcover_min(AbsLog{2}(), U) == soft_symcover_min(AbsLog{2}(), U; linsolve=:lsqr)
    # Asymmetric form; the cover products are gauge-free.
    G = sprand(rng, 50, 40, 0.12)
    ga, gb = soft_cover_min(AbsLog{2}(), G)
    @test (ga, gb) == soft_cover_min(AbsLog{2}(), G; linsolve=:lsqr)
    gad, gbd = soft_cover_min(AbsLog{2}(), Matrix(G); linsolve=:dense)
    @test ga .* gb' ≈ gad .* gbd' rtol=1e-6
    a0 = copy(a)
    @test soft_symcover_min!(AbsLog{2}(), a0, Ssp) == a
end

@testset "soft_symcover!/soft_cover! refiners" begin
    Asym = [4.0 1.0 0.5; 1.0 3.0 1.0; 0.5 1.0 2.5]
    Agen = [1.0 2.0 0.5; 0.25 3.0 1.0]

    @testset "refining the multistart's own start reproduces it: $ϕ" for ϕ in PENALTIES
        a = initialize_symcover(Asym; strategy=:geomean, feasible=:none)
        soft_symcover!(ϕ, a, Asym)
        @test cover_objective(ϕ, a, Asym) ≈ cover_objective(ϕ, soft_symcover(ϕ, Asym), Asym) rtol=1e-6
    end

    @testset "no-ϕ form defaults to AbsLinear{2}" begin
        a1, a2 = initialize_symcover(Asym; strategy=:geomean, feasible=:none), initialize_symcover(Asym; strategy=:geomean, feasible=:none)
        @test soft_symcover!(a1, Asym) == soft_symcover!(AbsLinear{2}(), a2, Asym)
        b1, c1 = initialize_cover(Agen; strategy=:geomean, feasible=:none)
        b2, c2 = initialize_cover(Agen; strategy=:geomean, feasible=:none)
        @test soft_cover!(b1, c1, Agen) == soft_cover!(AbsLinear{2}(), b2, c2, Agen)
    end

    # Convex refiners ignore the basin; nonconvex refiners do not.
    @testset "start-dependence" begin
        Abasins = [0.021778451276962405 1.5690256886348526
                   1.5690256886348526  0.20473123461805692]
        starts() = (initialize_symcover(Abasins; strategy=:geomean, feasible=:none),
                    initialize_symcover(Abasins; strategy=:hardcover))
        o(ϕ, a) = cover_objective(ϕ, a, Abasins)

        s1, s2 = starts()
        @test o(AbsLog{2}(), soft_symcover!(AbsLog{2}(), s1, Abasins)) ≈
              o(AbsLog{2}(), soft_symcover!(AbsLog{2}(), s2, Abasins))

        s1, s2 = starts()
        @test !isapprox(o(AbsLinear{2}(), soft_symcover!(AbsLinear{2}(), s1, Abasins)),
                        o(AbsLinear{2}(), soft_symcover!(AbsLinear{2}(), s2, Abasins)); rtol=1e-6)
    end

    # Unlike symcover_min!, a soft refiner imposes no coverage constraint on its start.
    @testset "start need not cover A: $ϕ" for ϕ in PENALTIES
        a = fill(0.01, 3)
        @test !iscover(a, Asym)
        @test soft_symcover!(ϕ, a, Asym) === a
        b, c = fill(0.01, 2), fill(0.01, 3)
        @test !iscover(b, c, Agen)
        @test soft_cover!(ϕ, b, c, Agen) === (b, c)
    end

    @testset "invalid starts throw, naming the function called" begin
        @test_throws "soft_symcover! requires a start with finite positive scale" soft_symcover!([1.0, -1.0, 1.0], Asym)
        @test_throws "soft_symcover! requires a start with finite positive scale" soft_symcover!([1.0, 0.0, 1.0], Asym)
        @test_throws "soft_symcover! requires a start with finite positive scale" soft_symcover!([1.0, Inf, 1.0], Asym)
        @test_throws "soft_symcover! requires a square matrix" soft_symcover!([1.0, 1.0], Agen)
        @test_throws DimensionMismatch soft_symcover!([1.0, 1.0], Asym)
        @test_throws "soft_cover! requires a start with finite positive scale" soft_cover!([1.0, -1.0], fill(1.0, 3), Agen)
        @test_throws "soft_cover! requires a start with finite positive scale" soft_cover!(fill(1.0, 2), [1.0, -1.0, 1.0], Agen)
        @test_throws DimensionMismatch soft_cover!(fill(1.0, 3), fill(1.0, 3), Agen)
    end

    # Scales on unsupported rows are inert and come back zero, matching every other
    # cover in the package.
    @testset "unsupported rows are zeroed: $ϕ" for ϕ in PENALTIES
        Az = [1.0 0.0 2.0; 0.0 0.0 0.0; 2.0 0.0 3.0]
        a = initialize_symcover(Az; strategy=:geomean, feasible=:none)
        @test soft_symcover!(ϕ, a, Az)[2] == 0
    end

    @testset "asymmetric refiners pin the balance convention: $ϕ" for ϕ in PENALTIES
        a, b = initialize_cover(Agen; strategy=:geomean, feasible=:none)
        a .*= 7           # move the gauge; the objective cannot see it
        b ./= 7
        soft_cover!(ϕ, a, b, Agen)
        @test isbalanced(a, b, Agen)
    end

    @testset "offset axes propagate: $ϕ" for ϕ in PENALTIES
        Ao = OffsetArray(Asym, -1:1, -1:1)
        ao = initialize_symcover(Ao; strategy=:geomean, feasible=:none)
        soft_symcover!(ϕ, ao, Ao)
        @test axes(ao, 1) == axes(Ao, 1)
        aref = initialize_symcover(Asym; strategy=:geomean, feasible=:none)
        soft_symcover!(ϕ, aref, Asym)
        @test collect(ao) ≈ aref rtol=1e-6
    end
end

# Matrix readable only through support hooks; `getindex` throws.
struct HookOnlyMatrix{T} <: AbstractMatrix{T}
    entries::Vector{Tuple{Int,Int,T}}   # `i <= j` only; the transpose is implied
    n::Int
end
Base.size(M::HookOnlyMatrix) = (M.n, M.n)
Base.getindex(::HookOnlyMatrix, ::Int, ::Int) =
    error("HookOnlyMatrix must be read through foreach_support")
function MatrixCovers.foreach_support_sym(f, M::HookOnlyMatrix)
    for (i, j, v) in M.entries
        iszero(v) || f(i, j, abs(v))
    end
    return nothing
end
function MatrixCovers.foreach_support(f, M::HookOnlyMatrix)
    for (i, j, v) in M.entries
        iszero(v) && continue
        f(i, j, abs(v))
        i == j || f(j, i, abs(v))
    end
    return nothing
end
# One stored orientation makes magnitude symmetry structural.
MatrixCovers.require_abs_symmetric(::HookOnlyMatrix, fname) = nothing

@testset "the soft-cover kernels read through the support hook" begin
    entries = [(1, 1, 2.0), (1, 3, 1.5), (2, 2, 3.0), (2, 4, 0.5), (3, 4, 4.0), (4, 4, 1.0)]
    M = HookOnlyMatrix(entries, 4)
    dense = zeros(4, 4)
    for (i, j, v) in entries
        dense[i, j] = dense[j, i] = v
    end
    start() = [1.3, 0.7, 2.1, 0.9]

    # Match the dense result without indexing `M`.
    for kernel! in (MatrixCovers._abslog1_iter!, MatrixCovers._abslinear1_iter!,
                    MatrixCovers._abslinear2_iter!)
        @test kernel!(start(), M, 50) ≈ kernel!(start(), dense, 50) rtol=1e-10
    end
    for kernel! in (MatrixCovers._abslog1_iter_asym!, MatrixCovers._abslinear1_iter_asym!,
                    MatrixCovers._msmc_als!)
        ah, bh = kernel!(start(), start(), M, 50)
        ad, bd = kernel!(start(), start(), dense, 50)
        @test ah ≈ ad rtol=1e-10
        @test bh ≈ bd rtol=1e-10
    end
end

# Exercise support-only traversal through public initialization paths.
@testset "the cover entry points read through the support hook" begin
    entries = [(1, 1, 2.0), (1, 3, 1.5), (2, 2, 3.0), (2, 4, 0.5), (3, 4, 4.0), (4, 4, 1.0)]
    M = HookOnlyMatrix(entries, 4)
    dense = zeros(4, 4)
    for (i, j, v) in entries
        dense[i, j] = dense[j, i] = v
    end

    # `:diagfeasible` runs `init_feasible_diag!` and `boost_feasible_seq!`;
    # `:leaveout` runs `_leaveout_logmean_init!`.
    for strategy in (:geomean, :hardcover, :leaveout, :diagfeasible)
        for feasible in (:none, :boost, :inflate)
            @test initialize_symcover(M; strategy, feasible) ≈
                  initialize_symcover(dense; strategy, feasible) rtol=1e-10
        end
    end

    for ϕ in (AbsLinear{1}(), AbsLinear{2}(), AbsLog{1}(), AbsLog{2}())
        @test symcover(ϕ, M) ≈ symcover(ϕ, dense) rtol=1e-10
        @test soft_symcover(ϕ, M) ≈ soft_symcover(ϕ, dense) rtol=1e-6
        @test iscover(symcover(ϕ, M), dense; rtol=8eps())
        @test cover_objective(ϕ, symcover(ϕ, M), M) ≈
              cover_objective(ϕ, symcover(ϕ, dense), dense) rtol=1e-10
    end

    for ϕ in (AbsLinear{1}(), AbsLinear{2}(), AbsLog{1}(), AbsLog{2}())
        ah, bh = cover(ϕ, M)
        ad, bd = cover(ϕ, dense)
        @test ah ≈ ad rtol=1e-10
        @test bh ≈ bd rtol=1e-10
        @test iscover(ah, bh, dense; rtol=8eps())
    end
end
