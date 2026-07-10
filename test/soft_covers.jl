# Unconstrained AbsLinear soft covers: descent, multistart, and start provenance.

# Committed 5x5 matrix libraries (`symmetric_matrices`, `general_matrices`); the guard
# permits re-inclusion of this file in an already-initialized session.
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
    @test_throws ArgumentError soft_symcover([1.0 2.0; 3.0 4.0; 5.0 6.0])
    for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
        @test_throws ArgumentError soft_symcover(ϕ, [1.0 2.0; 3.0 4.0; 5.0 6.0])
    end

    # Rank-1 is exact: A = [4 2; 2 1] = [2;1]*[2 1], so exact cover has all ratios = 1
    A = [4.0 2.0; 2.0 1.0]
    for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
        a = soft_symcover(ϕ, A)
        @test cover_objective(ϕ, a, A) ≈ 0.0 atol=1e-8
    end

    # Diagonal-scaling covariance: soft_symcover(ϕ, A .* d .* d') outer-product ≈ d .* d' scaled
    A = [2.0 1.0; 1.0 3.0]
    d = [2.0, 0.5]
    for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
        a0 = soft_symcover(ϕ, A)
        a_scaled = soft_symcover(ϕ, A .* d .* d')
        C0 = a0 .* a0'
        Cs = a_scaled .* a_scaled'
        @test Cs ≈ (d .* d') .* C0 rtol=1e-5
    end

    # Lower objective than naive uniform scaling perturbations
    A = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]
    for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
        a = soft_symcover(ϕ, A)
        E = cover_objective(ϕ, a, A)
        for scale in [0.5, 0.9, 1.1, 2.0]
            @test E <= cover_objective(ϕ, scale .* a, A) + 1e-8
        end
    end

    # Continuity as a near-zero entry vanishes: AbsLinear has no discontinuity at r=0, so
    # the soft cover varies continuously as A[2,2] → 0. The leave-one-out start drops the
    # most-outlying small entry, reaching the same basin as the exact-zero case.
    γ = 0.5
    A_zero  = [γ 1.0; 1.0 0.0]
    A_small = [γ 1.0; 1.0 1e-10]
    for ϕ in (AbsLinear{1}(), AbsLinear{2}())
        a_zero  = soft_symcover(ϕ, A_zero;  maxiter=50)
        a_small = soft_symcover(ϕ, A_small; maxiter=50)
        @test a_small ≈ a_zero atol=1e-5
    end

    # Covariance must survive the regime where the leave-one-out start wins: the entry
    # dropped is selected by the scale-invariant log-residuals, so which basin wins
    # cannot depend on the frame. (Weighting entries by raw |A[i,j]|² fails here: the
    # entries' physical units differ, so their sums are incommensurate and a rescaling
    # can flip the winning basin.) For A_small the residuals of the two diagonal entries
    # tie exactly (true of every symmetric 2×2), where the tie-break uses raw magnitude;
    # the scaling below preserves the magnitude ordering, as covariance under
    # order-flipping scalings is unachievable on that degenerate class.
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
    # Closed form on Aε = [1 ε; ε 1]: the uniform-product critical point has
    # a*b' ≡ (1+ε²)/(1+ε) on every entry. A single geometric-mean start converges to
    # it. For small ε this is only a local minimizer — a strongly asymmetric solution
    # that covers three entries and sacrifices one off-diagonal has lower objective —
    # so the default multistart may (correctly) return a different, better product.
    for ε in (0.5, 0.1, 1e-3)
        Aε = [1.0 ε; ε 1.0]
        target = (1 + ε^2) / (1 + ε)
        a, b = soft_cover(Aε; starts=1, maxiter=200)
        @test all(≈(target; atol=1e-10), a * b')
        # The multistart never does worse than this uniform-basin local minimizer.
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
    a0, b0 = soft_cover(A)
    as, bs = soft_cover(dr .* A .* dc')
    @test as * bs' ≈ (dr .* (a0 * b0') .* dc') rtol=1e-8

    # Zeros: an entirely-zero row/column gets scale 0; scattered zeros are handled.
    Az = [0.0 0.0 0.0; 1.0 2.0 3.0; 4.0 0.0 5.0]
    az, bz = soft_cover(Az)
    @test az[1] == 0
    @test all(isfinite, az) && all(isfinite, bz)
    @test isfinite(cover_objective(AbsLinear{2}(), az, bz, Az))

    # An all-zero column forces its scale to 0.
    Ac = [1.0 0.0 2.0; 3.0 0.0 4.0]
    ac, bc = soft_cover(Ac)
    @test bc[2] == 0

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
            a0, b0 = soft_cover(AbsLinear{2}(), M; maxiter=5, starts=8, rng=MersenneTwister(0))
            Einit = cover_objective(AbsLinear{1}(), a0, b0, M)
            ar, br = soft_cover(AbsLinear{1}(), M; rng=MersenneTwister(0))
            @test cover_objective(AbsLinear{1}(), ar, br, M) <= Einit + 1e-12
        end

        # Scale-covariance of the product under independent row/column rescalings.
        Ac1 = [2.0 1.0 0.5; 0.1 4.0 3.0; 1.0 2.0 0.2; 5.0 0.3 1.0]
        dr = [3.0, 0.5, 2.0, 0.25]; dc = [4.0, 0.1, 1.5]
        a0, b0 = soft_cover(AbsLinear{1}(), Ac1; rng=MersenneTwister(7))
        as, bs = soft_cover(AbsLinear{1}(), dr .* Ac1 .* dc'; rng=MersenneTwister(7))
        @test as * bs' ≈ (dr .* (a0 * b0') .* dc') rtol=1e-8

        # Zeros: an entirely-zero row gets scale 0; results stay finite.
        Az1 = [0.0 0.0 0.0; 1.0 2.0 3.0; 4.0 0.0 5.0]
        az, bz = soft_cover(AbsLinear{1}(), Az1)
        @test az[1] == 0
        @test all(isfinite, az) && all(isfinite, bz)
        @test isfinite(cover_objective(AbsLinear{1}(), az, bz, Az1))
    end

    # AbsLog penalties are unsupported for soft_cover.
    @test_throws MethodError soft_cover(AbsLog{2}(), A)
end

@testset "AbsLinear soft-cover multistart" begin
    # Determinism: the default fresh-seeded RNG makes repeated calls bit-identical.
    A = [1.0 2.0 3.0; 6.0 5.0 4.0]
    @test soft_cover(A) == soft_cover(A)
    As = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]
    @test soft_symcover(As) == soft_symcover(As)

    # A caller-supplied `rng` is threaded through: identical RNG state ⇒ identical result.
    @test soft_cover(A; rng=MersenneTwister(7)) == soft_cover(A; rng=MersenneTwister(7))
    @test soft_symcover(As; rng=MersenneTwister(7)) == soft_symcover(As; rng=MersenneTwister(7))

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

    # best-of-8 never exceeds the single-start objective on the committed libraries
    # (the multistart's incumbent is the single start, replaced only on improvement).
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
    a0, b0 = soft_cover(Ac); ac, bc = soft_cover(dr .* Ac .* dc')
    @test ac * bc' ≈ dr .* (a0 * b0') .* dc' rtol=1e-7
    d = [2.0, 0.5, 3.0]
    a0s = soft_symcover(As); ass = soft_symcover(As .* d .* d')
    @test ass * ass' ≈ (d .* d') .* (a0s * a0s') rtol=1e-7

    # On a hard lognormal-σ=5 ensemble the multistart strictly lowers the objective on a
    # substantial fraction of instances (fixed seed; loose statistical bound).
    rng = MersenneTwister(2024)
    imp_sym = 0; imp_gen = 0
    for _ in 1:40
        S = exp.(5 .* randn(rng, 6, 6)); S = (S + S') / 2
        e1 = cover_objective(AbsLinear{2}(), soft_symcover(S; starts=1), S)
        e8 = cover_objective(AbsLinear{2}(), soft_symcover(S; starts=8), S)
        e8 < e1 - 1e-9 && (imp_sym += 1)
        G = exp.(5 .* randn(rng, 5, 7))
        g1a, g1b = soft_cover(G; starts=1); g8a, g8b = soft_cover(G; starts=8)
        cover_objective(AbsLinear{2}(), g8a, g8b, G) <
            cover_objective(AbsLinear{2}(), g1a, g1b, G) - 1e-9 && (imp_gen += 1)
    end
    @test imp_sym >= 20
    @test imp_gen >= 15
end

@testset "feasible start and provenance" begin
    # The multistart fills caller-supplied `labels`/`objs` in place; the winner is
    # `labels[_multistart_select(objs)]`, using the same selection rule as the solver.
    function provenance(A; rng=MersenneTwister(0))
        labels = String[]; objs = Float64[]
        a = ScaleInvariantAnalysis._soft_symcover_abslinear2(A, 20, 8, 2.0, rng; labels, objs)
        return a, labels[ScaleInvariantAnalysis._multistart_select(objs)], labels, objs
    end

    # The greedy feasible cover (`init_feasible_diag!`) is offered as a start only when `A` has a
    # zero entry. On this matrix every geometric-mean-derived start lands in one basin while
    # `feasible` reaches a distinctly better one, so it is the selected winner.
    Afe = Float64[0 11 18 0 12; 11 0 1 0 20; 18 1 0 3 18; 0 0 3 18 0; 12 20 18 0 17]
    a, winner, labels, objs = provenance(Afe)
    @test winner == "feasible"
    # The instrumented call returns exactly what the public entry point selects.
    @test a == soft_symcover(Afe)
    # `feasible` wins by a genuine basin gap, not descent-tolerance noise: it is
    # co-optimal in the best basin (a perturbed start may also reach that basin and
    # tie it to within the descent tolerance), and that basin beats every start in a
    # different basin by a wide margin.
    fi = findfirst(==("feasible"), labels)
    @test objs[fi] <= minimum(objs) * (1 + 1e-6)
    other_basin = minimum(o for o in objs if o > objs[fi] * (1 + 1e-6))
    @test objs[fi] < other_basin * (1 - 1e-3)

    # Selecting the `feasible` start preserves scale-covariance of the returned cover.
    d = [1.5, 0.3, 4.0, 0.7, 2.2]
    a_scaled = soft_symcover(Afe .* d .* d')
    @test a_scaled * a_scaled' ≈ (d .* d') .* (a * a') rtol=1e-7

    # Gate off: a fully dense matrix (no zeros) never offers the `feasible` start.
    _, _, dense_labels, _ = provenance(Float64[4 1 2; 1 5 3; 2 3 6])
    @test "feasible" ∉ dense_labels

    # The asymmetric multistart exposes provenance the same way (no `feasible` start there).
    Ag = [1.0 2.0 3.0; 6.0 5.0 4.0]
    albs = String[]; aobjs = Float64[]
    ab = ScaleInvariantAnalysis._soft_cover_abslinear2(Ag, 100, 8, 2.0, MersenneTwister(0);
                                                       labels=albs, objs=aobjs)
    @test ab == soft_cover(Ag)
    @test albs[ScaleInvariantAnalysis._multistart_select(aobjs)] in albs
    @test "feasible" ∉ albs
end
