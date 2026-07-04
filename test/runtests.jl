using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: divmag, dotabs
using JuMP, HiGHS, Ipopt   # triggers SIAJuMP and SIAIpopt extensions
using SparseArrays  # triggers SIASparseArrays extension
using LinearAlgebra
using Statistics: median
using Random: MersenneTwister
using Test

@testset "ScaleInvariantAnalysis.jl" begin

    @testset "cover_objective" begin
        A = [4.0 1.5; 1.5 1.0]
        a = [2.0, 1.0]
        # AbsLog{1}: sum of log-domain excesses over nonzero entries
        @test cover_objective(AbsLog{1}(), a, A) ≈ sum(abs(log(a[i]*a[j]/abs(A[i,j]))) for i in 1:2, j in 1:2 if A[i,j] != 0)
        # AbsLog{2}: sum of squared log-domain excesses
        @test cover_objective(AbsLog{2}(), a, A) ≈ sum(abs(log(a[i]*a[j]/abs(A[i,j])))^2 for i in 1:2, j in 1:2 if A[i,j] != 0)
        # AbsLinear{1} and AbsLinear{2}: ratio deviations from 1 (ALL entries, including zeros)
        @test cover_objective(AbsLinear{1}(), a, A) ≈ sum(abs(abs(A[i,j])/(a[i]*a[j]) - 1) for i in 1:2, j in 1:2)
        @test cover_objective(AbsLinear{2}(), a, A) ≈ sum((abs(A[i,j])/(a[i]*a[j]) - 1)^2 for i in 1:2, j in 1:2)
        # Two-argument form equals one-argument form
        @test cover_objective(AbsLog{2}(), a, a, A) == cover_objective(AbsLog{2}(), a, A)
        @test cover_objective(AbsLinear{2}(), a, a, A) == cover_objective(AbsLinear{2}(), a, A)
        # AbsLog{p}: zero entries contribute 0
        A0 = [1.0 0.0; 0.0 5.0]
        a0 = [1.0, 2.0]
        @test cover_objective(AbsLog{1}(), a0, A0) ≈ log(5/4)
        @test cover_objective(AbsLog{2}(), a0, A0) ≈ log(5/4)^2
        # AbsLinear{p}: zero entries contribute 1 each
        @test cover_objective(AbsLinear{1}(), a0, A0) ≈ 0.0 + 1.0 + 1.0 + 1/4    # (0,1), (1,0) off-diag zeros
        @test cover_objective(AbsLinear{2}(), a0, A0) ≈ 0.0 + 1.0 + 1.0 + 1/16   # (0,1), (1,0) off-diag zeros
    end

    @testset "symcover" begin
        # Cover property: a[i]*a[j] >= abs(A[i,j]) for all i, j
        for A in ([2.0 1.0; 1.0 3.0], [1.0 -0.2; -0.2 0.0], [1.0 0.0; 0.0 0.0],
                  [100.0 1.0; 1.0 0.01])
            for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
                a = symcover(ϕ, A)
                @test all(a[i] * a[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
            end
            # Default dispatch
            a = symcover(A)
            @test all(a[i] * a[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
        end
        # All-zero row or column gives zero cover element for any ϕ
        for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
            a = symcover(ϕ, [1.0 0; 0 0])
            @test a[2] == 0
            a = symcover(ϕ, [0 0; 0 1.0])
            @test a[1] == 0
        end
        # All-zero diagonals
        A = [0.0 1.0; 1.0 0.0]
        for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
            @test symcover(ϕ, A) == [1.0, 1.0]
        end
        # Diagonal scaling covariance
        A = [2.0 1.0; 1.0 3.0]
        d = [2.0, 0.5]
        for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
            @test symcover(ϕ, A .* d .* d') ≈ d .* symcover(ϕ, A)
        end
        # Non-square input is rejected
        @test_throws ArgumentError symcover([1.0 2.0; 3.0 4.0; 5.0 6.0])
    end

    @testset "symcover log cache" begin
        # Passing a scratch `cache` reuses the geometric-mean logarithms in the
        # AbsLinear feasibility step; the result must be identical to the no-cache
        # path (bitwise, not merely approximately). One buffer is reused across
        # several differently-sized-support matrices to mimic the batch use case.
        rng = MersenneTwister(1)
        for n in (2, 5, 40)
            cache = Matrix{Float64}(undef, n, n)
            for _ in 1:5
                B = randn(rng, n, n); A = (B + B') / 2
                @test symcover(A) == symcover(A; cache)
                @test symcover(AbsLinear{2}(), A) == symcover(AbsLinear{2}(), A; cache)
            end
            # Matrices with zero rows/entries: cache cells for zero entries are never read.
            A = [1.0 0.0 2.0; 0.0 0.0 0.0; 2.0 0.0 3.0]
            @test symcover(A) == symcover(A; cache=Matrix{Float64}(undef, 3, 3))
        end
        # A cache whose axes do not match `A` is rejected.
        @test_throws DimensionMismatch symcover([2.0 1.0; 1.0 3.0]; cache=zeros(3, 3))
    end

    @testset "cover" begin
        # Cover property: a[i]*b[j] >= abs(A[i,j]) for all i, j
        for A in ([2.0 1.0; 1.0 3.0], [0.0 1.0; -2.0 0.0], [1.0 0.0; 0.0 0.0],
                  [100.0 1.0; 1.0 0.01])
            for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
                a, b = cover(ϕ, A)
                @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
            end
            # Default dispatch
            a, b = cover(A)
            @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
        end
        # All-zero row or column gives zero cover element for any ϕ
        for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
            a, b = cover(ϕ, [1.0 0; 0 0])
            @test b[2] == 0
            a, b = cover(ϕ, [0 0; 0 1.0])
            @test a[1] == 0
        end
        # Zero-diagonal matrix
        A = [0.0 1.0; -1.0 0.0]
        for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
            a, b = cover(ϕ, A)
            @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
        end
        # Rectangular matrix
        A = [1.0 2.0 3.0; 4.0 5.0 6.0]
        a, b = cover(A)
        @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
        # Diagonal scaling covariance: cover(A .* dr .* dc') is cover(A) scaled by dr, dc up to a scalar
        A = [2.0 1.0; 1.0 3.0]
        dr, dc = [2.0, 0.5], [3.0, 0.25]
        for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
            a, b = cover(ϕ, A .* dr .* dc')
            a0, b0 = cover(ϕ, A)
            c = a \ (dr .* a0)   # scalar: cover has a free global scale a→c*a, b→b/c
            @test c * a ≈ dr .* a0
            @test b / c ≈ dc .* b0
        end
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
            a_zero  = soft_symcover(ϕ, A_zero;  iter=50)
            a_small = soft_symcover(ϕ, A_small; iter=50)
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
                a0 = soft_symcover(ϕ, B; iter=100)
                as = soft_symcover(ϕ, B .* d .* d'; iter=100)
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
            a, b = soft_cover(Aε; starts=1, iter=200)
            @test all(≈(target; atol=1e-10), a * b')
            # The multistart never does worse than this uniform-basin local minimizer.
            am, bm = soft_cover(Aε; iter=200)
            @test cover_objective(AbsLinear{2}(), am, bm, Aε) <=
                  cover_objective(AbsLinear{2}(), a, b, Aε) + 1e-12
        end

        # Default dispatch uses AbsLinear{2}
        A = [1.0 2.0 3.0; 6.0 5.0 4.0]
        @test soft_cover(A) == soft_cover(AbsLinear{2}(), A)

        # Monotone descent: the returned objective never exceeds the geometric-mean init's.
        for A in ([1.0 2.0 3.0; 6.0 5.0 4.0],
                  [2.0 1.0 0.5; 0.1 4.0 3.0; 1.0 2.0 0.2; 5.0 0.3 1.0])
            a0, b0 = cover(A; iter=0)
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

        # Only AbsLinear{2} is supported.
        @test_throws MethodError soft_cover(AbsLog{2}(), A)
    end

    @testset "AbsLinear soft-cover multistart" begin
        if !isdefined(@__MODULE__, :general_matrices)
            include("testmatrices.jl")
        end

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

    @testset "dotabs" begin
        @test dotabs([1.0, -2.0, 3.0], [4.0, 5.0, -6.0]) ≈ 4.0 + 10.0 + 18.0
        @test dotabs([0.0, 1.0], [1.0, 0.0]) == 0.0
        @test dotabs(big.([1.0, 2.0]), big.([3.0, 4.0])) ≈ 3.0 + 8.0
    end

    @testset "divmag" begin
        A = [1.0 -0.2; -0.2 0]
        b = [0.75, 7.0]
        a, mag = divmag(A, b)
        @test abs(dotabs(A \ b, a) - dotabs(big.(A) \ big.(b), a)) <= 2 * eps(mag)
        # Uniform scaling covariance
        asc, magsc = divmag(1000 * A, 1000 * b)
        @test asc ≈ sqrt(1000) .* a
        @test magsc ≈ sqrt(1000) * mag
        # Diagonal scaling covariance
        for _ in 1:10
            d = -log.(rand(2))
            Ad = A .* d .* d'
            bd = b .* d
            a_d, mag_d = divmag(Ad, bd)
            @test abs(dotabs(Ad \ bd, a_d) - dotabs(big.(Ad) \ big.(bd), a_d)) <= 100 * eps(mag_d)
        end
        # Ill-conditioned matrix
        A_ill = [1.0 -0.9999; -0.9999 1]
        a_ill, mag_ill = divmag(A_ill, b)
        @test abs(dotabs(A_ill \ b, a_ill) - dotabs(big.(A_ill) \ big.(b), a_ill)) > 10^6 * eps(mag_ill)
        # With cond=true, accuracy is recovered
        a_ill2, mag_ill2 = divmag(A_ill, b; cond=true)
        @test abs(dotabs(A_ill \ b, a_ill2) - dotabs(big.(A_ill) \ big.(b), a_ill2)) <= 10^3 * eps(mag_ill2)
    end

    @testset "symcover_min and cover_min (JuMP/HiGHS)" begin
        for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 1.0 0.01], [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0])
            a_fast  = symcover(A)
            a_lmin  = symcover_min(AbsLog{1}(), A)
            a_qmin  = symcover_min(AbsLog{2}(), A)
            # qmin is a valid cover
            @test all(a_qmin[i] * a_qmin[j] >= abs(A[i, j]) - 1e-10 for i in axes(A, 1), j in axes(A, 2))
            # qmin achieves lower or equal AbsLog{2} objective than symcover and lmin
            # (up to the near-exact solver's penalty-continuation tolerance).
            @test cover_objective(AbsLog{2}(), a_qmin, A) <= cover_objective(AbsLog{2}(), a_fast, A) * (1 + 1e-6) + 1e-10
            @test cover_objective(AbsLog{2}(), a_qmin, A) <= cover_objective(AbsLog{2}(), a_lmin, A) * (1 + 1e-6) + 1e-10
        end
        # Exact case with zeros
        A = [0 0 1; 0 0 2; 1 2 1]
        a = symcover_min(AbsLog{1}(), A)
        @test all(a[i] * a[j] >= abs(A[i, j]) - 1e-10 for i in axes(A, 1), j in axes(A, 2))
        @test a ≈ [1, 2, 1]
        @test abs(cover_objective(AbsLog{1}(), a, A)) < 1e-10
        a = symcover_min(AbsLog{2}(), A)
        @test all(a[i] * a[j] >= abs(A[i, j]) - 1e-10 for i in axes(A, 1), j in axes(A, 2))
        @test a ≈ [1, 2, 1]
        @test abs(cover_objective(AbsLog{2}(), a, A)) < 1e-10

        for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 0.5 0.01], [1.0 2.0 3.0; 4.0 5.0 6.0])
            a_fast, b_fast = cover(A)
            a_lmin, b_lmin = cover_min(AbsLog{1}(), A)
            a_qmin, b_qmin = cover_min(AbsLog{2}(), A)
            @test all(a_qmin[i] * b_qmin[j] >= abs(A[i, j]) - 1e-10 for i in axes(A, 1), j in axes(A, 2))
            @test cover_objective(AbsLog{2}(), a_qmin, b_qmin, A) <= cover_objective(AbsLog{2}(), a_fast, b_fast, A) + 1e-8
            @test cover_objective(AbsLog{2}(), a_qmin, b_qmin, A) <= cover_objective(AbsLog{2}(), a_lmin, b_lmin, A) + 1e-8
        end
        A = [0 0 0 1; 1 1 0 2; 1 0 2 1]
        a, b = cover_min(AbsLog{1}(), A)
        @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-10 for i in axes(A, 1), j in axes(A, 2))
        @test cover_objective(AbsLog{1}(), a, b, A) ≈ log(2)
        a, b = cover_min(AbsLog{2}(), A)
        @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-10 for i in axes(A, 1), j in axes(A, 2))
        @test cover_objective(AbsLog{2}(), a, b, A) ≈ 2*log(sqrt(2))^2

        # soft_symcover_min(AbsLog{2}): unconstrained, lower objective than constrained
        for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 1.0 0.01], [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0])
            a_soft = soft_symcover_min(AbsLog{2}(), A)
            a_hard = symcover_min(AbsLog{2}(), A)
            @test cover_objective(AbsLog{2}(), a_soft, A) <= cover_objective(AbsLog{2}(), a_hard, A) + 1e-8
        end
        # Rank-1 matrix: both achieve zero AbsLog{2} objective
        A_rank1 = [2.0 1.0 4.0; 1.0 0.5 2.0; 4.0 2.0 8.0]
        a_soft = soft_symcover_min(AbsLog{2}(), A_rank1)
        @test cover_objective(AbsLog{2}(), a_soft, A_rank1) < 1e-8
    end

    @testset "symcover_min native AbsLog{2}" begin
        # Non-square rejected.
        @test_throws ArgumentError symcover_min(AbsLog{2}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])

        # Native solver matches the HiGHS reference in objective across the whole
        # committed symmetric library, and returns a feasible cover.
        if !isdefined(@__MODULE__, :symmetric_matrices)
            include("testmatrices.jl")
        end
        for (_, A) in symmetric_matrices
            Af = Float64.(A)
            a  = symcover_min(AbsLog{2}(), Af)
            aj = ScaleInvariantAnalysis.symcover_min_jump(AbsLog{2}(), Af)
            @test all(a[i] * a[j] >= abs(Af[i, j]) - 1e-8 for i in axes(Af, 1), j in axes(Af, 2))
            oj = cover_objective(AbsLog{2}(), aj, Af)
            o  = cover_objective(AbsLog{2}(), a, Af)
            @test o <= oj * (1 + 1e-6) + 1e-10
        end

        # Scale-covariance: for a positive diagonal D, the optimal cover of D*A*D
        # is D times the optimal cover of A (up to the a·aᵀ gauge), so the product
        # a[i]*a[j] covaries as d[i]*d[j].
        rng = MersenneTwister(1234)
        for (_, A) in symmetric_matrices[1:30]
            Af = Float64.(A); n = size(Af, 1)
            d = exp.(2 .* randn(rng, n))
            AD = (d .* Af) .* d'
            a  = symcover_min(AbsLog{2}(), Af)
            aD = symcover_min(AbsLog{2}(), AD)
            for i in 1:n, j in 1:n
                (a[i] == 0 || a[j] == 0) && continue
                @test aD[i] * aD[j] ≈ d[i] * d[j] * a[i] * a[j] rtol=1e-6
            end
        end

        # Edge cases.
        @test symcover_min(AbsLog{2}(), reshape([4.0], 1, 1)) ≈ [2.0]           # n = 1
        # [0 1; 1 0]: a₁a₂ = 1 is the (gauge-invariant) optimum, objective 0.
        a = symcover_min(AbsLog{2}(), [0.0 1.0; 1.0 0.0])
        @test a[1] * a[2] ≈ 1.0
        @test cover_objective(AbsLog{2}(), a, [0.0 1.0; 1.0 0.0]) < 1e-12
        # A zero row/column leaves its scale at 0 while the rest is covered.
        Az = [1.0 0.0 2.0; 0.0 0.0 0.0; 2.0 0.0 3.0]
        a = symcover_min(AbsLog{2}(), Az)
        @test a[2] == 0.0
        @test all(a[i] * a[j] >= abs(Az[i, j]) - 1e-8 for i in 1:3, j in 1:3)
        # Scattered zeros with an exact rank-1 cover.
        A = [0 0 1; 0 0 2; 1 2 1]
        a = symcover_min(AbsLog{2}(), A)
        @test a ≈ [1, 2, 1]
        @test abs(cover_objective(AbsLog{2}(), a, A)) < 1e-8
        # κs keyword is accepted.
        @test symcover_min(AbsLog{2}(), [2.0 1.0; 1.0 3.0]; κs=(1e2, 1e4, 1e6, 1e8, 1e10)) isa Vector
    end

    @testset "cover_min native AbsLog{2}" begin
        if !isdefined(@__MODULE__, :general_matrices)
            include("testmatrices.jl")
        end
        # Native solver returns a feasible cover across the whole committed general
        # library, and matches the HiGHS reference in objective on a deterministic
        # subsample (the full 4367-matrix JuMP cross-check is slow).
        idx_sub = Set(round.(Int, range(1, length(general_matrices), length=500)))
        for (k, (_, A)) in enumerate(general_matrices)
            Af = Float64.(A)
            a, b = cover_min(AbsLog{2}(), Af)
            @test all(a[i] * b[j] >= abs(Af[i, j]) - 1e-7 for i in axes(Af, 1), j in axes(Af, 2))
            if k in idx_sub
                aj, bj = ScaleInvariantAnalysis.cover_min_jump(AbsLog{2}(), Af)
                oj = cover_objective(AbsLog{2}(), aj, bj, Af)
                o  = cover_objective(AbsLog{2}(), a, b, Af)
                @test o <= oj * (1 + 1e-6) + 1e-10
                # The balance convention is shared, so a, b agree entrywise with JuMP.
                @test a ≈ aj rtol=1e-5
                @test b ≈ bj rtol=1e-5
            end
        end

        # Scale-covariance under independent row/column scalings: covering D_r*A*D_c
        # scales the product a[i]*b[j] by d_r[i]*d_c[j].
        rng = MersenneTwister(1234)
        for (_, A) in general_matrices[1:30]
            Af = Float64.(A); m, n = size(Af)
            dr = exp.(2 .* randn(rng, m)); dc = exp.(2 .* randn(rng, n))
            AD = (dr .* Af) .* dc'
            a, b   = cover_min(AbsLog{2}(), Af)
            aD, bD = cover_min(AbsLog{2}(), AD)
            for i in 1:m, j in 1:n
                (a[i] == 0 || b[j] == 0) && continue
                @test aD[i] * bD[j] ≈ dr[i] * dc[j] * a[i] * b[j] rtol=1e-6
            end
        end

        # Non-square matrices, both orientations (transpose swaps the roles of a, b).
        A = [1.0 2.0 3.0; 4.0 5.0 6.0]
        a, b = cover_min(AbsLog{2}(), A)
        @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-8 for i in 1:2, j in 1:3)
        aT, bT = cover_min(AbsLog{2}(), permutedims(A))
        @test aT ≈ b rtol=1e-6
        @test bT ≈ a rtol=1e-6

        # Edge cases.
        a, b = cover_min(AbsLog{2}(), reshape([4.0], 1, 1))   # 1×1
        @test a[1] * b[1] ≈ 4.0
        # [0 1; 1 0]: bipartite support (singular signless Laplacian), covered by
        # the gauge term v0*v0ᵀ; a₁b₂ = a₂b₁ = 1 is optimal, objective 0.
        a, b = cover_min(AbsLog{2}(), [0.0 1.0; 1.0 0.0])
        @test a[1] * b[2] ≈ 1.0
        @test a[2] * b[1] ≈ 1.0
        @test cover_objective(AbsLog{2}(), a, b, [0.0 1.0; 1.0 0.0]) < 1e-12
        # An all-zero row and column leave their scales at 0 while the rest is covered.
        Az = [1.0 0.0 2.0; 0.0 0.0 0.0; 3.0 0.0 4.0]
        a, b = cover_min(AbsLog{2}(), Az)
        @test a[2] == 0.0
        @test b[2] == 0.0
        @test all(iszero(Az[i, j]) || a[i] * b[j] >= abs(Az[i, j]) - 1e-8 for i in 1:3, j in 1:3)
        # A matrix with a zero column, exact objective known from the AbsLog{2} optimum.
        A = [0 0 0 1; 1 1 0 2; 1 0 2 1]
        a, b = cover_min(AbsLog{2}(), A)
        @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-8 for i in axes(A, 1), j in axes(A, 2))
        @test cover_objective(AbsLog{2}(), a, b, A) ≈ 2 * log(sqrt(2))^2
        # κs keyword is accepted.
        @test cover_min(AbsLog{2}(), [1.0 2.0; 3.0 4.0]; κs=(1e2, 1e4, 1e6, 1e8, 1e10)) isa Tuple
    end

    @testset "MCM native AbsLog{2} matrix-free LSQR path" begin
        if !isdefined(@__MODULE__, :symmetric_matrices)
            include("testmatrices.jl")
        end
        # Invalid solver selection is rejected.
        @test_throws ArgumentError symcover_min(AbsLog{2}(), [2.0 1.0; 1.0 3.0]; linsolve=:qr)
        @test_throws ArgumentError cover_min(AbsLog{2}(), [2.0 1.0; 1.0 3.0]; linsolve=:qr)

        # The matrix-free LSQR path reproduces the dense path and the HiGHS reference
        # across the committed symmetric library, and returns a feasible cover.
        for (_, A) in symmetric_matrices
            Af = Float64.(A)
            a  = symcover_min(AbsLog{2}(), Af; linsolve=:lsqr)
            aj = ScaleInvariantAnalysis.symcover_min_jump(AbsLog{2}(), Af)
            @test all(a[i] * a[j] >= abs(Af[i, j]) - 1e-8 for i in axes(Af, 1), j in axes(Af, 2))
            @test cover_objective(AbsLog{2}(), a, Af) <=
                  cover_objective(AbsLog{2}(), aj, Af) * (1 + 1e-6) + 1e-10
        end

        # Asymmetric LSQR path: feasible and matching HiGHS on a deterministic
        # subsample of the committed general library.
        idx_sub = Set(round.(Int, range(1, length(general_matrices), length=200)))
        for (k, (_, A)) in enumerate(general_matrices)
            k in idx_sub || continue
            Af = Float64.(A)
            a, b = cover_min(AbsLog{2}(), Af; linsolve=:lsqr)
            @test all(a[i] * b[j] >= abs(Af[i, j]) - 1e-7 for i in axes(Af, 1), j in axes(Af, 2))
            aj, bj = ScaleInvariantAnalysis.cover_min_jump(AbsLog{2}(), Af)
            @test cover_objective(AbsLog{2}(), a, b, Af) <=
                  cover_objective(AbsLog{2}(), aj, bj, Af) * (1 + 1e-6) + 1e-10
        end

        # Gauge/edge cases the dense path handles via a ridge or v0*v0ᵀ must also work
        # matrix-free: bipartite support, a scalar, and scattered zeros.
        a = symcover_min(AbsLog{2}(), [0.0 1.0; 1.0 0.0]; linsolve=:lsqr)
        @test a[1] * a[2] ≈ 1.0
        @test symcover_min(AbsLog{2}(), reshape([4.0], 1, 1); linsolve=:lsqr) ≈ [2.0]
        Az = [1.0 0.0 2.0; 0.0 0.0 0.0; 2.0 0.0 3.0]
        a = symcover_min(AbsLog{2}(), Az; linsolve=:lsqr)
        @test a[2] == 0.0
        @test all(a[i] * a[j] >= abs(Az[i, j]) - 1e-8 for i in 1:3, j in 1:3)
        a, b = cover_min(AbsLog{2}(), [0.0 1.0; 1.0 0.0]; linsolve=:lsqr)
        @test a[1] * b[2] ≈ 1.0
        @test a[2] * b[1] ≈ 1.0
    end

    @testset "symcover_min and soft_symcover_min (JuMP/Ipopt, AbsLinear)" begin
        # non-square rejected
        @test_throws ArgumentError symcover_min(AbsLinear{2}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])
        @test_throws ArgumentError symcover_min(AbsLinear{1}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])
        @test_throws ArgumentError soft_symcover_min(AbsLinear{2}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])
        @test_throws ArgumentError soft_symcover_min(AbsLinear{1}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])

        for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 1.0 0.01], [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0])
            a_fast = symcover(AbsLinear{2}(), A)
            for ϕ in (AbsLinear{1}(), AbsLinear{2}())
                # symcover_min: valid hard cover, at most as costly as heuristic
                a_min = symcover_min(ϕ, A)
                @test all(a_min[i] * a_min[j] >= abs(A[i, j]) - 1e-6
                          for i in axes(A, 1), j in axes(A, 2))
                @test cover_objective(ϕ, a_min, A) <= cover_objective(ϕ, a_fast, A) + 1e-8

                # soft_symcover_min: lower or equal objective than constrained version
                a_soft = soft_symcover_min(ϕ, A)
                @test cover_objective(ϕ, a_soft, A) <= cover_objective(ϕ, a_min, A) + 1e-8
            end
            # AbsLinear{2} ≤ AbsLinear{1} soft objectives (p=2 is a stricter lower bound)
            a2 = soft_symcover_min(AbsLinear{2}(), A)
            a1 = soft_symcover_min(AbsLinear{1}(), A)
            @test cover_objective(AbsLinear{1}(), a1, A) <= cover_objective(AbsLinear{1}(), a2, A) + 1e-8
        end

        # Rank-1 matrix: soft cover achieves near-zero objective for all nonzero entries
        A_rank1 = [4.0 2.0; 2.0 1.0]
        a2 = soft_symcover_min(AbsLinear{2}(), A_rank1)
        @test cover_objective(AbsLinear{2}(), a2, A_rank1) < 1e-8
        a1 = soft_symcover_min(AbsLinear{1}(), A_rank1)
        @test cover_objective(AbsLinear{1}(), a1, A_rank1) < 1e-8

        # Matrix with zeros: zero entries contribute 1 each regardless of cover
        A = [0.0 1.0; 1.0 0.0]  # only off-diagonal nonzero; min possible = 0 (off-diag) + 2 (diag zeros)
        a2 = soft_symcover_min(AbsLinear{2}(), A)
        @test cover_objective(AbsLinear{2}(), a2, A) ≈ 2.0 atol=1e-8
        a1 = soft_symcover_min(AbsLinear{1}(), A)
        @test cover_objective(AbsLinear{1}(), a1, A) ≈ 2.0 atol=1e-8

        # soft_symcover_min matches soft_symcover on the analytical minimum (AbsLog{2} minimum)
        # (agreement to within solver tolerance)
        for A in ([2.0 1.0; 1.0 3.0], [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0])
            a_opt  = soft_symcover_min(AbsLinear{2}(), A)
            a_heur = soft_symcover(AbsLinear{2}(), A; iter=50)
            @test cover_objective(AbsLinear{2}(), a_opt, A) <=
                  cover_objective(AbsLinear{2}(), a_heur, A) + 1e-6
        end
    end

    @testset "SparseMatrixCSC" begin
        for Adense in ([2.0 1.0; 1.0 3.0], [1.0 -0.2; -0.2 0.0], [0.0 12.0 9.0; 12.0 7.0 12.0; 9.0 12.0 0.0],
                       [100.0 1.0; 1.0 0.01])
            for A in (sparse(Adense), Symmetric(sparse(tril(Adense)), :L), Symmetric(sparse(triu(Adense)), :U),
                      Hermitian(sparse(tril(Adense)), :L), Hermitian(sparse(triu(Adense)), :U))
                for ϕ in (AbsLog{2}(), AbsLinear{2}())
                    a = symcover(ϕ, A)
                    @test all(a[i] * a[j] >= abs(Adense[i, j]) - 1e-12 for i in axes(Adense, 1), j in axes(Adense, 2))
                end
                # Default dispatch
                a = symcover(A)
                @test all(a[i] * a[j] >= abs(Adense[i, j]) - 1e-12 for i in axes(Adense, 1), j in axes(Adense, 2))
                # cover_objective matches dense for AbsLog
                a = symcover(AbsLog{2}(), A)
                @test cover_objective(AbsLog{2}(), a, A) ≈ cover_objective(AbsLog{2}(), a, Adense)
            end
        end
        for Adense in ([2.0 1.0; 1.0 3.0], [0.0 1.0; -2.0 0.0], [1.0 2.0 3.0; 4.0 5.0 6.0])
            A = sparse(Adense)
            a, b = cover(A)
            @test all(a[i] * b[j] >= abs(Adense[i, j]) - 1e-12 for i in axes(Adense, 1), j in axes(Adense, 2))
        end
        # Zero-diagonal sparse matrix
        A0 = sparse([0.0 1.0; 1.0 0.0])
        @test symcover(A0) == [1.0, 1.0]
    end

    @testset "structured matrices" begin
        @testset "Diagonal" begin
            for dv in ([4.0, 9.0, 1.0], [4.0, 0.0, 1.0], [0.25, 100.0])
                D = Diagonal(dv)
                Ddense = Matrix(D)
                for ϕ in (AbsLog{2}(), AbsLinear{2}())
                    a = symcover(ϕ, D)
                    @test all(a[i] * a[j] >= abs(Ddense[i, j]) - 1e-12 for i in axes(Ddense, 1), j in axes(Ddense, 2))
                end
                a3, b3 = cover(D)
                @test all(a3[i] * b3[j] >= abs(Ddense[i, j]) - 1e-12 for i in axes(Ddense, 1), j in axes(Ddense, 2))
            end
        end

        @testset "PlusMinus1Banded" begin
            asym_cases = [
                Bidiagonal([3.0, 2.0, 1.0], [6.0, 0.5], :U),
                Bidiagonal([3.0, 2.0, 1.0], [6.0, 0.5], :L),
                Tridiagonal([1.0, 0.5], [3.0, 2.0, 1.0], [4.0, 0.5]),
            ]
            for A in asym_cases
                Adense = Matrix(A)
                n = size(A, 1)
                a, b = cover(A)
                @test all(a[i] * b[j] >= abs(Adense[i, j]) - 1e-12 for i in 1:n, j in 1:n)
            end
            sym_cases = [
                SymTridiagonal([4.0, 3.0, 1.0], [2.0, 0.5]),
                SymTridiagonal([0.0, 3.0, 0.0], [2.0, 0.5]),
                Tridiagonal([2.0, 0.5], [4.0, 3.0, 1.0], [2.0, 0.5]),
            ]
            for A in sym_cases
                Adense = Matrix(A)
                n = size(A, 1)
                a, b = cover(A)
                @test all(a[i] * b[j] >= abs(Adense[i, j]) - 1e-12 for i in 1:n, j in 1:n)
                for ϕ in (AbsLog{2}(), AbsLinear{2}())
                    a = symcover(ϕ, A)
                    @test all(a[i] * a[j] >= abs(Adense[i, j]) - 1e-12 for i in 1:n, j in 1:n)
                end
                # cover_objective matches dense for AbsLog{2}
                a = symcover(AbsLog{2}(), A)
                @test cover_objective(AbsLog{2}(), a, A) ≈ cover_objective(AbsLog{2}(), a, Adense)
            end
        end

        @testset "Adjoint and Transpose" begin
            for Adense in ([1.0 2.0; 3.0 4.0], [1.0 2.0 3.0; 4.0 5.0 6.0])
                for wrapper in (adjoint, transpose)
                    A = wrapper(Adense)
                    Adense_wrap = Matrix(A)
                    a, b = cover(A)
                    @test all(a[i] * b[j] >= abs(Adense_wrap[i, j]) - 1e-12
                              for i in axes(Adense_wrap, 1), j in axes(Adense_wrap, 2))
                    # cover_objective matches dense
                    @test cover_objective(AbsLog{2}(), a, b, A) ≈ cover_objective(AbsLog{2}(), a, b, Adense_wrap)
                    # Objectives are same as computing cover on parent and swapping
                    a0, b0 = cover(Adense)
                    @test cover_objective(AbsLog{2}(), a, b, A) ≈ cover_objective(AbsLog{2}(), a0, b0, Adense)
                end
            end
        end
    end

    @testset "native solvers on sparse and structured inputs" begin
        # The native AbsLog{2} MCM solvers (`symcover_min`/`cover_min`) and the AbsLinear
        # soft covers must agree with the dense reference on `Matrix(A)` when handed a
        # sparse-backed or structured input, and the hard MCM covers must stay feasible.
        # On a `SparseMatrixCSC`/`Symmetric`/`Hermitian`-sparse the MCM solvers default to
        # the matrix-free LSQR inner solve; structured inputs use the generic dense path.
        feasible_sym(a, M) = all(a[i] * a[j] >= abs(M[i, j]) - 1e-7 for i in axes(M, 1), j in axes(M, 2))
        feasible_asym(a, b, M) = all(a[i] * b[j] >= abs(M[i, j]) - 1e-7 for i in axes(M, 1), j in axes(M, 2))

        symdenses = [[2.0 1.0 0.0; 1.0 3.0 2.0; 0.0 2.0 5.0],
                     [4.0 0.0 1.0; 0.0 0.0 0.0; 1.0 0.0 2.0]]   # second has a zero row/column
        for M in symdenses
            ad = symcover_min(AbsLog{2}(), M)
            asd = soft_symcover(AbsLinear{2}(), M)
            for A in (sparse(M), Symmetric(sparse(triu(M)), :U), Symmetric(sparse(tril(M)), :L),
                      Hermitian(sparse(triu(M)), :U))
                a = symcover_min(AbsLog{2}(), A)
                @test feasible_sym(a, M)
                @test cover_objective(AbsLog{2}(), a, M) ≈ cover_objective(AbsLog{2}(), ad, M) rtol = 1e-7
                as = soft_symcover(AbsLinear{2}(), A)
                @test cover_objective(AbsLinear{2}(), as, M) ≈ cover_objective(AbsLinear{2}(), asd, M) rtol = 1e-7
                # Scale vectors are dense: return a plain Vector, matching cover/symcover.
                @test as isa Vector{Float64}
            end
        end
        @test symcover_min(AbsLog{2}(), sparse(symdenses[1])) isa Vector{Float64}
        # Every soft_symcover penalty returns a dense Vector on sparse-backed input.
        let Ssp = sparse(symdenses[1])
            for ϕ in (AbsLog{2}(), AbsLog{1}(), AbsLinear{2}(), AbsLinear{1}())
                @test soft_symcover(ϕ, Ssp) isa Vector{Float64}
                @test soft_symcover(ϕ, Symmetric(Ssp)) isa Vector{Float64}
            end
        end

        gendenses = [[1.0 2.0 0.0; 0.0 5.0 4.0; 3.0 0.0 6.0],
                     [2.0 0.0; 0.0 3.0]]   # second is diagonal: disconnected support
        for M in gendenses
            A = sparse(M)
            ad, bd = cover_min(AbsLog{2}(), M)
            a, b = cover_min(AbsLog{2}(), A)
            @test feasible_asym(a, b, M)
            @test cover_objective(AbsLog{2}(), a, b, M) ≈ cover_objective(AbsLog{2}(), ad, bd, M) rtol = 1e-7
            asd, bsd = soft_cover(AbsLinear{2}(), M)
            as, bs = soft_cover(AbsLinear{2}(), A)
            @test cover_objective(AbsLinear{2}(), as, bs, M) ≈ cover_objective(AbsLinear{2}(), asd, bsd, M) rtol = 1e-7
            @test as isa Vector{Float64} && bs isa Vector{Float64}
        end
        let (a, b) = cover_min(AbsLog{2}(), sparse(gendenses[1]))
            @test a isa Vector{Float64} && b isa Vector{Float64}
        end

        for D in (Diagonal([4.0, 9.0, 1.0]), Diagonal([4.0, 0.0, 1.0]))
            M = Matrix(D)
            a = symcover_min(AbsLog{2}(), D)
            @test feasible_sym(a, M)
            @test cover_objective(AbsLog{2}(), a, M) ≈
                  cover_objective(AbsLog{2}(), symcover_min(AbsLog{2}(), M), M) rtol = 1e-7
            a2, b2 = cover_min(AbsLog{2}(), D)   # disconnected support: exercises the gauge ridge
            @test feasible_asym(a2, b2, M)
            @test cover_objective(AbsLog{2}(), a2, b2, M) ≈ 0.0 atol = 1e-8   # diagonal is exactly coverable
        end
        for A in (SymTridiagonal([4.0, 3.0, 1.0], [2.0, 0.5]),
                  Tridiagonal([1.0, 0.5], [3.0, 2.0, 1.0], [4.0, 0.5]))
            M = Matrix(A)
            a2, b2 = cover_min(AbsLog{2}(), A)
            @test feasible_asym(a2, b2, M)
            @test cover_objective(AbsLog{2}(), a2, b2, M) ≈
                  cover_objective(AbsLog{2}(), cover_min(AbsLog{2}(), M)..., M) rtol = 1e-7
        end
    end

    @testset "MCM disconnected-support gauge" begin
        # A support graph that splits into k connected components carries k independent
        # (e; −e) gauges. The asymmetric dense normal equations pin only the global one
        # with v0*v0ᵀ; a minimal scale-relative ridge lifts the remaining k−1, so
        # `cover_min` no longer hits a SingularException on block-disconnected supports.
        # The dense (`:auto`) and matrix-free (`:lsqr`) paths must agree.
        singletons(vals) = Matrix(sparse(1:length(vals), 1:length(vals), float.(vals)))  # k singleton components
        block2(k) = cat(([2.0+i i; i 3.0+i] for i in 1:k)...; dims = (1, 2))              # k dense 2×2 components
        for M in (singletons([4.0, 9.0, 1.0]), singletons(1.0:6.0), block2(3), block2(6))
            ad, bd = cover_min(AbsLog{2}(), M)                    # :auto = dense + ridge
            al, bl = cover_min(AbsLog{2}(), M; linsolve = :lsqr)
            @test all(ad[i] * bd[j] >= abs(M[i, j]) - 1e-8 for i in axes(M, 1), j in axes(M, 2))
            @test all(al[i] * bl[j] >= abs(M[i, j]) - 1e-8 for i in axes(M, 1), j in axes(M, 2))
            @test cover_objective(AbsLog{2}(), ad, bd, M) ≈
                  cover_objective(AbsLog{2}(), al, bl, M) rtol = 1e-6 atol = 1e-8
        end
        # The canonical failure mode: `cover_min` on a Diagonal (n singleton components).
        D = Diagonal([4.0, 9.0, 1.0])
        a, b = cover_min(AbsLog{2}(), D)
        @test cover_objective(AbsLog{2}(), a, b, Matrix(D)) ≈ 0.0 atol = 1e-10
    end

    @testset "quality vs optimal (testmatrices)" begin
        if !isdefined(@__MODULE__, :symmetric_matrices) || !isdefined(@__MODULE__, :general_matrices)
            include("testmatrices.jl")
        end

        sym_ratios = Float64[]
        for (_, A) in symmetric_matrices
            Af = Float64.(A)
            # Initialization should give a valid cover
            a0 = symcover(AbsLog{2}(), Af; iter=0)
            @test all(a0[i] * a0[j] >= abs(Af[i, j]) - 1e-12 for i in axes(Af, 1), j in axes(Af, 2))
            a0 = symcover(AbsLog{2}(), Af / 100; iter=0)
            @test all(a0[i] * a0[j] >= abs(Af[i, j])/100 - 1e-12 for i in axes(Af, 1), j in axes(Af, 2))
            # Covers are nearly quadratically optimal
            qopt  = cover_objective(AbsLog{2}(), symcover_min(AbsLog{2}(), Af), Af)
            qfast = cover_objective(AbsLog{2}(), symcover(AbsLog{2}(), Af; iter=10), Af)
            iszero(qopt) || push!(sym_ratios, qfast / qopt)
        end
        @test median(sym_ratios) < 1.02

        gen_ratios = Float64[]
        for (_, A) in general_matrices
            Af = Float64.(A)
            a0, b0 = cover(AbsLog{2}(), Af; iter=0)
            @test all(a0[i] * b0[j] >= abs(Af[i, j]) - 1e-12 for i in axes(Af, 1), j in axes(Af, 2))
            a0, b0 = cover(AbsLog{2}(), Af / 100; iter=0)
            @test all(a0[i] * b0[j] >= abs(Af[i, j])/100 - 1e-12 for i in axes(Af, 1), j in axes(Af, 2))
            qopt  = cover_objective(AbsLog{2}(), cover_min(AbsLog{2}(), Af)..., Af)
            qfast = cover_objective(AbsLog{2}(), cover(AbsLog{2}(), Af; iter=10)..., Af)
            iszero(qopt) || push!(gen_ratios, qfast / qopt)
        end
        @test median(gen_ratios) < 1.02
    end

end
