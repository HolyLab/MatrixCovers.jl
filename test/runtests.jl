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
