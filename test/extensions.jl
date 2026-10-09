# JuMP/HiGHS and JuMP/Ipopt extension solvers, and the missing-extension error hints.

@testset "symcover_min and cover_min (JuMP/HiGHS)" begin
    for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 1.0 0.01], [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0])
        a_fast  = symcover(A)
        a_lmin, _ = symcover_min(AbsLog{1}(), A)
        a_qmin, _ = symcover_min(AbsLog{2}(), A)
        # qmin is a valid cover
        @test iscover(a_qmin, A; atol=1e-10)
        # qmin achieves lower or equal AbsLog{2} objective than symcover and lmin
        # (up to the near-exact solver's penalty-continuation tolerance).
        @test cover_objective(AbsLog{2}(), a_qmin, A) <= cover_objective(AbsLog{2}(), a_fast, A) * (1 + 1e-6) + 1e-10
        @test cover_objective(AbsLog{2}(), a_qmin, A) <= cover_objective(AbsLog{2}(), a_lmin, A) * (1 + 1e-6) + 1e-10
    end
    # Exact case with zeros
    A = [0 0 1; 0 0 2; 1 2 1]
    a, _ = symcover_min(AbsLog{1}(), A)
    @test iscover(a, A; atol=1e-10)
    @test a ≈ [1, 2, 1]
    @test abs(cover_objective(AbsLog{1}(), a, A)) < 1e-10
    a, _ = symcover_min(AbsLog{2}(), A)
    @test iscover(a, A; atol=1e-10)
    @test a ≈ [1, 2, 1]
    @test abs(cover_objective(AbsLog{2}(), a, A)) < 1e-10

    for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 0.5 0.01], [1.0 2.0 3.0; 4.0 5.0 6.0])
        a_fast, b_fast = cover(A)
        a_lmin, b_lmin = cover_min(AbsLog{1}(), A)
        a_qmin, b_qmin = cover_min(AbsLog{2}(), A)
        @test iscover(a_qmin, b_qmin, A; atol=1e-10)
        @test cover_objective(AbsLog{2}(), a_qmin, b_qmin, A) <= cover_objective(AbsLog{2}(), a_fast, b_fast, A) + 1e-8
        @test cover_objective(AbsLog{2}(), a_qmin, b_qmin, A) <= cover_objective(AbsLog{2}(), a_lmin, b_lmin, A) + 1e-8
    end
    A = [0 0 0 1; 1 1 0 2; 1 0 2 1]
    a, b = cover_min(AbsLog{1}(), A)
    @test iscover(a, b, A; atol=1e-10)
    @test cover_objective(AbsLog{1}(), a, b, A) ≈ log(2)
    a, b = cover_min(AbsLog{2}(), A)
    @test iscover(a, b, A; atol=1e-10)
    @test cover_objective(AbsLog{2}(), a, b, A) ≈ 2*log(sqrt(2))^2

    # soft_symcover(AbsLog{p}): unconstrained, lower objective than constrained
    for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 1.0 0.01], [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0])
        for ϕ in (AbsLog{1}(), AbsLog{2}())
            a_soft, _ = soft_symcover(ϕ, A)
            a_hard, _ = symcover_min(ϕ, A)
            @test cover_objective(ϕ, a_soft, A) <= cover_objective(ϕ, a_hard, A) + 1e-8
        end
    end
    # Rank-1 matrix: both achieve zero objective
    A_rank1 = [2.0 1.0 4.0; 1.0 0.5 2.0; 4.0 2.0 8.0]
    @test cover_objective(AbsLog{2}(), first(soft_symcover(AbsLog{2}(), A_rank1)), A_rank1) < 1e-8
    @test cover_objective(AbsLog{1}(), first(soft_symcover(AbsLog{1}(), A_rank1)), A_rank1) < 1e-6

    # Non-optimal solver statuses are errors.
    @test MatrixCovers.check_solved(JuMP.OPTIMAL, "HiGHS", "symcover_min") === nothing
    @test MatrixCovers.check_solved(JuMP.LOCALLY_SOLVED, "Ipopt", "symcover_min!") === nothing
    @test_throws "terminated with status" MatrixCovers.check_solved(JuMP.DUAL_INFEASIBLE, "HiGHS", "symcover_min")
    @test_throws "symcover_min" MatrixCovers.check_solved(JuMP.DUAL_INFEASIBLE, "HiGHS", "symcover_min")
    @test_throws "HiGHS" MatrixCovers.check_solved(JuMP.INFEASIBLE, "HiGHS", "symcover_min")
    # A tolerance the caller did not ask for is not a solve.
    @test_throws "terminated with status" MatrixCovers.check_solved(JuMP.ALMOST_LOCALLY_SOLVED, "Ipopt", "symcover_min!")
end

@testset "symcover_min and soft_symcover (JuMP/Ipopt, AbsLinear)" begin
    # non-square rejected
    @test_throws "symcover_min requires a square matrix" symcover_min(AbsLinear{2}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])
    @test_throws "symcover_min requires a square matrix" symcover_min(AbsLinear{1}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])
    @test_throws "soft_symcover requires a square matrix" soft_symcover(AbsLinear{2}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])
    @test_throws "soft_symcover requires a square matrix" soft_symcover(AbsLinear{1}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])
    @test_throws "soft_symcover requires a square matrix" soft_symcover(AbsLog{1}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])

    for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 1.0 0.01], [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0])
        a_fast = symcover(AbsLinear{2}(), A)
        for ϕ in (AbsLinear{1}(), AbsLinear{2}())
            # symcover_min: valid hard cover, at most as costly as heuristic
            a_min, _ = symcover_min(ϕ, A)
            @test iscover(a_min, A; atol=1e-6)
            @test cover_objective(ϕ, a_min, A) <= cover_objective(ϕ, a_fast, A) + 1e-8

            # soft_symcover: lower or equal objective than constrained version
            a_soft, _ = soft_symcover(ϕ, A)
            @test cover_objective(ϕ, a_soft, A) <= cover_objective(ϕ, a_min, A) + 1e-8
        end
        # AbsLinear{2} ≤ AbsLinear{1} soft objectives (p=2 is a stricter lower bound)
        a2, _ = soft_symcover(AbsLinear{2}(), A)
        a1, _ = soft_symcover(AbsLinear{1}(), A)
        @test cover_objective(AbsLinear{1}(), a1, A) <= cover_objective(AbsLinear{1}(), a2, A) + 1e-8
    end

    # Rank-1 matrix: soft cover achieves near-zero objective for all nonzero entries
    A_rank1 = [4.0 2.0; 2.0 1.0]
    a2, _ = soft_symcover(AbsLinear{2}(), A_rank1)
    @test cover_objective(AbsLinear{2}(), a2, A_rank1) < 1e-8
    a1, _ = soft_symcover(AbsLinear{1}(), A_rank1)
    @test cover_objective(AbsLinear{1}(), a1, A_rank1) < 1e-8

    # Matrix with zeros: zero entries contribute 1 each regardless of cover
    A = [0.0 1.0; 1.0 0.0]  # only off-diagonal nonzero; min possible = 0 (off-diag) + 2 (diag zeros)
    a2, _ = soft_symcover(AbsLinear{2}(), A)
    @test cover_objective(AbsLinear{2}(), a2, A) ≈ 2.0 atol=1e-8
    a1, _ = soft_symcover(AbsLinear{1}(), A)
    @test cover_objective(AbsLinear{1}(), a1, A) ≈ 2.0 atol=1e-8
end

@testset "AbsLinear{2} soft refinement converges from dispersed starts" begin
    # Refinements from widely dispersed starts must reach a local minimum, not stop on the
    # flat part of the (1-r)^2 objective: sorted objectives either agree to 1e-8 (the same
    # minimum) or differ by more than 1e-3 (distinct minima; `Agen` has two).
    ϕ = AbsLinear{2}()
    rng = StableRNG(7)
    L = randn(rng, 5, 5)
    Asym = exp.((L .+ L') ./ 2)
    Agen = exp.(randn(rng, 5, 5))
    rng = StableRNG(1)
    for (A, sym) in ((Asym, true), (Agen, false))
        es = sort(map(1:12) do _
            a = exp.(3 .* randn(rng, 5))
            b = exp.(3 .* randn(rng, 5))
            sym ? cover_objective(ϕ, first(soft_symcover!(ϕ, a, A)), A) :
                  cover_objective(ϕ, soft_cover!(ϕ, a, b, A)[1:2]..., A)
        end)
        gaps = diff(es) ./ es[begin:end-1]
        @test all(g -> g < 1e-8 || g > 1e-3, gaps)
    end
end

@testset "AbsLinear refinement stopping on a plateau throws" begin
    # A start with one scale far too large puts every ratio in that row/column near 0, where
    # the |1-r|^p objective is flat; Ipopt reports the start region as solved. For p = 1,
    # Ipopt may instead fail numerically as the scales diverge.
    A = ones(3, 3)
    plateau = r"Ipopt stopped on a plateau: every supported entry in (row|column) 3"
    for (ϕ, msg) in ((AbsLinear{1}(), Regex(plateau.pattern * "|Ipopt terminated with status")),
                     (AbsLinear{2}(), plateau))
        @test_throws msg soft_cover!(ϕ, ones(3), [1.0, 1.0, 1e40], A)
        @test_throws msg soft_symcover!(ϕ, [1.0, 1.0, 1e40], A)
        @test_throws msg cover_min!(ϕ, ones(3), [1.0, 1.0, 1e40], A)
        @test_throws msg symcover_min!(ϕ, [1.0, 1.0, 1e40], A)
        @test_throws MatrixCovers.SolverFailure soft_cover!(ϕ, ones(3), [1.0, 1.0, 1e40], A)
    end
end

@testset "symcover_min!/cover_min! refiners (JuMP/HiGHS/Ipopt)" begin
    A = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]
    Aasym = [1.0 2.0 3.0; 4.0 5.0 6.0]

    for ϕ in PENALTIES
        # Refinement does not worsen the start beyond solver tolerance.
        a0 = initialize_symcover(A)
        a, _ = symcover_min!(ϕ, copy(a0), A)
        @test iscover(a, A; rtol=1e-6)
        @test cover_objective(ϕ, a, A) <= cover_objective(ϕ, a0, A) * (1 + 1e-6) + 1e-8

        ab0, bb0 = initialize_cover(Aasym)
        ab, bb, _ = cover_min!(ϕ, copy(ab0), copy(bb0), Aasym)
        @test iscover(ab, bb, Aasym; rtol=1e-6)
        @test cover_objective(ϕ, ab, bb, Aasym) <= cover_objective(ϕ, ab0, bb0, Aasym) * (1 + 1e-6) + 1e-8

        # The asymmetric result is pinned to the balance convention, so the start is
        # read only up to the gauge a -> c*a, b -> b/c that leaves a[i]*b[j] fixed.
        ga, gb, _ = cover_min!(ϕ, 4 .* ab0, bb0 ./ 4, Aasym)
        @test ga ≈ ab && gb ≈ bb
    end

    # `AbsLog{2}` tie-breaking makes the `AbsLog{1}` result start-independent.
    a_cold, _ = symcover_min(AbsLog{1}(), A)
    for strategy in (:hardcover, :geomean, :diagfeasible)
        @test first(symcover_min!(AbsLog{1}(), initialize_symcover(A; strategy), A)) ≈ a_cold
    end
    ab_cold, bb_cold = cover_min(AbsLog{1}(), Aasym)
    for strategy in (:hardcover, :geomean)
        ah, bh = cover_min!(AbsLog{1}(), initialize_cover(Aasym; strategy)..., Aasym)
        @test ah ≈ ab_cold && bh ≈ bb_cold
    end

    # Different starts reach different `AbsLinear` local minima.
    Abasin = [81.892035218799 1.06622031288736 29.4700945830419 0.0181293142917846;
              1.06622031288736 0.243512973596586 38.0236584552296 0.0279078887878805;
              29.4700945830419 38.0236584552296 8.96405068596511 26.5775238859338;
              0.0181293142917846 0.0279078887878805 26.5775238859338 42.6650094474717]
    ϕ = AbsLinear{2}()
    a_hard, _ = symcover_min!(ϕ, initialize_symcover(Abasin; strategy=:hardcover), Abasin)
    a_geo, _ = symcover_min!(ϕ, initialize_symcover(Abasin; strategy=:geomean), Abasin)
    @test iscover(a_hard, Abasin; rtol=1e-6)
    @test iscover(a_geo, Abasin; rtol=1e-6)
    @test cover_objective(ϕ, a_geo, Abasin) < cover_objective(ϕ, a_hard, Abasin) - 0.5

    # Offset axes survive the Ipopt position mapping.
    Ao = OffsetArray(A, -1, -1)
    ao, _ = symcover_min!(ϕ, initialize_symcover(Ao), Ao)
    @test axes(ao, 1) == axes(Ao, 1)
    @test collect(ao) ≈ first(symcover_min!(ϕ, initialize_symcover(A), A))
end

@testset "soft_symcover multistart and refiner (Ipopt)" begin
    # Soft refiners accept non-covering starts.
    A = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]
    a0 = initialize_symcover(A; strategy=:geomean, feasible=:none)
    @test !iscover(a0, A)
    @test_throws "requires a start that covers `A`" symcover_min!(AbsLinear{2}(), copy(a0), A)
    @test first(soft_symcover!(AbsLinear{2}(), copy(a0), A)) isa AbstractVector

    for ϕ in (AbsLinear{1}(), AbsLinear{2}(), AbsLog{1}())
        @test first(soft_symcover(ϕ, A)) == first(soft_symcover(ϕ, A))           # deterministic
    end

    # No start on the menu beats the driver.
    for ϕ in (AbsLinear{1}(), AbsLinear{2}())
        a, _ = soft_symcover(ϕ, A)
        for strategy in (:hardcover, :geomean, :leaveout)
            single, _ = soft_symcover!(ϕ, initialize_symcover(A; strategy, feasible=:none), A)
            @test cover_objective(ϕ, a, A) <= cover_objective(ϕ, single, A) * (1 + 1e-6) + 1e-8
        end
    end

    # Covariant starts must select corresponding nonconvex basins after rescaling.
    Abasins = [0.020358630644342735 0.53352144014074843 5.8899714528796077 0.23770314779348869 3.0721768720180109;
              0.53352144014074843 3.5416395788642903 37.199280652497748 49.972622569225109 333.53567816710364;
              5.8899714528796077 37.199280652497748 0.76014027958709862 2.4189759139690739 0.92571970793600067;
              0.23770314779348869 49.972622569225109 2.4189759139690739 8.1401049198051822 7.3601096491681046;
              3.0721768720180109 333.53567816710364 0.92571970793600067 7.3601096491681046 1.0844635712556003]
    d = [20.451338935482074, 0.69212569803171398, 35.401627522529395, 15.904906661932396, 0.19696509774727827]
    for ϕ in (AbsLinear{1}(), AbsLinear{2}())
        @test covaries(A -> first(soft_symcover(ϕ, A)), Abasins, d; rtol=1e-5)
    end

    @test_throws "positive scale on every supported row" soft_symcover!(AbsLinear{2}(), [1.0, -1.0, 2.0], A)
    @test_throws "soft_symcover! requires a square matrix" soft_symcover!(AbsLinear{2}(), [1.0, 2.0], [1.0 2.0 3.0; 4.0 5.0 6.0])
end

@testset "soft_cover multistart and refiner (Ipopt)" begin
    A = [1.0 2.0 3.0; 40.0 5.0 0.6]
    nza = vec(count(!iszero, A, dims=2))
    nzb = vec(count(!iszero, A, dims=1))
    balance(a, b) = sum(nza .* log.(a)) - sum(nzb .* log.(b))

    # The soft start need not cover `A`, so cover_min! rejects what soft_cover! accepts.
    a0, b0 = initialize_cover(A; strategy=:geomean, feasible=:none)
    @test !iscover(a0, b0, A)
    @test_throws "requires a start that covers `A`" cover_min!(AbsLinear{2}(), copy(a0), copy(b0), A)

    for ϕ in (AbsLinear{1}(), AbsLinear{2}(), AbsLog{1}(), AbsLog{2}())
        a, b = soft_cover(ϕ, A)
        @test soft_cover(ϕ, A)[1:2] == (a, b)                             # deterministic
        # The objective depends on `a`, `b` only through their products, so the split is
        # fixed by the balance convention rather than left to the solver.
        @test balance(a, b) ≈ 0 atol=1e-7
        # Gauge-invariant in the start: (a, b) and (2a, b/2) name the same point.
        ag, bg, _ = soft_cover!(ϕ, 2 .* copy(a0), copy(b0) ./ 2, A)
        ah, bh, _ = soft_cover!(ϕ, copy(a0), copy(b0), A)
        @test ag ≈ ah && bg ≈ bh
    end

    # No start on the menu beats the driver.
    for ϕ in (AbsLinear{1}(), AbsLinear{2}())
        a, b, _ = soft_cover(ϕ, A)
        for strategy in (:hardcover, :covariant)
            sa, sb, _ = soft_cover!(ϕ, initialize_cover(A; strategy, feasible=:none)..., A)
            @test cover_objective(ϕ, a, b, A) <= cover_objective(ϕ, sa, sb, A) * (1 + 1e-6) + 1e-8
        end

        # The asymmetric soft cover relaxes the symmetric one on a symmetric matrix.
        As = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]
        aa, ab, _ = soft_cover(ϕ, As)
        @test cover_objective(ϕ, aa, ab, As) <=
              cover_objective(ϕ, first(soft_symcover(ϕ, As)), As) * (1 + 1e-6) + 1e-8
    end

    # Covariant starts make the selected cover covariant on irregular support too.
    rng = StableRNG(3)
    for _ in 1:20
        B = exp.(2 .* randn(rng, 5, 6)) .* (rand(rng, 5, 6) .> 0.4)
        dr = exp.(3 .* randn(rng, 5))
        dc = exp.(3 .* randn(rng, 6))
        (any(iszero, sum(B; dims=1)) || any(iszero, sum(B; dims=2))) && continue
        @test covaries(B -> soft_cover(AbsLinear{2}(), B), B, dr, dc; rtol=1e-5)
        @test covaries(B -> cover_min(AbsLinear{2}(), B), B, dr, dc; rtol=1e-5)
    end

    @test_throws "positive scale on every supported row" soft_cover!(AbsLinear{2}(), [1.0, 0.0], [1.0, 1.0, 1.0], A)
    @test_throws "positive scale on every supported column" soft_cover!(AbsLinear{2}(), [1.0, 1.0], [1.0, -1.0, 1.0], A)
end

@testset "AbsLog{1} canonical selection on the optimal face (HiGHS)" begin
    # Select the `AbsLog{2}`-minimal member of the `AbsLog{1}` optimal face.
    ϕ1, ϕ2 = AbsLog{1}(), AbsLog{2}()
    rng = StableRNG(17)
    for n in (3, 5, 8)
        B = exp.(3 .* randn(rng, n, n)) .* (rand(rng, n, n) .> 0.25)
        A = (B + B') / 2
        all(iszero, A) && continue
        a, _ = symcover_min(ϕ1, A)
        @test iscover(a, A; rtol=1e-7)

        # No cover beats it on AbsLog{1}: the canonical selection does not cost optimality.
        # Any feasible cover is a witness; use the AbsLog{2}-minimal one, and the heuristic.
        for other in (first(symcover_min(ϕ2, A)), symcover(A))
            @test cover_objective(ϕ1, a, A) <= cover_objective(ϕ1, other, A) * (1 + 1e-6) + 1e-8
        end

        # Start-independent in the *result*, not merely in the objective value.
        for strategy in (:hardcover, :geomean, :diagfeasible)
            @test first(symcover_min!(ϕ1, initialize_symcover(A; strategy), A)) ≈ a
        end

        # Scale-covariant: both objectives see `A` only through the residuals, which a
        # rescaling leaves invariant, so the selected member co-varies with the frame.
        D = Diagonal(exp.(randn(rng, n)))
        @test first(symcover_min(ϕ1, D * A * D)) ≈ D * a
    end

    # Gauge balancing and `AbsLog{1}` face selection are independent.
    Aasym = [3.0 1.0 7.0; 2.0 5.0 1.0; 8.0 1.0 4.0]
    a, b, _ = cover_min(ϕ1, Aasym)
    nza = vec(count(!iszero, Aasym, dims=2))
    nzb = vec(count(!iszero, Aasym, dims=1))
    @test sum(nza .* log.(a)) ≈ sum(nzb .* log.(b)) atol=1e-8
    @test iscover(a, b, Aasym; rtol=1e-7)
    # Gauge-invariant in the start, and start-independent in the result.
    a0, b0 = initialize_cover(Aasym; strategy=:geomean)
    ag, bg, _ = cover_min!(ϕ1, 4 .* a0, b0 ./ 4, Aasym)
    @test ag ≈ a && bg ≈ b
end

@testset "AbsLinear multistart drivers (Ipopt)" begin
    # `:geomean` reaches the better `AbsLinear{2}` basin here.
    Abasin = [81.892035218799 1.06622031288736 29.4700945830419 0.0181293142917846;
              1.06622031288736 0.243512973596586 38.0236584552296 0.0279078887878805;
              29.4700945830419 38.0236584552296 8.96405068596511 26.5775238859338;
              0.0181293142917846 0.0279078887878805 26.5775238859338 42.6650094474717]
    Aasym = [1.0 2.0 3.0; 40.0 5.0 0.6]

    for ϕ in (AbsLinear{1}(), AbsLinear{2}())
        a, _ = symcover_min(ϕ, Abasin)
        @test iscover(a, Abasin; rtol=1e-6)
        # No start on the menu beats the driver.
        for strategy in (:hardcover, :geomean, :leaveout)
            single, _ = symcover_min!(ϕ, initialize_symcover(Abasin; strategy), Abasin)
            @test cover_objective(ϕ, a, Abasin) <=
                  cover_objective(ϕ, single, Abasin) * (1 + 1e-6) + 1e-8
        end
        @test first(symcover_min(ϕ, Abasin)) == a   # deterministic

        ab, bb, _ = cover_min(ϕ, Aasym)
        @test iscover(ab, bb, Aasym; rtol=1e-6)
        for strategy in (:hardcover, :covariant)
            sa, sb, _ = cover_min!(ϕ, initialize_cover(Aasym; strategy)..., Aasym)
            @test cover_objective(ϕ, ab, bb, Aasym) <=
                  cover_objective(ϕ, sa, sb, Aasym) * (1 + 1e-6) + 1e-8
        end
        @test cover_min(ϕ, Aasym)[1:2] == (ab, bb)   # deterministic

        # The asymmetric cover relaxes the symmetric one: independent row and column
        # scales can only do better on the same matrix.
        asym_a, asym_b, _ = cover_min(ϕ, Abasin)
        @test cover_objective(ϕ, asym_a, asym_b, Abasin) <=
              cover_objective(ϕ, first(symcover_min(ϕ, Abasin)), Abasin) * (1 + 1e-6) + 1e-8
    end

    # Restricting `strategies` changes the selected basin.
    ϕ = AbsLinear{2}()
    @test first(symcover_min(ϕ, Abasin; strategies=(:hardcover,))) ≈
          first(symcover_min!(ϕ, initialize_symcover(Abasin; strategy=:hardcover), Abasin))
    @test cover_objective(ϕ, first(symcover_min(ϕ, Abasin)), Abasin) <
          cover_objective(ϕ, first(symcover_min(ϕ, Abasin; strategies=(:hardcover,))), Abasin) - 0.5

    # A matrix whose every row carries a single support entry admits no :leaveout start;
    # that strategy forfeits its slot rather than failing the solve.
    Adiag = [4.0 0.0; 0.0 9.0]
    @test first(symcover_min(ϕ, Adiag)) ≈ [2.0, 3.0] rtol=1e-4

    @test_throws "unknown strategy :banana" symcover_min(ϕ, Abasin; strategies=(:banana,))
    @test_throws "no strategy in (:leaveout,)" symcover_min(ϕ, Adiag; strategies=(:leaveout,))
    @test_throws "has no asymmetric formulation" cover_min(ϕ, Aasym; strategies=(:leaveout,))
    # An empty menu is reported as such by both drivers, rather than as the
    # unrelated "no strategy yields a start" that a fall-through would produce.
    @test_throws "at least one starting cover" cover_min(ϕ, Aasym; strategies=())
    @test_throws "at least one starting cover" symcover_min(ϕ, Abasin; strategies=())
    @test_throws "symcover_min:" symcover_min(ϕ, Abasin; strategies=())
end

@testset "error hint gated on argument types" begin
    # Do not hint when loading an extension cannot fix the argument type.
    A = [4.0 1.0; 1.0 4.0]
    e = try
        first(symcover_min(AbsLog{2}(), "not a matrix"))
        nothing
    catch err
        err
    end
    @test e isa MethodError
    @test !occursin("loading JuMP", sprint(showerror, e))

    # Test missing-extension hints in a fresh process.
    script = """
    using MatrixCovers
    A = [4.0 1.0; 1.0 4.0]
    try
        symcover_min(AbsLog{1}(), A)
    catch e
        print(sprint(showerror, e))
    end
    """
    out = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`, String)
    @test occursin("loading JuMP", out)

    # A driver should preserve the inner missing-extension hint.
    script_driver = """
    using MatrixCovers
    A = [4.0 1.0; 1.0 4.0]
    for call in (() -> soft_symcover(AbsLinear{2}(), A), () -> soft_cover(AbsLog{1}(), A))
        try
            call()
        catch e
            println(sprint(showerror, e))
        end
    end
    """
    out_driver = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script_driver`, String)
    @test count("loading JuMP", out_driver) == 2


    # Every penalty cover_min does not solve natively lives in an extension, whether the
    # call reaches it directly (AbsLog{1}) or through the AbsLinear multistart driver,
    # whose MethodError is raised on the cover_min! kernel it calls.
    script_cover = """
    using MatrixCovers
    A = [4.0 1.0; 1.0 4.0]
    for call in (() -> cover_min(AbsLog{1}(), A), () -> cover_min(AbsLinear{2}(), A))
        try
            call()
        catch e
            print(sprint(showerror, e))
        end
    end
    """
    out_cover = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script_cover`, String)
    @test count("loading JuMP", out_cover) == 2

    # The refiners are themselves the extension's entry points, so a caller who reaches
    # for one directly must be advised of the load just as the drivers' callers are.
    script_bang = """
    using MatrixCovers
    A = [4.0 1.0; 1.0 4.0]
    for call in (() -> symcover_min!(AbsLinear{2}(), [2.0, 2.0], A),
                 () -> cover_min!(AbsLinear{2}(), [2.0, 2.0], [2.0, 2.0], A))
        try
            call()
        catch e
            print(sprint(showerror, e))
        end
    end
    """
    out_bang = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script_bang`, String)
    @test count("loading JuMP", out_bang) == 2

    # The widened gate still refuses to fire when no package load could help.
    e5 = try
        first(symcover_min!(AbsLog{2}(), [1.0, 1.0], "not a matrix"))
        nothing
    catch err
        err
    end
    @test e5 isa MethodError
    @test !occursin("loading JuMP", sprint(showerror, e5))
end

# `HookOnlyMatrix` verifies that model builders use support hooks.
@testset "the solver entry points read through the support hook" begin
    entries = [(1, 1, 2.0), (1, 3, 1.5), (2, 2, 3.0), (2, 4, 0.5), (3, 4, 4.0), (4, 4, 1.0)]
    M = HookOnlyMatrix(entries, 4)
    dense = zeros(4, 4)
    for (i, j, v) in entries
        dense[i, j] = dense[j, i] = v
    end

    @test first(symcover_min(AbsLog{2}(), M)) ≈ first(symcover_min(AbsLog{2}(), dense)) rtol=1e-8
    @test MatrixCovers.symcover_min_jump(AbsLog{2}(), M) ≈
          MatrixCovers.symcover_min_jump(AbsLog{2}(), dense) rtol=1e-6
    @test first(symcover_min(AbsLog{1}(), M)) ≈ first(symcover_min(AbsLog{1}(), dense)) rtol=1e-6
    @test iscover(first(symcover_min(AbsLog{1}(), M)), dense; rtol=1e-8)

    for (fh, fd) in ((cover_min(AbsLog{1}(), M), cover_min(AbsLog{1}(), dense)),
                     (MatrixCovers.cover_min_jump(AbsLog{2}(), M),
                      MatrixCovers.cover_min_jump(AbsLog{2}(), dense)))
        @test fh[1] ≈ fd[1] rtol=1e-6
        @test fh[2] ≈ fd[2] rtol=1e-6
    end
    for (fh, fd) in ((scales(cover_transversal(M)), scales(cover_transversal(dense))),
                     (MatrixCovers.cover_transversal_jump(M)[1],
                      MatrixCovers.cover_transversal_jump(dense)[1]))
        @test fh[1] ≈ fd[1] rtol=1e-6
        @test fh[2] ≈ fd[2] rtol=1e-6
    end

    # Ipopt: the AbsLinear kernels, whose starts the caller supplies.
    for p in (1, 2)
        ϕ = AbsLinear{p}()
        sh, _ = symcover_min!(ϕ, symcover(ϕ, M), M)
        sd, _ = symcover_min!(ϕ, symcover(ϕ, dense), dense)
        @test sh ≈ sd rtol=1e-5

        ah, bh, _ = cover_min!(ϕ, cover(ϕ, M)..., M)
        ad, bd, _ = cover_min!(ϕ, cover(ϕ, dense)..., dense)
        @test ah ≈ ad rtol=1e-5
        @test bh ≈ bd rtol=1e-5

        @test first(soft_symcover!(ϕ, first(soft_symcover(ϕ, M)), M)) ≈
              first(soft_symcover!(ϕ, first(soft_symcover(ϕ, dense)), dense)) rtol=1e-5

        sah, sbh, _ = soft_cover!(ϕ, soft_cover(ϕ, M)[1:2]..., M)
        sad, sbd, _ = soft_cover!(ϕ, soft_cover(ϕ, dense)[1:2]..., dense)
        @test sah ≈ sad rtol=1e-5
        @test sbh ≈ sbd rtol=1e-5
    end
end

# This matrix distinguishes full-grid symmetric weighting from triangle weighting.
@testset "cover_min balances a block-diagonal support (JuMP/HiGHS)" begin
    # Two connected components: the model constraint pins only the global gauge
    # direction, so the per-component balance must come from the post-solve
    # `_balance_cover!` shift `cover_min!` applies after the JuMP solve.
    B = [2.0 1.0; 0.5 3.0]
    C = [1.5 4.0 2.0; 3.0 0.5 1.0]
    A = [B zeros(2, 3); zeros(2, 2) C]
    a, b = cover_min(AbsLog{1}(), A)
    @test iscover(a, b, A; atol=1e-8)
    @test isbalanced(a, b, A)
end

@testset "cover_min balances a block-diagonal support (Ipopt)" begin
    B = [2.0 1.0; 0.5 3.0]
    C = [1.5 4.0 2.0; 3.0 0.5 1.0]
    A = [B zeros(2, 3); zeros(2, 2) C]
    a, b = cover_min(AbsLinear{2}(), A)
    @test iscover(a, b, A; atol=1e-8)
    @test isbalanced(a, b, A)
end

@testset "the Ipopt sym objective uses the full-grid weighting" begin
    A = Float64[4 1 0; 1 1 5; 0 5 2]
    a0 = fill(sqrt(maximum(A)), 3)
    a, _ = symcover_min!(AbsLinear{2}(), copy(a0), A)
    @test a ≈ [2.0, 1.0, 5.0] rtol=1e-6
    @test iscover(a, A; rtol=1e-6)
    # The objective reported for the result is the one that was minimized.
    @test cover_objective(AbsLinear{2}(), a, A) ≈ cover_objective(AbsLinear{2}(), [2.0, 1.0, 5.0], A) rtol=1e-6
end

@testset "solver statistics (JuMP/HiGHS/Ipopt)" begin
    ESS = MatrixCovers.ExternalSolverStats
    A = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]
    Aasym = [1.0 2.0 3.0; 4.0 5.0 6.0]

    # HiGHS: the AbsLog{1} LP, then the AbsLog{2} QP over its optimal face.
    a, stats = @inferred symcover_min(AbsLog{1}(), A)
    @test stats isa ESS
    @test stats.solver === :HiGHS
    @test stats.status == [:OPTIMAL, :OPTIMAL]
    @test length(stats.objective) == length(stats.niters) == length(stats.solvetime) == 2
    @test all(>=(0), stats.niters) && all(>=(0), stats.solvetime)
    @test stats.objective[2] ≈ cover_objective(AbsLog{2}(), a, A) rtol=1e-6
    @test MatrixCovers.converged(stats)
    @test_throws MethodError MatrixCovers.iterations(stats)
    @test_throws MethodError MatrixCovers.residual(stats)
    @test_throws MethodError MatrixCovers.tolerance(stats)
    for r in (@inferred(cover_min(AbsLog{1}(), Aasym)), @inferred(soft_cover(AbsLog{1}(), Aasym)),
              @inferred(soft_symcover(AbsLog{1}(), A)),
              @inferred(symcover_min!(AbsLog{1}(), symcover(A), A)),
              @inferred(soft_symcover!(AbsLog{1}(), symcover(A), A)),
              @inferred(cover_min!(AbsLog{1}(), cover(Aasym)..., Aasym)),
              @inferred(soft_cover!(AbsLog{1}(), cover(Aasym)..., Aasym)))
        s = last(r)
        @test s isa ESS && s.solver === :HiGHS && s.status == [:OPTIMAL, :OPTIMAL]
    end
    # Without support there is no optimal face to search, so only the LP runs.
    _, _, s0 = soft_cover(AbsLog{1}(), zeros(2, 3))
    @test s0.status == [:OPTIMAL]

    # Ipopt: one local refinement.
    a, stats = @inferred symcover_min!(AbsLinear{2}(), symcover(A), A)
    @test stats isa ESS
    @test stats.solver === :Ipopt
    @test stats.status == [:LOCALLY_SOLVED]
    @test only(stats.niters) > 0
    @test only(stats.objective) ≈ cover_objective(AbsLinear{2}(), a, A) rtol=1e-8
    @test MatrixCovers.converged(stats)
    @test_throws MethodError MatrixCovers.iterations(stats)
    for r in (@inferred(cover_min!(AbsLinear{1}(), cover(Aasym)..., Aasym)),
              @inferred(soft_symcover!(AbsLinear{1}(), symcover(A), A)),
              @inferred(soft_cover!(AbsLinear{2}(), cover(Aasym)..., Aasym)))
        s = last(r)
        @test s isa ESS && s.solver === :Ipopt && s.status == [:LOCALLY_SOLVED]
    end

    # Strategy multistart: one refinement per strategy that yields a start.
    Abasin = [81.892035218799 1.06622031288736 29.4700945830419 0.0181293142917846;
              1.06622031288736 0.243512973596586 38.0236584552296 0.0279078887878805;
              29.4700945830419 38.0236584552296 8.96405068596511 26.5775238859338;
              0.0181293142917846 0.0279078887878805 26.5775238859338 42.6650094474717]
    ϕ = AbsLinear{2}()
    a, ms = @inferred symcover_min(ϕ, Abasin)
    @test ms isa MatrixCovers.MultistartStats{ESS,Float64}
    @test ms.labels == [:hardcover, :geomean, :leaveout]
    @test ms.labels[ms.selected] === :geomean
    @test isempty(ms.failed)
    @test all(s -> s.solver === :Ipopt, ms.stats)
    @test ms.objectives[ms.selected] == cover_objective(ϕ, a, Abasin)
    @test ms.objectives[ms.selected] <= minimum(ms.objectives) * (1 + 1e-6)
    # Each record describes the refinement of its own start.
    _, s_hard = symcover_min!(ϕ, initialize_symcover(Abasin; strategy=:hardcover), Abasin)
    @test ms.stats[1].objective ≈ s_hard.objective
    @test MatrixCovers.converged(ms) === MatrixCovers.converged(ms.stats[ms.selected])
    @test_throws MethodError MatrixCovers.iterations(ms)
    # A strategy that yields no start for `A` is not recorded.
    _, ms = symcover_min(ϕ, [4.0 0.0; 0.0 9.0])
    @test ms.labels == [:hardcover, :geomean]
    a, b, ms = @inferred cover_min(AbsLinear{1}(), Aasym)
    @test ms.labels == [:hardcover, :covariant]
    @test ms.objectives[ms.selected] == cover_objective(AbsLinear{1}(), a, b, Aasym)

    # Random multistart: failed random starts are recorded and left out.
    a, b, ms = @inferred soft_cover(ϕ, ones(3, 3); starts=8, σ=10.0, rng=StableRNG(2))
    @test !isempty(ms.failed)
    @test all(l -> startswith(string(l), "rand"), ms.failed)
    @test isempty(intersect(ms.labels, ms.failed))
    @test length(ms.labels) + length(ms.failed) == 8
    @test length(ms.stats) == length(ms.objectives) == length(ms.labels)
    @test ms.objectives[ms.selected] == cover_objective(ϕ, a, b, ones(3, 3))
    a, ms = @inferred soft_symcover(ϕ, Float32.(A))
    @test ms isa MatrixCovers.MultistartStats{ESS,Float32}
    @test ms.labels[1:2] == [:geomean, :hardcover]

    # The constructor checks that its records line up.
    @test_throws DimensionMismatch MatrixCovers.MultistartStats{ESS,Float64}([:a, :b], [stats], [1.0], 1, Symbol[])
    @test_throws "is not an index" MatrixCovers.MultistartStats{ESS,Float64}([:a], [stats], [1.0], 2, Symbol[])
end
