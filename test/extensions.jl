# JuMP/HiGHS and JuMP/Ipopt extension solvers, and the missing-extension error hints.

@testset "symcover_min and cover_min (JuMP/HiGHS)" begin
    for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 1.0 0.01], [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0])
        a_fast  = symcover(A)
        a_lmin  = symcover_min(AbsLog{1}(), A)
        a_qmin  = symcover_min(AbsLog{2}(), A)
        # qmin is a valid cover
        @test iscover(a_qmin, A; atol=1e-10)
        # qmin achieves lower or equal AbsLog{2} objective than symcover and lmin
        # (up to the near-exact solver's penalty-continuation tolerance).
        @test cover_objective(AbsLog{2}(), a_qmin, A) <= cover_objective(AbsLog{2}(), a_fast, A) * (1 + 1e-6) + 1e-10
        @test cover_objective(AbsLog{2}(), a_qmin, A) <= cover_objective(AbsLog{2}(), a_lmin, A) * (1 + 1e-6) + 1e-10
    end
    # Exact case with zeros
    A = [0 0 1; 0 0 2; 1 2 1]
    a = symcover_min(AbsLog{1}(), A)
    @test iscover(a, A; atol=1e-10)
    @test a ≈ [1, 2, 1]
    @test abs(cover_objective(AbsLog{1}(), a, A)) < 1e-10
    a = symcover_min(AbsLog{2}(), A)
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

    # A solve that does not reach an optimum is an error, not a cover: this input
    # leaves the AbsLog{1} LP unbounded, and the point the solver holds is the base
    # of a ray rather than a minimizer.
    @test_throws "terminated with status" symcover_min(AbsLog{1}(), [0.0 0.0; 1.0 0.0])
    @test_throws "symcover_min" symcover_min(AbsLog{1}(), [0.0 0.0; 1.0 0.0])
end

@testset "symcover_min and soft_symcover_min (JuMP/Ipopt, AbsLinear)" begin
    # non-square rejected
    @test_throws "symcover_min requires a square matrix" symcover_min(AbsLinear{2}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])
    @test_throws "symcover_min requires a square matrix" symcover_min(AbsLinear{1}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])
    @test_throws "soft_symcover_min requires a square matrix" soft_symcover_min(AbsLinear{2}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])
    @test_throws "soft_symcover_min requires a square matrix" soft_symcover_min(AbsLinear{1}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])

    for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 1.0 0.01], [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0])
        a_fast = symcover(AbsLinear{2}(), A)
        for ϕ in (AbsLinear{1}(), AbsLinear{2}())
            # symcover_min: valid hard cover, at most as costly as heuristic
            a_min = symcover_min(ϕ, A)
            @test iscover(a_min, A; atol=1e-6)
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
        a_heur = soft_symcover(AbsLinear{2}(), A; maxiter=50)
        @test cover_objective(AbsLinear{2}(), a_opt, A) <=
              cover_objective(AbsLinear{2}(), a_heur, A) + 1e-6
    end
end

@testset "symcover_min!/cover_min! refiners (JuMP/HiGHS/Ipopt)" begin
    A = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]
    Aasym = [1.0 2.0 3.0; 4.0 5.0 6.0]

    for ϕ in PENALTIES
        # Refining a start yields a hard cover no worse than the start itself, to
        # within the tolerance the solvers converge to (on this matrix the :hardcover
        # start is already all but optimal, so there is nothing else separating them).
        a0 = initialize_symcover(A)
        a = symcover_min!(ϕ, copy(a0), A)
        @test iscover(a, A; rtol=1e-6)
        @test cover_objective(ϕ, a, A) <= cover_objective(ϕ, a0, A) * (1 + 1e-6) + 1e-8

        ab0, bb0 = initialize_cover(Aasym)
        ab, bb = cover_min!(ϕ, copy(ab0), copy(bb0), Aasym)
        @test iscover(ab, bb, Aasym; rtol=1e-6)
        @test cover_objective(ϕ, ab, bb, Aasym) <= cover_objective(ϕ, ab0, bb0, Aasym) * (1 + 1e-6) + 1e-8

        # The asymmetric result is pinned to the balance convention, so the start is
        # read only up to the gauge a -> c*a, b -> b/c that leaves a[i]*b[j] fixed.
        ga, gb = cover_min!(ϕ, 4 .* ab0, bb0 ./ 4, Aasym)
        @test ga ≈ ab && gb ≈ bb
    end

    # AbsLog{1}'s optimum is a whole face of equally-scoring covers, but the solver pins the
    # member minimizing the AbsLog{2} objective over it, so the start cannot be read off the
    # result — the same point comes back from any of them.
    a_cold = symcover_min(AbsLog{1}(), A)
    for strategy in (:hardcover, :geomean, :diagfeasible)
        @test symcover_min!(AbsLog{1}(), initialize_symcover(A; strategy), A) ≈ a_cold
    end
    ab_cold, bb_cold = cover_min(AbsLog{1}(), Aasym)
    for strategy in (:hardcover, :geomean)
        ah, bh = cover_min!(AbsLog{1}(), initialize_cover(Aasym; strategy)..., Aasym)
        @test ah ≈ ab_cold && bh ≈ bb_cold
    end

    # The AbsLinear objectives are non-convex: on this matrix the :hardcover and
    # :geomean starts descend into genuinely different local minima, which is what
    # makes a menu of starts worth having.
    Abasin = [6.609216272192496 1.032613546278995 55.276094662396076 0.5076927138328724;
              1.032613546278995 3.1390835186570034 11.658167585612446 38.315826566607555;
              55.276094662396076 11.658167585612446 0.001705708114264713 21.68951642774627;
              0.5076927138328724 38.315826566607555 21.68951642774627 0.006443251375371587]
    ϕ = AbsLinear{2}()
    a_hard = symcover_min!(ϕ, initialize_symcover(Abasin; strategy=:hardcover), Abasin)
    a_geo = symcover_min!(ϕ, initialize_symcover(Abasin; strategy=:geomean), Abasin)
    @test iscover(a_hard, Abasin; rtol=1e-6)
    @test iscover(a_geo, Abasin; rtol=1e-6)
    @test cover_objective(ϕ, a_geo, Abasin) < cover_objective(ϕ, a_hard, Abasin) - 1e-3

    # Offset axes survive the Ipopt position mapping.
    Ao = OffsetArray(A, -1, -1)
    ao = symcover_min!(ϕ, initialize_symcover(Ao), Ao)
    @test axes(ao, 1) == axes(Ao, 1)
    @test collect(ao) ≈ symcover_min!(ϕ, initialize_symcover(A), A)
end

@testset "soft_symcover_min multistart and refiner (Ipopt)" begin
    # A start for the soft cover need not cover `A`: the objective imposes no coverage
    # constraint, and the raw geometric mean — the exact soft AbsLog{2} optimum — does not
    # cover. The refiner must accept it where symcover_min! would reject it.
    A = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]
    a0 = initialize_symcover(A; strategy=:geomean, feasible=:none)
    @test !iscover(a0, A)
    @test_throws "requires a start that covers `A`" symcover_min!(AbsLinear{2}(), copy(a0), A)
    @test soft_symcover_min!(AbsLinear{2}(), copy(a0), A) isa AbstractVector

    for ϕ in (AbsLinear{1}(), AbsLinear{2}(), AbsLog{2}())
        a = soft_symcover_min(ϕ, A)
        @test soft_symcover_min(ϕ, A) == a                       # deterministic
        # The soft optimum is no worse than the heuristic soft cover.
        @test cover_objective(ϕ, a, A) <=
              cover_objective(ϕ, soft_symcover(ϕ, A), A) * (1 + 1e-6) + 1e-8
    end

    # No start on the menu beats the driver.
    for ϕ in (AbsLinear{1}(), AbsLinear{2}())
        a = soft_symcover_min(ϕ, A)
        for strategy in (:hardcover, :geomean, :leaveout)
            single = soft_symcover_min!(ϕ, initialize_symcover(A; strategy, feasible=:none), A)
            @test cover_objective(ϕ, a, A) <= cover_objective(ϕ, single, A) * (1 + 1e-6) + 1e-8
        end
    end

    # For a non-convex objective, scale covariance is a property of the *start*: a start that
    # does not co-vary with `A` can reach a different basin once the frame is rescaled, and
    # the objective is scale-invariant only within a basin. This matrix and rescaling separate
    # the basins sharply enough to catch that — an `A`-independent start scores 4.00 in the
    # rescaled frame where the co-varied answer scores 1.47 — so it pins the menu's covariance.
    Abasins = [0.020358630644342735 0.53352144014074843 5.8899714528796077 0.23770314779348869 3.0721768720180109;
              0.53352144014074843 3.5416395788642903 37.199280652497748 49.972622569225109 333.53567816710364;
              5.8899714528796077 37.199280652497748 0.76014027958709862 2.4189759139690739 0.92571970793600067;
              0.23770314779348869 49.972622569225109 2.4189759139690739 8.1401049198051822 7.3601096491681046;
              3.0721768720180109 333.53567816710364 0.92571970793600067 7.3601096491681046 1.0844635712556003]
    d = [20.451338935482074, 0.69212569803171398, 35.401627522529395, 15.904906661932396, 0.19696509774727827]
    for ϕ in (AbsLinear{1}(), AbsLinear{2}())
        @test covaries(A -> soft_symcover_min(ϕ, A), Abasins, d; rtol=1e-5)
    end

    @test_throws "positive scale on every supported row" soft_symcover_min!(AbsLinear{2}(), [1.0, -1.0, 2.0], A)
    @test_throws "soft_symcover_min! requires a square matrix" soft_symcover_min!(AbsLinear{2}(), [1.0, 2.0], [1.0 2.0 3.0; 4.0 5.0 6.0])
    @test_throws "unknown strategy :banana" soft_symcover_min(AbsLinear{2}(), A; strategies=(:banana,))
end

@testset "soft_cover_min multistart and refiner (Ipopt)" begin
    A = [1.0 2.0 3.0; 40.0 5.0 0.6]
    nza = vec(count(!iszero, A, dims=2))
    nzb = vec(count(!iszero, A, dims=1))
    balance(a, b) = sum(nza .* log.(a)) - sum(nzb .* log.(b))

    # The soft start need not cover `A`, so cover_min! rejects what soft_cover_min! accepts.
    a0, b0 = initialize_cover(A; strategy=:geomean, feasible=:none)
    @test !iscover(a0, b0, A)
    @test_throws "requires a start that covers `A`" cover_min!(AbsLinear{2}(), copy(a0), copy(b0), A)

    for ϕ in (AbsLinear{1}(), AbsLinear{2}(), AbsLog{2}())
        a, b = soft_cover_min(ϕ, A)
        @test soft_cover_min(ϕ, A) == (a, b)                      # deterministic
        # The objective depends on `a`, `b` only through their products, so the split is
        # fixed by the balance convention rather than left to the solver.
        @test balance(a, b) ≈ 0 atol=1e-7
        # Gauge-invariant in the start: (a, b) and (2a, b/2) name the same point.
        ag, bg = soft_cover_min!(ϕ, 2 .* copy(a0), copy(b0) ./ 2, A)
        ah, bh = soft_cover_min!(ϕ, copy(a0), copy(b0), A)
        @test ag ≈ ah && bg ≈ bh
    end

    # No start on the menu beats the driver, and the driver is no worse than the heuristic.
    for ϕ in (AbsLinear{1}(), AbsLinear{2}())
        a, b = soft_cover_min(ϕ, A)
        for strategy in (:hardcover, :geomean)
            sa, sb = soft_cover_min!(ϕ, initialize_cover(A; strategy, feasible=:none)..., A)
            @test cover_objective(ϕ, a, b, A) <= cover_objective(ϕ, sa, sb, A) * (1 + 1e-6) + 1e-8
        end
        ha, hb = soft_cover(ϕ, A)
        @test cover_objective(ϕ, a, b, A) <= cover_objective(ϕ, ha, hb, A) * (1 + 1e-6) + 1e-8

        # The asymmetric soft cover relaxes the symmetric one on a symmetric matrix.
        As = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]
        aa, ab = soft_cover_min(ϕ, As)
        @test cover_objective(ϕ, aa, ab, As) <=
              cover_objective(ϕ, soft_symcover_min(ϕ, As), As) * (1 + 1e-6) + 1e-8
    end

    @test_throws "positive scale on every supported row" soft_cover_min!(AbsLinear{2}(), [1.0, 0.0], [1.0, 1.0, 1.0], A)
    @test_throws "positive scale on every supported column" soft_cover_min!(AbsLinear{2}(), [1.0, 1.0], [1.0, -1.0, 1.0], A)
    @test_throws "at least one starting cover" soft_cover_min(AbsLinear{2}(), A; strategies=())
end

@testset "AbsLog{1} canonical selection on the optimal face (HiGHS)" begin
    # The AbsLog{1} optimum is a face of the feasible polytope, not a point: its members are
    # different covers scoring the same objective. The solver returns the member that also
    # minimizes the AbsLog{2} objective — L1-optimal still, but the tightest such cover
    # rather than whichever vertex the LP happened to reach.
    ϕ1, ϕ2 = AbsLog{1}(), AbsLog{2}()
    rng = StableRNG(17)
    for n in (3, 5, 8)
        B = exp.(3 .* randn(rng, n, n)) .* (rand(rng, n, n) .> 0.25)
        A = (B + B') / 2
        all(iszero, A) && continue
        a = symcover_min(ϕ1, A)
        @test iscover(a, A; rtol=1e-7)

        # No cover beats it on AbsLog{1}: the canonical selection does not cost optimality.
        # Any feasible cover is a witness; use the AbsLog{2}-minimal one, and the heuristic.
        for other in (symcover_min(ϕ2, A), symcover(A))
            @test cover_objective(ϕ1, a, A) <= cover_objective(ϕ1, other, A) * (1 + 1e-6) + 1e-8
        end

        # Start-independent in the *result*, not merely in the objective value.
        for strategy in (:hardcover, :geomean, :diagfeasible)
            @test symcover_min!(ϕ1, initialize_symcover(A; strategy), A) ≈ a
        end

        # Scale-covariant: both objectives see `A` only through the residuals, which a
        # rescaling leaves invariant, so the selected member co-varies with the frame.
        D = Diagonal(exp.(randn(rng, n)))
        @test symcover_min(ϕ1, D * A * D) ≈ D * a
    end

    # Asymmetric: the balance convention pins the gauge `a -> c*a, b -> b/c`, which leaves
    # every product a[i]*b[j] untouched; the canonical selection resolves the L1 face, whose
    # members have genuinely different products. The two rules are independent and both hold.
    Aasym = [3.0 1.0 7.0; 2.0 5.0 1.0; 8.0 1.0 4.0]
    a, b = cover_min(ϕ1, Aasym)
    nza = vec(count(!iszero, Aasym, dims=2))
    nzb = vec(count(!iszero, Aasym, dims=1))
    @test sum(nza .* log.(a)) ≈ sum(nzb .* log.(b)) atol=1e-8
    @test iscover(a, b, Aasym; rtol=1e-7)
    # Gauge-invariant in the start, and start-independent in the result.
    a0, b0 = initialize_cover(Aasym; strategy=:geomean)
    ag, bg = cover_min!(ϕ1, 4 .* a0, b0 ./ 4, Aasym)
    @test ag ≈ a && bg ≈ b
end

@testset "AbsLinear multistart drivers (Ipopt)" begin
    # The matrix on which the starts genuinely disagree: :geomean reaches a better
    # AbsLinear{2} minimum than :hardcover, so a driver that tried only the latter
    # would report the worse of the two.
    Abasin = [6.609216272192496 1.032613546278995 55.276094662396076 0.5076927138328724;
              1.032613546278995 3.1390835186570034 11.658167585612446 38.315826566607555;
              55.276094662396076 11.658167585612446 0.001705708114264713 21.68951642774627;
              0.5076927138328724 38.315826566607555 21.68951642774627 0.006443251375371587]
    Aasym = [1.0 2.0 3.0; 40.0 5.0 0.6]

    for ϕ in (AbsLinear{1}(), AbsLinear{2}())
        a = symcover_min(ϕ, Abasin)
        @test iscover(a, Abasin; rtol=1e-6)
        # No start on the menu beats the driver.
        for strategy in (:hardcover, :geomean, :leaveout)
            single = symcover_min!(ϕ, initialize_symcover(Abasin; strategy), Abasin)
            @test cover_objective(ϕ, a, Abasin) <=
                  cover_objective(ϕ, single, Abasin) * (1 + 1e-6) + 1e-8
        end
        @test symcover_min(ϕ, Abasin) == a   # deterministic

        ab, bb = cover_min(ϕ, Aasym)
        @test iscover(ab, bb, Aasym; rtol=1e-6)
        for strategy in (:hardcover, :geomean)
            sa, sb = cover_min!(ϕ, initialize_cover(Aasym; strategy)..., Aasym)
            @test cover_objective(ϕ, ab, bb, Aasym) <=
                  cover_objective(ϕ, sa, sb, Aasym) * (1 + 1e-6) + 1e-8
        end
        @test cover_min(ϕ, Aasym) == (ab, bb)   # deterministic

        # The asymmetric cover relaxes the symmetric one: independent row and column
        # scales can only do better on the same matrix.
        asym_a, asym_b = cover_min(ϕ, Abasin)
        @test cover_objective(ϕ, asym_a, asym_b, Abasin) <=
              cover_objective(ϕ, symcover_min(ϕ, Abasin), Abasin) * (1 + 1e-6) + 1e-8
    end

    # On Abasin the :geomean start wins, so restricting the menu to :hardcover is
    # observable — the driver is refining the menu it is given, not a fixed start.
    ϕ = AbsLinear{2}()
    @test symcover_min(ϕ, Abasin; strategies=(:hardcover,)) ≈
          symcover_min!(ϕ, initialize_symcover(Abasin; strategy=:hardcover), Abasin)
    @test cover_objective(ϕ, symcover_min(ϕ, Abasin), Abasin) <
          cover_objective(ϕ, symcover_min(ϕ, Abasin; strategies=(:hardcover,)), Abasin) - 1e-3

    # A matrix whose every row carries a single support entry admits no :leaveout start;
    # that strategy forfeits its slot rather than failing the solve.
    Adiag = [4.0 0.0; 0.0 9.0]
    @test symcover_min(ϕ, Adiag) ≈ [2.0, 3.0] rtol=1e-4

    @test_throws "unknown strategy :banana" symcover_min(ϕ, Abasin; strategies=(:banana,))
    @test_throws "no strategy in (:leaveout,)" symcover_min(ϕ, Adiag; strategies=(:leaveout,))
    @test_throws "has no asymmetric formulation" cover_min(ϕ, Aasym; strategies=(:leaveout,))
    @test_throws "at least one starting cover" cover_min(ϕ, Aasym; strategies=())
end

@testset "error hint gated on argument types" begin
    # Wrong-argument-type MethodError: no extension load would fix this,
    # so the hint must not fire.
    A = [4.0 1.0; 1.0 4.0]
    e = try
        symcover_min(AbsLog{2}(), "not a matrix")
        nothing
    catch err
        err
    end
    @test e isa MethodError
    @test !occursin("loading JuMP", sprint(showerror, e))

    # Genuine missing-extension MethodError: run in a fresh process with
    # JuMP/HiGHS/Ipopt unloaded (this test file loads them itself, which
    # would otherwise mask the failure), so the hint should fire.
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

    # The no-ϕ wrapper `soft_symcover_min(A)` exists in the base package, but the
    # `AbsLinear{2}` method it forwards to lives in the MatrixCoversIpoptExt extension. The
    # MethodError raised (and hinted on) is for the inner call, so the hint must
    # still fire even though the outer, no-ϕ call is what the user wrote.
    script_noϕ = """
    using MatrixCovers
    A = [4.0 1.0; 1.0 4.0]
    try
        soft_symcover_min(A)
    catch e
        print(sprint(showerror, e))
    end
    """
    out_noϕ = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script_noϕ`, String)
    @test occursin("loading JuMP", out_noϕ)

    # soft_cover_min's AbsLog{1} is genuinely unimplemented rather than gated behind an
    # extension, so its hint must say so rather than claim a package load would help — while
    # still advising the load for the AbsLinear penalties, which Ipopt does provide.
    e3 = try
        soft_cover_min(AbsLog{1}(), A)
        nothing
    catch err
        err
    end
    @test e3 isa MethodError
    msg3 = sprint(showerror, e3)
    @test occursin("AbsLog{1} is not yet supported", msg3)
    @test occursin("loading JuMP", msg3)

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
        symcover_min!(AbsLog{2}(), [1.0, 1.0], "not a matrix")
        nothing
    catch err
        err
    end
    @test e5 isa MethodError
    @test !occursin("loading JuMP", sprint(showerror, e5))
end
