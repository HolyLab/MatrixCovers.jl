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

    # AbsLog{1}'s optimum is a flat family rather than a point, so the start can pick
    # out a different member of it; the objective attained is the same either way.
    a_cold = symcover_min(AbsLog{1}(), A)
    a_hot = symcover_min!(AbsLog{1}(), initialize_symcover(A), A)
    @test cover_objective(AbsLog{1}(), a_hot, A) ≈ cover_objective(AbsLog{1}(), a_cold, A)

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
    using ScaleInvariantAnalysis
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
    # `AbsLinear{2}` method it forwards to lives in the SIAIpopt extension. The
    # MethodError raised (and hinted on) is for the inner call, so the hint must
    # still fire even though the outer, no-ϕ call is what the user wrote.
    script_noϕ = """
    using ScaleInvariantAnalysis
    A = [4.0 1.0; 1.0 4.0]
    try
        soft_symcover_min(A)
    catch e
        print(sprint(showerror, e))
    end
    """
    out_noϕ = read(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script_noϕ`, String)
    @test occursin("loading JuMP", out_noϕ)

    # soft_cover_min's AbsLinear penalties are simply unimplemented, not gated
    # behind an extension, so the hint must say so rather than claim a package
    # load would help.
    e3 = try
        soft_cover_min(AbsLinear{2}(), A)
        nothing
    catch err
        err
    end
    @test e3 isa MethodError
    msg3 = sprint(showerror, e3)
    @test occursin("not yet supported", msg3)
    @test !occursin("loading JuMP", msg3)

    # Every penalty cover_min does not solve natively lives in an extension, whether the
    # call reaches it directly (AbsLog{1}) or through the AbsLinear multistart driver,
    # whose MethodError is raised on the cover_min! kernel it calls.
    script_cover = """
    using ScaleInvariantAnalysis
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
    using ScaleInvariantAnalysis
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
