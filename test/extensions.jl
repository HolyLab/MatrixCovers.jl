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
end
