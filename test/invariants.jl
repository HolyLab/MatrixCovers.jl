# Cross-notion invariants: conventions documented for every cover notion,
# checked uniformly across all of them. Each entry supplies the solver as
# `A -> a` (symmetric) or `A -> (a, b)` (general), whether it promises hard
# feasibility, and the tolerance its algorithm warrants. The themed files pin
# each notion's algorithm-specific precision; this file pins the shared
# conventions:
#   - repeated calls return identical results
#   - hard covers are feasible
#   - results co-vary with diagonal rescaling of A
#   - an entirely unsupported row/column gets scale exactly 0
#   - results depend only on entry magnitudes (complex input ≡ abs.(A))
#   - offset axes propagate from A to the scale vectors

const SYM_NOTIONS = (
    (name = "symcover",                        f = A -> symcover(A),                          hard = true,  rtol = 1e-9),
    (name = "symcover_min(AbsLog{2})",         f = A -> symcover_min(AbsLog{2}(), A),         hard = true,  rtol = 1e-5),
    (name = "symcover_min(AbsLog{1})",         f = A -> symcover_min(AbsLog{1}(), A),         hard = true,  rtol = 1e-5),
    (name = "symcover_min(AbsLinear{1})",      f = A -> symcover_min(AbsLinear{1}(), A),      hard = true,  rtol = 1e-5),
    (name = "symcover_min(AbsLinear{2})",      f = A -> symcover_min(AbsLinear{2}(), A),      hard = true,  rtol = 1e-5),
    (name = "soft_symcover(AbsLog{2})",        f = A -> soft_symcover(AbsLog{2}(), A),        hard = false, rtol = 1e-9),
    (name = "soft_symcover(AbsLog{1})",        f = A -> soft_symcover(AbsLog{1}(), A),        hard = false, rtol = 1e-8),
    (name = "soft_symcover(AbsLinear{2})",     f = A -> soft_symcover(AbsLinear{2}(), A),     hard = false, rtol = 1e-8),
    (name = "soft_symcover(AbsLinear{1})",     f = A -> soft_symcover(AbsLinear{1}(), A),     hard = false, rtol = 1e-8),
    (name = "soft_symcover_min(AbsLog{2})",    f = A -> soft_symcover_min(AbsLog{2}(), A),    hard = false, rtol = 1e-5),
    (name = "soft_symcover_min(AbsLinear{1})", f = A -> soft_symcover_min(AbsLinear{1}(), A), hard = false, rtol = 1e-5),
    (name = "soft_symcover_min(AbsLinear{2})", f = A -> soft_symcover_min(AbsLinear{2}(), A), hard = false, rtol = 1e-5),
)

const GEN_NOTIONS = (
    (name = "cover",                          f = A -> cover(A),                          hard = true,  rtol = 1e-9),
    (name = "cover_min(AbsLog{2})",           f = A -> cover_min(AbsLog{2}(), A),         hard = true,  rtol = 1e-5),
    (name = "cover_min(AbsLog{1})",           f = A -> cover_min(AbsLog{1}(), A),         hard = true,  rtol = 1e-5),
    (name = "cover_min(AbsLinear{1})",        f = A -> cover_min(AbsLinear{1}(), A),      hard = true,  rtol = 1e-5),
    (name = "cover_min(AbsLinear{2})",        f = A -> cover_min(AbsLinear{2}(), A),      hard = true,  rtol = 1e-5),
    (name = "soft_cover(AbsLinear{2})",       f = A -> soft_cover(AbsLinear{2}(), A),     hard = false, rtol = 1e-8),
    (name = "soft_cover(AbsLinear{1})",       f = A -> soft_cover(AbsLinear{1}(), A),     hard = false, rtol = 1e-8),
    (name = "soft_cover_min(AbsLog{2})",      f = A -> soft_cover_min(AbsLog{2}(), A),    hard = false, rtol = 1e-9),
    (name = "soft_cover_min(AbsLinear{1})",   f = A -> soft_cover_min(AbsLinear{1}(), A), hard = false, rtol = 1e-5),
    (name = "soft_cover_min(AbsLinear{2})",   f = A -> soft_cover_min(AbsLinear{2}(), A), hard = false, rtol = 1e-5),
)

@testset "cross-notion invariants" begin
    Asym  = [2.0 1.0 0.5; 1.0 3.0 1.0; 0.5 1.0 2.5]
    Azsym = [1.0 0.0 2.0; 0.0 0.0 0.0; 2.0 0.0 3.0]     # row/column 2 unsupported
    Hc    = [2.0 1.0+1.0im 0.0; 1.0-1.0im 3.0 0.5im; 0.0 -0.5im 2.5]  # Hermitian values
    d     = [2.0, 0.5, 4.0]

    @testset "sym: $(nt.name)" for nt in SYM_NOTIONS
        a = nt.f(Asym)
        @test a == nt.f(Asym)
        if nt.hard
            @test iscover(a, Asym; rtol=1e-8, atol=1e-6)
        end
        @test covaries(nt.f, Asym, d; rtol=nt.rtol)
        az = nt.f(Azsym)
        @test az[2] == 0
        @test nt.f(Hermitian(Hc)) ≈ nt.f(abs.(Hc)) rtol=nt.rtol
        Ao = OffsetArray(Asym, -1:1, -1:1)
        ao = nt.f(Ao)
        @test axes(ao, 1) == axes(Ao, 1)
        @test collect(ao) ≈ a rtol=nt.rtol
    end

    Agen  = [1.0 2.0 0.5; 0.25 3.0 1.0]                  # rectangular
    Azgen = [1.0 0.0 2.0; 0.0 0.0 0.0; 3.0 0.0 4.0]      # row/column 2 unsupported
    Gc    = [1.0+1.0im 2.0 0.5; 0.25im 3.0 1.0-2.0im]
    dr, dc = [2.0, 0.5], [3.0, 0.25, 1.5]
    # Two connected components, so the balance convention pins two independent gauges.
    Ablk  = [1.0 2.0 0.0 0.0 0.0; 0.25 3.0 0.0 0.0 0.0; 0.0 0.0 1.5 2.5 0.75]

    @testset "gen: $(nt.name)" for nt in GEN_NOTIONS
        a, b = nt.f(Agen)
        @test (a, b) == nt.f(Agen)
        if nt.hard
            @test iscover(a, b, Agen; rtol=1e-8, atol=1e-6)
        end
        @test covaries(nt.f, Agen, dr, dc; rtol=nt.rtol)
        az, bz = nt.f(Azgen)
        @test az[2] == 0
        @test bz[2] == 0
        # Gauge-invariant comparison: complex input must reproduce abs.(A).
        aC, bC = nt.f(Gc)
        aR, bR = nt.f(abs.(Gc))
        @test aC .* transpose(bC) ≈ aR .* transpose(bR) rtol=nt.rtol
        Ao = OffsetArray(Agen, 0:1, -1:1)
        ao, bo = nt.f(Ao)
        @test axes(ao, 1) == axes(Ao, 1)
        @test axes(bo, 1) == axes(Ao, 2)
        @test collect(ao) .* transpose(collect(bo)) ≈ a .* transpose(b) rtol=nt.rtol
        # The gauge a -> c*a, b -> b/c is invisible to every objective and every coverage
        # constraint, so nothing in the problem fixes the split between `a` and `b`. The
        # balance convention does, and every asymmetric cover reports its result in it.
        @test isbalanced(a, b, Agen)
        @test isbalanced(az, bz, Azgen)
        # The balance convention is imposed per connected component: on a support with
        # more than one component, `isbalanced` must hold within each rather than only
        # in a single global sum.
        ablk, bblk = nt.f(Ablk)
        @test isbalanced(ablk, bblk, Ablk)
    end

    @testset "gen: initialize_cover($strategy, $feasible) is balanced" for
            strategy in (:hardcover, :geomean), feasible in (:inflate, :boost, :none)
        @test isbalanced(initialize_cover(Agen; strategy, feasible)..., Agen)
        @test isbalanced(initialize_cover(Ablk; strategy, feasible)..., Ablk)
    end

    # The gauge factor is a whole power of two, so imposing the convention is exact
    # in binary floating point: every product the coverage constraints see is
    # preserved bit for bit. A cover cannot be perturbed into infeasibility by the
    # act of pinning its gauge. Products across two components are not preserved,
    # and are not constrained either — the support is exactly where both hold.
    @testset "balancing preserves on-support products exactly" begin
        rng = StableRNG(11)
        for A in (Agen, Ablk, Azgen)
            a = exp.(randn(rng, size(A, 1)))
            b = exp.(randn(rng, size(A, 2)))
            before = Dict{Tuple{Int,Int},Float64}()
            MatrixCovers.foreach_support(A) do i, j, v
                before[(i, j)] = a[i] * b[j]
            end
            MatrixCovers._balance_cover!(a, b, A)
            MatrixCovers.foreach_support(A) do i, j, v
                @test a[i] * b[j] === before[(i, j)]
            end
        end
    end

    # The gauge is a function of the component alone, so balancing a block-diagonal
    # assembly reproduces balancing each block on its own, entrywise.
    @testset "balancing commutes with block-diagonal assembly" begin
        rng = StableRNG(12)
        A1 = [1.0 2.0; 3.0 0.5]
        A2 = [4.0 0.0 1.0; 0.0 2.0 6.0]
        Abd = [A1 zeros(2, 3); zeros(2, 2) A2]
        a1, b1 = exp.(randn(rng, 2)), exp.(randn(rng, 2))
        a2, b2 = exp.(randn(rng, 2)), exp.(randn(rng, 3))
        abd, bbd = vcat(a1, a2), vcat(b1, b2)
        MatrixCovers._balance_cover!(a1, b1, A1)
        MatrixCovers._balance_cover!(a2, b2, A2)
        MatrixCovers._balance_cover!(abd, bbd, Abd)
        @test abd == vcat(a1, a2)
        @test bbd == vcat(b1, b2)
    end

    # The sym objective is summed over the full grid: each off-diagonal pair counts
    # twice, each diagonal entry once. `cover_objective` is the reference, and every
    # sym solver must minimize that same weighting even though its constraints live on
    # the `i <= j` triangle. The references below impose the constraints on the full
    # grid explicitly, so agreeing with them pins both halves of the convention: that
    # the triangle is the equivalent constraint set, and that the objective is not
    # halved along with it.
    @testset "sym solvers minimize the full-grid objective" begin
        function ref_min(pow, A)
            n = size(A, 1)
            L = log.(abs.(A))
            supp = [(i, j) for i in 1:n, j in 1:n if !iszero(A[i, j])]
            model = JuMP.Model(HiGHS.Optimizer)
            JuMP.set_silent(model)
            JuMP.@variable(model, α[1:n])
            r(i, j) = α[i] + α[j] - L[i, j]
            JuMP.@objective(model, Min, sum(r(i, j)^pow for (i, j) in supp))
            for (i, j) in supp
                JuMP.@constraint(model, r(i, j) >= 0)
            end
            JuMP.optimize!(model)
            return JuMP.objective_value(model)
        end

        # The two conventions share a minimizer whenever the diagonal residuals
        # vanish at the optimum, which is the common case — so a discriminating
        # matrix is committed rather than left to the random draws. Weighting this
        # one's off-diagonal pairs once instead of twice moves the optimum.
        Mdisc = [2.4 1.2 1.2; 1.2 1.4 2.4; 1.2 2.4 0.6]

        rng = StableRNG(20)
        mats = Any[Asym, Azsym, abs.(Hc), Mdisc]
        for _ in 1:5
            n = rand(rng, 3:7)
            B = randn(rng, n, n) .* (rand(rng, n, n) .< 0.5)
            M = B + transpose(B)
            push!(mats, M)
        end
        for M in mats
            # AbsLog{2}: native solver and the HiGHS reference model.
            o2 = ref_min(2, M)
            @test cover_objective(AbsLog{2}(), symcover_min(AbsLog{2}(), M), M) ≈ o2 rtol=1e-5
            @test cover_objective(AbsLog{2}(), MatrixCovers.symcover_min_jump(AbsLog{2}(), M), M) ≈ o2 rtol=1e-5
            # AbsLog{1}: the HiGHS LP, whose objective is assembled from nonzero counts
            # rather than from the residuals directly.
            @test cover_objective(AbsLog{1}(), symcover_min(AbsLog{1}(), M), M) ≈ ref_min(1, M) rtol=1e-5
        end
    end
end
