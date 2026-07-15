# The named starting covers: the :hardcover / :geomean / :leaveout / :diagfeasible
# menu, and the `feasible` keyword that pushes any of them onto the coverage boundary.

@testset "cover initializers" begin
    # Rows/columns carrying at least one nonzero entry; every strategy must return a
    # strictly positive scale on these and exactly zero on the rest.
    rowsupport(A) = [any(!iszero, view(A, i, :)) for i in axes(A, 1)]
    colsupport(A) = [any(!iszero, view(A, :, j)) for j in axes(A, 2)]

    SYM_STRATEGIES = (:hardcover, :geomean, :leaveout, :diagfeasible)
    ASYM_STRATEGIES = (:hardcover, :geomean)

    # Every strategy must be available on each of these: :leaveout needs a support entry it can
    # drop without emptying a row, so every supported row carries at least two.
    rng = StableRNG(7)
    Asyms = ([2.0 1.0; 1.0 3.0],
             [1.0 10.0; 10.0 1.0],                                  # geomean falls short here
             [1.0 0.5 0.0; 0.5 2.0 0.0; 0.0 0.0 0.0],               # unsupported row/column
             Matrix(Symmetric(randn(rng, 5, 5))))
    Aasyms = ([1.0 2.0 3.0; 4.0 5.0 6.0], [1.0 0.0 3.0; 4.0 0.0 6.0])

    FEASIBLE = (:inflate, :boost, :none)

    @testset "symmetric postcondition" begin
        for A in Asyms, strategy in SYM_STRATEGIES, feasible in FEASIBLE
            a = initialize_symcover(A; strategy, feasible)
            supp = rowsupport(A)
            @test all(a[i] > 0 for i in axes(A, 1) if supp[i])
            @test all(a[i] == 0 for i in axes(A, 1) if !supp[i])
            # Both feasible routes promise coverage; `:none` promises nothing.
            feasible === :none || @test iscover(a, A; rtol=8eps())
            @test initialize_symcover(A; strategy, feasible) == a   # deterministic
        end
    end

    @testset "asymmetric postcondition" begin
        for A in Aasyms, strategy in ASYM_STRATEGIES, feasible in FEASIBLE
            a, b = initialize_cover(A; strategy, feasible)
            rs, cs = rowsupport(A), colsupport(A)
            @test all(a[i] > 0 for i in axes(A, 1) if rs[i])
            @test all(a[i] == 0 for i in axes(A, 1) if !rs[i])
            @test all(b[j] > 0 for j in axes(A, 2) if cs[j])
            @test all(b[j] == 0 for j in axes(A, 2) if !cs[j])
            feasible === :none || @test iscover(a, b, A; rtol=8eps())
            @test initialize_cover(A; strategy, feasible) == (a, b)
        end
    end

    @testset "`feasible` is what makes a cover" begin
        # The geometric mean is the soft AbsLog{2} optimum, not a cover: for this matrix it
        # falls short on the off-diagonal, and either route lifts it onto the boundary.
        A = [1.0 10.0; 10.0 1.0]
        @test !iscover(initialize_symcover(A; strategy=:geomean, feasible=:none), A)
        @test iscover(initialize_symcover(A; strategy=:geomean, feasible=:inflate), A; rtol=8eps())
        @test iscover(initialize_symcover(A; strategy=:geomean, feasible=:boost), A; rtol=8eps())
    end

    @testset "the two feasible routes are distinct points" begin
        # `:inflate` scales the whole point by one factor, so it lifts even the rows that were
        # already slack; `:boost` raises only the rows touching a violated entry. Rows 1-2 are
        # the violated block here and row 3 is slack, so only the inflation disturbs row 3.
        A = [1.0 0.1 0.0; 0.1 1.0 0.0; 0.0 0.0 100.0]
        ai = initialize_symcover(A; strategy=:geomean, feasible=:inflate)
        ab = initialize_symcover(A; strategy=:geomean, feasible=:boost)
        @test ai != ab
        @test ab[3] < ai[3]

        # `:hardcover` is the boosted geometric mean, tightened — so naming the middle stage
        # is what lets a caller stop there, and `cover` is that stage plus the tightening.
        # The decomposition is a statement about the cover, i.e. about the products a[i]*b[j];
        # the gauge is pinned once, at the end of whichever entry point the caller used.
        for B in Aasyms
            ag, bg = initialize_cover(B; strategy=:geomean, feasible=:boost)
            a0, b0 = cover(B; maxiter=0)
            @test ag * bg' ≈ a0 * b0'
            at, bt = ScaleInvariantAnalysis.tighten_cover!(copy(ag), copy(bg), B)
            ac, bc = cover(B)
            @test at * bt' ≈ ac * bc'
        end
    end

    @testset "the strategies are distinct starts" begin
        # :hardcover raises only the rows touching violated entries; an inflated :geomean moves
        # the whole point bodily to the boundary, so it lifts the already-slack rows too.
        # Reaching the boundary by different routes is what gives the AbsLinear multistart
        # different basins to choose between. Here rows 1-2 are the violated block and row 3 is
        # slack, so only the inflation disturbs row 3.
        A = [1.0 0.1 0.0; 0.1 1.0 0.0; 0.0 0.0 100.0]
        ah = initialize_symcover(A; strategy=:hardcover)
        ag = initialize_symcover(A; strategy=:geomean)
        @test ah != ag
        @test ah[3] < ag[3]
    end

    @testset "no penalty argument" begin
        # Every start on the menu is a property of `A` alone, so none of them needs a ϕ and
        # none is offered one. A regression check on that, not a promise it will never
        # change: a ϕ-tuned start would arrive as a new method, with the fallback dropping ϕ
        # and calling these — additive, so callers of the current forms are unaffected.
        @test_throws MethodError initialize_symcover(AbsLog{2}(), Asyms[1])
        @test_throws MethodError initialize_cover(AbsLog{2}(), Aasyms[1])
    end

    @testset ":hardcover reproduces the plain heuristics" begin
        # Without the exactifying inflation, `:hardcover` *is* symcover/cover.
        for A in Asyms
            @test initialize_symcover(A; strategy=:hardcover, feasible=:none) == symcover(A)
            @test initialize_symcover(A; strategy=:hardcover, feasible=:none, maxiter=1) == symcover(A; maxiter=1)
        end
        for A in Aasyms
            @test initialize_cover(A; strategy=:hardcover, feasible=:none) == cover(A)
            @test initialize_cover(A; strategy=:hardcover, feasible=:none, maxiter=0) == cover(A; maxiter=0)
        end
    end

    @testset "mutating forms" begin
        A = Asyms[1]
        a = similar(A, 2)
        for strategy in SYM_STRATEGIES
            @test initialize_symcover!(a, A; strategy) === a
            @test a == initialize_symcover(A; strategy)
        end
        B = Aasyms[1]
        aB, bB = similar(B, 2), similar(B, 3)
        for strategy in ASYM_STRATEGIES
            @test initialize_cover!(aB, bB, B; strategy) === (aB, bB)
            @test (aB, bB) == initialize_cover(B; strategy)
        end
    end

    @testset "offset axes" begin
        A, B = Asyms[1], Aasyms[1]
        Ao = OffsetArray(A, 0:1, 0:1)
        Bo = OffsetArray(B, 0:1, -1:1)
        for strategy in SYM_STRATEGIES
            ao = initialize_symcover(Ao; strategy)
            @test axes(ao, 1) == 0:1
            @test collect(ao) == initialize_symcover(A; strategy)
        end
        for strategy in ASYM_STRATEGIES
            aBo, bBo = initialize_cover(Bo; strategy)
            @test axes(aBo, 1) == 0:1 && axes(bBo, 1) == -1:1
            aB, bB = initialize_cover(B; strategy)
            @test collect(aBo) == aB && collect(bBo) == bB
        end
    end

    @testset "argument checking" begin
        A, B = Asyms[1], Aasyms[1]
        @test_throws "unknown strategy :bogus" initialize_symcover(A; strategy=:bogus)
        @test_throws "unknown strategy :bogus" initialize_cover(B; strategy=:bogus)
        @test_throws "unknown feasible :bogus" initialize_symcover(A; feasible=:bogus)
        @test_throws "unknown feasible :bogus" initialize_cover(B; feasible=:bogus)
        # The keyword names a route, not a yes/no.
        @test_throws TypeError initialize_symcover(A; feasible=true)
        # The symmetric-only strategies say so, rather than reporting an unknown name.
        @test_throws "strategy=:leaveout has no asymmetric formulation" initialize_cover(B; strategy=:leaveout)
        @test_throws "strategy=:diagfeasible has no asymmetric formulation" initialize_cover(B; strategy=:diagfeasible)
        # A strategy with no tunables of its own rejects keywords rather than dropping them.
        @test_throws "strategy=:geomean accepts no further keyword arguments" initialize_symcover(A; strategy=:geomean, maxiter=3)
        @test_throws "strategy=:geomean accepts no further keyword arguments" initialize_cover(B; strategy=:geomean, maxiter=3)
        # :leaveout needs an entry it can drop without emptying a row.
        @test_throws "requires a support entry that can be dropped" initialize_symcover([1.0 0.0; 0.0 1.0]; strategy=:leaveout)
        @test_throws "initialize_symcover requires a square matrix" initialize_symcover(B)
        @test_throws "initialize_symcover! requires a square matrix" initialize_symcover!(zeros(2), B)
        @test_throws "indices of `a` must match the indexing of `A`" initialize_symcover!(zeros(3), A)
        @test_throws "indices of `b` must match column-indexing of `A`" initialize_cover!(zeros(2), zeros(2), B)
    end

    @testset "inflate_feasible! rejects a zero scale on a supported row" begin
        A = [2.0 1.0; 1.0 3.0]
        @test_throws "positive scale on every supported row" ScaleInvariantAnalysis.inflate_feasible!([0.0, 1.0], A)
        @test_throws "positive scale on every supported row/column" ScaleInvariantAnalysis.inflate_feasible!([0.0, 1.0], [1.0, 1.0], A)
    end
end
