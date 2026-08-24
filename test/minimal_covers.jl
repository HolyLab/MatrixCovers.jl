# The *_min family: native AbsLog{2} minimal-cover solvers and their edge cases.

# Reuse the committed 5×5 matrix libraries when already loaded.
if !isdefined(@__MODULE__, :symmetric_matrices)
    include("testmatrices.jl")
end

@testset "symcover_min native AbsLog{2}" begin
    # Non-square rejected.
    @test_throws "symcover_min requires a square matrix" symcover_min(AbsLog{2}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])

    # Match HiGHS across the symmetric corpus.
    for (_, A) in symmetric_matrices
        Af = Float64.(A)
        a  = symcover_min(AbsLog{2}(), Af)
        aj = MatrixCovers.symcover_min_jump(AbsLog{2}(), Af)
        @test iscover(a, Af; atol=1e-8)
        oj = cover_objective(AbsLog{2}(), aj, Af)
        o  = cover_objective(AbsLog{2}(), a, Af)
        @test o <= oj * (1 + 1e-6) + 1e-10
    end

    # Scale-covariance: for a positive diagonal D, the optimal cover of D*A*D
    # is D times the optimal cover of A (up to the a·aᵀ gauge), so the product
    # a[i]*a[j] covaries as d[i]*d[j].
    rng = StableRNG(1234)
    for (_, A) in symmetric_matrices[1:30]
        Af = Float64.(A); n = size(Af, 1)
        d = exp.(2 .* randn(rng, n))
        @test covaries(A -> symcover_min(AbsLog{2}(), A), Af, d; rtol=1e-6)
    end

    # Edge cases.
    @test symcover_min(AbsLog{2}(), reshape([4.0], 1, 1)) ≈ [2.0]           # n = 1
    # [0 1; 1 0]: a₁a₂ = 1 is the (gauge-invariant) optimum, objective 0.
    a = symcover_min(AbsLog{2}(), [0.0 1.0; 1.0 0.0])
    @test a[1] * a[2] ≈ 1.0
    @test cover_objective(AbsLog{2}(), a, [0.0 1.0; 1.0 0.0]) < 1e-12
    # Scattered zeros with an exact rank-1 cover.
    A = [0 0 1; 0 0 2; 1 2 1]
    a = symcover_min(AbsLog{2}(), A)
    @test a ≈ [1, 2, 1]
    @test abs(cover_objective(AbsLog{2}(), a, A)) < 1e-8
    # κs keyword is accepted.
    @test symcover_min(AbsLog{2}(), [2.0 1.0; 1.0 3.0]; κs=(1e2, 1e4, 1e6, 1e8, 1e10)) isa Vector
end

@testset "cover_min native AbsLog{2}" begin
    # Match HiGHS on a deterministic sample of the general corpus.
    idx_sub = Set(round.(Int, range(1, length(general_matrices), length=500)))
    for (k, (_, A)) in enumerate(general_matrices)
        Af = Float64.(A)
        a, b = cover_min(AbsLog{2}(), Af)
        @test iscover(a, b, Af; atol=1e-7)
        if k in idx_sub
            aj, bj = MatrixCovers.cover_min_jump(AbsLog{2}(), Af)
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
    rng = StableRNG(1234)
    for (_, A) in general_matrices[1:30]
        Af = Float64.(A); m, n = size(Af)
        dr = exp.(2 .* randn(rng, m)); dc = exp.(2 .* randn(rng, n))
        @test covaries(A -> cover_min(AbsLog{2}(), A), Af, dr, dc; rtol=1e-6)
    end

    # Non-square matrices, both orientations (transpose swaps the roles of a, b).
    A = [1.0 2.0 3.0; 4.0 5.0 6.0]
    a, b = cover_min(AbsLog{2}(), A)
    @test iscover(a, b, A; atol=1e-8)
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
    # A matrix with a zero column, exact objective known from the AbsLog{2} optimum.
    A = [0 0 0 1; 1 1 0 2; 1 0 2 1]
    a, b = cover_min(AbsLog{2}(), A)
    @test iscover(a, b, A; atol=1e-8)
    @test cover_objective(AbsLog{2}(), a, b, A) ≈ 2 * log(sqrt(2))^2
    # κs keyword is accepted.
    @test cover_min(AbsLog{2}(), [1.0 2.0; 3.0 4.0]; κs=(1e2, 1e4, 1e6, 1e8, 1e10)) isa Tuple
end

@testset "MMC native AbsLog{2} matrix-free LSQR path" begin
    # Invalid solver selection is rejected.
    @test_throws "linsolve must be :auto, :dense, :lsqr, or :woodbury" symcover_min(AbsLog{2}(), [2.0 1.0; 1.0 3.0]; linsolve=:qr)
    @test_throws "linsolve must be :auto, :dense, :lsqr, or :woodbury" cover_min(AbsLog{2}(), [2.0 1.0; 1.0 3.0]; linsolve=:qr)

    # The matrix-free LSQR path reproduces the dense path and the HiGHS reference
    # across the committed symmetric library, and returns a feasible cover.
    for (_, A) in symmetric_matrices
        Af = Float64.(A)
        a  = symcover_min(AbsLog{2}(), Af; linsolve=:lsqr)
        aj = MatrixCovers.symcover_min_jump(AbsLog{2}(), Af)
        @test iscover(a, Af; atol=1e-8)
        @test cover_objective(AbsLog{2}(), a, Af) <=
              cover_objective(AbsLog{2}(), aj, Af) * (1 + 1e-6) + 1e-10
    end

    # Asymmetric LSQR path: feasible and matching HiGHS on a deterministic
    # stride subsample of the committed general library.
    for (_, A) in general_matrices[firstindex(general_matrices):43:lastindex(general_matrices)]
        Af = Float64.(A)
        a, b = cover_min(AbsLog{2}(), Af; linsolve=:lsqr)
        @test iscover(a, b, Af; atol=1e-7)
        aj, bj = MatrixCovers.cover_min_jump(AbsLog{2}(), Af)
        @test cover_objective(AbsLog{2}(), a, b, Af) <=
              cover_objective(AbsLog{2}(), aj, bj, Af) * (1 + 1e-6) + 1e-10
    end

    # Gauge/edge cases the dense path handles via a ridge or v0*v0ᵀ must also work
    # matrix-free: bipartite support, a scalar, and a zero row/column.
    a = symcover_min(AbsLog{2}(), [0.0 1.0; 1.0 0.0]; linsolve=:lsqr)
    @test a[1] * a[2] ≈ 1.0
    @test symcover_min(AbsLog{2}(), reshape([4.0], 1, 1); linsolve=:lsqr) ≈ [2.0]
    Az = [1.0 0.0 2.0; 0.0 0.0 0.0; 2.0 0.0 3.0]
    a = symcover_min(AbsLog{2}(), Az; linsolve=:lsqr)
    @test a[2] == 0.0
    @test iscover(a, Az; atol=1e-8)
    a, b = cover_min(AbsLog{2}(), [0.0 1.0; 1.0 0.0]; linsolve=:lsqr)
    @test a[1] * b[2] ≈ 1.0
    @test a[2] * b[1] ≈ 1.0
end

# Dense and Woodbury solve the same regularized equations. Factor comparisons use
# `sqrt(eps)` because the objective is more tightly determined than the factors.
@testset "MMC native AbsLog{2} Woodbury path" begin
    rng = StableRNG(9)
    lognormal(m, n) = exp.(randn(rng, m, n))
    symlognormal(n) = (X = lognormal(n, n); (X .+ X') ./ 2)

    @testset "symmetric, n = $n" for n in (6, 30, 120)
        A = symlognormal(n)
        # One zero per row satisfies both Woodbury sparsity guards.
        Z = symlognormal(n)
        for k in 1:(n ÷ 2)
            Z[2k-1, 2k] = 0.0
            Z[2k, 2k-1] = 0.0
        end
        variants = ["all nonzero" => A,
                    "zero diagonal" => A - Diagonal(A),
                    "paired zeros" => Z]
        # Diagonal and off-diagonal zeros give two per row.
        n ÷ 4 >= 2 && push!(variants, "zero diagonal and paired zeros" => Z - Diagonal(Z))
        for (name, M) in variants
            @testset "$name" begin
                ad, sd = MatrixCovers._symcover_min_abslog2(M; linsolve=:dense)
                aw, sw = MatrixCovers._symcover_min_abslog2(M; linsolve=:woodbury)
                aa, sa = MatrixCovers._symcover_min_abslog2(M)
                @test sd.linsolve === :dense
                @test sw.linsolve === :woodbury
                @test sa.linsolve === :woodbury
                @test aw ≈ ad rtol=1e-7
                @test aa == aw
                @test iscover(aw, M; atol=1e-8)
                # Exercise both CG and factorized Woodbury solves.
                @test sw.cgiters > 0
                @test sw.cholsolves > 0
                @test sd.cgiters == 0
                @test sd.cholsolves == 0
            end
        end
    end

    @testset "asymmetric, ($m, $n)" for (m, n) in ((8, 6), (30, 22), (120, 90))
        A = lognormal(m, n)
        # The gauge direction (e; −e) leaves every product a[i]*b[j] fixed, so only the
        # products are required to agree between the two solves.
        k = min(m, n) ÷ 4   # the largest zero band the guard admits
        for (name, M) in ("all nonzero" => A, "one zero band" => (B = copy(A); B[1:k, 1] .= 0.0; B))
            @testset "$name" begin
                ad, bd, sd = MatrixCovers._cover_min_abslog2(M; linsolve=:dense)
                aw, bw, sw = MatrixCovers._cover_min_abslog2(M; linsolve=:woodbury)
                _, _, sa = MatrixCovers._cover_min_abslog2(M)
                @test sd.linsolve === :dense
                @test sw.linsolve === :woodbury
                @test sa.linsolve === :woodbury
                @test aw .* bw' ≈ ad .* bd' rtol=1e-7
                @test iscover(aw, bw, M; atol=1e-7)
                @test sw.cgiters > 0
                @test sw.cholsolves > 0
                @test sd.cgiters == 0
                @test sd.cholsolves == 0
            end
        end
    end

    # Only abs.(A) is read, so a complex Hermitian takes the same path and lands on
    # the same cover as its magnitude matrix.
    n = 8
    M = randn(rng, ComplexF64, n, n)
    H = Hermitian(M + M')
    aw, sw = MatrixCovers._symcover_min_abslog2(H; linsolve=:woodbury)
    @test sw.linsolve === :woodbury
    @test aw ≈ MatrixCovers._symcover_min_abslog2(abs.(Matrix(H)); linsolve=:dense)[1] rtol=1e-7

    # Offset axes and views index the support through `axes(A)`, not `1:n`, on this
    # path as on the others.
    A = symlognormal(12)
    aref = symcover_min(AbsLog{2}(), A; linsolve=:woodbury)
    Ao = OffsetArray(A, -3, -3)
    ao = symcover_min(AbsLog{2}(), Ao; linsolve=:woodbury)
    @test axes(ao, 1) == axes(Ao, 1)
    @test collect(ao) ≈ aref rtol=1e-10
    Av = view(symlognormal(16), 3:14, 3:14)
    @test symcover_min(AbsLog{2}(), Matrix(Av); linsolve=:woodbury) ≈
          symcover_min(AbsLog{2}(), Av; linsolve=:woodbury) rtol=1e-10
    Ag = lognormal(14, 10)
    Agv = view(Ag, 2:13, 2:9)
    av, bv = cover_min(AbsLog{2}(), Agv; linsolve=:woodbury)
    am, bm = cover_min(AbsLog{2}(), Matrix(Agv); linsolve=:woodbury)
    @test axes(av, 1) == axes(Agv, 1)
    @test axes(bv, 1) == axes(Agv, 2)
    @test av .* bv' ≈ am .* bm' rtol=1e-10
    Ago = OffsetArray(Ag, -2, 4)
    ago, bgo = cover_min(AbsLog{2}(), Ago; linsolve=:woodbury)
    agm, bgm = cover_min(AbsLog{2}(), Ag; linsolve=:woodbury)
    @test axes(ago, 1) == axes(Ago, 1)
    @test axes(bgo, 1) == axes(Ago, 2)
    @test collect(ago) .* collect(bgo)' ≈ agm .* bgm' rtol=1e-10

    # A single stage at κ = 1e2 stays inside the conjugate-gradient regime throughout,
    # so the factorization is never reached.
    A1 = symlognormal(24)
    c1, s1 = MatrixCovers._symcover_min_abslog2(A1; κs=(1e2,), linsolve=:woodbury)
    @test s1.cgiters > 0
    @test s1.cholsolves == 0
    @test c1 ≈ MatrixCovers._symcover_min_abslog2(A1; κs=(1e2,), linsolve=:dense)[1] rtol=1e-8
    G1 = lognormal(24, 18)
    p1, q1, t1 = MatrixCovers._cover_min_abslog2(G1; κs=(1e2,), linsolve=:woodbury)
    pd1, qd1, _ = MatrixCovers._cover_min_abslog2(G1; κs=(1e2,), linsolve=:dense)
    @test t1.cgiters > 0
    @test t1.cholsolves == 0
    @test p1 .* q1' ≈ pd1 .* qd1' rtol=1e-8

    # No row may carry more than a quarter zeros — that is what keeps `C` positive
    # definite — and the arithmetic must be Float64.
    holey = symlognormal(12)
    holey[1, 1:5] .= 0.0
    holey[1:5, 1] .= 0.0
    @test_throws "at most n ÷ 4 = 3 zeros; got 5" symcover_min(AbsLog{2}(), holey; linsolve=:woodbury)
    @test MatrixCovers._symcover_min_abslog2(holey)[2].linsolve === :dense
    gholey = lognormal(12, 12)
    gholey[1, 1:5] .= 0.0
    @test_throws "at most min(m, n) ÷ 4 = 3 zeros; got 5" cover_min(AbsLog{2}(), gholey; linsolve=:woodbury)
    @test MatrixCovers._cover_min_abslog2(gholey)[3].linsolve === :dense

    # A support thin enough per row can still carry a quadratic number of zeros, which
    # would make the "sparse" correction dense work; the total budget rejects it.
    wide = symlognormal(40)
    for i in 1:40, j in 1:40
        (i != j && (i + j) % 4 == 0) && (wide[i, j] = 0.0)
    end
    @test maximum(count(iszero, wide; dims=2)) <= 40 ÷ 4
    @test count(iszero, wide) > 4 * 40
    @test_throws "at most 4n = 160 zeros in total" symcover_min(AbsLog{2}(), wide; linsolve=:woodbury)
    @test MatrixCovers._symcover_min_abslog2(wide)[2].linsolve === :dense
    gwide = lognormal(40, 40)
    for i in 1:40, j in 1:40
        (i + j) % 4 == 0 && (gwide[i, j] = 0.0)
    end
    @test_throws "at most 4·max(m, n) = 160 zeros in total" cover_min(AbsLog{2}(), gwide; linsolve=:woodbury)
    @test MatrixCovers._cover_min_abslog2(gwide)[3].linsolve === :dense

    # A working type narrower than Float64 is promoted, so it reaches the Woodbury
    # path; a wider one keeps its precision and is refused by the CHOLMOD-backed solve.
    A32 = Float32.(symlognormal(8))
    a32, s32 = MatrixCovers._symcover_min_abslog2(A32)
    @test s32.linsolve === :woodbury && eltype(a32) === Float32
    @test symcover_min(AbsLog{2}(), A32; linsolve=:woodbury) ≈ symcover_min(AbsLog{2}(), A32; linsolve=:dense) rtol=1e-6
    Abig = BigFloat.(symlognormal(8))
    @test_throws "requires Float64 arithmetic" symcover_min(AbsLog{2}(), Abig; linsolve=:woodbury)
    @test MatrixCovers._symcover_min_abslog2(Abig)[2].linsolve === :dense
    G32 = Float32.(lognormal(8, 6))
    @test MatrixCovers._cover_min_abslog2(G32)[3].linsolve === :woodbury
    Gbig = BigFloat.(lognormal(8, 6))
    @test_throws "requires Float64 arithmetic" cover_min(AbsLog{2}(), Gbig; linsolve=:woodbury)
    @test MatrixCovers._cover_min_abslog2(Gbig)[3].linsolve === :dense
end

@testset "MMC :auto measures the stored support" begin
    T3 = SymTridiagonal(fill(4.0, 30), fill(1.0, 29))
    aT, sT = MatrixCovers._symcover_min_abslog2(T3)
    @test sT.linsolve === :lsqr
    @test aT ≈ MatrixCovers._symcover_min_abslog2(Matrix(T3); linsolve=:dense)[1] rtol=1e-6
    @test iscover(aT, T3; rtol=1e-8)

    # Wrapped sparse support selects LSQR.
    rng = StableRNG(3)
    S0 = sprand(rng, 40, 40, 0.05)
    Sw = Symmetric(S0 + S0' + I)
    aw, bw, stw = MatrixCovers._cover_min_abslog2(Sw)
    @test stw.linsolve === :lsqr
    @test iscover(aw, bw, Sw; rtol=1e-8)

    X8 = exp.(randn(rng, 8, 8))
    @test MatrixCovers._symcover_min_abslog2((X8 .+ X8') ./ 2)[2].linsolve === :woodbury

    # The threshold includes equality.
    D4 = Matrix(Diagonal([4.0, 9.0, 1.0, 16.0]))
    @test MatrixCovers._symcover_min_abslog2(D4)[2].linsolve === :lsqr
    D4[1, 2] = D4[2, 1] = 1.0
    @test MatrixCovers._symcover_min_abslog2(D4)[2].linsolve === :dense
end

# Exact inner solves stop a stage when the violated set is unchanged; LSQR uses
# the decrease test.
@testset "MMC exact paths stop on a sign-stable Newton step" begin
    rng = StableRNG(31)
    A = (X = exp.(randn(rng, 60, 60)); (X .+ X') ./ 2)
    ad, sd = MatrixCovers._symcover_min_abslog2(A; linsolve=:dense)
    aw, sw = MatrixCovers._symcover_min_abslog2(A; linsolve=:woodbury)
    al, sl = MatrixCovers._symcover_min_abslog2(A; linsolve=:lsqr)
    @test ad ≈ al rtol=1e-6
    @test aw ≈ al rtol=1e-6
    @test sd.nsolves == sw.nsolves
    @test sd.nsolves <= 26
    # Exact paths save one solve per continuation stage. The comparison runs both
    # paths on one schedule, because the default is solver-dependent, and from one
    # start, because the LSQR path otherwise supplies its own: only a shared
    # starting iterate leaves the stage-exit rule as the difference between them.
    κs8 = MatrixCovers._kappa_schedule(Float64, false)
    a0 = symcover(A)
    _, sd8 = MatrixCovers._symcover_min_abslog2(A; linsolve=:dense, κs=κs8, start=a0)
    _, sl8 = MatrixCovers._symcover_min_abslog2(A; linsolve=:lsqr, κs=κs8, start=a0)
    @test sd8.nsolves <= sl8.nsolves - length(κs8)

    G = exp.(randn(rng, 60, 45))
    gd, hd, td = MatrixCovers._cover_min_abslog2(G; linsolve=:dense)
    gw, hw, tw = MatrixCovers._cover_min_abslog2(G; linsolve=:woodbury)
    gl, hl, tl = MatrixCovers._cover_min_abslog2(G; linsolve=:lsqr)
    @test gd .* hd' ≈ gl .* hl' rtol=1e-6
    @test gw .* hw' ≈ gl .* hl' rtol=1e-6
    @test td.nsolves == tw.nsolves
    # Leave a small margin in the solve-count bound.
    @test td.nsolves <= 28
    g0, h0 = cover(G)
    _, _, td8 = MatrixCovers._cover_min_abslog2(G; linsolve=:dense, κs=κs8, start=(g0, h0))
    _, _, tl8 = MatrixCovers._cover_min_abslog2(G; linsolve=:lsqr, κs=κs8, start=(g0, h0))
    @test td8.nsolves <= tl8.nsolves - length(κs8)
end

# LSQR continuation starts from the heuristic cover.
@testset "MMC :lsqr continuation starts from the heuristic cover" begin
    rng = StableRNG(17)
    A = (X = exp.(randn(rng, 50, 50)); (X .+ X') ./ 2)
    al, sl = MatrixCovers._symcover_min_abslog2(A; linsolve=:lsqr)
    ah, sh = MatrixCovers._symcover_min_abslog2(A; linsolve=:lsqr, start=symcover(A))
    @test al == ah
    @test (sl.nsolves, sl.lsqriters) == (sh.nsolves, sh.lsqriters)
    ad, _ = MatrixCovers._symcover_min_abslog2(A; linsolve=:dense)
    @test ad ≈ al rtol=1e-6

    G = exp.(randn(rng, 40, 30))
    gl, hl, tl = MatrixCovers._cover_min_abslog2(G; linsolve=:lsqr)
    g0, h0 = cover(G)
    gh, hh, th = MatrixCovers._cover_min_abslog2(G; linsolve=:lsqr, start=(g0, h0))
    @test gl == gh
    @test hl == hh
    @test (tl.nsolves, tl.lsqriters) == (th.nsolves, th.lsqriters)

    # With no stages the unweighted fit is the answer, not a start.
    @test soft_symcover_min(AbsLog{2}(), A; linsolve=:lsqr) != symcover(A)
end

# Bound iterations for Cholesky-preconditioned `Float64` LSQR.
@testset "MMC :lsqr iteration count is bounded across the continuation" begin
    rng = StableRNG(5)
    A = (X = exp.(randn(rng, 120, 120)); (X .+ X') ./ 2)
    ad, _ = MatrixCovers._symcover_min_abslog2(A; linsolve=:dense)
    al, sl = MatrixCovers._symcover_min_abslog2(A; linsolve=:lsqr)
    @test al ≈ ad rtol=1e-6
    # Bound the preconditioned iteration count with margin.
    @test sl.lsqriters <= 12 * sl.nsolves

    G = exp.(randn(rng, 120, 90))
    gd, hd, _ = MatrixCovers._cover_min_abslog2(G; linsolve=:dense)
    gl, hl, tl = MatrixCovers._cover_min_abslog2(G; linsolve=:lsqr)
    @test gl .* hl' ≈ gd .* hd' rtol=1e-6
    # Apply the same iteration bound to asymmetric problems.
    @test tl.lsqriters <= 12 * tl.nsolves

    # The ridge handles bipartite support.
    rngb = StableRNG(11)
    Tsp = sparse(Matrix(SymTridiagonal(exp.(randn(rngb, 40)), exp.(randn(rngb, 39)))))
    at, st = MatrixCovers._symcover_min_abslog2(Tsp; linsolve=:lsqr)
    atd, _ = MatrixCovers._symcover_min_abslog2(Matrix(Tsp); linsolve=:dense)
    @test at ≈ atd rtol=1e-6
    @test iscover(at, at, Tsp)
    @test st.lsqriters <= 12 * st.nsolves

    # A working type CHOLMOD cannot factor keeps the plain matrix-free iteration.
    A32 = Float32.([4.0 1.0 0.5; 1.0 3.0 1.0; 0.5 1.0 2.5])
    a32 = symcover_min(AbsLog{2}(), A32; linsolve=:lsqr)
    @test a32 isa Vector{Float32}
    @test a32 ≈ symcover_min(AbsLog{2}(), A32; linsolve=:dense) rtol=1e-5
    @test iscover(a32, A32; rtol=1e-5)
end

@testset "MMC disconnected-support gauge" begin
    # Dense and LSQR must handle independent gauges on disconnected support.
    singletons(vals) = Matrix(sparse(1:length(vals), 1:length(vals), float.(vals)))  # k singleton components
    block2(k) = cat(([2.0+i i; i 3.0+i] for i in 1:k)...; dims = (1, 2))              # k dense 2×2 components
    for M in (singletons([4.0, 9.0, 1.0]), singletons(1.0:6.0), block2(3), block2(6))
        ad, bd = cover_min(AbsLog{2}(), M; linsolve = :dense)   # ridge lifts the unpinned gauges
        al, bl = cover_min(AbsLog{2}(), M; linsolve = :lsqr)
        @test iscover(ad, bd, M; atol=1e-8)
        @test iscover(al, bl, M; atol=1e-8)
        @test cover_objective(AbsLog{2}(), ad, bd, M) ≈
              cover_objective(AbsLog{2}(), al, bl, M) rtol = 1e-6 atol = 1e-8
    end
    # Diagonal input has one support component per entry.
    D = Diagonal([4.0, 9.0, 1.0])
    a, b = cover_min(AbsLog{2}(), D)
    @test cover_objective(AbsLog{2}(), a, b, Matrix(D)) ≈ 0.0 atol = 1e-10
end

@testset "MMC block-diagonal assembly consistency" begin
    # The balance convention is imposed per connected component, so the (a, b) split
    # is a function of the support and the products alone: covering a block-diagonal
    # matrix must reproduce the factors of covering each block separately.
    rng = StableRNG(23)
    for _ in 1:5
        mB, nB = rand(rng, 2:5), rand(rng, 2:5)
        mC, nC = rand(rng, 2:5), rand(rng, 2:5)
        B = rand(rng, mB, nB) .+ 0.1
        C = rand(rng, mC, nC) .+ 0.1
        A = [B zeros(mB, nC); zeros(mC, nB) C]
        aA, bA = cover_min(AbsLog{2}(), A)
        aB, bB = cover_min(AbsLog{2}(), B)
        aC, bC = cover_min(AbsLog{2}(), C)
        @test aA ≈ vcat(aB, aC) rtol=1e-5
        @test bA ≈ vcat(bB, bC) rtol=1e-5
        @test isbalanced(aA, bA, A)
    end
end

@testset "symcover_min(AbsLog{2}) on complex Hermitian/Symmetric input" begin
    # The cover problem only depends on abs.(A), so a complex Hermitian (real
    # diagonal, conjugate-symmetric off-diagonals) or complex Symmetric matrix
    # must give the same result as symcover_min on the real magnitude matrix.
    rng = StableRNG(42)
    n = 6
    M = randn(rng, ComplexF64, n, n)
    Href = symcover_min(AbsLog{2}(), abs.(Matrix(Hermitian(M + M'))))

    Hdense = Hermitian(M + M')
    @test symcover_min(AbsLog{2}(), Hdense) ≈ Href rtol = 1e-10

    Msp = sprandn(rng, ComplexF64, n, n, 0.5)
    Hsp = Hermitian(sparse(Msp + Msp'))
    @test symcover_min(AbsLog{2}(), Hsp) ≈ symcover_min(AbsLog{2}(), abs.(Matrix(Hsp))) rtol = 1e-7

    # Symmetric{<:Complex} carries no conjugate-symmetry guarantee, but the cover
    # problem still only depends on abs.(A), so the same identity holds.
    Ssp = Symmetric(sparse(Msp + transpose(Msp)))
    @test symcover_min(AbsLog{2}(), Ssp) ≈ symcover_min(AbsLog{2}(), abs.(Matrix(Ssp))) rtol = 1e-7

    # Real input is unaffected by deriving T from real(eltype(A)).
    Ar = [4.0 1.0; 1.0 4.0]
    @test symcover_min(AbsLog{2}(), Ar) ≈ [2.0, 2.0]
end

@testset "soft_cover_min native AbsLog{2}" begin
    # On dense support, the geometric mean equals the minimum up to roundoff.
    A = [1.0 2.0 3.0; 6.0 5.0 4.0]
    a, b = soft_cover_min(AbsLog{2}(), A)
    a_ref, b_ref = similar(a), similar(b)
    MatrixCovers.unconstrained_min!(AbsLog{2}(), a_ref, b_ref, A)
    @test a ≈ a_ref && b ≈ b_ref

    # It's the unconstrained minimum: any perturbation can only raise the objective.
    obj0 = cover_objective(AbsLog{2}(), a, b, A)
    rng = StableRNG(42)
    for _ in 1:20
        ap = a .* exp.(0.1 .* randn(rng, length(a)))
        bp = b .* exp.(0.1 .* randn(rng, length(b)))
        @test cover_objective(AbsLog{2}(), ap, bp, A) >= obj0 - 1e-10
    end

    # Non-1-based axes propagate from A, not from 1:n.
    Ao = OffsetArray(A, 10, 20)
    ao, bo = soft_cover_min(AbsLog{2}(), Ao)
    @test axes(ao, 1) == axes(Ao, 1)
    @test axes(bo, 1) == axes(Ao, 2)
    @test collect(ao) == a && collect(bo) == b
end

@testset "no-ϕ convenience methods for the *_min family" begin
    # symcover_min(A) and cover_min(A) default to AbsLog{2}, matching
    # symcover(A)/cover(A). Solved natively, so no extension is needed.
    A = [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0]
    @test symcover_min(A) == symcover_min(AbsLog{2}(), A)
    Aasym = [1.0 2.0 3.0; 4.0 5.0 6.0]
    @test cover_min(Aasym) == cover_min(AbsLog{2}(), Aasym)

    # soft_symcover_min(A) defaults to AbsLinear{2}, matching soft_symcover(A);
    # requires JuMP+Ipopt.
    @test soft_symcover_min(A) == soft_symcover_min(AbsLinear{2}(), A)

    # soft_cover_min(A) also defaults to AbsLinear{2}, matching soft_cover(A);
    # requires JuMP+Ipopt.
    @test soft_cover_min(Aasym) == soft_cover_min(AbsLinear{2}(), Aasym)
end

@testset "symcover_min!/cover_min! native AbsLog{2}" begin
    A = [4.0 1.0 0.0; 1.0 9.0 2.0; 0.0 2.0 16.0]
    Aasym = [4.0 1.0 0.0; 1.0 9.0 2.0]

    # AbsLog{2} has a unique minimizer, so refining any valid start reproduces the
    # cold solve exactly, and the no-ϕ form selects the same penalty.
    a = initialize_symcover(A)
    @test symcover_min!(AbsLog{2}(), copy(a), A) ≈ symcover_min(AbsLog{2}(), A)
    @test symcover_min!(copy(a), A) == symcover_min!(AbsLog{2}(), copy(a), A)
    ab, bb = initialize_cover(Aasym)
    ra, rb = cover_min!(AbsLog{2}(), copy(ab), copy(bb), Aasym)
    ca, cb = cover_min(AbsLog{2}(), Aasym)
    @test ra ≈ ca && rb ≈ cb
    @test cover_min!(copy(ab), copy(bb), Aasym) == (ra, rb)

    # The refiners write through the buffers they are handed.
    a = initialize_symcover(A)
    @test symcover_min!(AbsLog{2}(), a, A) === a
    @test iscover(a, A; atol=1e-9)

    # The heuristics land on the coverage boundary only to within the roundoff of
    # their log-domain updates, so their output must be accepted as a start.
    @test symcover_min!(AbsLog{2}(), symcover(A), A) ≈ symcover_min(AbsLog{2}(), A)
    @test cover_min!(AbsLog{2}(), cover(Aasym)..., Aasym)[1] ≈ ca

    # Every start is read only up to the gauge a -> c*a, b -> b/c, which leaves
    # every product a[i]*b[j] fixed.
    ga, gb = cover_min!(AbsLog{2}(), 8 .* ab, bb ./ 8, Aasym)
    @test ga ≈ ra && gb ≈ rb

    # Scales on rows carrying no support are inert: ignored on input, zero on output.
    Anosupp = [4.0 1.0 0.0; 1.0 9.0 0.0; 0.0 0.0 0.0]
    a = initialize_symcover(Anosupp)
    a[3] = 7.0
    @test symcover_min!(AbsLog{2}(), a, Anosupp)[3] == 0.0

    # A start must cover A, and be positive wherever A carries support.
    @test_throws "requires a start that covers `A`" symcover_min!(AbsLog{2}(), fill(0.1, 3), A)
    @test_throws "requires a start that covers `A`" symcover_min!(AbsLog{2}(), symcover(A) .* (1 - 1e-8), A)
    @test_throws "finite positive scale on every supported row" symcover_min!(AbsLog{2}(), [100.0, 0.0, 100.0], A)
    @test_throws "requires a start that covers `A`" cover_min!(AbsLog{2}(), fill(0.1, 2), fill(0.1, 3), Aasym)
    @test_throws "finite positive scale on every supported column" cover_min!(AbsLog{2}(), fill(100.0, 2), [100.0, 0.0, 100.0], Aasym)
    @test_throws DimensionMismatch symcover_min!(AbsLog{2}(), zeros(2), A)
    @test_throws DimensionMismatch cover_min!(AbsLog{2}(), zeros(2), zeros(2), Aasym)

    # Offset axes propagate through the start and the result.
    Ao = OffsetArray(A, -1, -1)
    ao = initialize_symcover(Ao)
    @test symcover_min!(AbsLog{2}(), ao, Ao) === ao
    @test axes(ao, 1) == axes(Ao, 1)
    @test collect(ao) ≈ symcover_min(AbsLog{2}(), A)
end

@testset "quality vs optimal (testmatrices)" begin
    sym_ratios = Float64[]
    for (_, A) in symmetric_matrices
        Af = Float64.(A)
        # Initialization should give a valid cover
        a0 = symcover(AbsLog{2}(), Af; maxiter=0)
        @test iscover(a0, Af; atol=1e-12)
        a0 = symcover(AbsLog{2}(), Af / 100; maxiter=0)
        @test iscover(a0, Af / 100; atol=1e-12)
        # Covers are nearly quadratically optimal
        qopt  = cover_objective(AbsLog{2}(), symcover_min(AbsLog{2}(), Af), Af)
        qfast = cover_objective(AbsLog{2}(), symcover(AbsLog{2}(), Af; maxiter=10), Af)
        iszero(qopt) || push!(sym_ratios, qfast / qopt)
    end
    @test median(sym_ratios) < 1.02

    gen_ratios = Float64[]
    for (_, A) in general_matrices
        Af = Float64.(A)
        a0, b0 = cover(AbsLog{2}(), Af; maxiter=0)
        @test iscover(a0, b0, Af; atol=1e-12)
        a0, b0 = cover(AbsLog{2}(), Af / 100; maxiter=0)
        @test iscover(a0, b0, Af / 100; atol=1e-12)
        qopt  = cover_objective(AbsLog{2}(), cover_min(AbsLog{2}(), Af)..., Af)
        qfast = cover_objective(AbsLog{2}(), cover(AbsLog{2}(), Af; maxiter=10)..., Af)
        iszero(qopt) || push!(gen_ratios, qfast / qopt)
    end
    @test median(gen_ratios) < 1.02
end

# LSQR allocation should scale with support rather than matrix area.
@testset ":lsqr allocates in proportion to the support" begin
    function lsqr_alloc(n)
        rng = StableRNG(4)
        B = sprandn(rng, n, n, 3.5 / n)
        A = Symmetric(B + transpose(B))
        MatrixCovers._symcover_min_abslog2(A; linsolve=:lsqr)   # compile before measuring
        return @allocated MatrixCovers._symcover_min_abslog2(A; linsolve=:lsqr)
    end
    small, large = lsqr_alloc(200), lsqr_alloc(800)
    # Allow iteration growth while rejecting an added dense workspace.
    @test large < 10 * small
end

@testset "MMC continuation reports how each stage ended" begin
    rng = StableRNG(17)
    A = (X = exp.(randn(rng, 40, 40)); (X .+ X') ./ 2)
    _, s = MatrixCovers._symcover_min_abslog2(A)
    @test length(s.exits) == 8
    @test length(s.stagedrops) == length(s.exits)
    @test all(in((:stable, :decrease, :maxiter)), s.exits)
    # With room to converge, no stage runs out of Newton steps.
    @test all(!=(:maxiter), s.exits)

    # A one-step limit truncates every `:lsqr` stage and emits a warning.
    @test_logs (:warn, r"reached maxiter=1") match_mode=:any begin
        _, sl = MatrixCovers._symcover_min_abslog2(A; maxiter=1, linsolve=:lsqr)
        @test all(==(:maxiter), sl.exits)
    end

    G = exp.(randn(rng, 40, 30))
    _, _, t = MatrixCovers._cover_min_abslog2(G)
    @test length(t.exits) == 8
    @test all(!=(:maxiter), t.exits)

    # Exact solves default to eight stages; `:lsqr` defaults to four.
    _, sl = MatrixCovers._symcover_min_abslog2(A; linsolve=:lsqr)
    @test length(sl.exits) == 4
    # Check the default `Float64` LSQR schedule exactly.
    @test MatrixCovers._kappa_schedule(Float64, true) === (1e2, 1e4, 1e6, 1e8)
    @test length(MatrixCovers._kappa_schedule(Float64, false)) == 8
    @test MatrixCovers._kappa_schedule(Float64, false)[1] == 1e2
    @test MatrixCovers._kappa_schedule(Float64, false)[end] == 1e8
    @test issorted(MatrixCovers._kappa_schedule(Float64, false))
    # The schedule uses the working precision.
    @test eltype(MatrixCovers._kappa_schedule(BigFloat, false)) === BigFloat
    @test eltype(MatrixCovers._kappa_schedule(BigFloat, true)) === BigFloat

    # Explicit schedules override the defaults.
    _, s4 = MatrixCovers._symcover_min_abslog2(A; κs=(1e2, 1e4, 1e6, 1e8))
    @test length(s4.exits) == 4
    _, s4l = MatrixCovers._symcover_min_abslog2(A; linsolve=:lsqr, κs=10 .^ range(2, 8, length=8))
    @test length(s4l.exits) == 8
end
