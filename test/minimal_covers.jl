# The *_min family: native AbsLog{2} minimal-cover solvers and their edge cases.

# Committed 5x5 matrix libraries (`symmetric_matrices`, `general_matrices`); the guard
# permits re-inclusion of this file in an already-initialized session.
if !isdefined(@__MODULE__, :symmetric_matrices)
    include("testmatrices.jl")
end

@testset "symcover_min native AbsLog{2}" begin
    # Non-square rejected.
    @test_throws "symcover_min requires a square matrix" symcover_min(AbsLog{2}(), [1.0 2.0; 3.0 4.0; 5.0 6.0])

    # Native solver matches the HiGHS reference in objective across the whole
    # committed symmetric library, and returns a feasible cover.
    for (_, A) in symmetric_matrices
        Af = Float64.(A)
        a  = symcover_min(AbsLog{2}(), Af)
        aj = ScaleInvariantAnalysis.symcover_min_jump(AbsLog{2}(), Af)
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
    # Native solver returns a feasible cover across the whole committed general
    # library, and matches the HiGHS reference in objective on a deterministic
    # subsample (the full 4367-matrix JuMP cross-check is slow).
    idx_sub = Set(round.(Int, range(1, length(general_matrices), length=500)))
    for (k, (_, A)) in enumerate(general_matrices)
        Af = Float64.(A)
        a, b = cover_min(AbsLog{2}(), Af)
        @test iscover(a, b, Af; atol=1e-7)
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

@testset "MCM native AbsLog{2} matrix-free LSQR path" begin
    # Invalid solver selection is rejected.
    @test_throws "linsolve must be :auto, :dense, or :lsqr" symcover_min(AbsLog{2}(), [2.0 1.0; 1.0 3.0]; linsolve=:qr)
    @test_throws "linsolve must be :auto, :dense, or :lsqr" cover_min(AbsLog{2}(), [2.0 1.0; 1.0 3.0]; linsolve=:qr)

    # The matrix-free LSQR path reproduces the dense path and the HiGHS reference
    # across the committed symmetric library, and returns a feasible cover.
    for (_, A) in symmetric_matrices
        Af = Float64.(A)
        a  = symcover_min(AbsLog{2}(), Af; linsolve=:lsqr)
        aj = ScaleInvariantAnalysis.symcover_min_jump(AbsLog{2}(), Af)
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
        aj, bj = ScaleInvariantAnalysis.cover_min_jump(AbsLog{2}(), Af)
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
        @test iscover(ad, bd, M; atol=1e-8)
        @test iscover(al, bl, M; atol=1e-8)
        @test cover_objective(AbsLog{2}(), ad, bd, M) ≈
              cover_objective(AbsLog{2}(), al, bl, M) rtol = 1e-6 atol = 1e-8
    end
    # The canonical failure mode: `cover_min` on a Diagonal (n singleton components).
    D = Diagonal([4.0, 9.0, 1.0])
    a, b = cover_min(AbsLog{2}(), D)
    @test cover_objective(AbsLog{2}(), a, b, Matrix(D)) ≈ 0.0 atol = 1e-10
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
    # `A` has no zero entry, the case in which the geometric mean coincides with the
    # minimum; the two compute it differently, so they agree to roundoff, not bitwise.
    # On a sparse support they part company -- see the oracle in `test/soft_covers.jl`.
    A = [1.0 2.0 3.0; 6.0 5.0 4.0]
    a, b = soft_cover_min(AbsLog{2}(), A)
    a_ref, b_ref = similar(a), similar(b)
    ScaleInvariantAnalysis.unconstrained_min!(AbsLog{2}(), a_ref, b_ref, A)
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
