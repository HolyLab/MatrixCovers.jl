# The fast heuristic covers: symcover/cover, their in-place forms, and the
# feasibility boost.

@testset "symcover" begin
    # Cover property: a[i]*a[j] >= abs(A[i,j]) for all i, j
    for A in ([2.0 1.0; 1.0 3.0], [1.0 -0.2; -0.2 0.0], [1.0 0.0; 0.0 0.0],
              [100.0 1.0; 1.0 0.01])
        for ϕ in PENALTIES
            a = symcover(ϕ, A)
            @test iscover(a, A; rtol=8eps())
        end
        # Default dispatch
        a = symcover(A)
        @test iscover(a, A; rtol=8eps())
    end
    # All-zero row or column gives zero cover element
    a = symcover([1.0 0; 0 0])
    @test a[2] == 0
    a = symcover([0 0; 0 1.0])
    @test a[1] == 0
    # All-zero diagonals
    A = [0.0 1.0; 1.0 0.0]
    @test symcover(A) == [1.0, 1.0]
    # Diagonal scaling covariance
    A = [2.0 1.0; 1.0 3.0]
    d = [2.0, 0.5]
    for ϕ in PENALTIES
        @test covaries(A -> symcover(ϕ, A), A, d)
    end
    # Non-square input is rejected, with the message naming the called function
    @test_throws "symcover requires a square matrix" symcover([1.0 2.0; 3.0 4.0; 5.0 6.0])
    @test_throws "symcover! requires a square matrix" symcover!(zeros(3), [1.0 2.0; 3.0 4.0; 5.0 6.0])
end

@testset "symcover does not currently consult ϕ" begin
    # The current heuristic accepts but ignores the penalty.
    rng = StableRNG(1)
    for n in (2, 5, 40)
        B = randn(rng, n, n); A = (B + B') / 2
        a = symcover(A)
        for ϕ in PENALTIES
            @test symcover(ϕ, A) == a
        end
    end

    # The first argument must still be a penalty.
    A = [4.0 1.5; 1.5 1.0]
    a = symcover(A)
    b = copy(a)
    for bad in (42, "nope", :whatever)
        @test_throws MethodError symcover(bad, A)
        @test_throws MethodError symcover!(bad, copy(a), A)
        @test_throws MethodError cover(bad, A)
        @test_throws MethodError cover!(bad, copy(a), copy(b), A)
    end
end

@testset "symcover with unequal row degrees" begin
    # Rows with differing numbers of nonzeros must not destabilize the initialization.
    # Sparse matrices of this size and density are the stressing case: the returned cover
    # must be finite and feasible, never NaN/Inf.
    for (n, seed) in ((20, 44), (60, 21))
        rng = StableRNG(seed)
        M = randn(rng, n, n) .* (rand(rng, n, n) .< 0.3)
        A = Matrix(Symmetric(M))
        a = symcover(A)
        @test all(isfinite, a)
        @test iscover(a, A; rtol=1e-9)
    end
    # Arrow matrix: one dense row/column, everything else diagonal (degrees 2,…,2,n).
    n = 6
    A = Matrix(Diagonal(fill(2.0, n))); A[1, :] .= 1.0; A[:, 1] .= 1.0; A[1, 1] = 2.0
    a = symcover(A)
    @test all(isfinite, a)
    @test iscover(a, A; rtol=1e-9)
end

@testset "cover" begin
    # Cover property: a[i]*b[j] >= abs(A[i,j]) for all i, j
    for A in ([2.0 1.0; 1.0 3.0], [0.0 1.0; -2.0 0.0], [1.0 0.0; 0.0 0.0],
              [100.0 1.0; 1.0 0.01])
        for ϕ in PENALTIES
            a, b = cover(ϕ, A)
            @test iscover(a, b, A; rtol=8eps())
        end
        # Default dispatch
        a, b = cover(A)
        @test iscover(a, b, A; rtol=8eps())
    end
    # All-zero row or column gives zero cover element
    a, b = cover([1.0 0; 0 0])
    @test b[2] == 0
    a, b = cover([0 0; 0 1.0])
    @test a[1] == 0
    # Zero-diagonal matrix
    A = [0.0 1.0; -1.0 0.0]
    for ϕ in PENALTIES
        a, b = cover(ϕ, A)
        @test iscover(a, b, A; rtol=8eps())
    end
    # Rectangular matrix
    A = [1.0 2.0 3.0; 4.0 5.0 6.0]
    a, b = cover(A)
    @test iscover(a, b, A; rtol=8eps())
    # Diagonal scaling covariance: cover(A .* dr .* dc') is cover(A) scaled by dr, dc up to a scalar
    A = [2.0 1.0; 1.0 3.0]
    dr, dc = [2.0, 0.5], [3.0, 0.25]
    for ϕ in PENALTIES
        @test covaries(A -> cover(ϕ, A), A, dr, dc)
    end
end

@testset "symcover! and cover!" begin
    A = [2.0 1.0; 1.0 3.0]
    a = symcover(A)
    abuf = similar(a)
    @test symcover!(abuf, A) === abuf
    @test abuf == a

    B = [1.0 2.0 3.0; 4.0 5.0 6.0]
    aB, bB = cover(B)
    abuf2, bbuf2 = similar(aB), similar(bB)
    r = cover!(abuf2, bbuf2, B)
    @test r === (abuf2, bbuf2)
    @test abuf2 == aB && bbuf2 == bB

    # The bang forms take the same ϕ as the allocating ones, and likewise ignore it.
    for ϕ in PENALTIES
        @test symcover!(ϕ, similar(a), A) == a
        @test cover!(ϕ, similar(aB), similar(bB), B) == (aB, bB)
    end

    # Adjoint/Transpose wrappers match the allocating forms.
    aT, bT = cover(B')
    abufT, bbufT = similar(aT), similar(bT)
    cover!(abufT, bbufT, B')
    @test abufT == aT && bbufT == bT

    # Buffer axes must match A's axes, including non-1-based indexing.
    Ao = OffsetArray(A, 0:1, 0:1)
    abufo = OffsetArray(similar(a), 0:1)
    symcover!(abufo, Ao)
    @test collect(abufo) == a

    @test_throws "indices of `a` must match the indexing of `A`" symcover!(zeros(3), A)
    @test_throws "indices of `b` must match column-indexing of `A`" cover!(zeros(2), zeros(2), B)
    @test_throws "indices of `b` must match column-indexing of `A`" cover!(zeros(2), zeros(4), B)
end

@testset "feasibility boost" begin
    rng = StableRNG(42)

    # Feasibility on randomized dense (with zero rows/diagonal), sparse, banded, and
    # offset-axes inputs.
    for n in (5, 12)
        for _ in 1:5
            B = randn(rng, n, n); A = (B + B') / 2
            A[1, :] .= 0; A[:, 1] .= 0   # zero row/column
            A[2, 2] = 0                  # zero diagonal entry
            a = symcover(AbsLog{2}(), A)
            @test iscover(a, A; rtol=4eps())
            a2, b2 = cover(AbsLog{2}(), A)
            @test iscover(a2, b2, A; rtol=4eps())
        end
    end
    for _ in 1:5
        S = sprandn(rng, 10, 10, 0.3); A = S + S'
        a = symcover(AbsLog{2}(), A)
        @test iscover(a, Matrix(A); rtol=4eps())
    end
    for _ in 1:5
        dv, ev = randn(rng, 8), randn(rng, 7)
        A = SymTridiagonal(dv, ev)
        a = symcover(AbsLog{2}(), A)
        @test iscover(a, Matrix(A); rtol=4eps())
    end
    let B = randn(rng, 6, 6), Asym = (B + B') / 2
        Ao = OffsetArray(Asym, -3:2, -3:2)
        a = symcover(AbsLog{2}(), Ao)
        @test axes(a, 1) == axes(Ao, 1)
        @test iscover(a, Ao; rtol=4eps())
    end

    # Scale-covariance of the boosted (untightened) cover under diagonal/row-col rescaling.
    n = 8
    B = randn(rng, n, n); A = (B + B') / 2
    d = exp.(randn(rng, n))
    @test covaries(A -> symcover(AbsLog{2}(), A; maxiter=0), A, d; rtol=1e-10)

    m = 6
    Ag = randn(rng, n, m)
    dr, dc = exp.(randn(rng, n)), exp.(randn(rng, m))
    @test covaries(A -> cover(AbsLog{2}(), A; maxiter=0), Ag, dr, dc; rtol=1e-10)

    # Bound the median objective gap across the fixed corpus.
    qrng = StableRNG(20260708)
    gaps = Float64[]
    for _ in 1:15
        n = rand(qrng, 5:15)
        B = randn(qrng, n, n) .* exp.(rand(qrng) * 3 * randn(qrng, n, n))
        A = (B + B') / 2
        Emin = cover_objective(AbsLog{2}(), symcover_min(AbsLog{2}(), A), A)
        E3   = cover_objective(AbsLog{2}(), symcover(AbsLog{2}(), A; maxiter=3), A)
        iszero(Emin) || push!(gaps, log(E3 / Emin))
    end
    for _ in 1:10
        n = rand(qrng, 5:15)
        S = sprand(qrng, n, n, 0.3); A = Matrix(S + S')
        Emin = cover_objective(AbsLog{2}(), symcover_min(AbsLog{2}(), A), A)
        E3   = cover_objective(AbsLog{2}(), symcover(AbsLog{2}(), A; maxiter=3), A)
        iszero(Emin) || push!(gaps, log(E3 / Emin))
    end
    @test median(gaps) < 0.0286 * 1.5
end

@testset "permutation equivariance" begin
    rng = StableRNG(2026)
    # `:geomean` and the simultaneous boost depend on the support only through
    # sums and maxima, so permuting rows and columns permutes the result.
    # Sizes above and below `MatrixCovers.DENSE_GRID_MIN` exercise both the
    # dense-grid and the flat-support paths.
    for A in (sprandn(rng, 100, 100, 0.05), randn(rng, 80, 80), randn(rng, 12, 12),
              Float64.(rand(rng, 1:4, 40, 40)), Float64.(rand(rng, 1:4, 90, 90)))
        p = randperm(rng, size(A, 1))
        q = randperm(rng, size(A, 2))
        a, b = cover(A; start=:geomean)
        ap, bp = cover(A[p, q]; start=:geomean)
        @test ap ≈ a[p] rtol=1e-10
        @test bp ≈ b[q] rtol=1e-10
    end

    # `symcover` under a simultaneous row/column permutation.
    n = 60
    for A in (let B = randn(rng, n, n); (B + B') / 2 end,
              let S = sprandn(rng, n, n, 0.08); Matrix(S + S') end,
              let M = Float64.(rand(rng, 1:4, n, n)); (M + M') / 2 end,
              let B = randn(rng, n, n); C = (B + B') / 2; C[3, 3] = C[7, 7] = 0; C end)
        p = randperm(rng, n)
        a = symcover(A)
        @test symcover(A[p, p]) ≈ a[p] rtol=1e-10
    end
end

@testset "boost feasibility: dominant off-diagonal bands and extreme dynamic range" begin
    # An off-diagonal band far larger than the diagonal: the boost must raise both
    # endpoints of each band entry, not just the diagonal it started from.
    a = symcover(AbsLog{2}(), SymTridiagonal([0.01, 0.01, 0.01], [100.0, 100.0]); maxiter=0)
    @test a[1] * a[2] >= 100 * (1 - 8eps())
    @test a[2] * a[3] >= 100 * (1 - 8eps())
    T40 = Tridiagonal(fill(50.0, 39), fill(0.01, 40), fill(50.0, 39))
    a = symcover(AbsLog{2}(), T40; maxiter=0)
    M = Matrix(T40)
    @test iscover(a, M; rtol=8eps())

    # Float32 range where linear-domain deficit ratios overflow.
    A32 = fill(1f-35, 6, 6); A32[1, 2] = A32[2, 1] = 3f37
    a32 = symcover(AbsLog{2}(), A32)
    @test all(isfinite, a32)
    @test iscover(a32, A32; rtol=64eps(Float32))

    # Float64 range where the geometric-mean init underflows without clamping.
    A = fill(1e308, 6, 6); A[1, :] .= 0; A[:, 1] .= 0; A[1, 2] = A[2, 1] = 1e-308
    a = symcover(AbsLog{2}(), A)
    @test all(isfinite, a)
    @test iscover(a, A; rtol=8eps())
end

@testset "tighten_cover! leaves zero-product scales unchanged" begin
    a, b = tighten_cover!(zeros(3), zeros(3), Diagonal([1.0, 2.0, 3.0]))
    @test all(iszero, a) && all(iszero, b)
end

@testset "symcover scale covariance on irregular support" begin
    # The unconstrained start references each row to its diagonal entry, so
    # symcover co-varies with D*A*D on any support whose components contain a
    # nonzero diagonal entry, not only on complete support.
    rng = StableRNG(7)
    start(A) = (a = zeros(size(A, 1)); unconstrained_min!(AbsLog{2}(), a, A); a)
    for n in (12, 40, 100)   # n = 100 uses the dense-grid kernel for Matrix storage
        S = sprandn(rng, n, n, 0.2)
        A = Matrix(S + S') + Diagonal(randn(rng, n) .+ 0.1)
        d = exp.(2 .* randn(rng, n))
        @test covaries(start, A, d; rtol=1e-9)
        @test covaries(A -> symcover(A; maxiter=0), A, d; rtol=1e-9)
        @test covaries(symcover, A, d; rtol=1e-9)
        @test covaries(symcover, sparse(A), d; rtol=1e-9)
        @test symcover(sparse(A)) ≈ symcover(A) rtol=1e-9
    end
    # A power-of-two rescaling is exact in floating point.
    n = 30
    S = sprandn(rng, n, n, 0.2)
    A = Matrix(S + S') + Diagonal(randn(rng, n) .+ 0.1)
    d = exp2.(rand(rng, -20:20, n))
    @test covaries(symcover, A, d; rtol=4eps())

    # Rows with a zero diagonal take their reference from neighbors, layer by
    # layer: a path whose only diagonal entry sits at one end propagates through
    # every row.
    for n in (10, 70)
        A = Matrix(SymTridiagonal([1.0; zeros(n - 1)], exp.(randn(rng, n - 1))))
        d = exp.(2 .* randn(rng, n))
        @test covaries(start, A, d; rtol=1e-9)
        @test covaries(symcover, A, d; rtol=1e-9)
        @test iscover(symcover(A), A; rtol=8eps())
    end

    # On complete support the start is the exact minimizer of the unconstrained
    # objective: (diag(n) + S) α = 𝔞 with S the all-ones support.
    n = 8
    B = randn(rng, n, n); A = B + B'
    L = log.(abs.(A))
    α = (n * I + ones(n, n)) \ vec(sum(L, dims=2))
    @test start(A) ≈ exp.(α) rtol=1e-10
end

@testset "cover: covariant start and conjugate-gradient refinement" begin
    # Centered log-product deviation under row and column scaling.
    function covdev(coverfn, A, dr, dc)
        a, b = coverfn(A)
        aD, bD = coverfn(dr .* A .* dc')
        d = log.((aD .* bD') ./ ((dr .* a) .* (dc .* b)'))[A .!= 0]
        return maximum(abs, d .- sum(d) / length(d))
    end
    rng = StableRNG(20260905)
    n = 12
    A = zeros(n, n)
    for i in 1:n, j in max(1, i - 1):min(n, i + 1)
        A[i, j] = exp(2 * randn(rng))
    end
    dr, dc = exp.(4 .* randn(rng, n)), exp.(4 .* randn(rng, n))
    # Covariance holds at every refinement count.
    for k in (0, 1, 4, 4n)
        @test covdev(A -> cover(A; cgiter=k), A, dr, dc) < 1e-9
        @test covaries(A -> cover(A; cgiter=k), A, dr, dc; rtol=1e-9)
        a, b = cover(A; cgiter=k)
        @test iscover(a, b, A; rtol=8eps())
    end
    @test_throws ArgumentError cover(A; cgiter=-1)

    # Refinement reduces the least-squares residual.
    sup = MatrixCovers.flat_support(A, Float64)
    function fitresidual(k)
        a, b = zeros(n), zeros(n)
        MatrixCovers.covariant_start!(a, b, sup)
        MatrixCovers.cg_refine_start!(a, b, sup, k)
        return sum((log(a[i]) + log(b[j]) - log(A[i, j]))^2
                   for i in 1:n, j in 1:n if A[i, j] != 0)
    end
    @test fitresidual(4) < fitresidual(0)
    @test fitresidual(4n) < fitresidual(4)

    # Powers of two rescale the input exactly in binary floating point.
    d2r, d2c = exp2.(rand(rng, -10:10, n)), exp2.(rand(rng, -10:10, n))
    for k in (0, 1, 4, 4n)
        @test covdev(A -> cover(A; cgiter=k), A, d2r, d2c) < 1e-12
    end

    # The dense-grid and flattened-support paths start and refine identically.
    m = 2 * MatrixCovers.DENSE_GRID_MIN
    B = zeros(m, m)
    for i in 1:m, j in max(1, i - 2):min(m, i + 1)
        B[i, j] = exp(2 * randn(rng)) * (rand(rng) < 0.7)
    end
    for i in 1:m
        B[i, i] = exp(2 * randn(rng))
    end
    for k in (0, 4, 40)
        ad, bd = cover(B; cgiter=k)
        as, bs = cover(sparse(B); cgiter=k)
        @test ad .* bd' ≈ as .* bs' rtol = 1e-12
    end
    # A grid with scattered zeros exercises the same fallback on both paths.
    C = exp.(2 .* randn(rng, m, m)) .* (rand(rng, m, m) .< 0.9)
    C[7, :] .= 0.0                        # an unsupported row
    for k in (0, 4)
        ad, bd = cover(C; cgiter=k)
        as, bs = cover(sparse(C); cgiter=k)
        @test ad .* bd' ≈ as .* bs' rtol = 1e-12
    end

    # Refinement preserves exact starts on complete and tree-shaped support.
    F = exp.(2 .* randn(rng, 9, 7))
    Fbig = exp.(2 .* randn(rng, m, m))
    P = Matrix(Bidiagonal(exp.(randn(rng, n)), exp.(randn(rng, n - 1)), :U))
    for A0 in (F, sparse(F), Fbig, P, sparse(P))
        a0, b0 = cover(A0; cgiter=0)
        for k in (4, 50)
            a, b = cover(A0; cgiter=k)
            @test all(isfinite, a) && all(isfinite, b)
            @test a .* b' ≈ a0 .* b0' rtol = 1e-12
        end
    end

    # Empty rows and columns keep zero scales through start and refinement.
    Z = [0.0 2.0 0.0; 0.0 0.0 0.0; 1.0 0.0 0.0]
    a, b = cover(Z)
    @test a[2] == 0 && b[3] == 0 && iscover(a, b, Z; rtol=8eps())
end

@testset "cover scale covariance on irregular support" begin
    # Compare supported products: balancing disconnected components can change
    # off-support products and individual factors under rescaling.
    function covaries_on_support(A, d1, d2; cgiter=4, rtol=1e-12)
        a, b = cover(A; cgiter)
        B = d1 .* A .* transpose(d2)
        aB, bB = cover(B; cgiter)
        ok = iscover(aB, bB, B; rtol=8eps())
        foreach_support(A) do i, j, v
            ok &= isapprox(aB[i] * bB[j], d1[i] * a[i] * d2[j] * b[j]; rtol)
        end
        return ok
    end

    rng = StableRNG(2718)
    dyadic(k) = exp2.(rand(rng, -10:10, k))

    n = 12
    tridiag = Matrix(Tridiagonal(randn(rng, n - 1), randn(rng, n), randn(rng, n - 1)))
    banded = [abs(i - j) <= 2 ? randn(rng) : 0.0 for i in 1:15, j in 1:15]
    S = sprandn(rng, 30, 25, 0.15)
    # Exercise the dense-grid kernel with incomplete support.
    densezeros = randn(rng, 70, 70)
    densezeros[3, 4] = 0.0
    densezeros[10, :] .= 0.0
    densezeros[:, 20] .= 0.0
    rect = randn(rng, 7, 13)
    blocks = Matrix(blockdiag(sparse(randn(rng, 4, 3)), sparse(randn(rng, 5, 6))))

    for A in (tridiag, banded, S, Matrix(S), densezeros, rect, blocks)
        m, k = size(A)
        d1, d2 = dyadic(m), dyadic(k)
        for cgiter in (0, 1, 4)
            @test covaries_on_support(A, d1, d2; cgiter)
        end
    end

    # On complete support the start is the row/column geometric mean.
    for (m, k) in ((5, 7), (70, 70))
        A = exp.(3 .* randn(rng, m, k))
        sup = MatrixCovers.flat_support(A, Float64)
        a1, b1 = zeros(m), zeros(k)
        MatrixCovers.covariant_start!(a1, b1, sup)
        a2, b2 = zeros(m), zeros(k)
        unconstrained_min!(AbsLog{2}(), a2, b2, sup)
        @test a1 .* b1' ≈ a2 .* b2' rtol = 1e-12
    end

    # Offset axes preserve the scales.
    Ao = OffsetArray(densezeros, -2:67, 0:69)
    a, b = cover(densezeros)
    ao, bo = cover(Ao)
    @test axes(ao, 1) == axes(Ao, 1) && axes(bo, 1) == axes(Ao, 2)
    @test collect(ao) ≈ a rtol = 1e-12
    @test collect(bo) ≈ b rtol = 1e-12
    @test iscover(ao, bo, Ao; rtol=8eps())
end

@testset "cover: geometric-mean start" begin
    # Centered log-product deviation under row and column scaling; see the
    # covariant-start testset.
    function covdev(coverfn, A, dr, dc)
        a, b = coverfn(A)
        aD, bD = coverfn(dr .* A .* dc')
        d = log.((aD .* bD') ./ ((dr .* a) .* (dc .* b)'))[A .!= 0]
        return maximum(abs, d .- sum(d) / length(d))
    end

    rng = StableRNG(31415)
    n = 12
    trid = Matrix(Tridiagonal(exp.(randn(rng, n - 1)), exp.(randn(rng, n)), exp.(randn(rng, n - 1))))

    # The default is the covariant start.
    for A in (exp.(randn(rng, 9, 7)), trid, sparse(trid))
        @test cover(A) == cover(A; start=:covariant)
    end

    # The geometric-mean start yields covers obeying the package's conventions.
    for A in (exp.(2 .* randn(rng, 6, 9)), trid, sparse(trid),
              Matrix(sprandn(rng, 20, 14, 0.3)), sprandn(rng, 20, 14, 0.3))
        a, b = cover(A; start=:geomean)
        @test iscover(a, b, A; rtol=8eps())
        @test isbalanced(a, b, A)
    end

    # On complete support the two starts coincide, on both the flattened-support
    # and dense-grid paths.
    m = 2 * MatrixCovers.DENSE_GRID_MIN
    for A in (exp.(2 .* randn(rng, 6, 9)), exp.(2 .* randn(rng, m, m)),
              sparse(exp.(2 .* randn(rng, m, m))))
        ac, bc = cover(A; start=:covariant)
        ag, bg = cover(A; start=:geomean)
        @test ac .* bc' ≈ ag .* bg' rtol = 1e-12
    end

    # On an irregular support the geometric mean is not scale-covariant without
    # refinement, while the covariant start is.
    dr, dc = exp2.(rand(rng, -10:10, n)), exp2.(rand(rng, -10:10, n))
    @test covdev(A -> cover(A; cgiter=0, start=:covariant), trid, dr, dc) < 1e-12
    @test covdev(A -> cover(A; cgiter=0, start=:geomean), trid, dr, dc) > 0.1

    # The dense-grid and flattened-support paths agree on a grid with zeros.
    C = exp.(2 .* randn(rng, m, m)) .* (rand(rng, m, m) .< 0.9)
    C[7, :] .= 0.0                        # an unsupported row
    C[:, 11] .= 0.0                       # an unsupported column
    for k in (0, 4)
        ad, bd = cover(C; cgiter=k, start=:geomean)
        as, bs = cover(sparse(C); cgiter=k, start=:geomean)
        @test ad .* bd' ≈ as .* bs' rtol = 1e-12
    end

    # Offset axes preserve the scales.
    Co = OffsetArray(C, -2:m-3, 0:m-1)
    a, b = cover(C; start=:geomean)
    ao, bo = cover(Co; start=:geomean)
    @test axes(ao, 1) == axes(Co, 1) && axes(bo, 1) == axes(Co, 2)
    @test collect(ao) ≈ a rtol = 1e-12
    @test collect(bo) ≈ b rtol = 1e-12
    @test iscover(ao, bo, Co; rtol=8eps())

    # The mutating form and the Adjoint/Transpose wrappers take the keyword.
    B = exp.(randn(rng, 5, 8)) .* (rand(rng, 5, 8) .< 0.8)
    ab, bb = cover(B; start=:geomean)
    abuf, bbuf = similar(ab), similar(bb)
    @test cover!(abuf, bbuf, B; start=:geomean) === (abuf, bbuf)
    @test abuf == ab && bbuf == bb
    for Bw in (B', transpose(B))
        aw, bw = cover(Bw; start=:geomean)
        @test aw ≈ bb && bw ≈ ab
        awbuf, bwbuf = similar(aw), similar(bw)
        cover!(awbuf, bwbuf, Bw; start=:geomean)
        @test awbuf == aw && bwbuf == bw
    end

    # `initialize_cover` forwards the keyword to the hard cover.
    @test initialize_cover(B; strategy=:hardcover, feasible=:none, start=:geomean) == (ab, bb)

    # Any other start is rejected before work begins.
    @test_throws "unknown start :nope; expected one of :covariant, :geomean" cover(B; start=:nope)
    @test_throws "unknown start :nope; expected one of :covariant, :geomean" cover!(abuf, bbuf, B; start=:nope)
end
