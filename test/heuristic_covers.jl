# The fast heuristic covers: symcover/cover, their in-place forms, and the
# bucketed feasibility boost.

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
    # A regression check on the heuristic as it stands, not an API guarantee: the docstring
    # says the heuristic covers ignore `ϕ` *currently*, and that this may change. Should a
    # penalty-tuned heuristic land, this test records what changes — update it; do not read a
    # failure here as a broken promise to callers.
    rng = StableRNG(1)
    for n in (2, 5, 40)
        B = randn(rng, n, n); A = (B + B') / 2
        a = symcover(A)
        for ϕ in PENALTIES
            @test symcover(ϕ, A) == a
        end
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

@testset "bucket boost" begin
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

    # Quality gate: median log-optimality-gap of the 3-iteration heuristic over
    # this fixed corpus, with a generous 1.5x margin over the measured value; a
    # tighter algorithm may lower it, a regression will trip it.
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
    @test median(gaps) < 0.0195 * 1.5
end

@testset "boost feasibility: lower-dominant bands and extreme dynamic range" begin
    # Lower-dominant bands: the symmetric-contract value is the max over both
    # triangles, so the subdiagonal must be covered even when it dominates.
    a = symcover(AbsLog{2}(), Bidiagonal([0.01, 0.01, 0.01], [100.0, 100.0], 'L'); maxiter=0)
    @test a[1] * a[2] >= 100 * (1 - 8eps())
    @test a[2] * a[3] >= 100 * (1 - 8eps())
    T40 = Tridiagonal(fill(50.0, 39), fill(0.01, 40), fill(0.02, 39))
    a = symcover(AbsLog{2}(), T40; maxiter=0)
    M = Matrix(T40)
    @test iscover(a, M; rtol=8eps())

    # Float32 dynamic range wide enough that linear-domain deficit ratios overflow.
    # The boost's apply! step shifts log(a[i]) by h = z/2, where z ~ 120 for this
    # matrix; exp(log(a[i]) + h) then carries forward the rounding error already
    # present in h at that magnitude, so the achievable relative precision is set
    # by eps(Float32) scaled by |h|, not by a fixed few-ulp bound.
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
