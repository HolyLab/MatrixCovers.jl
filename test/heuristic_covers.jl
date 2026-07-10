# The fast heuristic covers: symcover/cover, their in-place forms, and the
# bucketed feasibility boost.

@testset "symcover" begin
    # Cover property: a[i]*a[j] >= abs(A[i,j]) for all i, j
    for A in ([2.0 1.0; 1.0 3.0], [1.0 -0.2; -0.2 0.0], [1.0 0.0; 0.0 0.0],
              [100.0 1.0; 1.0 0.01])
        for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
            a = symcover(ϕ, A)
            @test all(a[i] * a[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
        end
        # Default dispatch
        a = symcover(A)
        @test all(a[i] * a[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
    end
    # All-zero row or column gives zero cover element for any ϕ
    for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
        a = symcover(ϕ, [1.0 0; 0 0])
        @test a[2] == 0
        a = symcover(ϕ, [0 0; 0 1.0])
        @test a[1] == 0
    end
    # All-zero diagonals
    A = [0.0 1.0; 1.0 0.0]
    for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
        @test symcover(ϕ, A) == [1.0, 1.0]
    end
    # Diagonal scaling covariance
    A = [2.0 1.0; 1.0 3.0]
    d = [2.0, 0.5]
    for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
        @test symcover(ϕ, A .* d .* d') ≈ d .* symcover(ϕ, A)
    end
    # Non-square input is rejected
    @test_throws ArgumentError symcover([1.0 2.0; 3.0 4.0; 5.0 6.0])
end

@testset "symcover ignores ϕ" begin
    # The initialization is the geometric mean plus the greedy max-deficit boost for
    # every penalty, so `ϕ` (and `p`) cannot change the result.
    rng = MersenneTwister(1)
    for n in (2, 5, 40)
        B = randn(rng, n, n); A = (B + B') / 2
        a = symcover(A)
        for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
            @test symcover(ϕ, A) == a
        end
    end
end

@testset "symcover with unequal row degrees" begin
    # Rows with differing numbers of nonzeros must not destabilize the initialization.
    # Sparse matrices of this size and density are the stressing case: the returned cover
    # must be finite and feasible, never NaN/Inf.
    for (n, seed) in ((20, 44), (60, 21))
        rng = MersenneTwister(seed)
        M = randn(rng, n, n) .* (rand(rng, n, n) .< 0.3)
        A = Matrix(Symmetric(M))
        a = symcover(A)
        @test all(isfinite, a)
        @test all(a[i] * a[j] >= abs(A[i, j]) - 1e-9 for i in 1:n, j in 1:n)
    end
    # Arrow matrix: one dense row/column, everything else diagonal (degrees 2,…,2,n).
    n = 6
    A = Matrix(Diagonal(fill(2.0, n))); A[1, :] .= 1.0; A[:, 1] .= 1.0; A[1, 1] = 2.0
    a = symcover(A)
    @test all(isfinite, a)
    @test all(a[i] * a[j] >= abs(A[i, j]) - 1e-9 for i in 1:n, j in 1:n)
end

@testset "cover" begin
    # Cover property: a[i]*b[j] >= abs(A[i,j]) for all i, j
    for A in ([2.0 1.0; 1.0 3.0], [0.0 1.0; -2.0 0.0], [1.0 0.0; 0.0 0.0],
              [100.0 1.0; 1.0 0.01])
        for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
            a, b = cover(ϕ, A)
            @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
        end
        # Default dispatch
        a, b = cover(A)
        @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
    end
    # All-zero row or column gives zero cover element for any ϕ
    for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
        a, b = cover(ϕ, [1.0 0; 0 0])
        @test b[2] == 0
        a, b = cover(ϕ, [0 0; 0 1.0])
        @test a[1] == 0
    end
    # Zero-diagonal matrix
    A = [0.0 1.0; -1.0 0.0]
    for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
        a, b = cover(ϕ, A)
        @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
    end
    # Rectangular matrix
    A = [1.0 2.0 3.0; 4.0 5.0 6.0]
    a, b = cover(A)
    @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
    # Diagonal scaling covariance: cover(A .* dr .* dc') is cover(A) scaled by dr, dc up to a scalar
    A = [2.0 1.0; 1.0 3.0]
    dr, dc = [2.0, 0.5], [3.0, 0.25]
    for ϕ in (AbsLog{1}(), AbsLog{2}(), AbsLinear{1}(), AbsLinear{2}())
        a, b = cover(ϕ, A .* dr .* dc')
        a0, b0 = cover(ϕ, A)
        c = a \ (dr .* a0)   # scalar: cover has a free global scale a→c*a, b→b/c
        @test c * a ≈ dr .* a0
        @test b / c ≈ dc .* b0
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

    @test_throws DimensionMismatch symcover!(zeros(3), A)
    @test_throws DimensionMismatch cover!(zeros(2), zeros(2), B)
    @test_throws DimensionMismatch cover!(zeros(2), zeros(4), B)
end

@testset "bucket boost" begin
    rng = MersenneTwister(42)

    # Feasibility on randomized dense (with zero rows/diagonal), sparse, banded, and
    # offset-axes inputs.
    feasible_sym(a, M) = all(a[i] * a[j] >= abs(M[i, j]) - 4 * eps(max(a[i] * a[j], abs(M[i, j])))
                              for i in axes(M, 1), j in axes(M, 2))
    for n in (5, 12)
        for _ in 1:5
            B = randn(rng, n, n); A = (B + B') / 2
            A[1, :] .= 0; A[:, 1] .= 0   # zero row/column
            A[2, 2] = 0                  # zero diagonal entry
            a = symcover(AbsLog{2}(), A)
            @test feasible_sym(a, A)
            a2, b2 = cover(AbsLog{2}(), A)
            @test all(a2[i] * b2[j] >= abs(A[i, j]) - 4 * eps(max(a2[i] * b2[j], abs(A[i, j])))
                      for i in axes(A, 1), j in axes(A, 2))
        end
    end
    for _ in 1:5
        S = sprandn(rng, 10, 10, 0.3); A = S + S'
        a = symcover(AbsLog{2}(), A)
        @test feasible_sym(a, Matrix(A))
    end
    for _ in 1:5
        dv, ev = randn(rng, 8), randn(rng, 7)
        A = SymTridiagonal(dv, ev)
        a = symcover(AbsLog{2}(), A)
        @test feasible_sym(a, Matrix(A))
    end
    let B = randn(rng, 6, 6), Asym = (B + B') / 2
        Ao = OffsetArray(Asym, -3:2, -3:2)
        a = symcover(AbsLog{2}(), Ao)
        @test axes(a, 1) == axes(Ao, 1)
        @test feasible_sym(a, Ao)
    end

    # Scale-covariance of the boosted (untightened) cover under diagonal/row-col rescaling.
    n = 8
    B = randn(rng, n, n); A = (B + B') / 2
    d = exp.(randn(rng, n))
    @test symcover(AbsLog{2}(), A .* d .* d'; maxiter=0) ≈ d .* symcover(AbsLog{2}(), A; maxiter=0) rtol=1e-10

    m = 6
    Ag = randn(rng, n, m)
    dr, dc = exp.(randn(rng, n)), exp.(randn(rng, m))
    a1, b1 = cover(AbsLog{2}(), Ag; maxiter=0)
    a2, b2 = cover(AbsLog{2}(), dr .* Ag .* dc'; maxiter=0)
    # cover has a free global gauge a→c*a, b→b/c; only the product co-varies.
    c = a2 \ (dr .* a1)
    @test c * a2 ≈ dr .* a1 rtol=1e-10
    @test b2 / c ≈ dc .* b1 rtol=1e-10

    # Quality gate: on a seeded lognormal + sparse corpus, the median log-optimality-gap
    # after 3 tighten iterations stays close to the measured value for this boost
    # (0.0181 on 2026-07-08); the bound is a generous 1.5x margin, not a tight pin.
    qrng = MersenneTwister(20260708)
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
    @test median(gaps) < 0.0181 * 1.5
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
    @test all(a[i] * a[j] >= max(abs(M[i, j]), abs(M[j, i])) * (1 - 8eps()) for i in axes(M, 1), j in axes(M, 2))

    # Float32 dynamic range wide enough that linear-domain deficit ratios overflow.
    # The boost's apply! step shifts log(a[i]) by h = z/2, where z ~ 120 for this
    # matrix; exp(log(a[i]) + h) then carries forward the rounding error already
    # present in h at that magnitude, so the achievable relative precision is set
    # by eps(Float32) scaled by |h|, not by a fixed few-ulp bound.
    A32 = fill(1f-35, 6, 6); A32[1, 2] = A32[2, 1] = 3f37
    a32 = symcover(AbsLog{2}(), A32)
    @test all(isfinite, a32)
    @test all(a32[i] * a32[j] >= abs(A32[i, j]) * (1 - 64 * eps(Float32)) for i in axes(A32, 1), j in axes(A32, 2))

    # Float64 range where the geometric-mean init underflows without clamping.
    A = fill(1e308, 6, 6); A[1, :] .= 0; A[:, 1] .= 0; A[1, 2] = A[2, 1] = 1e-308
    a = symcover(AbsLog{2}(), A)
    @test all(isfinite, a)
    @test all(a[i] * a[j] >= abs(A[i, j]) * (1 - 8eps()) for i in axes(A, 1), j in axes(A, 2))
end

@testset "tighten_cover! leaves zero-product scales unchanged" begin
    a, b = tighten_cover!(zeros(3), zeros(3), Diagonal([1.0, 2.0, 3.0]))
    @test all(iszero, a) && all(iszero, b)
end
