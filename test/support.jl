@testset "support traversal" begin
    # Reference implementations built directly from `getindex`, independent of storage.
    naive_support(A) = Set((i, j, Float64(abs(A[i, j]))) for i in axes(A, 1), j in axes(A, 2) if !iszero(A[i, j]))
    function naive_support_sym(A; valat=(i, j) -> abs(A[i, j]))
        ax = axes(A, 1)
        axes(A, 2) == ax || throw(ArgumentError("naive_support_sym requires a square matrix"))
        return Set((i, j, Float64(valat(i, j))) for j in ax for i in first(ax):j if !iszero(valat(i, j)))
    end

    collect_support(A) = begin
        out = Tuple{Int,Int,Float64}[]
        foreach_support((i, j, v) -> push!(out, (i, j, Float64(v))), A)
        Set(out)
    end
    collect_support_sym(A) = begin
        out = Tuple{Int,Int,Float64}[]
        foreach_support_sym((i, j, v) -> push!(out, (i, j, Float64(v))), A)
        Set(out)
    end

    rng = MersenneTwister(42)

    @testset "dense AbstractMatrix" begin
        A = randn(rng, 5, 6)
        A[2, 3] = 0.0
        A[4, 1] = 0.0
        @test collect_support(A) == naive_support(A)

        S = randn(rng, 5, 5)
        S[1, 4] = 0.0
        @test collect_support_sym(S) == naive_support_sym(S)

        @test_throws DimensionMismatch foreach_support_sym((i, j, v) -> nothing, randn(rng, 3, 4))
    end

    @testset "OffsetArray" begin
        A = OffsetArray(randn(rng, 5, 5), -2:2, -2:2)
        A[-1, 1] = 0.0
        @test collect_support(A) == naive_support(A)
        @test collect_support_sym(A) == naive_support_sym(A)
    end

    @testset "Diagonal" begin
        d = randn(rng, 6)
        d[3] = 0.0
        D = Diagonal(d)
        @test collect_support(D) == naive_support(D)
        @test collect_support_sym(D) == naive_support_sym(D)
    end

    @testset "SymTridiagonal" begin
        n = 7
        dv = randn(rng, n); dv[2] = 0.0
        ev = randn(rng, n - 1); ev[4] = 0.0
        A = SymTridiagonal(dv, ev)
        @test collect_support_sym(A) == naive_support_sym(A)
        @test collect_support(A) == naive_support(A)
    end

    @testset "Bidiagonal" begin
        n = 6
        dv = randn(rng, n); dv[1] = 0.0
        ev = randn(rng, n - 1); ev[3] = 0.0
        for uplo in ('U', 'L')
            A = Bidiagonal(dv, ev, uplo)
            @test collect_support(A) == naive_support(A)
            valat(i, j) = max(abs(A[i, j]), abs(A[j, i]))
            @test collect_support_sym(A) == naive_support_sym(A; valat)
        end
    end

    @testset "Tridiagonal" begin
        n = 6
        dl = randn(rng, n - 1); dl[2] = 0.0
        d  = randn(rng, n); d[4] = 0.0
        du = randn(rng, n - 1); du[2] = 0.0
        A = Tridiagonal(dl, d, du)
        @test collect_support(A) == naive_support(A)
        valat(i, j) = max(abs(A[i, j]), abs(A[j, i]))
        @test collect_support_sym(A) == naive_support_sym(A; valat)
    end

    @testset "SparseMatrixCSC" begin
        A = sparse([1, 2, 2, 3, 5], [1, 2, 3, 3, 5], [1.0, 0.0, 2.0, 3.0, 4.0], 5, 5)
        A = A + A'   # symmetric-valued, includes a stored explicit zero (2,2) and structural zeros elsewhere
        @test collect_support(A) == naive_support(A)
        @test collect_support_sym(A) == naive_support_sym(A)
    end

    @testset "Symmetric{SparseMatrixCSC} / Hermitian{SparseMatrixCSC}" begin
        P = sparse([1, 2, 2, 3, 5, 4], [1, 2, 3, 3, 5, 2], [1.0, 0.0, 2.0, 3.0, 4.0, 5.0], 5, 5)
        for uplo in (:U, :L)
            S = Symmetric(P, uplo)
            @test collect_support_sym(S) == naive_support_sym(Matrix(S))
            H = Hermitian(P, uplo)
            @test collect_support_sym(H) == naive_support_sym(Matrix(H))
        end
    end
end

@testset "the symmetry precondition" begin
    # A genuinely asymmetric matrix is refused rather than covered as some
    # unspecified symmetrization of itself.
    for A in ([0.0 0.0; 1.0 0.0], [1.0 2.0; 3.0 1.0], [4.0 1.0; -1.5 4.0])
        @test_throws "abs.(A)` to be symmetric" symcover(A)
        @test_throws "abs.(A)` to be symmetric" soft_symcover(A)
        @test_throws "abs.(A)` to be symmetric" symcover_min(AbsLog{2}(), A)
        @test_throws "abs.(A)` to be symmetric" initialize_symcover(A)
        @test_throws "abs.(A)` to be symmetric" sparse(A) |> symcover
        # Every solver family, including the ones behind the extensions.
        @test_throws "abs.(A)` to be symmetric" symcover_min(AbsLog{1}(), A)
        @test_throws "abs.(A)` to be symmetric" symcover_min(AbsLinear{2}(), A)
        @test_throws "abs.(A)` to be symmetric" soft_symcover_min(AbsLinear{2}(), A)
    end
    # The message names the offending pair and the entry point.
    @test_throws "symcover!" symcover([0.0 0.0; 1.0 0.0])
    @test_throws "abs(A[2,1])" symcover([0.0 0.0; 1.0 0.0])

    # Only the magnitudes must agree, so a sign flip is fine and a complex
    # Hermitian — where A[i,j] == conj(A[j,i]) — is a legitimate input.
    @test iscover(symcover([4.0 1.0; -1.0 4.0]), [4.0 1.0; -1.0 4.0]; rtol=8eps())
    P = sparse([1, 2, 1], [1, 2, 2], ComplexF64[4.0, 2.0, 0.5+0.5im], 2, 2)
    @test symcover_min(AbsLog{2}(), Hermitian(P, :U)) isa AbstractVector

    # Exact symmetry is not required: a symmetric matrix that has been through
    # floating-point arithmetic lands a ULP or so off, and must still be accepted.
    rng = StableRNG(7)
    B = randn(rng, 12, 12); S = B + B'
    d = exp.(randn(rng, 12))
    Sd = (d .* S) .* d'          # symmetric in exact arithmetic only
    @test !issymmetric(Sd)
    @test symcover(Sd) isa AbstractVector

    # Wrapper types are exempt because their storage makes the property structural.
    M = [4.0 1.0; 2.0 4.0]       # asymmetric as written
    @test symcover(Symmetric(M, :U)) isa AbstractVector
    @test symcover(Symmetric(M, :L)) isa AbstractVector
    @test symcover(Diagonal([1.0, 2.0])) isa AbstractVector

    # Banded storage earns no exemption. A `Bidiagonal` reads one of its
    # off-diagonals as a structural zero, so any nonzero band makes it asymmetric;
    # a `Tridiagonal` stores both bands and qualifies only when they agree.
    for A in (Bidiagonal([3.0, 2.0, 1.0], [6.0, 0.5], :U),
              Bidiagonal([3.0, 2.0, 1.0], [6.0, 0.5], :L),
              Tridiagonal([1.0, 0.5], [3.0, 2.0, 1.0], [4.0, 0.5]))
        @test_throws "abs.(A)` to be symmetric" symcover(A)
        @test_throws "abs.(A)` to be symmetric" symcover_min(AbsLog{2}(), A)
        # The asymmetric family covers them as stored.
        @test iscover(cover(A)..., Matrix(A); rtol=8eps())
    end
    # Bands that do agree are ordinary symmetric input.
    @test iscover(symcover(Tridiagonal([2.0, 0.5], [3.0, 2.0, 1.0], [2.0, 0.5])),
                  Matrix(Tridiagonal([2.0, 0.5], [3.0, 2.0, 1.0], [2.0, 0.5])); rtol=8eps())
    @test symcover(Bidiagonal([3.0, 2.0, 1.0], [0.0, 0.0], :U)) isa AbstractVector
end
