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
