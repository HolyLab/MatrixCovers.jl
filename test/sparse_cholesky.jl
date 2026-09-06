using MatrixCovers: SparseCholesky, analyze!, factorize!, solve!, solve_ptl!, solve_up!,
                    factor_entries, CHOLMOD_A, _eachindex

@testset "SparseCholesky matches SparseArrays" begin
    rng = MersenneTwister(0)
    n = 120
    S = sprand(rng, n, n, 0.03)
    S = sparse(S + S' + 4n * I)
    Su = triu(S)
    F = SparseCholesky()
    analyze!(F, Su)
    factorize!(F, Su)
    Fref = cholesky(Symmetric(Su, :U))
    B = rand(rng, n, 3)
    X = zeros(n, 3)
    @test solve!(X, F, CHOLMOD_A, B) === X
    @test X == Fref \ B
    v = rand(rng, n)
    y = zeros(n)
    @test solve_ptl!(y, F, v) == Fref.PtL \ v
    @test solve_up!(y, F, v) == Fref.UP \ v
    # Strided column views work on both sides.
    Bw = zeros(n + 2, 3)
    Bw[1:n, :] .= B
    Xw = zeros(n + 3, 3)
    solve!(view(Xw, 1:n, :), F, CHOLMOD_A, view(Bw, 1:n, :))
    @test Xw[1:n, :] == Fref \ B
    @test all(iszero, Xw[n+1:end, :])
    # In-place use: the solution may overwrite its right-hand side.
    y2 = copy(v)
    @test solve_ptl!(y2, F, y2) == Fref.PtL \ v
    # Refactorization with new values on the same pattern.
    Su2 = copy(Su)
    nonzeros(Su2) .*= 2
    factorize!(F, Su2)
    @test solve!(X, F, CHOLMOD_A, B) == cholesky(Symmetric(Su2, :U)) \ B
    # The fill estimate is the number of stored factor values.
    Fs = SparseCholesky()
    Sd = sparse(Diagonal(1.0:8.0))
    analyze!(Fs, Sd)
    @test factor_entries(Fs) == 8
    # Failures are reported through exceptions.
    Sb = copy(Su)
    nonzeros(Sb)[1] = -1e6
    @test_throws PosDefException factorize!(F, Sb)
    Sm = sparse(Diagonal(ones(n)))
    @test_throws "differs from the analyzed pattern" factorize!(F, Sm)
    @test_throws "must be square" analyze!(F, sparse(ones(2, 3)))
    @test_throws "`analyze!` must run" factorize!(SparseCholesky(), Su)
    @test_throws DimensionMismatch solve!(zeros(n, 2), F, CHOLMOD_A, B)
    @test_throws DimensionMismatch solve!(zeros(n - 1), F, CHOLMOD_A, zeros(n - 1))
end

@testset "_eachindex" begin
    a, b, c, d, e = zeros(3), zeros(3), zeros(3), zeros(3), zeros(3)
    @test _eachindex(a, b, c) == eachindex(a)
    @test _eachindex(a, b, c, d) == eachindex(a)
    @test _eachindex(a, b, c, d, e) == eachindex(a)
    oa, ob, oc = OffsetArray(zeros(3), -1), OffsetArray(zeros(3), -1), OffsetArray(zeros(3), -1)
    @test _eachindex(oa, ob, oc) == eachindex(oa)
    @test_throws "must have the same indices" _eachindex(a, b, zeros(4))
    @test_throws "must have the same indices" _eachindex(a, b, c, zeros(4))
    @test_throws "must have the same indices" _eachindex(a, b, c, d, oa)
end
