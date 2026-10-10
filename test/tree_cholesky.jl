using MatrixCovers: TreeCholesky, tree_cholesky, max_weight_forest, solve_ptl!, solve_up!, lt_mul!

@testset "tree Cholesky" begin
    rng = StableRNG(31)
    for T in (Float64, BigFloat), trial in 1:5
        N = 40
        # A random forest: most vertices attach to an earlier one.
        I = Int[]
        J = Int[]
        for v in 2:N
            rand(rng) < 0.8 && (push!(I, v); push!(J, rand(rng, 1:v-1)))
        end
        V = T.(randn(rng, length(I)))
        # Diagonal dominance makes the matrix positive definite.
        diag = T.(abs.(randn(rng, N)) .+ 1)
        for k in eachindex(I)
            diag[I[k]] += abs(V[k])
            diag[J[k]] += abs(V[k])
        end
        M = Matrix(sparse([I; J; 1:N], [J; I; 1:N], [V; V; diag], N, N))
        F = tree_cholesky(N, I, J, V, diag)
        @test F isa TreeCholesky{T}
        b = T.(randn(rng, N))
        x = solve_up!(zeros(T, N), F, solve_ptl!(zeros(T, N), F, b))
        @test M * x ≈ b
        # `L \ (M x) == Lᵀ x` shows `M == L Lᵀ`.
        @test solve_ptl!(zeros(T, N), F, M * x) ≈ lt_mul!(zeros(T, N), F, x)
        # Every operation works in place.
        xa = copy(b)
        solve_ptl!(xa, F, xa)
        solve_up!(xa, F, xa)
        @test xa ≈ x
        ya = copy(x)
        lt_mul!(ya, F, ya)
        @test ya ≈ lt_mul!(zeros(T, N), F, x)
    end
    @test_throws "contain a cycle" tree_cholesky(3, [1, 2, 3], [2, 3, 1], ones(3), fill(10.0, 3))
    @test_throws "not positive definite" tree_cholesky(2, [1], [2], [5.0], [1.0, 1.0])
    @test_throws DimensionMismatch tree_cholesky(2, [1], [2], [0.5], [1.0, 1.0, 1.0])
    # Kruskal keeps the two heaviest edges of the triangle and the pendant edge.
    @test sort(max_weight_forest(4, [1, 2, 3, 1], [2, 3, 1, 4], [1.0, 3.0, 2.0, 0.5])) == [2, 3, 4]
    @test max_weight_forest(3, Int[], Int[], Float64[]) == Int[]
end
