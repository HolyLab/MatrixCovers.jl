using MatrixCovers: _max_product_transversal, _row_support, _cover_min_abslog2

# Permutations of 1:n, for brute-force enumeration of transversals.
function permutations_of(n)
    n == 1 && return [[1]]
    out = Vector{Int}[]
    for p in permutations_of(n - 1), k in 1:n
        push!(out, [p[1:k-1]; n; p[k:end]])
    end
    return out
end

# All transversals attaining the maximum log product, and that maximum.
function max_product_transversals(A)
    n = size(A, 1)
    ps = permutations_of(n)
    lp = [all(i -> !iszero(A[i, p[i]]), 1:n) ? sum(i -> log(abs(A[i, p[i]])), 1:n) : -Inf for p in ps]
    m = maximum(lp)
    return ps[abs.(lp .- m) .<= 1e-12 * max(1, abs(m))], m
end

# Largest log-domain discrepancy between two covers' products over the support of `A`.
function logproduct_gap(a, b, a2, b2, A)
    g = 0.0
    foreach_support(A) do i, j, _
        g = max(g, abs(log(a[i] * b[j]) - log(a2[i] * b2[j])))
    end
    return g
end

@testset "maximum-product transversal" begin
    for (_, A) in general_matrices[1:400]
        Af = Float64.(A)
        σ, α, β, logπ = _max_product_transversal(_row_support(Af, Float64), axes(Af, 2), :test)
        _, lπ = max_product_transversals(Af)
        @test logπ ≈ lπ rtol=1e-12
        @test sort(σ) == 1:5
        # The duals are a hard cover, tight on the transversal.
        @test all(α[i] + β[j] >= log(Af[i, j]) - 1e-12 for i in 1:5, j in 1:5)
        @test all(abs(α[i] + β[σ[i]] - log(Af[i, σ[i]])) <= 1e-12 for i in 1:5)
    end
    # Sparse support, where the greedy start leaves rows for the augmenting paths.
    A = [1.0 1 0 0; 1 0 0 0; 0 1 1 1; 0 0 1 0]
    σ, _, _, logπ = _max_product_transversal(_row_support(A, Float64), axes(A, 2), :test)
    @test σ == [2, 1, 4, 3]
    @test logπ == 0
end

@testset "cover_transversal" begin
    # The paper's example: a single transversal (1,2), (2,3), (3,1), which the
    # minimal cover covers loosely at (1,2).
    η = 1e-3
    A = [1 1 0; η^2 1 1; 1 0 0]
    a, b = cover_transversal(A)
    @test a .* b' ≈ ones(3, 3) rtol=1e-8
    @test cond(A ./ (a .* b')) ≈ 4.0489 rtol=1e-4
    am, bm = cover_min(A)
    @test cond(A ./ (am .* bm')) > 2000
    @test isbalanced(a, b, A)

    # Agreement with the two-stage JuMP reference, and the defining properties.
    idx_sub = Set(round.(Int, range(1, length(general_matrices), length=200)))
    for (k, (_, A)) in enumerate(general_matrices)
        k in idx_sub || continue
        Af = Float64.(A)
        a, b = cover_transversal(Af)
        @test iscover(a, b, Af; atol=1e-7)
        σs, lπ = max_product_transversals(Af)
        @test sum(log, a) + sum(log, b) ≈ lπ atol=1e-7
        # Tight on every maximum-product transversal.
        @test all(σ -> all(i -> abs(log(a[i] * b[σ[i]] / Af[i, σ[i]])) <= 1e-7, 1:5), σs)
        (aj, bj), lπj = MatrixCovers.cover_transversal_jump(Af)
        @test lπj ≈ lπ rtol=1e-8
        @test logproduct_gap(a, b, aj, bj, Af) <= 1e-5
        @test cover_objective(AbsLog{2}(), a, b, Af) >= cover_objective(AbsLog{2}(), scales(cover_min(Af))..., Af) * (1 - 1e-6) - 1e-10
    end

    # Independence from the transversal used: solve with each maximum-product
    # transversal in turn on matrices with ties.
    ntie = 0
    for (_, A) in general_matrices
        ntie >= 20 && break
        Af = Float64.(A)
        σs, _ = max_product_transversals(Af)
        length(σs) > 1 || continue
        ntie += 1
        a, b = cover_transversal(Af)
        for σ in σs
            aσ, bσ, _ = _cover_min_abslog2(Af; transversal=σ, fname=:cover_transversal)
            @test logproduct_gap(a, b, aσ, bσ, Af) <= 1e-7
        end
    end
    @test ntie == 20
    Af = Float64.(general_matrices[1][2])
    σs, _ = max_product_transversals(Af)
    σlow = first(σ for σ in permutations_of(5) if σ ∉ σs)
    @test_throws "maximum is" _cover_min_abslog2(Af; transversal=σlow, fname=:cover_transversal)
    @test_throws "permutation" _cover_min_abslog2(Af; transversal=[1, 1, 3, 4, 5], fname=:cover_transversal)

    # Coincides with the minimal cover when that is already tight on the transversal.
    D = [4.0 1 1; 1 4 1; 1 1 4]
    @test logproduct_gap(scales(cover_transversal(D))..., scales(cover_min(D))..., D) <= 1e-7

    # Scale covariance, including power-of-two rescalings, which change the
    # floating-point costs seen by the matching.
    rng = StableRNG(1)
    for (_, A) in general_matrices[1:30]
        Af = Float64.(A)
        dr = exp.(2 .* randn(rng, 5)); dc = exp.(2 .* randn(rng, 5))
        @test covaries(cover_transversal, Af, dr, dc; rtol=1e-6)
    end
    for seed in 1:3
        rng = StableRNG(seed)
        n = 40
        A = sprandn(rng, n, n, 0.1) + spdiagm(0 => exp.(3 .* randn(rng, n)))
        d = 2.0 .^ rand(rng, -30:30, n)
        e = 2.0 .^ rand(rng, -30:30, n)
        a, b = cover_transversal(A)
        ã, b̃ = scales(cover_transversal(Diagonal(d) * A * Diagonal(e)))
        @test logproduct_gap(ã ./ d, b̃ ./ e, a, b, A) <= 1e-6
        # The sparse default (`:lsqr`) and the dense solve agree.
        ad, bd = cover_transversal(Matrix(A); linsolve=:dense)
        @test logproduct_gap(a, b, ad, bd, A) <= 1e-6
        @test iscover(a, b, A)
        @test isbalanced(a, b, A)
    end

    # Entries (i,j) and (k,σ(i)) with j = σ(k) join the same pair of rows, so the
    # preconditioner pattern must merge them.
    T3 = sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
    a, b, st = _cover_min_abslog2(T3; transversal=true, linsolve=:lsqr, fname=:cover_transversal)
    @test st.precond === :factor
    @test logproduct_gap(a, b, scales(cover_transversal(Matrix(T3); linsolve=:dense))..., T3) <= 1e-7

    # Several support components, each balanced separately.
    Ablk = [1.0 2.0 0 0; 0.25 3.0 0 0; 0 0 1.5 2.5; 0 0 0.75 0]
    a, b = cover_transversal(Ablk)
    @test iscover(a, b, Ablk; atol=1e-9)
    @test isbalanced(a, b, Ablk)
    @test a[3] * b[4] ≈ 2.5 && a[4] * b[3] ≈ 0.75

    # Generic indexing.
    A = [1.0 1 0; 0.25 1 1; 1 0 0]
    a, b = cover_transversal(A)
    Ao = OffsetArray(A, -1:1, 5:7)
    ao, bo = cover_transversal(Ao)
    @test axes(ao, 1) == axes(Ao, 1) && axes(bo, 1) == axes(Ao, 2)
    @test parent(ao) .* parent(bo)' ≈ a .* b'
    B = [9.0 9 9 9; 9 1 1 0; 9 0.25 1 1; 9 1 0 0]
    av, bv = cover_transversal(view(B, 2:4, 2:4))
    @test av .* bv' ≈ a .* b'
    ao, bo = cover_transversal!(OffsetArray(zeros(3), -1:1), OffsetArray(zeros(3), 5:7), Ao)
    @test parent(ao) .* parent(bo)' ≈ a .* b'

    # Element types.
    a32, b32 = cover_transversal(Float32.(A))
    @test eltype(a32) == Float32
    @test iscover(a32, b32, Float32.(A))
    aC, bC = cover_transversal(A .* cis.(reshape(1:9, 3, 3)))
    @test aC .* bC' ≈ a .* b'

    # Mutating form writes through its buffers.
    a2, b2 = zeros(3), zeros(3)
    @test scales(cover_transversal!(a2, b2, A)) === (a2, b2)
    @test (a2, b2) == (a, b)
    @test_throws DimensionMismatch cover_transversal!(zeros(2), zeros(3), A)

    # Errors.
    @test_throws "requires a square matrix" cover_transversal([1.0 2.0 3.0; 4.0 5.0 6.0])
    @test_throws "structurally nonsingular" cover_transversal([1.0 1 1; 1 0 0; 1 0 0])
    @test_throws "structurally nonsingular" cover_transversal([1.0 1; 0 0])
    @test_throws "structurally nonsingular" cover_transversal(sparse([1.0 1 1; 1 0 0; 1 0 0]))
    @test_throws "finite entries" cover_transversal([1.0 Inf; 1 1])
    @test_throws "linsolve must be :auto, :dense, :lsqr, or :woodbury" cover_transversal(A; linsolve=:bad)
end

@testset "cover_transversal, linsolve=:woodbury" begin
    tt(A; kw...) = _cover_min_abslog2(A; transversal=true, fname=:cover_transversal, kw...)
    # Zeros in bands off the diagonal, within the limits of `:woodbury`: at most
    # `n ÷ 4` per row and column and `4n` in total.
    function addzeros(A)
        n = size(A, 1)
        B = copy(A)
        for s in 1:min(4, n ÷ 4), i in 1:n
            B[i, mod1(i + s, n)] = 0
        end
        return B
    end
    function compare(A)
        ad, bd, sd = tt(A; linsolve=:dense)
        aw, bw, sw = tt(A; linsolve=:woodbury)
        @test sd.linsolve === :dense
        @test sw.linsolve === :woodbury
        @test sw.converged
        # The objective is stationary at the minimizer, so it agrees far more
        # tightly than the products.
        mask = A .!= 0
        @test (aw .* bw')[mask] ≈ (ad .* bd')[mask] rtol=1e-7
        @test cover_objective(AbsLog{2}(), aw, bw, A) ≈ cover_objective(AbsLog{2}(), ad, bd, A) rtol=1e-10
        @test iscover(aw, bw, A)
        @test isbalanced(aw, bw, A)
    end
    rng = StableRNG(11)
    for n in (5, 20, 60, 150)
        A = exp.(randn(rng, n, n))
        compare(A)
        compare(addzeros(A))
        dr = exp.(2 .* randn(rng, n))
        dc = exp.(2 .* randn(rng, n))
        @test covaries(M -> tt(M; linsolve=:woodbury)[1:2], addzeros(A), dr, dc; rtol=1e-6)
    end
    # Every transversal through the leading 3×3 block attains the maximum, so
    # its rows form one tight component.
    for n in (6, 20, 60)
        A = exp.(randn(rng, n, n))
        A[1:3, 1:3] .= 100
        compare(A)
        B = addzeros(A)
        B[1:3, 1:3] .= 100
        compare(B)
    end

    # Selection by `:auto`, and the limits of `:woodbury`.
    A = exp.(randn(rng, 20, 20))
    @test tt(A)[3].linsolve === :woodbury
    S = sprand(rng, 40, 40, 0.1) + 5I
    @test tt(S)[3].linsolve === :lsqr
    Z = copy(A)
    Z[1:6, 20] .= 0
    @test tt(Z)[3].linsolve === :dense
    @test_throws "at most min(m, n) ÷ 4 = 5 zeros; got 6" tt(Z; linsolve=:woodbury)
    # Narrow types compute in Float64 and so take the same path; wider ones cannot.
    a32, b32, s32 = tt(Float32.(A); linsolve=:woodbury)
    @test s32.linsolve === :woodbury
    @test eltype(a32) === Float32
    @test iscover(a32, b32, Float32.(A))
    @test_throws "requires Float64 arithmetic" tt(Double64.(A); linsolve=:woodbury)

    # A large penalty weight pushes the condition estimate past the CG
    # threshold, reaching the sparse Cholesky solve.
    _, _, sc = tt(A; linsolve=:woodbury, κ=1e7, maxouter=2)
    @test sc.cholsolves > 0
end
