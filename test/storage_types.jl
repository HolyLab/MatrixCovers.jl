# Storage-type equivalence: sparse, structured, and wrapped inputs must match the
# dense reference.

@testset "SparseMatrixCSC" begin
    for Adense in ([2.0 1.0; 1.0 3.0], [1.0 -0.2; -0.2 0.0], [0.0 12.0 9.0; 12.0 7.0 12.0; 9.0 12.0 0.0],
                   [100.0 1.0; 1.0 0.01])
        for A in (sparse(Adense), Symmetric(sparse(tril(Adense)), :L), Symmetric(sparse(triu(Adense)), :U),
                  Hermitian(sparse(tril(Adense)), :L), Hermitian(sparse(triu(Adense)), :U))
            for ϕ in PENALTIES
                a = symcover(ϕ, A)
                @test iscover(a, Adense; atol=1e-12)
            end
            # Default dispatch
            a = symcover(A)
            @test iscover(a, Adense; atol=1e-12)
            # cover_objective matches dense for AbsLog
            a = symcover(AbsLog{2}(), A)
            @test cover_objective(AbsLog{2}(), a, A) ≈ cover_objective(AbsLog{2}(), a, Adense)
        end
    end
    for Adense in ([2.0 1.0; 1.0 3.0], [0.0 1.0; -2.0 0.0], [1.0 2.0 3.0; 4.0 5.0 6.0])
        A = sparse(Adense)
        a, b = cover(A)
        @test iscover(a, b, Adense; atol=1e-12)
    end
    # Zero-diagonal sparse matrix
    A0 = sparse([0.0 1.0; 1.0 0.0])
    @test symcover(A0) == [1.0, 1.0]
end

@testset "structured matrices" begin
    @testset "Diagonal" begin
        for dv in ([4.0, 9.0, 1.0], [4.0, 0.0, 1.0], [0.25, 100.0])
            D = Diagonal(dv)
            Ddense = Matrix(D)
            for ϕ in PENALTIES
                a = symcover(ϕ, D)
                @test iscover(a, Ddense; atol=1e-12)
            end
            a3, b3 = cover(D)
            @test iscover(a3, b3, Ddense; atol=1e-12)
        end
    end

    @testset "PlusMinus1Banded" begin
        asym_cases = [
            Bidiagonal([3.0, 2.0, 1.0], [6.0, 0.5], :U),
            Bidiagonal([3.0, 2.0, 1.0], [6.0, 0.5], :L),
            Tridiagonal([1.0, 0.5], [3.0, 2.0, 1.0], [4.0, 0.5]),
        ]
        for A in asym_cases
            Adense = Matrix(A)
            a, b = cover(A)
            @test iscover(a, b, Adense; atol=1e-12)
        end
        sym_cases = [
            SymTridiagonal([4.0, 3.0, 1.0], [2.0, 0.5]),
            SymTridiagonal([0.0, 3.0, 0.0], [2.0, 0.5]),
            Tridiagonal([2.0, 0.5], [4.0, 3.0, 1.0], [2.0, 0.5]),
        ]
        for A in sym_cases
            Adense = Matrix(A)
            a, b = cover(A)
            @test iscover(a, b, Adense; atol=1e-12)
            for ϕ in PENALTIES
                a = symcover(ϕ, A)
                @test iscover(a, Adense; atol=1e-12)
            end
            # cover_objective matches dense for AbsLog{2}
            a = symcover(AbsLog{2}(), A)
            @test cover_objective(AbsLog{2}(), a, A) ≈ cover_objective(AbsLog{2}(), a, Adense)
        end
    end

    @testset "Adjoint and Transpose" begin
        for Adense in ([1.0 2.0; 3.0 4.0], [1.0 2.0 3.0; 4.0 5.0 6.0])
            for wrapper in (adjoint, transpose)
                A = wrapper(Adense)
                Adense_wrap = Matrix(A)
                a, b = cover(A)
                @test iscover(a, b, Adense_wrap; atol=1e-12)
                # cover_objective matches dense
                @test cover_objective(AbsLog{2}(), a, b, A) ≈ cover_objective(AbsLog{2}(), a, b, Adense_wrap)
                # Objectives are same as computing cover on parent and swapping
                a0, b0 = cover(Adense)
                @test cover_objective(AbsLog{2}(), a, b, A) ≈ cover_objective(AbsLog{2}(), a0, b0, Adense)
            end
        end
    end
end

@testset "traversal-based kernels match dense reference" begin
    # unconstrained_min! and tighten_cover! are order-insensitive folds over
    # foreach_support(_sym) (sum/min accumulations), so structured/sparse
    # storage must agree with the dense form up to floating-point summation
    # order (rtol=1e-12). The full symcover pipeline includes the bucketed
    # feasibility boost, whose within-bucket processing order follows the
    # storage type's traversal order: storages that traverse the canonical
    # triangle in the dense fallback's column-major order are compared
    # elementwise, the rest on feasibility and objective value. cover's boost
    # order is likewise storage-dependent and is not compared elementwise.
    rng = StableRNG(11)
    n = 8
    Adense = randn(rng, n, n); Adense = Adense + Adense'
    Asp = sparse(Adense)
    Ssp_U = Symmetric(sparse(triu(Adense)), :U)
    Ssp_L = Symmetric(sparse(tril(Adense)), :L)
    Hsp_U = Hermitian(sparse(triu(Adense)), :U)
    D = Diagonal(rand(rng, n) .+ 0.1)
    dv, ev = rand(rng, n) .+ 0.1, rand(rng, n - 1) .+ 0.1
    St = SymTridiagonal(dv, ev)
    Tsym = Tridiagonal(ev, dv, ev)

    # Traversal order matches the dense fallback's column-major canonical
    # triangle: compare elementwise.
    a_ref = symcover(AbsLog{2}(), Adense)
    for A in (Asp, Ssp_U, Hsp_U)
        @test symcover(AbsLog{2}(), A) ≈ a_ref rtol=1e-12
    end
    @test symcover(AbsLog{2}(), D) ≈ symcover(AbsLog{2}(), Matrix(D)) rtol=1e-12
    @test symcover(AbsLog{2}(), St) ≈ symcover(AbsLog{2}(), Tsym) rtol=1e-12   # same symmetric-valued matrix

    # Traversal order differs from the dense fallback's (Ssp_L keys pairs by
    # the smaller index; St/Tsym visit all diagonal entries before any
    # off-diagonal, dense interleaves them), so the bucketed boost's
    # within-bucket order can differ: compare on feasibility and objective
    # value rather than elementwise.
    objclose(a1, M1, a2, M2) = isapprox(cover_objective(AbsLog{2}(), a1, M1),
        cover_objective(AbsLog{2}(), a2, M2); rtol=1e-2, atol=1e-10)
    for A in (Ssp_L,)
        aA, aref = symcover(AbsLog{2}(), A), symcover(AbsLog{2}(), Adense)
        @test iscover(aA, Adense; rtol=8eps())
        @test objclose(aA, Adense, aref, Adense)
    end
    for A in (St, Tsym)
        aA, aM = symcover(AbsLog{2}(), A), symcover(AbsLog{2}(), Matrix(A))
        @test iscover(aA, Matrix(A); rtol=8eps())
        @test objclose(aA, Matrix(A), aM, Matrix(A))
    end

    # Kernel 3 (tighten_cover!) directly, starting from a Kernel-1-derived vector.
    for A in (Asp, Ssp_U, D, St, Tsym)
        a_start = zeros(size(A, 1))
        unconstrained_min!(AbsLog{2}(), a_start, A)
        @test tighten_cover!(copy(a_start), A) ≈ tighten_cover!(copy(a_start), Matrix(A)) rtol=1e-12
    end
    for A in (Asp, D, St, Tsym)
        a_start, b_start = zeros(size(A, 1)), zeros(size(A, 2))
        unconstrained_min!(AbsLog{2}(), a_start, b_start, A)
        a1, b1 = tighten_cover!(copy(a_start), copy(b_start), A)
        a2, b2 = tighten_cover!(copy(a_start), copy(b_start), Matrix(A))
        @test a1 ≈ a2 rtol=1e-12
        @test b1 ≈ b2 rtol=1e-12
    end
end

@testset "AbsLinear cross-type equivalence" begin
    # symcover(::AbsLinear) reaches sparse/structured inputs solely through
    # the generic AbstractMatrix method and foreach_support(_sym) dispatch (no
    # per-type specialization remains), so it must agree with the dense result
    # up to the traversal-order ties noted above.
    rng = StableRNG(13)
    n = 7
    Adense = randn(rng, n, n); Adense = Adense + Adense'
    Asp = sparse(Adense)
    Ssp_U = Symmetric(sparse(triu(Adense)), :U)
    dv, ev = rand(rng, n) .+ 0.1, rand(rng, n - 1) .+ 0.1
    St = SymTridiagonal(dv, ev)
    Tsym = Tridiagonal(ev, dv, ev)

    a_ref = symcover(AbsLinear{2}(), Adense)
    for A in (Asp, Ssp_U)
        @test symcover(AbsLinear{2}(), A) ≈ a_ref rtol=1e-12
    end
    @test symcover(AbsLinear{2}(), St) ≈ symcover(AbsLinear{2}(), Tsym) rtol=1e-12
end

@testset "native solvers on sparse and structured inputs" begin
    # The native AbsLog{2} MCM solvers (`symcover_min`/`cover_min`) and the AbsLinear
    # soft covers must agree with the dense reference on `Matrix(A)` when handed a
    # sparse-backed or structured input, and the hard MCM covers must stay feasible.
    # On a `SparseMatrixCSC`/`Symmetric`/`Hermitian`-sparse the MCM solvers default to
    # the matrix-free LSQR inner solve; structured inputs use the generic dense path.

    symdenses = [[2.0 1.0 0.0; 1.0 3.0 2.0; 0.0 2.0 5.0],
                 [4.0 0.0 1.0; 0.0 0.0 0.0; 1.0 0.0 2.0]]   # second has a zero row/column
    for M in symdenses
        ad = symcover_min(AbsLog{2}(), M)
        asd = soft_symcover(AbsLinear{2}(), M)
        for A in (sparse(M), Symmetric(sparse(triu(M)), :U), Symmetric(sparse(tril(M)), :L),
                  Hermitian(sparse(triu(M)), :U))
            a = symcover_min(AbsLog{2}(), A)
            @test iscover(a, M; atol=1e-7)
            @test cover_objective(AbsLog{2}(), a, M) ≈ cover_objective(AbsLog{2}(), ad, M) rtol = 1e-7 atol = 1e-10
            as = soft_symcover(AbsLinear{2}(), A)
            @test cover_objective(AbsLinear{2}(), as, M) ≈ cover_objective(AbsLinear{2}(), asd, M) rtol = 1e-7 atol = 1e-10
            # Scale vectors are dense: return a plain Vector, matching cover/symcover.
            @test as isa Vector{Float64}
        end
    end
    @test symcover_min(AbsLog{2}(), sparse(symdenses[1])) isa Vector{Float64}
    # Every soft_symcover penalty returns a dense Vector on sparse-backed input.
    let Ssp = sparse(symdenses[1])
        for ϕ in PENALTIES
            @test soft_symcover(ϕ, Ssp) isa Vector{Float64}
            @test soft_symcover(ϕ, Symmetric(Ssp)) isa Vector{Float64}
        end
    end

    gendenses = [[1.0 2.0 0.0; 0.0 5.0 4.0; 3.0 0.0 6.0],
                 [2.0 0.0; 0.0 3.0]]   # second is diagonal: disconnected support
    for M in gendenses
        A = sparse(M)
        ad, bd = cover_min(AbsLog{2}(), M)
        a, b = cover_min(AbsLog{2}(), A)
        @test iscover(a, b, M; atol=1e-7)
        @test cover_objective(AbsLog{2}(), a, b, M) ≈ cover_objective(AbsLog{2}(), ad, bd, M) rtol = 1e-7 atol = 1e-10
        asd, bsd = soft_cover(AbsLinear{2}(), M)
        as, bs = soft_cover(AbsLinear{2}(), A)
        @test cover_objective(AbsLinear{2}(), as, bs, M) ≈ cover_objective(AbsLinear{2}(), asd, bsd, M) rtol = 1e-7 atol = 1e-10
        @test as isa Vector{Float64} && bs isa Vector{Float64}
    end
    let (a, b) = cover_min(AbsLog{2}(), sparse(gendenses[1]))
        @test a isa Vector{Float64} && b isa Vector{Float64}
    end

    for D in (Diagonal([4.0, 9.0, 1.0]), Diagonal([4.0, 0.0, 1.0]))
        M = Matrix(D)
        a = symcover_min(AbsLog{2}(), D)
        @test iscover(a, M; atol=1e-7)
        @test cover_objective(AbsLog{2}(), a, M) ≈
              cover_objective(AbsLog{2}(), symcover_min(AbsLog{2}(), M), M) rtol = 1e-7
        a2, b2 = cover_min(AbsLog{2}(), D)   # disconnected support: exercises the gauge ridge
        @test iscover(a2, b2, M; atol=1e-7)
        @test cover_objective(AbsLog{2}(), a2, b2, M) ≈ 0.0 atol = 1e-8   # diagonal is exactly coverable
    end
    for A in (SymTridiagonal([4.0, 3.0, 1.0], [2.0, 0.5]),
              Tridiagonal([1.0, 0.5], [3.0, 2.0, 1.0], [4.0, 0.5]))
        M = Matrix(A)
        a2, b2 = cover_min(AbsLog{2}(), A)
        @test iscover(a2, b2, M; atol=1e-7)
        @test cover_objective(AbsLog{2}(), a2, b2, M) ≈
              cover_objective(AbsLog{2}(), cover_min(AbsLog{2}(), M)..., M) rtol = 1e-7
    end
end
