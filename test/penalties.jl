# Penalty objectives (cover_objective).

@testset "cover_objective" begin
    A = [4.0 1.5; 1.5 1.0]
    a = [2.0, 1.0]
    # AbsLog{1}: sum of log-domain excesses over nonzero entries
    @test cover_objective(AbsLog{1}(), a, A) ≈ sum(abs(log(a[i]*a[j]/abs(A[i,j]))) for i in 1:2, j in 1:2 if A[i,j] != 0)
    # AbsLog{2}: sum of squared log-domain excesses
    @test cover_objective(AbsLog{2}(), a, A) ≈ sum(abs(log(a[i]*a[j]/abs(A[i,j])))^2 for i in 1:2, j in 1:2 if A[i,j] != 0)
    # AbsLinear{1} and AbsLinear{2}: ratio deviations from 1 (ALL entries, including zeros)
    @test cover_objective(AbsLinear{1}(), a, A) ≈ sum(abs(abs(A[i,j])/(a[i]*a[j]) - 1) for i in 1:2, j in 1:2)
    @test cover_objective(AbsLinear{2}(), a, A) ≈ sum((abs(A[i,j])/(a[i]*a[j]) - 1)^2 for i in 1:2, j in 1:2)
    # Two-argument form equals one-argument form
    @test cover_objective(AbsLog{2}(), a, a, A) == cover_objective(AbsLog{2}(), a, A)
    @test cover_objective(AbsLinear{2}(), a, a, A) == cover_objective(AbsLinear{2}(), a, A)
    # AbsLog{p}: zero entries contribute 0
    A0 = [1.0 0.0; 0.0 5.0]
    a0 = [1.0, 2.0]
    @test cover_objective(AbsLog{1}(), a0, A0) ≈ log(5/4)
    @test cover_objective(AbsLog{2}(), a0, A0) ≈ log(5/4)^2
    # AbsLinear{p}: zero entries contribute 1 each
    @test cover_objective(AbsLinear{1}(), a0, A0) ≈ 0.0 + 1.0 + 1.0 + 1/4    # (0,1), (1,0) off-diag zeros
    @test cover_objective(AbsLinear{2}(), a0, A0) ≈ 0.0 + 1.0 + 1.0 + 1/16   # (0,1), (1,0) off-diag zeros
end

@testset "PowerMean" begin
    φ(p, r) = (r^p - 1 - p * log(r)) / p
    for p in (1, 2, 3.5, 0.5), r in (1e-3, 0.3, 1.0, 1.7, 40.0)
        @test PowerMean{p}()(r) ≈ φ(p, r) rtol=1e-12 atol=1e-15
    end
    # Accurate near r = 1, where the formula cancels to O(log(r)^2).
    @test PowerMean{2}()(1 + 1e-9) ≈ 1e-18 rtol=1e-6
    @test PowerMean{2}()(1.0) == 0
    # Zero is the structural-zero convention; an uncovered entry is infinite.
    @test PowerMean{2}()(0.0) === 0.0
    @test PowerMean{2}()(Inf) === Inf
    @test PowerMean{2}()(Float32(0.5)) isa Float32
    @test_throws "requires a real exponent p > 0" PowerMean{0}()
    @test_throws "requires a real exponent p > 0" PowerMean{-1.0}()
    @test_throws "requires a real exponent p > 0" PowerMean{:two}()

    # Zero entries contribute nothing: the objective is the sum over the support.
    A = [4.0 0.0 1.5; 0.0 0.0 2.0; 1.5 3.0 0.0]
    a, b = [2.0, 0.5, 1.0], [1.5, 3.0, 0.7]
    for p in (1, 2)
        E = cover_objective(PowerMean{p}(), a, b, A)
        @test isfinite(E)
        @test E ≈ sum(φ(p, abs(A[i, j]) / (a[i] * b[j])) for i in 1:3, j in 1:3 if A[i, j] != 0)
        @test cover_objective(PowerMean{p}(), a, b, sparse(A)) ≈ E
    end
end

# A matrix readable only through the traversal hook: `getindex` throws, so any
# full-grid scan fails outright.
struct SupportOnlyMatrix{T} <: AbstractMatrix{T}
    entries::Vector{Tuple{Int,Int,T}}
    sz::Tuple{Int,Int}
end
Base.size(M::SupportOnlyMatrix) = M.sz
Base.getindex(::SupportOnlyMatrix, ::Int, ::Int) =
    error("SupportOnlyMatrix must be read through foreach_support")
function MatrixCovers.foreach_support(f, M::SupportOnlyMatrix)
    for (i, j, v) in M.entries
        iszero(v) || f(i, j, abs(v))
    end
    return nothing
end

@testset "cover_objective reads through the support hook" begin
    entries = [(1, 2, 3.0), (2, 1, -1.5), (3, 3, 4.0)]
    M = SupportOnlyMatrix(entries, (3, 4))
    dense = zeros(3, 4)
    for (i, j, v) in entries
        dense[i, j] = v
    end
    a, b = [2.0, 1.0, 0.5], [1.5, 0.5, 3.0, 1.0]
    for ϕ in SOFT_PENALTIES
        # Agreement with the dense reference pins the zero-entry accounting: the
        # entries the hook skips still carry `ϕ(0)`, which `AbsLinear` makes nonzero.
        @test cover_objective(ϕ, a, b, M) ≈ cover_objective(ϕ, a, b, dense)
    end

    # A zero scale under a nonzero entry is uncovered, and under a zero entry is not.
    az = [0.0, 1.0, 0.5]
    @test isinf(cover_objective(AbsLog{2}(), az, b, M))
    # Column 4 carries no support, so zeroing its scale constrains nothing.
    @test isfinite(cover_objective(AbsLog{2}(), a, [1.5, 0.5, 3.0, 0.0], M))

    @test cover_objective(AbsLog{2}(), b, a, M') ≈ cover_objective(AbsLog{2}(), a, b, dense)

    # A penalty infinite at `r = 0` is legal, and a matrix with no zero entries must
    # not pick up its value: the zero-entry term is skipped, not multiplied by zero.
    infatzero(r) = iszero(r) ? Inf : abs(log(r))
    @test cover_objective(infatzero, [1.0, 1.0], [1.0, 1.0], [1.0 2.0; 3.0 4.0]) ≈
          sum(abs(log(v)) for v in (1.0, 2.0, 3.0, 4.0))
    @test isinf(cover_objective(infatzero, [1.0, 1.0], [1.0, 1.0], [1.0 0.0; 3.0 4.0]))
end

@testset "cover_objective: index checking" begin
    A = [4.0 1.5; 1.5 1.0]
    @test_throws "indices of `a` must match row-indexing of `A`" cover_objective(AbsLog{2}(), [1.0], [1.0, 1.0], A)
    @test_throws "indices of `b` must match column-indexing of `A`" cover_objective(AbsLog{2}(), [1.0, 1.0], [1.0], A)
    # Offset axes are honored, not merely tolerated: the score is unchanged.
    Ao = OffsetArray(A, -1:0, 2:3)
    ao, bo = OffsetArray([2.0, 1.0], -1:0), OffsetArray([1.0, 3.0], 2:3)
    for ϕ in SOFT_PENALTIES
        @test cover_objective(ϕ, ao, bo, Ao) ≈ cover_objective(ϕ, [2.0, 1.0], [1.0, 3.0], A)
    end
end

@testset "cover_objective: complex input" begin
    # The objective depends only on entry magnitudes, so complex A and abs.(A)
    # give identical results, and the accumulator stays real.
    Ac = [1.0+2.0im 0.5-1.0im; 0.3+0.1im 3.0+0.0im]
    a, b = [2.0, 1.0], [1.5, 0.5]
    for ϕ in SOFT_PENALTIES
        v = cover_objective(ϕ, a, b, Ac)
        @test v isa Real
        @test v == cover_objective(ϕ, a, b, abs.(Ac))
    end
end

