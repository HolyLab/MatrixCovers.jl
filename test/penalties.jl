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
    for ϕ in PENALTIES
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
    for ϕ in PENALTIES
        @test cover_objective(ϕ, ao, bo, Ao) ≈ cover_objective(ϕ, [2.0, 1.0], [1.0, 3.0], A)
    end
end

@testset "cover_objective: complex input" begin
    # The objective depends only on entry magnitudes, so complex A and abs.(A)
    # give identical results, and the accumulator stays real.
    Ac = [1.0+2.0im 0.5-1.0im; 0.3+0.1im 3.0+0.0im]
    a, b = [2.0, 1.0], [1.5, 0.5]
    for ϕ in PENALTIES
        v = cover_objective(ϕ, a, b, Ac)
        @test v isa Real
        @test v == cover_objective(ϕ, a, b, abs.(Ac))
    end
end

