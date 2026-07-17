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

