# Penalty objectives (cover_objective) and the scalar utilities dotabs/divmag.

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

@testset "dotabs" begin
    @test dotabs([1.0, -2.0, 3.0], [4.0, 5.0, -6.0]) ≈ 4.0 + 10.0 + 18.0
    @test dotabs([0.0, 1.0], [1.0, 0.0]) == 0.0
    @test dotabs(big.([1.0, 2.0]), big.([3.0, 4.0])) ≈ 3.0 + 8.0
    @test dotabs([1.0 + 2.0im, -1.0im], [3.0, 1.0 + 1.0im]) ≈ abs((1.0 + 2.0im) * 3.0) + abs(-1.0im * (1.0 + 1.0im))
end

@testset "divmag" begin
    A = [1.0 -0.2; -0.2 0]
    b = [0.75, 7.0]
    a, mag = divmag(A, b)
    @test abs(dotabs(A \ b, a) - dotabs(big.(A) \ big.(b), a)) <= 2 * eps(mag)
    # Uniform scaling covariance
    asc, magsc = divmag(1000 * A, 1000 * b)
    @test asc ≈ sqrt(1000) .* a
    @test magsc ≈ sqrt(1000) * mag
    # Diagonal scaling covariance, from mild to strong anisotropy
    for d in ([1.0, 1.0], [0.5, 2.0], [0.05, 3.0], [10.0, 0.01])
        Ad = A .* d .* d'
        bd = b .* d
        a_d, mag_d = divmag(Ad, bd)
        @test abs(dotabs(Ad \ bd, a_d) - dotabs(big.(Ad) \ big.(bd), a_d)) <= 100 * eps(mag_d)
    end
    # Ill-conditioned matrix
    A_ill = [1.0 -0.9999; -0.9999 1]
    a_ill, mag_ill = divmag(A_ill, b)
    @test abs(dotabs(A_ill \ b, a_ill) - dotabs(big.(A_ill) \ big.(b), a_ill)) > 10^6 * eps(mag_ill)
    # With use_cond=true, accuracy is recovered
    a_ill2, mag_ill2 = divmag(A_ill, b; use_cond=true)
    @test abs(dotabs(A_ill \ b, a_ill2) - dotabs(big.(A_ill) \ big.(b), a_ill2)) <= 10^3 * eps(mag_ill2)
end
