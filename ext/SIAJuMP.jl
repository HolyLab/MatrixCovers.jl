module SIAJuMP

using JuMP: JuMP, @variable, @objective, @constraint
using HiGHS: HiGHS
using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: AbsLog
using LinearAlgebra: dot

# Exact reference for the native `symcover_min(::AbsLog{2})`: same QP, solved by
# HiGHS. Not exported; used by the test suite to cross-check the native solver.
function ScaleInvariantAnalysis.symcover_min_jump(::AbsLog{2}, A)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min_jump requires a square matrix"))
    logA = log.(abs.(A))
    n = size(A, 1)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:n])
    @objective(model, Min, sum(abs2, α[i] + α[j] - logA[i, j] for i in 1:n, j in 1:n if A[i, j] != 0))
    for i in 1:n, j in i:n
        A[i, j] != 0 && @constraint(model, α[i] + α[j] - logA[i, j] >= 0)
    end
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α))
end

function ScaleInvariantAnalysis.symcover_min(::AbsLog{1}, A)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min requires a square matrix"))
    logA = log.(abs.(A))
    n = size(A, 1)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:n])
    nonzero_rows = count(!iszero, A, dims=1)'
    nonzero_cols = count(!iszero, A, dims=2)
    @objective(model, Min, dot(α, nonzero_rows .+ nonzero_cols))
    for i in 1:n, j in i:n
        A[i, j] != 0 && @constraint(model, α[i] + α[j] - logA[i, j] >= 0)
    end
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α))
end

function ScaleInvariantAnalysis.cover_min_jump(::AbsLog{2}, A)
    logA = log.(abs.(A))
    m, n = size(A)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:m])
    @variable(model, β[1:n])
    @objective(model, Min, sum(abs2, α[i] + β[j] - logA[i, j] for i in 1:m, j in 1:n if A[i, j] != 0))
    for i in 1:m, j in 1:n
        A[i, j] != 0 && @constraint(model, α[i] + β[j] - logA[i, j] >= 0)
    end
    nza, nzb = sum(!iszero, A; dims=2)[:], sum(!iszero, A; dims=1)[:]
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α)), exp.(JuMP.value.(β))
end

function ScaleInvariantAnalysis.cover_min(::AbsLog{1}, A)
    logA = log.(abs.(A))
    m, n = size(A)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:m])
    @variable(model, β[1:n])
    nonzero_rows = count(!iszero, A, dims=1)'
    nonzero_cols = count(!iszero, A, dims=2)
    @objective(model, Min, dot(α, nonzero_cols) + dot(β, nonzero_rows))
    for i in 1:m, j in 1:n
        A[i, j] != 0 && @constraint(model, α[i] + β[j] - logA[i, j] >= 0)
    end
    nza, nzb = sum(!iszero, A; dims=2)[:], sum(!iszero, A; dims=1)[:]
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α)), exp.(JuMP.value.(β))
end

# Soft (unconstrained) symmetric cover: minimize ∑ (log r_ij)² with no constraints.
# The objective is quadratic in α = log a, solved as a QP.
function ScaleInvariantAnalysis.soft_symcover_min(::AbsLog{2}, A)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover_min requires a square matrix"))
    logA = log.(abs.(A))
    n = size(A, 1)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:n])
    @objective(model, Min, sum(abs2, α[i] + α[j] - logA[i, j] for i in 1:n, j in 1:n if A[i, j] != 0))
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α))
end

end
