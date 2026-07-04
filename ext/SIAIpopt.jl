module SIAIpopt

using JuMP: JuMP, @variable, @objective, @constraint, @NLobjective, @NLconstraint
using Ipopt: Ipopt
using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: AbsLinear, symcover, soft_symcover

# ============================================================
# Hard cover: symcover_min(::AbsLinear{p}, A)
# Minimizes ∑_{i≤j: A[i,j]≠0} |1 - |A[i,j]|/(a[i]*a[j])|^p
# subject to a[i]*a[j] ≥ |A[i,j]| for all i≤j with A[i,j]≠0.
# Variables: α[i] = log(a[i]); constraint: α[i]+α[j] ≥ log|A[i,j]|.
# Warm-started from symcover(AbsLog{2}(), A).
# ============================================================

function ScaleInvariantAnalysis.symcover_min(::AbsLinear{2}, A)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min requires a square matrix"))
    n = size(A, 1)
    # Precompute nonzero upper-triangle entries
    idx = [(i, j) for i in 1:n for j in i:n if A[i, j] != 0]
    logA = Dict((i, j) => log(abs(A[i, j])) for (i, j) in idx)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    # Warm start from AbsLog{2} cover
    a0 = symcover(ScaleInvariantAnalysis.AbsLog{2}(), A)
    @variable(model, α[k=1:n], start = log(a0[k]))
    @NLobjective(model, Min,
        sum((1 - exp(logA[(i,j)] - α[i] - α[j]))^2 for (i,j) in idx))
    for (i, j) in idx
        @constraint(model, α[i] + α[j] >= logA[(i, j)])
    end
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α))
end

function ScaleInvariantAnalysis.symcover_min(::AbsLinear{1}, A)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_min requires a square matrix"))
    n = size(A, 1)
    idx = [(i, j) for i in 1:n for j in i:n if A[i, j] != 0]
    logA = Dict((i, j) => log(abs(A[i, j])) for (i, j) in idx)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    a0 = symcover(ScaleInvariantAnalysis.AbsLog{2}(), A)
    @variable(model, α[k=1:n], start = log(a0[k]))
    # |1 - exp(lA - αi - αj)| via auxiliary variables t ≥ 0 and slack s
    @variable(model, t[eachindex(idx)] >= 0)
    for (k, (i, j)) in enumerate(idx)
        lA = logA[(i, j)]
        @NLconstraint(model,  1 - exp(lA - α[i] - α[j]) <= t[k])
        @NLconstraint(model, -1 + exp(lA - α[i] - α[j]) <= t[k])
    end
    @objective(model, Min, sum(t))
    for (i, j) in idx
        @constraint(model, α[i] + α[j] >= logA[(i, j)])
    end
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α))
end

# ============================================================
# Soft cover: soft_symcover_min(::AbsLinear{p}, A)
# Same objective, no coverage constraints.
# ============================================================

function ScaleInvariantAnalysis.soft_symcover_min(::AbsLinear{2}, A)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover_min requires a square matrix"))
    n = size(A, 1)
    # Include ALL entries (including zeros, which contribute (1-0)^2=1 regardless of α)
    nonzero_idx = [(i, j) for i in 1:n, j in 1:n if A[i, j] != 0]
    logA = Dict((i, j) => log(abs(A[i, j])) for (i, j) in nonzero_idx)
    n_zeros = n^2 - length(nonzero_idx)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    # a0 = soft_symcover(ScaleInvariantAnalysis.AbsLinear{2}(), A)
    a0 = ones(n)
    @variable(model, α[k=1:n], start = log(a0[k]))
    @NLobjective(model, Min,
        sum((1 - exp(logA[(i,j)] - α[i] - α[j]))^2 for (i,j) in nonzero_idx) + n_zeros)
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α))
end

function ScaleInvariantAnalysis.soft_symcover_min(::AbsLinear{1}, A)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("soft_symcover_min requires a square matrix"))
    n = size(A, 1)
    nonzero_idx = [(i, j) for i in 1:n, j in 1:n if A[i, j] != 0]
    logA = Dict((i, j) => log(abs(A[i, j])) for (i, j) in nonzero_idx)
    n_zeros = n^2 - length(nonzero_idx)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    a0 = soft_symcover(ScaleInvariantAnalysis.AbsLinear{1}(), A)
    @variable(model, α[k=1:n], start = log(a0[k]))
    @variable(model, t[eachindex(nonzero_idx)] >= 0)
    for (k, (i, j)) in enumerate(nonzero_idx)
        lA = logA[(i, j)]
        @NLconstraint(model,  1 - exp(lA - α[i] - α[j]) <= t[k])
        @NLconstraint(model, -1 + exp(lA - α[i] - α[j]) <= t[k])
    end
    @objective(model, Min, sum(t) + n_zeros)
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α))
end

end  # module SIAIpopt
