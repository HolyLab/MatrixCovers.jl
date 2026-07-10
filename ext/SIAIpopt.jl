module SIAIpopt

using JuMP: JuMP, @variable, @objective, @constraint, @NLobjective, @NLconstraint
using Ipopt: Ipopt
using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: AbsLinear, symcover, soft_symcover

# The models are built over 1-based positions 1:n; `pr` maps each position to the
# corresponding axis index of `A`, and results are scattered back onto a vector
# whose axes match `A`'s so offset axes are honored. A row/column of `A` with no
# nonzero entry appears in no constraint or objective term; its scale is set to
# exactly 0, matching the native solvers.

# ============================================================
# Hard cover: symcover_min(::AbsLinear{p}, A)
# Minimizes ∑_{i≤j: A[i,j]≠0} |1 - |A[i,j]|/(a[i]*a[j])|^p
# subject to a[i]*a[j] ≥ |A[i,j]| for all i≤j with A[i,j]≠0.
# Variables: α[i] = log(a[i]); constraint: α[i]+α[j] ≥ log|A[i,j]|.
# Warm-started from symcover(AbsLog{2}(), A).
# ============================================================

function ScaleInvariantAnalysis.symcover_min(::AbsLinear{2}, A)
    axr = axes(A, 1)
    axes(A, 2) == axr || throw(ArgumentError("symcover_min requires a square matrix"))
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    Apos = [A[pr[i], pr[j]] for i in 1:n, j in 1:n]
    supported = [any(!iszero, @view Apos[i, :]) || any(!iszero, @view Apos[:, i]) for i in 1:n]
    # Precompute nonzero upper-triangle entries
    idx = [(i, j) for i in 1:n for j in i:n if Apos[i, j] != 0]
    logA = Dict((i, j) => log(abs(Apos[i, j])) for (i, j) in idx)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    # Warm start from AbsLog{2} cover
    a0 = symcover(ScaleInvariantAnalysis.AbsLog{2}(), A)
    start0 = [supported[k] && !iszero(a0[pr[k]]) ? log(a0[pr[k]]) : zero(T) for k in 1:n]
    @variable(model, α[k=1:n], start = start0[k])
    @NLobjective(model, Min,
        sum((1 - exp(logA[(i,j)] - α[i] - α[j]))^2 for (i,j) in idx))
    for (i, j) in idx
        @constraint(model, α[i] + α[j] >= logA[(i, j)])
    end
    JuMP.optimize!(model)
    a = similar(Array{T}, axr)
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

function ScaleInvariantAnalysis.symcover_min(::AbsLinear{1}, A)
    axr = axes(A, 1)
    axes(A, 2) == axr || throw(ArgumentError("symcover_min requires a square matrix"))
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    Apos = [A[pr[i], pr[j]] for i in 1:n, j in 1:n]
    supported = [any(!iszero, @view Apos[i, :]) || any(!iszero, @view Apos[:, i]) for i in 1:n]
    idx = [(i, j) for i in 1:n for j in i:n if Apos[i, j] != 0]
    logA = Dict((i, j) => log(abs(Apos[i, j])) for (i, j) in idx)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    a0 = symcover(ScaleInvariantAnalysis.AbsLog{2}(), A)
    start0 = [supported[k] && !iszero(a0[pr[k]]) ? log(a0[pr[k]]) : zero(T) for k in 1:n]
    @variable(model, α[k=1:n], start = start0[k])
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
    a = similar(Array{T}, axr)
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

# ============================================================
# Soft cover: soft_symcover_min(::AbsLinear{p}, A)
# Same objective, no coverage constraints.
# ============================================================

function ScaleInvariantAnalysis.soft_symcover_min(::AbsLinear{2}, A)
    axr = axes(A, 1)
    axes(A, 2) == axr || throw(ArgumentError("soft_symcover_min requires a square matrix"))
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    Apos = [A[pr[i], pr[j]] for i in 1:n, j in 1:n]
    supported = [any(!iszero, @view Apos[i, :]) || any(!iszero, @view Apos[:, i]) for i in 1:n]
    # Include ALL entries (including zeros, which contribute (1-0)^2=1 regardless of α)
    nonzero_idx = [(i, j) for i in 1:n, j in 1:n if Apos[i, j] != 0]
    logA = Dict((i, j) => log(abs(Apos[i, j])) for (i, j) in nonzero_idx)
    n_zeros = n^2 - length(nonzero_idx)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[k=1:n], start = zero(T))
    @NLobjective(model, Min,
        sum((1 - exp(logA[(i,j)] - α[i] - α[j]))^2 for (i,j) in nonzero_idx) + n_zeros)
    JuMP.optimize!(model)
    a = similar(Array{T}, axr)
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

function ScaleInvariantAnalysis.soft_symcover_min(::AbsLinear{1}, A)
    axr = axes(A, 1)
    axes(A, 2) == axr || throw(ArgumentError("soft_symcover_min requires a square matrix"))
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    Apos = [A[pr[i], pr[j]] for i in 1:n, j in 1:n]
    supported = [any(!iszero, @view Apos[i, :]) || any(!iszero, @view Apos[:, i]) for i in 1:n]
    nonzero_idx = [(i, j) for i in 1:n, j in 1:n if Apos[i, j] != 0]
    logA = Dict((i, j) => log(abs(Apos[i, j])) for (i, j) in nonzero_idx)
    n_zeros = n^2 - length(nonzero_idx)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    a0 = soft_symcover(ScaleInvariantAnalysis.AbsLinear{1}(), A)
    start0 = [supported[k] && !iszero(a0[pr[k]]) ? log(a0[pr[k]]) : zero(T) for k in 1:n]
    @variable(model, α[k=1:n], start = start0[k])
    @variable(model, t[eachindex(nonzero_idx)] >= 0)
    for (k, (i, j)) in enumerate(nonzero_idx)
        lA = logA[(i, j)]
        @NLconstraint(model,  1 - exp(lA - α[i] - α[j]) <= t[k])
        @NLconstraint(model, -1 + exp(lA - α[i] - α[j]) <= t[k])
    end
    @objective(model, Min, sum(t) + n_zeros)
    JuMP.optimize!(model)
    a = similar(Array{T}, axr)
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

end  # module SIAIpopt
