module SIAIpopt

using JuMP: JuMP, @variable, @objective, @constraint, @NLobjective, @NLconstraint
using Ipopt: Ipopt
using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: AbsLinear, soft_symcover, initialize_symcover

# The models are built over 1-based positions 1:n; `pr`/`pc` map each position to the
# corresponding axis index of `A`, and results are scattered back onto vectors
# whose axes match `A`'s so offset axes are honored. A row/column of `A` with no
# nonzero entry appears in no constraint or objective term; its scale is set to
# exactly 0, matching the native solvers.

# The AbsLinear objectives are non-convex, so Ipopt returns a local minimum of
# whichever basin it descends into from the start it is given. That makes the start a
# genuine input rather than a hint: the `*_min!` refiners take it from the caller, and
# the non-mutating entry points pick one.

# Ipopt is a local solver on a non-convex problem; any termination other than a
# solved one means the returned point does not solve the problem posed, so it is an
# error rather than a result to be quietly handed back.
function check_solved(model, fname)
    status = JuMP.termination_status(model)
    status == JuMP.LOCALLY_SOLVED || status == JuMP.OPTIMAL ||
        error("$fname: Ipopt terminated with status $status")
    return nothing
end

# ============================================================
# Hard cover: symcover_min!(::AbsLinear{p}, a, A)
# Minimizes ∑_{i≤j: A[i,j]≠0} |1 - |A[i,j]|/(a[i]*a[j])|^p
# subject to a[i]*a[j] ≥ |A[i,j]| for all i≤j with A[i,j]≠0.
# Variables: α[i] = log(a[i]); constraint: α[i]+α[j] ≥ log|A[i,j]|.
# ============================================================

function ScaleInvariantAnalysis.symcover_min!(::AbsLinear{2}, a::AbstractVector, A)
    ScaleInvariantAnalysis._prepare_symcover_start!(a, A)
    axr = axes(A, 1)
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
    start0 = [supported[k] && !iszero(a[pr[k]]) ? log(T(a[pr[k]])) : zero(T) for k in 1:n]
    @variable(model, α[k=1:n], start = start0[k])
    @NLobjective(model, Min,
        sum((1 - exp(logA[(i,j)] - α[i] - α[j]))^2 for (i,j) in idx))
    for (i, j) in idx
        @constraint(model, α[i] + α[j] >= logA[(i, j)])
    end
    JuMP.optimize!(model)
    check_solved(model, "symcover_min!")
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

function ScaleInvariantAnalysis.symcover_min!(::AbsLinear{1}, a::AbstractVector, A)
    ScaleInvariantAnalysis._prepare_symcover_start!(a, A)
    axr = axes(A, 1)
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    Apos = [A[pr[i], pr[j]] for i in 1:n, j in 1:n]
    supported = [any(!iszero, @view Apos[i, :]) || any(!iszero, @view Apos[:, i]) for i in 1:n]
    idx = [(i, j) for i in 1:n for j in i:n if Apos[i, j] != 0]
    logA = Dict((i, j) => log(abs(Apos[i, j])) for (i, j) in idx)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    start0 = [supported[k] && !iszero(a[pr[k]]) ? log(T(a[pr[k]])) : zero(T) for k in 1:n]
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
    check_solved(model, "symcover_min!")
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

function ScaleInvariantAnalysis.symcover_min(ϕ::Union{AbsLinear{1},AbsLinear{2}}, A)
    axes(A, 2) == axes(A, 1) || throw(ArgumentError("symcover_min requires a square matrix"))
    a = initialize_symcover(A)
    return ScaleInvariantAnalysis.symcover_min!(ϕ, a, A)
end

# ============================================================
# Hard cover: cover_min!(::AbsLinear{p}, a, b, A)
# The bipartite analog of symcover_min!: row scales α = log a, column scales
# β = log b, residuals over every stored (i, j) rather than over i ≤ j. The product
# a[i]*b[j] is invariant under (α, β) → (α + s, β - s), so — unlike the symmetric
# problem, which has no such freedom — the model is degenerate along that direction
# until the balance constraint ∑ nzaᵢ αᵢ = ∑ nzbⱼ βⱼ pins it, exactly as
# cover_min(::AbsLog{1}) does. Without it the split Ipopt reports between `a` and `b`
# would be arbitrary.
# ============================================================

function ScaleInvariantAnalysis.cover_min!(::AbsLinear{2}, a::AbstractVector, b::AbstractVector, A)
    ScaleInvariantAnalysis._prepare_cover_start!(a, b, A)
    axr, axc = axes(A, 1), axes(A, 2)
    T = float(real(eltype(A)))
    pr, pc = collect(axr), collect(axc)
    m, n = length(pr), length(pc)
    Apos = [A[pr[i], pc[j]] for i in 1:m, j in 1:n]
    nza = vec(count(!iszero, Apos, dims=2))
    nzb = vec(count(!iszero, Apos, dims=1))
    idx = [(i, j) for i in 1:m for j in 1:n if Apos[i, j] != 0]
    logA = Dict((i, j) => log(abs(Apos[i, j])) for (i, j) in idx)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    α0 = [nza[i] > 0 ? log(T(a[pr[i]])) : zero(T) for i in 1:m]
    β0 = [nzb[j] > 0 ? log(T(b[pc[j]])) : zero(T) for j in 1:n]
    @variable(model, α[i=1:m], start = α0[i])
    @variable(model, β[j=1:n], start = β0[j])
    @NLobjective(model, Min,
        sum((1 - exp(logA[(i,j)] - α[i] - β[j]))^2 for (i,j) in idx))
    for (i, j) in idx
        @constraint(model, α[i] + β[j] >= logA[(i, j)])
    end
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    check_solved(model, "cover_min!")
    for (i, k) in pairs(pr)
        a[k] = nza[i] > 0 ? exp(JuMP.value(α[i])) : zero(T)
    end
    for (j, k) in pairs(pc)
        b[k] = nzb[j] > 0 ? exp(JuMP.value(β[j])) : zero(T)
    end
    return a, b
end

function ScaleInvariantAnalysis.cover_min!(::AbsLinear{1}, a::AbstractVector, b::AbstractVector, A)
    ScaleInvariantAnalysis._prepare_cover_start!(a, b, A)
    axr, axc = axes(A, 1), axes(A, 2)
    T = float(real(eltype(A)))
    pr, pc = collect(axr), collect(axc)
    m, n = length(pr), length(pc)
    Apos = [A[pr[i], pc[j]] for i in 1:m, j in 1:n]
    nza = vec(count(!iszero, Apos, dims=2))
    nzb = vec(count(!iszero, Apos, dims=1))
    idx = [(i, j) for i in 1:m for j in 1:n if Apos[i, j] != 0]
    logA = Dict((i, j) => log(abs(Apos[i, j])) for (i, j) in idx)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    α0 = [nza[i] > 0 ? log(T(a[pr[i]])) : zero(T) for i in 1:m]
    β0 = [nzb[j] > 0 ? log(T(b[pc[j]])) : zero(T) for j in 1:n]
    @variable(model, α[i=1:m], start = α0[i])
    @variable(model, β[j=1:n], start = β0[j])
    @variable(model, t[eachindex(idx)] >= 0)
    for (k, (i, j)) in enumerate(idx)
        lA = logA[(i, j)]
        @NLconstraint(model,  1 - exp(lA - α[i] - β[j]) <= t[k])
        @NLconstraint(model, -1 + exp(lA - α[i] - β[j]) <= t[k])
    end
    @objective(model, Min, sum(t))
    for (i, j) in idx
        @constraint(model, α[i] + β[j] >= logA[(i, j)])
    end
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    check_solved(model, "cover_min!")
    for (i, k) in pairs(pr)
        a[k] = nza[i] > 0 ? exp(JuMP.value(α[i])) : zero(T)
    end
    for (j, k) in pairs(pc)
        b[k] = nzb[j] > 0 ? exp(JuMP.value(β[j])) : zero(T)
    end
    return a, b
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
