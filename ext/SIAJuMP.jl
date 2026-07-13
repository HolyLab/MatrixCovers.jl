module SIAJuMP

using JuMP: JuMP, @variable, @objective, @constraint
using HiGHS: HiGHS
using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: AbsLog
using LinearAlgebra: dot

# The models are built over 1-based positions 1:n; `pr`/`pc` map each position to
# the corresponding axis index of `A`, and results are scattered back onto vectors
# whose axes match `A`'s so offset axes are honored. A row/column of `A` with no
# nonzero entry carries no constraint or objective term; its scale is set to exactly
# 0, matching the native solvers.

# Exact reference for the native `symcover_min(::AbsLog{2})`: same QP, solved by
# HiGHS. Not exported; used by the test suite to cross-check the native solver.
function ScaleInvariantAnalysis.symcover_min_jump(::AbsLog{2}, A)
    axr = axes(A, 1)
    axes(A, 2) == axr || throw(ArgumentError("symcover_min_jump requires a square matrix"))
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    Apos = [A[pr[i], pr[j]] for i in 1:n, j in 1:n]
    logA = log.(abs.(Apos))
    supported = [any(!iszero, @view Apos[i, :]) || any(!iszero, @view Apos[:, i]) for i in 1:n]
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:n])
    @objective(model, Min, sum(abs2, α[i] + α[j] - logA[i, j] for i in 1:n, j in 1:n if Apos[i, j] != 0))
    for i in 1:n, j in i:n
        Apos[i, j] != 0 && @constraint(model, α[i] + α[j] - logA[i, j] >= 0)
    end
    JuMP.optimize!(model)
    a = similar(Array{T}, axr)
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

ScaleInvariantAnalysis.symcover_min(::AbsLog{1}, A) = _symcover_min_abslog1(A, nothing)

function ScaleInvariantAnalysis.symcover_min!(::AbsLog{1}, a::AbstractVector, A)
    ScaleInvariantAnalysis._prepare_symcover_start!(a, A)
    a .= _symcover_min_abslog1(A, a)
    return a
end

# The AbsLog{1} hard cover is an LP: the coverage constraint forces every residual
# α[i]+α[j]-log|A[i,j]| to be nonnegative, so |·| drops away and the objective is
# linear in α. `start`, when given, is a cover of `A` supplying the initial point;
# the problem is convex, so it is a hint to the solver and not visible in the result.
function _symcover_min_abslog1(A, start)
    axr = axes(A, 1)
    axes(A, 2) == axr || throw(ArgumentError("symcover_min requires a square matrix"))
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    Apos = [A[pr[i], pr[j]] for i in 1:n, j in 1:n]
    logA = log.(abs.(Apos))
    colcount = vec(count(!iszero, Apos, dims=1))
    rowcount = vec(count(!iszero, Apos, dims=2))
    supported = [colcount[i] > 0 || rowcount[i] > 0 for i in 1:n]
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    if start === nothing
        @variable(model, α[1:n])
    else
        α0 = [supported[k] ? log(T(start[pr[k]])) : zero(T) for k in 1:n]
        @variable(model, α[k=1:n], start = α0[k])
    end
    @objective(model, Min, dot(α, colcount .+ rowcount))
    for i in 1:n, j in i:n
        Apos[i, j] != 0 && @constraint(model, α[i] + α[j] - logA[i, j] >= 0)
    end
    JuMP.optimize!(model)
    a = similar(Array{T}, axr)
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

function ScaleInvariantAnalysis.cover_min_jump(::AbsLog{2}, A)
    axr = axes(A, 1)
    axc = axes(A, 2)
    T = float(real(eltype(A)))
    pr = collect(axr)
    pc = collect(axc)
    m = length(pr)
    n = length(pc)
    Apos = [A[pr[i], pc[j]] for i in 1:m, j in 1:n]
    logA = log.(abs.(Apos))
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:m])
    @variable(model, β[1:n])
    @objective(model, Min, sum(abs2, α[i] + β[j] - logA[i, j] for i in 1:m, j in 1:n if Apos[i, j] != 0))
    for i in 1:m, j in 1:n
        Apos[i, j] != 0 && @constraint(model, α[i] + β[j] - logA[i, j] >= 0)
    end
    nza, nzb = vec(sum(!iszero, Apos; dims=2)), vec(sum(!iszero, Apos; dims=1))
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    a = similar(Array{T}, axr)
    b = similar(Array{T}, axc)
    for (i, k) in pairs(pr)
        a[k] = nza[i] > 0 ? exp(JuMP.value(α[i])) : zero(T)
    end
    for (j, k) in pairs(pc)
        b[k] = nzb[j] > 0 ? exp(JuMP.value(β[j])) : zero(T)
    end
    return a, b
end

ScaleInvariantAnalysis.cover_min(::AbsLog{1}, A) = _cover_min_abslog1(A, nothing)

function ScaleInvariantAnalysis.cover_min!(::AbsLog{1}, a::AbstractVector, b::AbstractVector, A)
    ScaleInvariantAnalysis._prepare_cover_start!(a, b, A)
    anew, bnew = _cover_min_abslog1(A, (a, b))
    a .= anew
    b .= bnew
    return a, b
end

# Asymmetric counterpart of `_symcover_min_abslog1`, on the bipartite support: the
# same LP over row scales α and column scales β, with the row/column gauge pinned by
# the balance constraint so the split between `a` and `b` is deterministic.
function _cover_min_abslog1(A, start)
    axr = axes(A, 1)
    axc = axes(A, 2)
    T = float(real(eltype(A)))
    pr = collect(axr)
    pc = collect(axc)
    m = length(pr)
    n = length(pc)
    Apos = [A[pr[i], pc[j]] for i in 1:m, j in 1:n]
    logA = log.(abs.(Apos))
    colcount = vec(count(!iszero, Apos, dims=1))
    rowcount = vec(count(!iszero, Apos, dims=2))
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    if start === nothing
        @variable(model, α[1:m])
        @variable(model, β[1:n])
    else
        sa, sb = start
        α0 = [rowcount[i] > 0 ? log(T(sa[pr[i]])) : zero(T) for i in 1:m]
        β0 = [colcount[j] > 0 ? log(T(sb[pc[j]])) : zero(T) for j in 1:n]
        @variable(model, α[i=1:m], start = α0[i])
        @variable(model, β[j=1:n], start = β0[j])
    end
    @objective(model, Min, dot(α, rowcount) + dot(β, colcount))
    for i in 1:m, j in 1:n
        Apos[i, j] != 0 && @constraint(model, α[i] + β[j] - logA[i, j] >= 0)
    end
    nza, nzb = rowcount, colcount
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    a = similar(Array{T}, axr)
    b = similar(Array{T}, axc)
    for (i, k) in pairs(pr)
        a[k] = nza[i] > 0 ? exp(JuMP.value(α[i])) : zero(T)
    end
    for (j, k) in pairs(pc)
        b[k] = nzb[j] > 0 ? exp(JuMP.value(β[j])) : zero(T)
    end
    return a, b
end

# Soft (unconstrained) symmetric cover: minimize ∑ (log r_ij)² with no constraints.
# The objective is quadratic in α = log a, solved as a QP.
function ScaleInvariantAnalysis.soft_symcover_min(::AbsLog{2}, A)
    axr = axes(A, 1)
    axes(A, 2) == axr || throw(ArgumentError("soft_symcover_min requires a square matrix"))
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    Apos = [A[pr[i], pr[j]] for i in 1:n, j in 1:n]
    logA = log.(abs.(Apos))
    supported = [any(!iszero, @view Apos[i, :]) || any(!iszero, @view Apos[:, i]) for i in 1:n]
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:n])
    @objective(model, Min, sum(abs2, α[i] + α[j] - logA[i, j] for i in 1:n, j in 1:n if Apos[i, j] != 0))
    JuMP.optimize!(model)
    a = similar(Array{T}, axr)
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

end
