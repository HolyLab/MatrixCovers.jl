module MatrixCoversIpoptExt

using JuMP: JuMP, @variable, @objective, @constraint
using Ipopt: Ipopt
using MatrixCovers
using MatrixCovers: AbsLinear
using MatrixCovers: _edge_list, _sym_edge_list, _degrees

# The models are built over 1-based positions 1:n; `pr`/`pc` map each position to the
# corresponding axis index of `A`, and results are scattered back onto vectors
# whose axes match `A`'s so offset axes are honored. A row/column of `A` with no
# nonzero entry appears in no constraint or objective term; its scale is set to
# exactly 0, matching the native solvers.
#
# `A` is read through the support hook and gathered into a flat edge list in position
# space — `ei`/`ej` the endpoints, `elog` the log-magnitude — so a model costs O(nnz)
# to build rather than O(length(A)). The symmetric list is the full-grid reading; the
# hard-cover models below sum over its `ei <= ej` half instead, per the objective each
# one is defined by.

# The AbsLinear objectives are non-convex, so Ipopt returns a local minimum of
# whichever basin it descends into from the start it is given. That makes the start a
# genuine input rather than a hint, and it is why the hard-cover entry points here are the
# `*_min!` refiners, which take the start from the caller. The non-mutating `symcover_min`
# and `cover_min` are multistart drivers over these kernels and live in the main package.

check_solved(model, fname) =
    MatrixCovers.check_solved(JuMP.termination_status(model), "Ipopt", fname)

# The `i ≤ j` half of a symmetric gather, paired with the multiplicity `w` each entry
# stands for in the full grid: an off-diagonal pair is two entries of `A`, a diagonal
# entry one. Carrying the weight is equivalent to summing over both orientations and
# costs half the terms — and, for the AbsLinear{1} models, half the auxiliary variables.
function _triangle(fi, fj, flog)
    keep = [e for e in eachindex(fi) if fi[e] <= fj[e]]
    return fi[keep], fj[keep], flog[keep], [fi[e] == fj[e] ? 1 : 2 for e in keep]
end

# ============================================================
# Hard cover: symcover_min!(::AbsLinear{p}, a, A)
# Minimizes ∑_{i,j: A[i,j]≠0} |1 - |A[i,j]|/(a[i]*a[j])|^p over the full grid, so an
# off-diagonal pair counts twice and a diagonal entry once — the weighting
# `cover_objective` reports and the rest of the package minimizes.
# subject to a[i]*a[j] ≥ |A[i,j]| for all i≤j with A[i,j]≠0, which is the same
# constraint set as imposing it on the full grid.
# Variables: α[i] = log(a[i]); constraint: α[i]+α[j] ≥ log|A[i,j]|.
# ============================================================

function MatrixCovers.symcover_min!(::AbsLinear{2}, a::AbstractVector, A)
    MatrixCovers._prepare_symcover_start!(a, A)
    axr = axes(A, 1)
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    fi, fj, flog = _sym_edge_list(A, T)
    supported = _degrees(fi, n) .> 0
    ti, tj, tlog, tw = _triangle(fi, fj, flog)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    start0 = [supported[k] && !iszero(a[pr[k]]) ? log(T(a[pr[k]])) : zero(T) for k in 1:n]
    @variable(model, α[k=1:n], start = start0[k])
    @objective(model, Min,
        sum(tw[k] * (1 - exp(tlog[k] - α[ti[k]] - α[tj[k]]))^2 for k in eachindex(ti)))
    for k in eachindex(ti)
        @constraint(model, α[ti[k]] + α[tj[k]] >= tlog[k])
    end
    JuMP.optimize!(model)
    check_solved(model, "symcover_min!")
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

function MatrixCovers.symcover_min!(::AbsLinear{1}, a::AbstractVector, A)
    MatrixCovers._prepare_symcover_start!(a, A)
    axr = axes(A, 1)
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    fi, fj, flog = _sym_edge_list(A, T)
    supported = _degrees(fi, n) .> 0
    ti, tj, tlog, tw = _triangle(fi, fj, flog)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    start0 = [supported[k] && !iszero(a[pr[k]]) ? log(T(a[pr[k]])) : zero(T) for k in 1:n]
    @variable(model, α[k=1:n], start = start0[k])
    # |1 - exp(lA - αi - αj)| via auxiliary variables t ≥ 0 and slack s
    @variable(model, t[eachindex(ti)] >= 0)
    for k in eachindex(ti)
        @constraint(model,  1 - exp(tlog[k] - α[ti[k]] - α[tj[k]]) <= t[k])
        @constraint(model, -1 + exp(tlog[k] - α[ti[k]] - α[tj[k]]) <= t[k])
    end
    @objective(model, Min, sum(tw[k] * t[k] for k in eachindex(ti)))
    for k in eachindex(ti)
        @constraint(model, α[ti[k]] + α[tj[k]] >= tlog[k])
    end
    JuMP.optimize!(model)
    check_solved(model, "symcover_min!")
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
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

function MatrixCovers.cover_min!(::AbsLinear{2}, a::AbstractVector, b::AbstractVector, A)
    MatrixCovers._prepare_cover_start!(a, b, A)
    axr, axc = axes(A, 1), axes(A, 2)
    T = float(real(eltype(A)))
    pr, pc = collect(axr), collect(axc)
    m, n = length(pr), length(pc)
    ei, ej, elog = _edge_list(A, T)
    nza = _degrees(ei, m)
    nzb = _degrees(ej, n)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    α0 = [nza[i] > 0 ? log(T(a[pr[i]])) : zero(T) for i in 1:m]
    β0 = [nzb[j] > 0 ? log(T(b[pc[j]])) : zero(T) for j in 1:n]
    @variable(model, α[i=1:m], start = α0[i])
    @variable(model, β[j=1:n], start = β0[j])
    @objective(model, Min,
        sum((1 - exp(elog[e] - α[ei[e]] - β[ej[e]]))^2 for e in eachindex(ei)))
    for e in eachindex(ei)
        @constraint(model, α[ei[e]] + β[ej[e]] >= elog[e])
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

function MatrixCovers.cover_min!(::AbsLinear{1}, a::AbstractVector, b::AbstractVector, A)
    MatrixCovers._prepare_cover_start!(a, b, A)
    axr, axc = axes(A, 1), axes(A, 2)
    T = float(real(eltype(A)))
    pr, pc = collect(axr), collect(axc)
    m, n = length(pr), length(pc)
    ei, ej, elog = _edge_list(A, T)
    nza = _degrees(ei, m)
    nzb = _degrees(ej, n)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    α0 = [nza[i] > 0 ? log(T(a[pr[i]])) : zero(T) for i in 1:m]
    β0 = [nzb[j] > 0 ? log(T(b[pc[j]])) : zero(T) for j in 1:n]
    @variable(model, α[i=1:m], start = α0[i])
    @variable(model, β[j=1:n], start = β0[j])
    @variable(model, t[eachindex(ei)] >= 0)
    for e in eachindex(ei)
        @constraint(model,  1 - exp(elog[e] - α[ei[e]] - β[ej[e]]) <= t[e])
        @constraint(model, -1 + exp(elog[e] - α[ei[e]] - β[ej[e]]) <= t[e])
    end
    @objective(model, Min, sum(t))
    for e in eachindex(ei)
        @constraint(model, α[ei[e]] + β[ej[e]] >= elog[e])
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
# Soft cover: soft_symcover_min!(::AbsLinear{p}, a, A)
# Same objective, no coverage constraints — so the start need not cover `A`, and the
# raw geometric mean (the exact soft AbsLog{2} optimum) is a natural one. The multistart
# driver over these kernels is native; see soft_symcover_min.
# ============================================================

function MatrixCovers.soft_symcover_min!(::AbsLinear{2}, a::AbstractVector, A)
    MatrixCovers._prepare_soft_symcover_start!(a, A)
    axr = axes(A, 1)
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    fi, fj, flog = _sym_edge_list(A, T)
    supported = _degrees(fi, n) .> 0
    ti, tj, tlog, tw = _triangle(fi, fj, flog)
    n_zeros = n^2 - length(fi)   # a zero entry contributes (1-0)^2 = 1 regardless of α

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    start0 = [supported[k] ? log(T(a[pr[k]])) : zero(T) for k in 1:n]
    @variable(model, α[k=1:n], start = start0[k])
    @objective(model, Min,
        sum(tw[k] * (1 - exp(tlog[k] - α[ti[k]] - α[tj[k]]))^2 for k in eachindex(ti)) + n_zeros)
    JuMP.optimize!(model)
    check_solved(model, "soft_symcover_min!")
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

function MatrixCovers.soft_symcover_min!(::AbsLinear{1}, a::AbstractVector, A)
    MatrixCovers._prepare_soft_symcover_start!(a, A)
    axr = axes(A, 1)
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    fi, fj, flog = _sym_edge_list(A, T)
    supported = _degrees(fi, n) .> 0
    ti, tj, tlog, tw = _triangle(fi, fj, flog)
    n_zeros = n^2 - length(fi)   # a zero entry contributes |1 - 0| = 1 regardless of α

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    start0 = [supported[k] ? log(T(a[pr[k]])) : zero(T) for k in 1:n]
    @variable(model, α[k=1:n], start = start0[k])
    @variable(model, t[eachindex(ti)] >= 0)
    for k in eachindex(ti)
        @constraint(model,  1 - exp(tlog[k] - α[ti[k]] - α[tj[k]]) <= t[k])
        @constraint(model, -1 + exp(tlog[k] - α[ti[k]] - α[tj[k]]) <= t[k])
    end
    @objective(model, Min, sum(tw[k] * t[k] for k in eachindex(ti)) + n_zeros)
    JuMP.optimize!(model)
    check_solved(model, "soft_symcover_min!")
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

# ============================================================
# Soft cover: soft_cover_min!(::AbsLinear{p}, a, b, A)
# The bipartite analog of soft_symcover_min!, and the unconstrained analog of cover_min!:
# no coverage constraints, but the same row/column gauge, pinned by the same balance
# constraint ∑ nzaᵢ αᵢ = ∑ nzbⱼ βⱼ. A zero entry of `A` contributes ϕ(0) = 1 whatever the
# scales, so the count of zeros enters the objective as a constant, matching cover_objective.
# ============================================================

function MatrixCovers.soft_cover_min!(::AbsLinear{2}, a::AbstractVector, b::AbstractVector, A)
    MatrixCovers._prepare_soft_cover_start!(a, b, A)
    axr, axc = axes(A, 1), axes(A, 2)
    T = float(real(eltype(A)))
    pr, pc = collect(axr), collect(axc)
    m, n = length(pr), length(pc)
    ei, ej, elog = _edge_list(A, T)
    nza = _degrees(ei, m)
    nzb = _degrees(ej, n)
    n_zeros = m * n - length(ei)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    α0 = [nza[i] > 0 ? log(T(a[pr[i]])) : zero(T) for i in 1:m]
    β0 = [nzb[j] > 0 ? log(T(b[pc[j]])) : zero(T) for j in 1:n]
    @variable(model, α[i=1:m], start = α0[i])
    @variable(model, β[j=1:n], start = β0[j])
    @objective(model, Min,
        sum((1 - exp(elog[e] - α[ei[e]] - β[ej[e]]))^2 for e in eachindex(ei)) + n_zeros)
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    check_solved(model, "soft_cover_min!")
    for (i, k) in pairs(pr)
        a[k] = nza[i] > 0 ? exp(JuMP.value(α[i])) : zero(T)
    end
    for (j, k) in pairs(pc)
        b[k] = nzb[j] > 0 ? exp(JuMP.value(β[j])) : zero(T)
    end
    return a, b
end

function MatrixCovers.soft_cover_min!(::AbsLinear{1}, a::AbstractVector, b::AbstractVector, A)
    MatrixCovers._prepare_soft_cover_start!(a, b, A)
    axr, axc = axes(A, 1), axes(A, 2)
    T = float(real(eltype(A)))
    pr, pc = collect(axr), collect(axc)
    m, n = length(pr), length(pc)
    ei, ej, elog = _edge_list(A, T)
    nza = _degrees(ei, m)
    nzb = _degrees(ej, n)
    n_zeros = m * n - length(ei)

    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    α0 = [nza[i] > 0 ? log(T(a[pr[i]])) : zero(T) for i in 1:m]
    β0 = [nzb[j] > 0 ? log(T(b[pc[j]])) : zero(T) for j in 1:n]
    @variable(model, α[i=1:m], start = α0[i])
    @variable(model, β[j=1:n], start = β0[j])
    @variable(model, t[eachindex(ei)] >= 0)
    for e in eachindex(ei)
        @constraint(model,  1 - exp(elog[e] - α[ei[e]] - β[ej[e]]) <= t[e])
        @constraint(model, -1 + exp(elog[e] - α[ei[e]] - β[ej[e]]) <= t[e])
    end
    @objective(model, Min, sum(t) + n_zeros)
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    check_solved(model, "soft_cover_min!")
    for (i, k) in pairs(pr)
        a[k] = nza[i] > 0 ? exp(JuMP.value(α[i])) : zero(T)
    end
    for (j, k) in pairs(pc)
        b[k] = nzb[j] > 0 ? exp(JuMP.value(β[j])) : zero(T)
    end
    return a, b
end

end  # module MatrixCoversIpoptExt
