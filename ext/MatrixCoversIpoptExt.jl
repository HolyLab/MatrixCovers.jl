module MatrixCoversIpoptExt

using JuMP: JuMP, @variable, @objective, @constraint
using Ipopt: Ipopt
using MatrixCovers
using MatrixCovers: AbsLinear
using MatrixCovers: _edge_list, _sym_edge_list, _degrees

# Models use 1-based positions and scatter results back to `A`'s axes. Support is
# gathered as an O(nnz) edge list; unsupported scales are zero.

# Ipopt kernels refine one start; the main package supplies multistart drivers.

check_solved(model, fname) =
    MatrixCovers.check_solved(JuMP.termination_status(model), "Ipopt", fname)

# Suppress both solver output and Ipopt's startup banner.
function _ipopt_model()
    model = JuMP.Model(Ipopt.Optimizer)
    JuMP.set_silent(model)
    JuMP.set_attribute(model, "sb", "yes")
    return model
end

# Symmetric triangle with full-grid multiplicities.
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

    model = _ipopt_model()
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

    model = _ipopt_model()
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
# Asymmetric hard-cover model in row and column log scales. The model pins the
# global gauge; post-processing balances components and restores feasibility.
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

    model = _ipopt_model()
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
    MatrixCovers._balance_cover!(a, b, A)
    return MatrixCovers.inflate_feasible!(a, b, A)
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

    model = _ipopt_model()
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
    MatrixCovers._balance_cover!(a, b, A)
    return MatrixCovers.inflate_feasible!(a, b, A)
end

# ============================================================
# Soft cover: soft_symcover_min!(::AbsLinear{p}, a, A)
# Same objective without coverage constraints; starts need not cover `A`.
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

    model = _ipopt_model()
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

    model = _ipopt_model()
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
# Asymmetric soft-cover model. Post-processing balances component gauges; zero
# entries contribute the constant `ϕ(0) = 1`.
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

    model = _ipopt_model()
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
    return MatrixCovers._balance_cover!(a, b, A)
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

    model = _ipopt_model()
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
    return MatrixCovers._balance_cover!(a, b, A)
end

end  # module MatrixCoversIpoptExt
