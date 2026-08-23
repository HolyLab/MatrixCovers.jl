module MatrixCoversJuMPExt

using JuMP: JuMP, @variable, @objective, @constraint
using HiGHS: HiGHS
using MatrixCovers
using MatrixCovers: AbsLog
using MatrixCovers: _edge_list, _sym_edge_list, _degrees
using LinearAlgebra: dot

check_solved(model, fname) =
    MatrixCovers.check_solved(JuMP.termination_status(model), "HiGHS", fname)

# Models use 1-based positions and scatter results back to `A`'s axes. Support is
# gathered as an O(nnz) edge list; unsupported scales are zero.

# HiGHS reference for tests of native symmetric `AbsLog{2}`.
function MatrixCovers.symcover_min_jump(::AbsLog{2}, A)
    axr = axes(A, 1)
    axes(A, 2) == axr || throw(ArgumentError("symcover_min_jump requires a square matrix"))
    MatrixCovers.require_abs_symmetric(A, :symcover_min_jump)
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    ei, ej, elog = _sym_edge_list(A, T)
    supported = _degrees(ei, n) .> 0
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:n])
    @objective(model, Min, sum(abs2, α[ei[e]] + α[ej[e]] - elog[e] for e in eachindex(ei)))
    for e in eachindex(ei)
        ei[e] <= ej[e] && @constraint(model, α[ei[e]] + α[ej[e]] - elog[e] >= 0)
    end
    JuMP.optimize!(model)
    check_solved(model, "symcover_min_jump")
    a = similar(Array{T}, axr)
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

MatrixCovers.symcover_min(::AbsLog{1}, A) = _symcover_min_abslog1(A, nothing)

function MatrixCovers.symcover_min!(::AbsLog{1}, a::AbstractVector, A)
    MatrixCovers._prepare_symcover_start!(a, A)
    a .= _symcover_min_abslog1(A, a)
    return a
end

# Slack for re-evaluating the `AbsLog{1}` optimum during tie-breaking.
const LEX_L1_SLACK = 1e-9

# Break `AbsLog{1}` ties with `AbsLog{2}` using full-grid support weights.
function _minimize_l2_over_l1_face!(model, lin, residuals, fname)
    isempty(residuals) && return nothing
    linopt = JuMP.value(lin)
    l1 = sum(JuMP.value, residuals)          # the AbsLog{1} objective attained
    @constraint(model, lin <= linopt + LEX_L1_SLACK * max(one(l1), l1))
    @objective(model, Min, sum(r^2 for r in residuals))
    JuMP.optimize!(model)
    check_solved(model, fname)
    return nothing
end

# Symmetric `AbsLog{1}` LP. A second stage selects the canonical point on the
# optimal face; `start` is only a solver hint.
function _symcover_min_abslog1(A, start)
    axr = axes(A, 1)
    axes(A, 2) == axr || throw(ArgumentError("symcover_min requires a square matrix"))
    MatrixCovers.require_abs_symmetric(A, :symcover_min)
    T = float(real(eltype(A)))
    pr = collect(axr)
    n = length(pr)
    ei, ej, elog = _sym_edge_list(A, T)
    # The gather is already the full grid, so a position's degree is both its row
    # count and its column count.
    cnt = _degrees(ei, n)
    supported = cnt .> 0
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    if start === nothing
        @variable(model, α[1:n])
    else
        α0 = [supported[k] ? log(T(start[pr[k]])) : zero(T) for k in 1:n]
        @variable(model, α[k=1:n], start = α0[k])
    end
    lin = dot(α, 2 .* cnt)
    @objective(model, Min, lin)
    for e in eachindex(ei)
        ei[e] <= ej[e] && @constraint(model, α[ei[e]] + α[ej[e]] - elog[e] >= 0)
    end
    JuMP.optimize!(model)
    check_solved(model, "symcover_min")
    residuals = [α[ei[e]] + α[ej[e]] - elog[e] for e in eachindex(ei)]
    _minimize_l2_over_l1_face!(model, lin, residuals, "symcover_min")
    a = similar(Array{T}, axr)
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(JuMP.value(α[i])) : zero(T)
    end
    return a
end

function MatrixCovers.cover_min_jump(::AbsLog{2}, A)
    axr = axes(A, 1)
    axc = axes(A, 2)
    T = float(real(eltype(A)))
    pr = collect(axr)
    pc = collect(axc)
    m = length(pr)
    n = length(pc)
    ei, ej, elog = _edge_list(A, T)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:m])
    @variable(model, β[1:n])
    @objective(model, Min, sum(abs2, α[ei[e]] + β[ej[e]] - elog[e] for e in eachindex(ei)))
    for e in eachindex(ei)
        @constraint(model, α[ei[e]] + β[ej[e]] - elog[e] >= 0)
    end
    nza, nzb = _degrees(ei, m), _degrees(ej, n)
    # The post-solve balance handles component gauges not pinned here.
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    check_solved(model, "cover_min_jump")
    a = similar(Array{T}, axr)
    b = similar(Array{T}, axc)
    for (i, k) in pairs(pr)
        a[k] = nza[i] > 0 ? exp(JuMP.value(α[i])) : zero(T)
    end
    for (j, k) in pairs(pc)
        b[k] = nzb[j] > 0 ? exp(JuMP.value(β[j])) : zero(T)
    end
    MatrixCovers._balance_cover!(a, b, A)
    return MatrixCovers.inflate_feasible!(a, b, A)
end

MatrixCovers.cover_min(::AbsLog{1}, A) = _cover_min_abslog1(A, nothing)

function MatrixCovers.cover_min!(::AbsLog{1}, a::AbstractVector, b::AbstractVector, A)
    MatrixCovers._prepare_cover_start!(a, b, A)
    anew, bnew = _cover_min_abslog1(A, (a, b))
    a .= anew
    b .= bnew
    return a, b
end

# Asymmetric `AbsLog{1}` LP. Balance globally in the model and per component
# after solving.
function _cover_min_abslog1(A, start)
    axr = axes(A, 1)
    axc = axes(A, 2)
    T = float(real(eltype(A)))
    pr = collect(axr)
    pc = collect(axc)
    m = length(pr)
    n = length(pc)
    ei, ej, elog = _edge_list(A, T)
    rowcount = _degrees(ei, m)
    colcount = _degrees(ej, n)
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
    lin = dot(α, rowcount) + dot(β, colcount)
    @objective(model, Min, lin)
    for e in eachindex(ei)
        @constraint(model, α[ei[e]] + β[ej[e]] - elog[e] >= 0)
    end
    nza, nzb = rowcount, colcount
    # Pin the global row/column gauge; post-processing handles components.
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    check_solved(model, "cover_min")
    residuals = [α[ei[e]] + β[ej[e]] - elog[e] for e in eachindex(ei)]
    _minimize_l2_over_l1_face!(model, lin, residuals, "cover_min")
    a = similar(Array{T}, axr)
    b = similar(Array{T}, axc)
    for (i, k) in pairs(pr)
        a[k] = nza[i] > 0 ? exp(JuMP.value(α[i])) : zero(T)
    end
    for (j, k) in pairs(pc)
        b[k] = nzb[j] > 0 ? exp(JuMP.value(β[j])) : zero(T)
    end
    MatrixCovers._balance_cover!(a, b, A)
    return MatrixCovers.inflate_feasible!(a, b, A)
end


end
