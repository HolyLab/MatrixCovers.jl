module MatrixCoversJuMPExt

using JuMP: JuMP, @variable, @objective, @constraint
using HiGHS: HiGHS
using MatrixCovers
using MatrixCovers: AbsLog, ExternalSolverStats
using MatrixCovers: _edge_list, _sym_edge_list, _degrees
using LinearAlgebra: dot

check_solved(model, fname) =
    MatrixCovers.check_solved(JuMP.termination_status(model), "HiGHS", fname)

_highs_stats() = ExternalSolverStats(:HiGHS, Symbol[], Float64[], Int[], Float64[])

# Iterations reported through MathOptInterface; HiGHS's QP iterations are not among them.
_highs_iterations(model) = JuMP.simplex_iterations(model) + JuMP.barrier_iterations(model)

# Check that the last solve of `model` succeeded and append it to `stats` as a stage.
function record_solved!(stats::ExternalSolverStats, model, fname)
    check_solved(model, fname)
    push!(stats.status, Symbol(JuMP.termination_status(model)))
    push!(stats.objective, JuMP.objective_value(model))
    push!(stats.niters, _highs_iterations(model))
    push!(stats.solvetime, JuMP.solve_time(model))
    return stats
end

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
    anew, stats = _symcover_min_abslog1(A, a)
    a .= anew
    return a, stats
end

# Slack for re-evaluating the `AbsLog{1}` optimum during tie-breaking.
const LEX_L1_SLACK = 1e-9

# Break `AbsLog{1}` ties with `AbsLog{2}` using full-grid support weights.
function _minimize_l2_over_l1_face!(stats, model, lin, residuals, fname)
    isempty(residuals) && return stats
    linopt = JuMP.value(lin)
    l1 = sum(JuMP.value, residuals)          # the AbsLog{1} objective attained
    @constraint(model, lin <= linopt + LEX_L1_SLACK * max(one(l1), l1))
    @objective(model, Min, sum(r^2 for r in residuals))
    JuMP.optimize!(model)
    return record_solved!(stats, model, fname)
end

# The soft `AbsLog{1}` objective is convex (an L1 fit in the log scales), so it
# is solved exactly by the same LPs without the coverage constraints.
MatrixCovers.soft_symcover(::AbsLog{1}, A::AbstractMatrix) = _symcover_min_abslog1(A, nothing; soft=true)

function MatrixCovers.soft_symcover!(::AbsLog{1}, a::AbstractVector, A::AbstractMatrix)
    MatrixCovers._prepare_soft_symcover_start!(a, A)
    anew, stats = _symcover_min_abslog1(A, a; soft=true)
    a .= anew
    return a, stats
end

MatrixCovers.soft_cover(::AbsLog{1}, A::AbstractMatrix) = _cover_min_abslog1(A, nothing; soft=true)

function MatrixCovers.soft_cover!(::AbsLog{1}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    MatrixCovers._prepare_soft_cover_start!(a, b, A)
    anew, bnew, stats = _cover_min_abslog1(A, (a, b); soft=true)
    a .= anew
    b .= bnew
    return a, b, stats
end

# Symmetric `AbsLog{1}` LP. A second stage selects the canonical point on the
# optimal face; `start` is only a solver hint. With `soft=true` the residuals
# are unconstrained and the LP minimizes their absolute values through slacks.
function _symcover_min_abslog1(A, start; soft::Bool=false)
    fname = soft ? "soft_symcover" : "symcover_min"
    axr = axes(A, 1)
    axes(A, 2) == axr || throw(ArgumentError("$fname requires a square matrix"))
    MatrixCovers.require_abs_symmetric(A, Symbol(fname))
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
    if soft
        # One slack per lower-triangle entry, weighted by its full-grid multiplicity.
        tri = [e for e in eachindex(ei) if ei[e] <= ej[e]]
        @variable(model, t[eachindex(tri)] >= 0)
        for (k, e) in pairs(tri)
            r = α[ei[e]] + α[ej[e]] - elog[e]
            @constraint(model, t[k] >= r)
            @constraint(model, t[k] >= -r)
        end
        lin = sum((ei[e] == ej[e] ? 1 : 2) * t[k] for (k, e) in pairs(tri); init=zero(JuMP.AffExpr))
    else
        lin = dot(α, 2 .* cnt)
        for e in eachindex(ei)
            ei[e] <= ej[e] && @constraint(model, α[ei[e]] + α[ej[e]] - elog[e] >= 0)
        end
    end
    @objective(model, Min, lin)
    JuMP.optimize!(model)
    stats = record_solved!(_highs_stats(), model, fname)
    residuals = [α[ei[e]] + α[ej[e]] - elog[e] for e in eachindex(ei)]
    _minimize_l2_over_l1_face!(stats, model, lin, residuals, fname)
    αv = [JuMP.value(α[i]) for i in 1:n]
    if soft
        # Without the coverage constraints, bipartite components keep their gauge.
        αo = similar(Array{T}, axr)
        for (i, k) in pairs(pr)
            αo[k] = αv[i]
        end
        MatrixCovers._balance_bipartite_sym!(αo, MatrixCovers._sym_support(A, T))
        αv = [αo[k] for k in pr]
    end
    a = similar(Array{T}, axr)
    for (i, k) in pairs(pr)
        a[k] = supported[i] ? exp(αv[i]) : zero(T)
    end
    return a, stats
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

# Two-stage reference for `cover_transversal`: the LP `min ∑α + ∑β` over hard
# covers attains `log π*`; the QP then minimizes the `AbsLog{2}` objective with
# `∑α + ∑β` fixed at that value.
function MatrixCovers.cover_transversal_jump(A)
    axr = axes(A, 1)
    axc = axes(A, 2)
    length(axr) == length(axc) || throw(ArgumentError("cover_transversal_jump requires a square matrix"))
    T = float(real(eltype(A)))
    m = n = length(axr)
    ei, ej, elog = _edge_list(A, T)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:m])
    @variable(model, β[1:n])
    for e in eachindex(ei)
        @constraint(model, α[ei[e]] + β[ej[e]] - elog[e] >= 0)
    end
    @objective(model, Min, sum(α) + sum(β))
    JuMP.optimize!(model)
    check_solved(model, "cover_transversal_jump")
    logπ = JuMP.objective_value(model)
    @constraint(model, sum(α) + sum(β) == logπ)
    nza, nzb = _degrees(ei, m), _degrees(ej, n)
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    @objective(model, Min, sum(abs2, α[ei[e]] + β[ej[e]] - elog[e] for e in eachindex(ei)))
    JuMP.optimize!(model)
    check_solved(model, "cover_transversal_jump")
    a = similar(Array{T}, axr)
    b = similar(Array{T}, axc)
    for (i, k) in enumerate(axr)
        a[k] = exp(JuMP.value(α[i]))
    end
    for (j, k) in enumerate(axc)
        b[k] = exp(JuMP.value(β[j]))
    end
    MatrixCovers._balance_cover!(a, b, A)
    return MatrixCovers.inflate_feasible!(a, b, A), logπ
end

MatrixCovers.cover_min(::AbsLog{1}, A) = _cover_min_abslog1(A, nothing)

function MatrixCovers.cover_min!(::AbsLog{1}, a::AbstractVector, b::AbstractVector, A)
    MatrixCovers._prepare_cover_start!(a, b, A)
    anew, bnew, stats = _cover_min_abslog1(A, (a, b))
    a .= anew
    b .= bnew
    return a, b, stats
end

# Asymmetric `AbsLog{1}` LP. Balance globally in the model and per component
# after solving. With `soft=true` the residuals are unconstrained and the LP
# minimizes their absolute values through slacks.
function _cover_min_abslog1(A, start; soft::Bool=false)
    fname = soft ? "soft_cover" : "cover_min"
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
    if soft
        @variable(model, t[eachindex(ei)] >= 0)
        for e in eachindex(ei)
            r = α[ei[e]] + β[ej[e]] - elog[e]
            @constraint(model, t[e] >= r)
            @constraint(model, t[e] >= -r)
        end
        lin = sum(t; init=zero(JuMP.AffExpr))
    else
        lin = dot(α, rowcount) + dot(β, colcount)
        for e in eachindex(ei)
            @constraint(model, α[ei[e]] + β[ej[e]] - elog[e] >= 0)
        end
    end
    @objective(model, Min, lin)
    nza, nzb = rowcount, colcount
    # Pin the global row/column gauge; post-processing handles components.
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    stats = record_solved!(_highs_stats(), model, fname)
    residuals = [α[ei[e]] + β[ej[e]] - elog[e] for e in eachindex(ei)]
    _minimize_l2_over_l1_face!(stats, model, lin, residuals, fname)
    a = similar(Array{T}, axr)
    b = similar(Array{T}, axc)
    for (i, k) in pairs(pr)
        a[k] = nza[i] > 0 ? exp(JuMP.value(α[i])) : zero(T)
    end
    for (j, k) in pairs(pc)
        b[k] = nzb[j] > 0 ? exp(JuMP.value(β[j])) : zero(T)
    end
    MatrixCovers._balance_cover!(a, b, A)
    soft || MatrixCovers.inflate_feasible!(a, b, A)
    return a, b, stats
end


end
