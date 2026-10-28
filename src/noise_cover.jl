# Noise covers: the diagonal scaling that gives the noise of a data matrix unit
# average variance in every row and column, computed from a matrix of variance
# estimates by the `PowerMean{1}` solver with constant targets.

"""
    a, b, stats = noise_cover(V; shrink=0, tol=nothing, kwargs...)

Return scales `a` and `b` that normalize a data matrix whose entries have
noise variances estimated by `V`: after normalization `Y ./ (a .* b')`, the
noise of every row and column has average variance one. `V` is a nonnegative
matrix of the size of the data, and `a` and `b` satisfy

    ∑_j V[i,j] / (a[i]*b[j])^2 = n′,    ∑_i V[i,j] / (a[i]*b[j])^2 = m′

for every nonzero row `i` and column `j` of `V`, where `m′` and `n′` are the
numbers of nonzero rows and columns. The sums include the zero entries of `V`. A
zero row or column of `V` receives zero scale. For Poisson counts `Y`, `V = Y`
gives the biwhitening of Landa, Zhang, and Kluger (2022).

With `shrink = δ > 0`, `V` is replaced by `(V + δ*r*c'/s) / (1 + δ)`, where `r`,
`c`, and `s` are the row sums, column sums, and total of `V`. This matrix has the
same total as `V` and is positive on the nonzero rows and columns, so the scaling
always exists. For Poisson counts, it is the posterior mean of the variances under
a gamma prior centered on the rank-one fit `r*c'/s` with the weight of `δ`
observations. With `shrink = 0`, the scaling exists only for some zero patterns
of `V`, which are checked before solving; a pattern without one throws a
[`SolverFailure`](@ref) that describes the obstruction.

The products `(a[i]*b[j])^2` are the `PowerMean{1}` soft cover of this matrix,
except that the targets `n′` and `m′` count its zero entries; they are computed by
the solver of [`soft_cover`](@ref). `tol` and the remaining keywords are the
`PowerMean` keywords of `soft_cover`, with the same meanings and defaults. A solve
that does not converge warns and reports `converged(stats) == false`. `a` and `b` follow the balance convention of
[`cover_min`](@ref). `stats` is a [`PowerMeanStats`](@ref MatrixCovers.PowerMeanStats).

The [guide](@ref noise-covers) describes how to estimate `V` for different kinds
of data and how to use the normalized matrix.

# Examples

```jldoctest
julia> Y = [3 0 1; 0 2 5; 4 1 0];

julia> a, b, stats = noise_cover(Y);

julia> round.(sum(Y ./ (a .* b').^2; dims=2); digits=8)
3×1 Matrix{Float64}:
 3.0
 3.0
 3.0
```
"""
function noise_cover(V::AbstractMatrix; shrink::Real=0, tol::Union{Real,Nothing}=nothing, kwargs...)
    _check_noise_variance(V)
    (isfinite(shrink) && shrink >= 0) ||
        throw(ArgumentError("noise_cover: `shrink` must be finite and nonnegative, got $shrink"))
    T =_powermean_type(real(eltype(V)))
    Ta = float(real(eltype(V)))
    rtol = _powermean_tol(T, tol)
    δ = T(shrink)
    R, C = _row_support(V, T), _col_support(V, T)
    r, c = _group_sums(R), _group_sums(C)
    m′, n′ = count(>(0), r), count(>(0), c)
    a = similar(Array{Ta}, axes(V, 1))
    b = similar(Array{Ta}, axes(V, 2))
    if m′ == 0
        fill!(a, zero(Ta)); fill!(b, zero(Ta))
        return a, b, PowerMeanStats{T}(true, rtol, zero(T), zero(T), 0, false, 0, 0, 0, 0, :none)
    end
    full = length(R.val) == m′ * n′
    iszero(δ) && !full && _check_noise_scaling_exists(R, axes(V, 2), m′, n′)

    # The variance products `(a[i]*b[j])^2` are the `PowerMean{1}` soft cover of
    # `M = (V + δ*r*c'/s) / (1 + δ)`, with targets `n′` and `m′` that count the
    # zeros of `M`: its stored entries are those of `V` divided by `1 + δ`, and the
    # rank-one term is `κ*r*c'` with `κ = δ / ((1 + δ) s)`. When `V` has no zeros in
    # its nonzero block and `δ = 0`, the targets are the support counts.
    s = sum(r)
    lδ = log1p(δ)
    for S in (R, C)
        map!(v -> log(v) - lδ, S.val, S.val)
    end
    terms = if full && iszero(δ)
        nothing
    else
        lt = T[k <= length(r) ? log(T(n′)) : log(T(m′)) for k in 1:length(r)+length(c)]
        lw = iszero(δ) ? T[] : log.(vcat(r, c))
        PowerMeanTerms{T}(lt, lw, log(δ) - lδ - log(s))
    end
    # Start from the solution for the rank-one matrix `r*c'/s`.
    ls2 = log(s) / 2
    α = similar(Array{T}, axes(V, 1))
    β = similar(Array{T}, axes(V, 2))
    for (i, ri) in zip(eachindex(α), r)
        α[i] = ri > 0 ? log(ri) - ls2 : zero(T)
    end
    for (j, cj) in zip(eachindex(β), c)
        β[j] = cj > 0 ? log(cj) - ls2 : zero(T)
    end
    # The solver keywords pass through to `_powermean_solve!`, which holds their defaults.
    stats = _powermean_solve!(α, β, R, C, one(T), rtol; kwargs..., terms)
    _warn_powermean_unconverged(:noise_cover, stats, :sweeps)
    _warn_powermean_unsettled(:noise_cover, stats)

    α ./= 2
    β ./= 2
    _check_representable(α, i -> !isempty(_slots(R, i)), β, j -> !isempty(_slots(C, j)), Ta, :noise_cover; gauge=true)
    for (i, ri) in zip(eachindex(a, α), r)
        a[i] = ri > 0 ? exp(α[i]) : zero(Ta)
    end
    for (j, cj) in zip(eachindex(b, β), c)
        b[j] = cj > 0 ? exp(β[j]) : zero(Ta)
    end
    if iszero(δ)
        _balance_cover!(a, b, V)
    else
        # The support of `M` is the whole nonzero block, a single component.
        rowcomp = Int[ri > 0 for ri in r]
        colcomp = Int[cj > 0 for cj in c]
        _balance_cover!(a, b, rowcomp, colcomp, 1, n′ .* rowcomp, m′ .* colcomp)
    end
    return a, b, PowerMeanStats{T}(stats)
end

# Sum of the values of each group of `S`, by position.
_group_sums(S::GroupedSupport{T}) where {T} =
    T[sum(view(S.val, S.ptr[p]:S.ptr[p+1]-1); init=zero(T)) for p in 1:length(S.ax)]

_noise_variance_ok(v) = isfinite(v) && v >= 0

function _check_noise_variance(V::AbstractMatrix)
    for v in V
        _noise_variance_ok(v) || _throw_noise_variance(v)
    end
    return nothing
end

function _check_noise_variance(V::SparseMatrixCSC)
    for v in nonzeros(V)
        _noise_variance_ok(v) || _throw_noise_variance(v)
    end
    return nothing
end

@noinline _throw_noise_variance(v) =
    throw(ArgumentError("noise_cover: the variance estimates `V` must be finite and nonnegative, but `V` has an entry $v"))

# ============================================================
# Existence of the scaling for a zero pattern
# ============================================================

# A positive scaling of `V` to row sums `nt` and column sums `mt` (on its nonzero
# rows and columns) exists if and only if some nonnegative matrix with exactly the
# zero pattern of `V` has these sums. That is decided by a maximum flow from a
# source through each nonzero row (capacity `nt`), each nonzero entry (unbounded),
# and each nonzero column (capacity `mt`) to a sink:
#
# - if the maximum flow does not saturate the source, no matrix with this pattern
#   has these sums;
# - otherwise an entry can be positive in some saturating flow exactly when its row
#   and column lie in the same strongly connected component of the residual graph
#   (arcs row → column for every entry, column → row where the flow is positive).
#   Saturating flows form a convex set, so the scaling exists if and only if every
#   entry passes this test; when one fails, Sinkhorn iteration converges only to a
#   limit in which some scales diverge.
#
# `R` holds the row-grouped support of `V`, `colax` the column axis of `V`.
function _check_noise_scaling_exists(R::GroupedSupport, colax::AbstractUnitRange, m′::Int, n′::Int)
    m, n = length(R.ax), length(colax)
    roff, coff = first(R.ax) - 1, first(colax) - 1
    rowptr = R.ptr
    rowcol = Int[k - coff for k in R.idx]
    # Column-grouped copy of the pattern; `coledge[k]` is the row-grouped slot of
    # the same entry, which indexes the flow.
    colptr = zeros(Int, n + 1)
    for q in rowcol
        colptr[q+1] += 1
    end
    colptr[1] = 1
    cumsum!(colptr, colptr)
    colrow = Vector{Int}(undef, length(rowcol))
    coledge = Vector{Int}(undef, length(rowcol))
    cursor = colptr[1:n]
    for p in 1:m, e in rowptr[p]:rowptr[p+1]-1
        q = rowcol[e]
        k = cursor[q]
        colrow[k] = p
        coledge[k] = e
        cursor[q] = k + 1
    end
    rcap = Int[rowptr[p+1] > rowptr[p] ? n′ : 0 for p in 1:m]
    ccap = Int[colptr[q+1] > colptr[q] ? m′ : 0 for q in 1:n]
    flow, srem, lr, lc = _noise_maxflow(rowptr, rowcol, colptr, colrow, coledge, rcap, ccap)
    if any(>(0), srem)
        # `lr` and `lc` mark the rows and columns reachable from the source in the
        # final residual graph: a minimum cut, whose rows have nonzero entries only
        # in its columns and require more than those columns can absorb.
        rows = [p + roff for p in 1:m if lr[p] >= 0]
        k, l = length(rows), count(>=(0), lc)
        throw(SolverFailure("noise_cover: no scaling of `V` to these row and column sums exists, because no nonnegative matrix with the zero pattern of `V` has them: the $k row(s) $(_index_list(rows)) have their nonzero entries in only $l column(s), and $k*n′ = $(k*n′) exceeds $l*m′ = $(l*m′) (n′ = $(n′), m′ = $(m′)). Remove such rows or columns, or pass `shrink > 0`."))
    end
    comp = _noise_scc(rowptr, rowcol, colptr, colrow, coledge, flow)
    nbad, first_bad = 0, (0, 0)
    for p in 1:m, e in rowptr[p]:rowptr[p+1]-1
        q = rowcol[e]
        if comp[p] != comp[m+q]
            nbad == 0 && (first_bad = (p + roff, q + coff))
            nbad += 1
        end
    end
    nbad == 0 && return nothing
    throw(SolverFailure("noise_cover: `V` can be scaled to these row and column sums only in a limit in which some scale factors diverge: every nonnegative matrix with the zero pattern of `V` and these sums vanishes on $nbad of the nonzero entries of `V`, including V[$(first_bad[1]), $(first_bad[2])]. Pass `shrink > 0`, or remove the rows or columns that force this."))
end

function _index_list(idx; nshow=5)
    s = join(string.(idx[1:min(end, nshow)]), ", ")
    return length(idx) > nshow ? "$s, …" : s
end

# Maximum flow by Dinic's algorithm on the network of
# `_check_noise_scaling_exists`. Nodes are rows `1:m` and columns `1:n`; the source
# and sink are implicit, with remaining capacities `srem` (source → row) and `trem`
# (column → sink). Returns the flow on each entry, `srem`, and the BFS levels of the
# last phase, in which the rows and columns with level `>= 0` are those reachable
# from the source.
function _noise_maxflow(rowptr, rowcol, colptr, colrow, coledge, rcap, ccap)
    m, n = length(rcap), length(ccap)
    flow = zeros(Int, length(rowcol))
    srem, trem = copy(rcap), copy(ccap)
    lr, lc = fill(-1, m), fill(-1, n)
    queue = Int[]
    currow = Vector{Int}(undef, m)   # next forward arc of each row
    curcol = Vector{Int}(undef, n)   # next arc of each column; `colptr[q] - 1` is the sink arc
    stk = Int[]                      # path: row `p` as `p`, column `q` as `m + q`, sink as `0`
    arc = Int[]                      # entry through which each path node is reached (0 from source or to sink)
    while true
        # Levels of the residual graph.
        fill!(lr, -1); fill!(lc, -1)
        lt = -1
        empty!(queue)
        for p in 1:m
            srem[p] > 0 && (lr[p] = 1; push!(queue, p))
        end
        head = 1
        while head <= length(queue)
            u = queue[head]; head += 1
            if u <= m
                for e in rowptr[u]:rowptr[u+1]-1
                    q = rowcol[e]
                    lc[q] < 0 && (lc[q] = lr[u] + 1; push!(queue, m + q))
                end
            else
                q = u - m
                trem[q] > 0 && lt < 0 && (lt = lc[q] + 1)
                for k in colptr[q]:colptr[q+1]-1
                    p = colrow[k]
                    flow[coledge[k]] > 0 && lr[p] < 0 && (lr[p] = lc[q] + 1; push!(queue, p))
                end
            end
        end
        lt < 0 && return flow, srem, lr, lc
        # Blocking flow by depth-first search along level-increasing arcs.
        for p in 1:m
            currow[p] = rowptr[p]
        end
        for q in 1:n
            curcol[q] = colptr[q] - 1
        end
        for p0 in 1:m
            while srem[p0] > 0 && lr[p0] == 1
                empty!(stk); empty!(arc)
                push!(stk, p0); push!(arc, 0)
                while !isempty(stk) && last(stk) != 0
                    u = last(stk)
                    found = false
                    if u <= m
                        while currow[u] < rowptr[u+1]
                            e = currow[u]
                            q = rowcol[e]
                            if lc[q] == lr[u] + 1
                                push!(stk, m + q); push!(arc, e)
                                found = true
                                break
                            end
                            currow[u] = e + 1
                        end
                        found || (lr[u] = -2)
                    else
                        q = u - m
                        if curcol[q] < colptr[q]
                            if trem[q] > 0 && lt == lc[q] + 1
                                push!(stk, 0); push!(arc, 0)
                                found = true
                            else
                                curcol[q] = colptr[q]
                            end
                        end
                        while !found && curcol[q] < colptr[q+1]
                            k = curcol[q]
                            e, p = coledge[k], colrow[k]
                            if flow[e] > 0 && lr[p] == lc[q] + 1
                                push!(stk, p); push!(arc, e)
                                found = true
                                break
                            end
                            curcol[q] = k + 1
                        end
                        found || (lc[q] = -2)
                    end
                    # A dead end leaves the path; its level no longer matches, so the
                    # arc into it is skipped when its parent resumes.
                    found || (pop!(stk); pop!(arc))
                end
                isempty(stk) && break
                # `stk` alternates row, column, row, …, column, sink; reverse arcs
                # enter rows after the first.
                f = srem[p0]
                for t in 3:2:length(stk)-1
                    f = min(f, flow[arc[t]])
                end
                qlast = stk[end-1] - m
                f = min(f, trem[qlast])
                srem[p0] -= f
                trem[qlast] -= f
                for t in 2:length(stk)-1
                    flow[arc[t]] += iseven(t) ? f : -f
                end
            end
        end
    end
end

# Strongly connected components of the residual graph of a saturating flow: arcs
# row → column for every entry and column → row where the flow is positive. Rows are
# nodes `1:m` and columns `m+1:m+n`. Tarjan's algorithm with an explicit stack.
function _noise_scc(rowptr, rowcol, colptr, colrow, coledge, flow)
    m, n = length(rowptr) - 1, length(colptr) - 1
    N = m + n
    index = zeros(Int, N)
    low = zeros(Int, N)
    onstack = falses(N)
    comp = zeros(Int, N)
    it = zeros(Int, N)
    stack = Int[]
    calls = Int[]
    counter = 0
    ncomp = 0
    first_arc(v) = v <= m ? rowptr[v] : colptr[v-m]
    function next_neighbor!(v)
        if v <= m
            k = it[v]
            k < rowptr[v+1] || return 0
            it[v] = k + 1
            return m + rowcol[k]
        end
        q = v - m
        k = it[v]
        while k < colptr[q+1]
            if flow[coledge[k]] > 0
                it[v] = k + 1
                return colrow[k]
            end
            k += 1
        end
        it[v] = k
        return 0
    end
    for v0 in 1:N
        index[v0] == 0 || continue
        counter += 1
        index[v0] = low[v0] = counter
        push!(stack, v0); onstack[v0] = true
        it[v0] = first_arc(v0)
        push!(calls, v0)
        while !isempty(calls)
            v = last(calls)
            w = next_neighbor!(v)
            if w != 0
                if index[w] == 0
                    counter += 1
                    index[w] = low[w] = counter
                    push!(stack, w); onstack[w] = true
                    it[w] = first_arc(w)
                    push!(calls, w)
                elseif onstack[w]
                    low[v] = min(low[v], index[w])
                end
            else
                pop!(calls)
                if low[v] == index[v]
                    ncomp += 1
                    while true
                        w = pop!(stack)
                        onstack[w] = false
                        comp[w] = ncomp
                        w == v && break
                    end
                end
                isempty(calls) || (u = last(calls); low[u] = min(low[u], low[v]))
            end
        end
    end
    return comp
end
