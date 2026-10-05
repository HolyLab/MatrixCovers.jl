# Transversal-tight minimal covers and the maximum-product transversal they rest on.

"""
    a, b = cover_transversal(A; kwargs...)

Return the transversal-tight minimal cover of the square matrix `A`: the hard
cover minimizing `∑ log(a[i]*b[j]/|A[i,j]|)^2` over the support of `A`, among
the hard covers with `prod(a) * prod(b) == π*`, where

    π* = max over permutations σ of ∏ᵢ |A[i,σ(i)]|.

Every hard cover has `prod(a) * prod(b) >= π*`, with equality exactly when it
is tight (`a[i]*b[j] == |A[i,j]|`) on every entry of every transversal whose
product is `π*`. Among hard covers, these maximize `|det(A ./ (a .* b'))|`.

The products `a[i]*b[j]` are unique on the support, they do not depend on which
maximum-product transversal the solver finds, and they are scale-covariant:
for `D₁ * A * D₂` with positive diagonal `D₁`, `D₂`, they are multiplied by
`D₁[i,i] * D₂[j,j]`. When the minimal cover [`cover_min`](@ref) is already
tight on a maximum-product transversal, the two coincide. The factors use the
balance convention of [`cover_min`](@ref).

`A` must be square and structurally nonsingular, i.e., have at least one
transversal of nonzero entries; otherwise an `ArgumentError` is thrown.

# Extended help

A maximum-product transversal `σ` is found by shortest augmenting paths on the
costs `log(max_k |A[k,j]|) - log|A[i,j]|`. Tightness on `σ` determines `b` from
`a`, `b[σ[i]] = |A[i,σ[i]]| / a[i]`, so the remaining problem is posed on the
row scales alone, with one constraint per entry off the transversal. The
augmented-Lagrangian solver of [`cover_min`](@ref) solves it, starting from the
dual variables of the transversal, which are a hard cover tight on `σ`.

The keywords `κ`, `maxouter`, `maxiter`, `fillbudget`, `flopbudget`, and
`linsolve` are as for [`cover_min`](@ref), except that `linsolve=:woodbury` is
not supported; `:auto` chooses `:lsqr` when the stored support fills at most a
quarter of the grid and `:dense` otherwise. Sparse matrices default to `:lsqr`.
As for [`cover_min`](@ref), an active-set method finishes a solve whose
multiplier iteration stops early.

See also: [`cover_transversal!`](@ref), [`cover_min`](@ref).
"""
function cover_transversal(A::AbstractMatrix; kwargs...)
    a, b, _ = _cover_min_abslog2(A; transversal=true, fname=:cover_transversal, kwargs...)
    return a, b
end

"""
    a, b = cover_transversal!(a, b, A; kwargs...)

Mutating counterpart of [`cover_transversal`](@ref): writes the cover into `a`
and `b` and returns them. The initial contents of `a` and `b` are ignored.
`eachindex(a)` must match `axes(A, 1)` and `eachindex(b)` must match
`axes(A, 2)`.
"""
function cover_transversal!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...)
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(string(eachindex(b))), axes(A, 2)=$(string(axes(A, 2)))"))
    anew, bnew = cover_transversal(A; kwargs...)
    a .= anew
    b .= bnew
    return a, b
end

# Binary min-heap of `(key, index)` pairs. Entries are never decreased in place;
# a stale entry is skipped when popped.
function _heappush!(h::Vector{Tuple{T,Int}}, item::Tuple{T,Int}) where {T}
    push!(h, item)
    k = length(h)
    while k > 1
        p = k >> 1
        h[p][1] <= h[k][1] && break
        h[p], h[k] = h[k], h[p]
        k = p
    end
    return h
end

function _heappop!(h::Vector{Tuple{T,Int}}) where {T}
    top = h[1]
    last = pop!(h)
    n = length(h)
    if n > 0
        h[1] = last
        k = 1
        while true
            l = 2k
            l > n && break
            r = l + 1
            c = r <= n && h[r][1] < h[l][1] ? r : l
            h[k][1] <= h[c][1] && break
            h[k], h[c] = h[c], h[k]
            k = c
        end
    end
    return top
end

# Maximum-product transversal of the square support grouped by row in `G`, whose
# partners are the column indices `axc`. Solves the assignment problem with costs
# `w[i,j] = cmax[j] - log|A[i,j]|` (`cmax[j]` the largest log magnitude in column
# `j`) by shortest augmenting paths, with Dijkstra on the reduced costs
# `w[i,j] - u[i] - v[j] >= 0`, which vanish on matched entries.
#
# Returns `(σ, α, β, logπ)` in positions: row `i` is matched to column `σ[i]`,
# `logπ = ∑ᵢ log|A[i,σ[i]]|`, and the duals `α = -u`, `β = cmax - v` satisfy
# `α[i] + β[j] >= log|A[i,j]|` on the support, with equality on the transversal,
# up to roundoff.
function _max_product_transversal(G::GroupedSupport, axc::AbstractUnitRange, fname)
    T = eltype(G.val)
    axr = G.ax
    n = length(axr)
    length(axc) == n || throw(ArgumentError("$fname requires a square matrix, got size ($n, $(length(axc)))"))
    oc = first(axc) - 1
    cmax = fill(T(-Inf), n)
    for (ip, i) in enumerate(axr)
        for s in _slots(G, i)
            lv = log(G.val[s])
            isfinite(lv) || throw(ArgumentError("$fname requires finite entries, got abs(A[$i, $(G.idx[s])]) = $(G.val[s])"))
            jp = G.idx[s] - oc
            cmax[jp] = max(cmax[jp], lv)
        end
    end
    # Columns of the transversal entries, by row; rows of matched columns.
    σ = zeros(Int, n)
    τ = zeros(Int, n)
    u = zeros(T, n)
    v = zeros(T, n)
    # The column shift makes each column's smallest cost zero, so `v = 0` is
    # feasible; `u[i]` takes row `i`'s smallest cost, and zero-cost entries are
    # matched greedily.
    for (ip, i) in enumerate(axr)
        isempty(_slots(G, i)) && _throw_singular(fname)
        ui = T(Inf)
        for s in _slots(G, i)
            ui = min(ui, cmax[G.idx[s]-oc] - log(G.val[s]))
        end
        u[ip] = ui
        for s in _slots(G, i)
            jp = G.idx[s] - oc
            if τ[jp] == 0 && cmax[jp] - log(G.val[s]) - ui == 0
                σ[ip] = jp
                τ[jp] = ip
                break
            end
        end
    end
    dist = fill(T(Inf), n)
    pred = zeros(Int, n)
    final = falses(n)
    touched = Int[]   # columns whose `dist` was set in this search
    rows = Int[]      # rows reached in this search
    cols = Int[]      # columns finalized in this search
    heap = Tuple{T,Int}[]
    for sp in 1:n
        σ[sp] == 0 || continue
        empty!(heap)
        empty!(rows)
        empty!(cols)
        ip = sp
        δ = zero(T)
        sink = 0
        while true
            push!(rows, ip)
            i = first(axr) + ip - 1
            for s in _slots(G, i)
                jp = G.idx[s] - oc
                final[jp] && continue
                r = δ + (cmax[jp] - log(G.val[s]) - u[ip] - v[jp])
                if r < dist[jp]
                    isinf(dist[jp]) && push!(touched, jp)
                    dist[jp] = r
                    pred[jp] = ip
                    _heappush!(heap, (r, jp))
                end
            end
            jp = 0
            while !isempty(heap)
                r, k = _heappop!(heap)
                if !final[k] && r == dist[k]
                    jp = k
                    break
                end
            end
            jp == 0 && _throw_singular(fname)
            final[jp] = true
            push!(cols, jp)
            δ = dist[jp]
            if τ[jp] == 0
                sink = jp
                break
            end
            ip = τ[jp]
        end
        # Dual update keeping reduced costs nonnegative and zero on the new matching.
        u[sp] += δ
        for ip in rows
            ip == sp || (u[ip] += δ - dist[σ[ip]])
        end
        for jp in cols
            v[jp] -= δ - dist[jp]
        end
        # Augment along the shortest path.
        jp = sink
        while true
            ip = pred[jp]
            τ[jp] = ip
            jnext = σ[ip]
            σ[ip] = jp
            ip == sp && break
            jp = jnext
        end
        for jp in touched
            dist[jp] = T(Inf)
            final[jp] = false
        end
        empty!(touched)
    end
    α = -u
    β = cmax .- v
    logπ = zero(T)
    for (ip, i) in enumerate(axr)
        for s in _slots(G, i)
            if G.idx[s] - oc == σ[ip]
                logπ += log(G.val[s])
                break
            end
        end
    end
    return σ, α, β, logπ
end

@noinline _throw_singular(fname) =
    throw(ArgumentError("$fname requires a structurally nonsingular matrix, but no transversal of nonzero entries exists"))

# Rows joined by entries that lie on some maximum-product transversal.
#
# In the difference system on the row log scales `α`, entry `e = (p, q)` has
# residual `α[p] - α[q] - cvals[e]`, and the transversal duals `α0` make every
# residual nonnegative. An entry with zero residual at `α0` that lies on a cycle
# of such entries completes an alternating cycle, and so a second
# maximum-product transversal; every cover tight on one is tight on all of
# them, which forces a zero residual on the whole cycle. These equalities leave
# the feasible set without interior. Merging each strongly connected component
# of the zero-residual entries into one unknown removes them.
#
# Returns `(comp, o, nc)`: row `p` belongs to component `comp[p]` in `1:nc`, and
# `α[p] = y[comp[p]] + o[p]` for the component scales `y`.
function _tight_components(edges::Vector{Tuple{Int,Int}}, cvals::Vector{T}, α0::Vector{T}) where {T}
    m = length(α0)
    # Zero-residual entries, as adjacency lists in both directions.
    tight = falses(length(edges))
    for (e, (p, q)) in enumerate(edges)
        z = α0[p] - α0[q] - cvals[e]
        tight[e] = abs(z) <= TIGHT_ULPS * eps(T) * max(oneunit(T), abs(α0[p]), abs(α0[q]), abs(cvals[e]))
    end
    outptr = zeros(Int, m + 1)
    for (e, (p, _)) in enumerate(edges)
        tight[e] && (outptr[p+1] += 1)
    end
    outptr[1] = 1
    cumsum!(outptr, outptr)
    outedge = zeros(Int, outptr[end] - 1)
    cursor = outptr[1:m]
    for (e, (p, _)) in enumerate(edges)
        tight[e] || continue
        outedge[cursor[p]] = e
        cursor[p] += 1
    end
    # Iterative Tarjan.
    comp = zeros(Int, m)
    index = zeros(Int, m)
    low = zeros(Int, m)
    onstack = falses(m)
    stack = Int[]
    callstack = Tuple{Int,Int}[]   # (row, next slot in its adjacency list)
    counter = 0
    nc = 0
    for r in 1:m
        index[r] == 0 || continue
        push!(callstack, (r, outptr[r]))
        counter += 1
        index[r] = low[r] = counter
        push!(stack, r); onstack[r] = true
        while !isempty(callstack)
            p, s = callstack[end]
            if s < outptr[p+1]
                callstack[end] = (p, s + 1)
                q = edges[outedge[s]][2]
                if index[q] == 0
                    counter += 1
                    index[q] = low[q] = counter
                    push!(stack, q); onstack[q] = true
                    push!(callstack, (q, outptr[q]))
                elseif onstack[q]
                    low[p] = min(low[p], index[q])
                end
            else
                pop!(callstack)
                if low[p] == index[p]
                    nc += 1
                    while true
                        q = pop!(stack)
                        onstack[q] = false
                        comp[q] = nc
                        q == p && break
                    end
                end
                isempty(callstack) || (low[callstack[end][1]] = min(low[callstack[end][1]], low[p]))
            end
        end
    end
    # Offsets from a breadth-first spanning tree of each component, crossing
    # tight entries within the component in either direction.
    inptr = zeros(Int, m + 1)
    for (e, (_, q)) in enumerate(edges)
        tight[e] && (inptr[q+1] += 1)
    end
    inptr[1] = 1
    cumsum!(inptr, inptr)
    inedge = zeros(Int, inptr[end] - 1)
    cursor = inptr[1:m]
    for (e, (_, q)) in enumerate(edges)
        tight[e] || continue
        inedge[cursor[q]] = e
        cursor[q] += 1
    end
    o = zeros(T, m)
    seen = falses(m)
    queue = Int[]
    for r in 1:m
        seen[r] && continue
        seen[r] = true
        push!(queue, r)
        while !isempty(queue)
            p = popfirst!(queue)
            for s in outptr[p]:outptr[p+1]-1
                e = outedge[s]
                q = edges[e][2]
                (seen[q] || comp[q] != comp[p]) && continue
                o[q] = o[p] - cvals[e]
                seen[q] = true
                push!(queue, q)
            end
            for s in inptr[p]:inptr[p+1]-1
                e = inedge[s]
                q = edges[e][1]
                (seen[q] || comp[q] != comp[p]) && continue
                o[q] = o[p] + cvals[e]
                seen[q] = true
                push!(queue, q)
            end
        end
    end
    return comp, o, nc
end

# Roundoff allowance, in units of `eps`, for a zero residual at the transversal duals.
const TIGHT_ULPS = 64
