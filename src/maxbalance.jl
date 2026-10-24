# Max-balancing of the transversal-tight difference system within its strongly
# connected components (Hook, Pestana, Tisseur & Hogg, SIAM J. Matrix Anal.
# Appl. 40:320–346, 2019, Algorithm 3.10).
#
# In the difference system on the component scales `y`, entry `e` joins
# `P[e] → Q[e]` with log slack `d[e] = y[P[e]] - y[Q[e]] - C[e] >= 0`, and so log
# coverage ratio `w[e] = -d[e] <= 0`. A shift `x` of the scales changes `w[e]` to
# `w[e] - x[P[e]] + x[Q[e]]`, which leaves the sum of `w` around every cycle
# unchanged. On a strongly connected digraph, the max-balanced shift makes the
# largest `w` leaving every set of vertices equal to the largest `w` entering it.
# It is unique up to a constant, and every `w` is then at most the largest mean
# weight of a cycle.

# Arcs `e` with `keep(e)` grouped by tail `P[e]` among the vertices `1:nv`:
# `adj[ptr[v]:ptr[v+1]-1]` are the arcs leaving `v`.
function _out_arcs(nv::Int, P::AbstractVector{Int}, keep)
    ptr = zeros(Int, nv + 1)
    for e in eachindex(P)
        keep(e) && (ptr[P[e]+1] += 1)
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    adj = Vector{Int}(undef, ptr[end] - 1)
    cursor = ptr[1:nv]
    for e in eachindex(P)
        keep(e) || continue
        adj[cursor[P[e]]] = e
        cursor[P[e]] += 1
    end
    return ptr, adj
end

# Max-plus policy iteration (Cochet-Terrasson, Cohen, Gaubert, McGettrick &
# Quadrat, 1998) on the strongly connected digraph on `1:nv` whose arcs leaving
# `v` are `adj[ptr[v]:ptr[v+1]-1]`, arc `e` going to `dst[e]` with weight `w[e]`.
# `pol[v]` is the arc chosen at `v`, or 0 for none; it is the starting policy and
# is updated in place. `tol` is the resolution of the weights; a cycle mean
# accumulates the roundoff of up to `nv` of them, so cycle means within
# `nv * tol` of each other are treated as equal.
#
# Returns `(λ, s, cycle)`: the largest mean weight `λ` of a cycle, a vector `s`
# with `w[e] - s[v] + s[dst[e]] <= λ` on every arc `e` leaving `v`, up to
# roundoff, and the vertices of a cycle of mean `λ`, in order.
function _howard!(pol::Vector{Int}, nv::Int, ptr::Vector{Int}, adj::Vector{Int},
                  dst::Vector{Int}, w::Vector{T}, tol::T) where {T}
    for v in 1:nv
        ptr[v] < ptr[v+1] || error("internal error: a vertex has no arc leaving it")
        pol[v] == 0 || continue
        best = adj[ptr[v]]
        for s in ptr[v]:ptr[v+1]-1
            w[adj[s]] > w[best] && (best = adj[s])
        end
        pol[v] = best
    end
    ηtol = nv * tol
    η = zeros(T, nv)   # mean weight of the cycle each vertex's policy path reaches
    s = zeros(T, nv)   # bias, relative to that cycle
    state = zeros(Int, nv)
    path = Int[]
    cyc = Int[]
    best = Int[]
    for _ in 1:(100 + 10 * nv)
        # Value determination on the functional graph `v → dst[pol[v]]`.
        fill!(state, 0)    # 0 unvisited, 1 on the current path, 2 done
        λ = T(-Inf)
        for r in 1:nv
            state[r] == 0 || continue
            empty!(path)
            v = r
            while state[v] == 0
                state[v] = 1
                push!(path, v)
                v = dst[pol[v]]
            end
            if state[v] == 1
                # A new cycle through `v`: its mean fixes `η`, and `s[v]`
                # keeps its value.
                empty!(cyc)
                u = v
                μ = zero(T)
                while true
                    push!(cyc, u)
                    μ += w[pol[u]]
                    u = dst[pol[u]]
                    u == v && break
                end
                μ /= length(cyc)
                for u in cyc
                    η[u] = μ
                end
                for k in length(cyc):-1:2
                    u = cyc[k]
                    s[u] = w[pol[u]] - μ + s[dst[pol[u]]]
                end
                for u in cyc
                    state[u] = 2
                end
                if μ > λ
                    λ = μ
                    copy!(best, cyc)
                end
            end
            for u in Iterators.reverse(path)
                state[u] == 2 && continue
                t = dst[pol[u]]
                η[u] = η[t]
                s[u] = w[pol[u]] - η[t] + s[t]
                state[u] = 2
            end
        end
        # Policy improvement: first the cycle means, then the bias.
        changed = false
        for v in 1:nv
            ηv = η[dst[pol[v]]]
            e = pol[v]
            for k in ptr[v]:ptr[v+1]-1
                a = adj[k]
                η[dst[a]] > ηv + ηtol && (ηv = η[dst[a]]; e = a)
            end
            e == pol[v] || (pol[v] = e; changed = true)
        end
        if !changed
            for v in 1:nv
                cur = w[pol[v]] + s[dst[pol[v]]]
                e = pol[v]
                for k in ptr[v]:ptr[v+1]-1
                    a = adj[k]
                    abs(η[dst[a]] - η[v]) <= ηtol || continue
                    val = w[a] + s[dst[a]]
                    val > cur + tol + TIGHT_ULPS * eps(T) * abs(cur) && (cur = val; e = a)
                end
                e == pol[v] || (pol[v] = e; changed = true)
            end
        end
        if !changed
            ηmin, ηmax = extrema(η)
            ηmax - ηmin <= ηtol ||
                error("internal error: the cycle means differ ($ηmin to $ηmax) on a strongly connected digraph")
            return λ, s, best
        end
    end
    error("internal error: policy iteration did not converge")
end

# Max-balancing shift `x` of the strongly connected digraph on `1:nv` with arcs
# `src[e] → dst[e]` of weight `w[e]`: repeatedly find a cycle of largest mean in
# the contracted digraph, shift by the bias of `_howard!`, which lowers every
# arc to at most that mean, and contract the cycle, whose relative shifts are
# then fixed. Returns `x`, with `x[1] == 0`.
function _maxbalance(nv::Int, src::Vector{Int}, dst::Vector{Int}, w::Vector{T}) where {T}
    x = zeros(T, nv)
    nv == 1 && return x
    tol = TIGHT_ULPS * eps(T) * max(one(T), maximum(abs, w))
    parent = collect(1:nv)
    function find(v)
        while parent[v] != v
            parent[v] = parent[parent[v]]
            v = parent[v]
        end
        return v
    end
    pos = zeros(Int, nv)      # position of each representative in the contracted digraph
    pol = zeros(Int, nv)      # policy arc of each representative, by arc of `src`
    ne = length(src)
    live = falses(ne)
    cs = zeros(Int, ne)
    cd = zeros(Int, ne)
    cw = zeros(T, ne)
    reps = Int[]
    nnodes = nv
    while nnodes > 1
        empty!(reps)
        for v in 1:nv
            find(v) == v && (push!(reps, v); pos[v] = length(reps))
        end
        for e in 1:ne
            r, q = find(src[e]), find(dst[e])
            live[e] = r != q
            live[e] || continue
            cs[e], cd[e] = pos[r], pos[q]
            cw[e] = w[e] - x[src[e]] + x[dst[e]]
        end
        nr = length(reps)
        ptr, adj = _out_arcs(nr, cs, e -> live[e])
        cpol = [pol[r] != 0 && live[pol[r]] ? pol[r] : 0 for r in reps]
        _, sh, cyc = _howard!(cpol, nr, ptr, adj, cd, cw, tol)
        for v in 1:nv
            x[v] += sh[pos[find(v)]]
        end
        for (k, r) in enumerate(reps)
            pol[r] = cpol[k]
        end
        root = reps[cyc[1]]
        for k in cyc
            parent[reps[k]] = root
        end
        pol[root] = 0
        nnodes -= length(cyc) - 1
    end
    x .-= x[1]
    return x
end

# The scales `y` of the difference system `sys` shifted by the max-balancing
# shift of each strongly connected component, `scc` and `nscc` as returned by
# `_strong_components`. The shift of each component is fixed by making its
# first vertex keep its scale.
function _maxbalance_components(sys::EdgeList{T}, y::Vector{T}, scc::Vector{Int}, nscc::Int) where {T}
    P = first.(sys.edges)
    Q = last.(sys.edges)
    C = sys.cvals
    ynew = copy(y)
    vptr, verts = _out_arcs(nscc, scc, _ -> true)
    eptr, edges = _out_arcs(nscc, [scc[P[e]] for e in eachindex(P)], e -> scc[P[e]] == scc[Q[e]])
    loc = zeros(Int, length(y))
    for k in 1:nscc
        vr = vptr[k]:vptr[k+1]-1
        length(vr) > 1 || continue
        for (t, s) in enumerate(vr)
            loc[verts[s]] = t
        end
        er = eptr[k]:eptr[k+1]-1
        src = [loc[P[edges[s]]] for s in er]
        dst = [loc[Q[edges[s]]] for s in er]
        w = [C[edges[s]] - y[P[edges[s]]] + y[Q[edges[s]]] for s in er]
        x = _maxbalance(length(vr), src, dst, w)
        for (t, s) in enumerate(vr)
            ynew[verts[s]] += x[t]
        end
    end
    return ynew
end

# Feasible scales `Y` of the components of the difference system `csys` between
# strongly connected components, whose edges go from higher to lower labels:
# each component, from the sources down, takes the largest scale at most zero
# that covers the entries entering it.
function _feasible_component_scales(csys::EdgeList{T}, nscc::Int) where {T}
    Y = zeros(T, nscc)
    iptr, inarcs = _out_arcs(nscc, last.(csys.edges), _ -> true)
    for k in nscc:-1:1
        for s in iptr[k]:iptr[k+1]-1
            e = inarcs[s]
            j, _ = csys.edges[e]
            j > k || error("internal error: an edge between components does not follow their order")
            Y[k] = min(Y[k], Y[j] - csys.cvals[e])
        end
    end
    return Y
end
