# Max-balancing of the transversal-tight difference system within its strongly
# connected components: the max-balanced Hungarian scaling of Hook, Pestana,
# Tisseur & Hogg (SIAM J. Matrix Anal. Appl. 40:320–346, 2019).
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

# Indexed binary max-heap over the integers `1:n`: `key[e]` is the key of `e`,
# `heap` holds the members, and `pos[e]` is the position of `e` in `heap`, or 0.
struct IndexedHeap{T}
    key::Vector{T}
    heap::Vector{Int}
    pos::Vector{Int}
end
IndexedHeap{T}(n::Int) where {T} = IndexedHeap{T}(Vector{T}(undef, n), Int[], zeros(Int, n))

function _heap_swap!(h::IndexedHeap, i::Int, j::Int)
    h.heap[i], h.heap[j] = h.heap[j], h.heap[i]
    h.pos[h.heap[i]] = i
    h.pos[h.heap[j]] = j
    return nothing
end
function _heap_up!(h::IndexedHeap, i::Int)
    while i > 1
        p = i >> 1
        h.key[h.heap[p]] >= h.key[h.heap[i]] && break
        _heap_swap!(h, p, i)
        i = p
    end
    return nothing
end
function _heap_down!(h::IndexedHeap, i::Int)
    n = length(h.heap)
    while true
        c = 2i
        c <= n || break
        c < n && h.key[h.heap[c+1]] > h.key[h.heap[c]] && (c += 1)
        h.key[h.heap[c]] > h.key[h.heap[i]] || break
        _heap_swap!(h, c, i)
        i = c
    end
    return nothing
end
# Inserts `e` with key `k`, or changes its key.
function _heap_set!(h::IndexedHeap, e::Int, k)
    h.key[e] = k
    i = h.pos[e]
    if i == 0
        push!(h.heap, e)
        h.pos[e] = length(h.heap)
        _heap_up!(h, length(h.heap))
    else
        _heap_up!(h, i)
        _heap_down!(h, h.pos[e])
    end
    return nothing
end
# Removes `e` if present.
function _heap_remove!(h::IndexedHeap, e::Int)
    i = h.pos[e]
    i == 0 && return nothing
    last = pop!(h.heap)
    h.pos[e] = 0
    if i <= length(h.heap)
        h.heap[i] = last
        h.pos[last] = i
        _heap_up!(h, i)
        _heap_down!(h, h.pos[last])
    end
    return nothing
end
_heap_pop!(h::IndexedHeap) = (e = h.heap[1]; _heap_remove!(h, e); e)

# The parametric shortest-path tree of a digraph whose arcs have length `λ - w`
# for a decreasing parameter `λ`, with each cycle contracted when its length
# reaches zero. The nodes are the vertices `1:nv`; a contracted cycle is
# represented by its node nearest the root, and the other nodes of the cycle are
# no longer live.
#
# - Arc `e` joins the live nodes `src[e] → dst[e]` with weight `w[e]`;
#   `alive[e]` is false once both ends lie in one node.
# - Live node `v` has depth `a[v]`, and the arcs of its tree path have total
#   weight `b[v]`, so its distance from the root is `a[v] * λ - b[v]`. `tarc[v]`
#   is the tree arc entering `v`, or 0 at the root.
# - The children of `v` form a doubly linked list with first member
#   `fchild[v]`, last member `lchild[v]`, and links `nsib`, `psib`.
# - The arcs incident to `v` form a doubly linked list with first member
#   `lhead[v]`, last member `ltail[v]`, and links `lnext`, `lprev`. Its members
#   are slots: slot `e` is arc `e` as it leaves `src[e]`, and slot
#   `e + length(w)` is arc `e` as it enters `dst[e]`.
# - `heap` holds every live non-tree arc `p → q` with `a[p] + 1 > a[q]`, keyed
#   by the `λ` at which it becomes tight.
struct ParametricTree{T}
    src::Vector{Int}
    dst::Vector{Int}
    w::Vector{T}
    alive::BitVector
    a::Vector{Int}
    b::Vector{T}
    tarc::Vector{Int}
    fchild::Vector{Int}
    lchild::Vector{Int}
    nsib::Vector{Int}
    psib::Vector{Int}
    lhead::Vector{Int}
    ltail::Vector{Int}
    lnext::Vector{Int}
    lprev::Vector{Int}
    heap::IndexedHeap{T}
end
function ParametricTree(nv::Int, src::Vector{Int}, dst::Vector{Int}, w::Vector{T}) where {T}
    ne = length(w)
    z() = zeros(Int, nv)
    return ParametricTree{T}(copy(src), copy(dst), copy(w), BitVector(src[e] != dst[e] for e in 1:ne),
                             z(), zeros(T, nv), z(), z(), z(), z(), z(), z(), z(),
                             zeros(Int, 2ne), zeros(Int, 2ne), IndexedHeap{T}(ne))
end

function _child_append!(S::ParametricTree, p::Int, c::Int)
    S.psib[c] = S.lchild[p]
    S.nsib[c] = 0
    S.lchild[p] == 0 ? (S.fchild[p] = c) : (S.nsib[S.lchild[p]] = c)
    S.lchild[p] = c
    return nothing
end
function _child_remove!(S::ParametricTree, p::Int, c::Int)
    pr, nx = S.psib[c], S.nsib[c]
    pr == 0 ? (S.fchild[p] = nx) : (S.nsib[pr] = nx)
    nx == 0 ? (S.lchild[p] = pr) : (S.psib[nx] = pr)
    return nothing
end
# Moves the children of `z` to the end of the children of `p`.
function _child_concat!(S::ParametricTree, p::Int, z::Int)
    S.fchild[z] == 0 && return nothing
    if S.lchild[p] == 0
        S.fchild[p] = S.fchild[z]
    else
        S.nsib[S.lchild[p]] = S.fchild[z]
        S.psib[S.fchild[z]] = S.lchild[p]
    end
    S.lchild[p] = S.lchild[z]
    S.fchild[z] = S.lchild[z] = 0
    return nothing
end
function _arcs_append!(S::ParametricTree, v::Int, s::Int)
    S.lprev[s] = S.ltail[v]
    S.lnext[s] = 0
    S.ltail[v] == 0 ? (S.lhead[v] = s) : (S.lnext[S.ltail[v]] = s)
    S.ltail[v] = s
    return nothing
end
function _arcs_concat!(S::ParametricTree, v::Int, z::Int)
    S.lhead[z] == 0 && return nothing
    if S.lhead[v] == 0
        S.lhead[v] = S.lhead[z]
    else
        S.lnext[S.ltail[v]] = S.lhead[z]
        S.lprev[S.lhead[z]] = S.ltail[v]
    end
    S.ltail[v] = S.ltail[z]
    S.lhead[z] = S.ltail[z] = 0
    return nothing
end
function _arcs_unlink!(S::ParametricTree, v::Int, s::Int)
    pr, nx = S.lprev[s], S.lnext[s]
    pr == 0 ? (S.lhead[v] = nx) : (S.lnext[pr] = nx)
    nx == 0 ? (S.ltail[v] = pr) : (S.lprev[nx] = pr)
    return nothing
end
# The arc of slot `s` and the node at the other end of it.
function _slot_arc(S::ParametricTree, s::Int)
    ne = length(S.w)
    return s > ne ? (s - ne, S.src[s-ne]) : (s, S.dst[s])
end

# Updates the heap entry of the live arc `e` after its ends have moved. The arc
# `p → q` has reduced length `(a[p] + 1 - a[q]) * λ - (b[p] + w[e] - b[q])`,
# which falls to zero as `λ` decreases only when `a[p] + 1 > a[q]`. Such a
# non-tree arc is keyed by the `λ` at which it does; any other arc is removed.
function _key!(S::ParametricTree, e::Int)
    p, q = S.src[e], S.dst[e]
    c = S.a[p] + 1 - S.a[q]
    if S.tarc[q] == e || c <= 0
        _heap_remove!(S.heap, e)
    else
        _heap_set!(S.heap, e, (S.b[p] + S.w[e] - S.b[q]) / c)
    end
    return nothing
end

# Adds `Δa` to the depth and `Δb` to the path weight of each node `y` of the
# subtree listed in `sub`, marks it with `stamp[y] = st`, and rekeys the arcs
# between the subtree and the rest of the digraph. Arcs within the subtree keep
# their keys, since both ends shift alike.
function _shift_subtree!(S::ParametricTree, sub::Vector{Int}, Δa::Int, Δb, stamp::Vector{Int}, st::Int)
    for y in sub
        S.a[y] += Δa
        S.b[y] += Δb
        stamp[y] = st
    end
    for y in sub
        s = S.lhead[y]
        while s != 0
            f, o = _slot_arc(S, s)
            stamp[o] == st || _key!(S, f)
            s = S.lnext[s]
        end
    end
    return nothing
end

# Appends the descendants of `v` to `sub`, with `stack` as workspace.
function _subtree!(sub::Vector{Int}, S::ParametricTree, v::Int, stack::Vector{Int})
    empty!(stack)
    c = S.fchild[v]
    while c != 0
        push!(stack, c)
        c = S.nsib[c]
    end
    while !isempty(stack)
        y = pop!(stack)
        push!(sub, y)
        c = S.fchild[y]
        while c != 0
            push!(stack, c)
            c = S.nsib[c]
        end
    end
    return sub
end

# Max-balancing shift `x` of the strongly connected digraph on `1:nv` with arcs
# `src[e] → dst[e]` of weight `w[e]`, by the minimum-balance algorithm of Young,
# Tarjan & Orlin (Networks 21:205–221, 1991) with the signs reversed; the shift
# is unique up to a constant (Schneider & Schneider, Math. Oper. Res.
# 16:208–222, 1991). Returns `x`, with `x[1] == 0`.
#
# Give arc `e` the length `λ - w[e]` and let `d` be the shortest-path distances
# from vertex 1. Then `w[e] - d[src[e]] + d[dst[e]] <= λ` on every arc, with
# equality on the tree arcs. As `λ` decreases from `+∞`, the tree changes each
# time a non-tree arc becomes tight. If the tail of the arc is not a descendant
# of its head, the arc replaces the tree arc into its head. Otherwise the arc
# and the tree path from its head to its tail form a cycle of zero length,
# whose mean weight `λ` is the largest cycle mean of the digraph left by the
# earlier contractions. The distances within the cycle are then fixed relative
# to one another, and the cycle is contracted into its node nearest the root.
# When node `z` is absorbed, `uf[z]` records the node that absorbed it and
# `uΔ[z]` its distance relative to that node; the shift of `z` is the sum of
# `uΔ` along the chain `z`, `uf[z]`, `uf[uf[z]]`, … up to vertex 1.
#
# The tree starts as the shortest-path tree for `λ → ∞`: fewest arcs, then
# largest weight. Each time an arc becomes tight, the work is proportional to
# the size of the subtrees whose depths change plus the number of arcs incident
# to them: `O(nm + n² log n)` in all, and far less when the tree stays shallow.
function _maxbalance(nv::Int, src::Vector{Int}, dst::Vector{Int}, w::Vector{T}) where {T}
    x = zeros(T, nv)
    nv == 1 && return x
    ne = length(w)
    S = ParametricTree(nv, src, dst, w)
    a, b, tarc, heap = S.a, S.b, S.tarc, S.heap
    csrc, cdst, cw = S.src, S.dst, S.w
    fill!(a, -1)
    ptr, adj = _out_arcs(nv, src, e -> S.alive[e])
    queue = Int[1]
    a[1] = 0
    i = 1
    while i <= length(queue)
        v = queue[i]
        i += 1
        for s in ptr[v]:ptr[v+1]-1
            e = adj[s]
            q = dst[e]
            if a[q] == -1
                a[q] = a[v] + 1
                push!(queue, q)
            end
            if a[q] == a[v] + 1 && (tarc[q] == 0 || b[v] + w[e] > b[q])
                b[q] = b[v] + w[e]
                tarc[q] = e
            end
        end
    end
    length(queue) == nv || error("internal error: the digraph is not strongly connected")
    for q in 2:nv
        _child_append!(S, src[tarc[q]], q)
    end
    for e in 1:ne
        S.alive[e] || continue
        _arcs_append!(S, src[e], e)
        _arcs_append!(S, dst[e], e + ne)
        _key!(S, e)
    end
    uf = collect(1:nv)
    uΔ = zeros(T, nv)
    stamp = zeros(Int, nv)
    st = 0
    sub = Int[]
    stack = Int[]
    path = Int[]
    touched = Int[]
    nlive = nv
    while nlive > 1
        isempty(heap.heap) && error("internal error: no arc can become tight on a strongly connected digraph")
        e = _heap_pop!(heap)
        λ = heap.key[e]
        u, v = csrc[e], cdst[e]
        # Decide whether `u` is a descendant of `v` by alternating a step up the
        # tree from `u` with a step of listing the subtree of `v`, so that a
        # cycle costs the smaller of its length and that subtree. If `u` is not
        # a descendant, `sub` ends up listing the descendants of `v`.
        cycle = false
        z = u
        empty!(sub)
        empty!(stack)
        push!(stack, v)
        while true
            if z != 0
                z == v && (cycle = true; break)
                t = tarc[z]
                z = t == 0 ? 0 : csrc[t]
            end
            isempty(stack) && break
            y = pop!(stack)
            y == v || push!(sub, y)
            y == u && (cycle = true; break)
            c = S.fchild[y]
            while c != 0
                push!(stack, c)
                c = S.nsib[c]
            end
        end
        if !cycle
            # `e` replaces the tree arc into `v`: the subtree of `v` moves under
            # `u` and becomes deeper.
            Δa = a[u] + 1 - a[v]
            Δb = b[u] + cw[e] - b[v]
            _child_remove!(S, csrc[tarc[v]], v)
            _child_append!(S, u, v)
            tarc[v] = e
            push!(sub, v)
            _shift_subtree!(S, sub, Δa, Δb, stamp, st += 1)
        else
            # Contract the cycle `v → ⋯ → u → v` into `v`.
            empty!(path)
            z = u
            while z != v
                push!(path, z)
                z = csrc[tarc[z]]
            end
            for z in path
                uf[z] = v
                uΔ[z] = (a[z] - a[v]) * λ - (b[z] - b[v])
                _child_remove!(S, csrc[tarc[z]], z)
            end
            empty!(touched)
            for z in path
                δ = uΔ[z]
                s = S.lhead[z]
                while s != 0
                    f, _ = _slot_arc(S, s)
                    if s > ne
                        cdst[f] = v
                        cw[f] += δ
                    else
                        csrc[f] = v
                        cw[f] -= δ
                    end
                    push!(touched, f)
                    s = S.lnext[s]
                end
                _arcs_concat!(S, v, z)
                # The children of `z` move under `v`, which is shallower, and
                # their subtrees rise by `a[z] - a[v]`.
                Δa = a[v] - a[z]
                _shift_subtree!(S, _subtree!(empty!(sub), S, z, stack), Δa, Δa * λ, stamp, st += 1)
                _child_concat!(S, v, z)
            end
            for f in touched
                S.alive[f] || continue
                if csrc[f] == cdst[f]
                    S.alive[f] = false
                    _arcs_unlink!(S, v, f)
                    _arcs_unlink!(S, v, f + ne)
                    _heap_remove!(heap, f)
                else
                    _key!(S, f)
                end
            end
            nlive -= length(path)
        end
    end
    # Sum `uΔ` along each vertex's chain of contractions, compressing the
    # chains as they are summed.
    for z in 1:nv
        empty!(stack)
        y = z
        while uf[y] != y
            push!(stack, y)
            y = uf[y]
        end
        r = y
        acc = zero(T)
        for i in length(stack):-1:1
            y = stack[i]
            acc += uΔ[y]
            uΔ[y] = acc
            uf[y] = r
        end
        x[z] = acc
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
