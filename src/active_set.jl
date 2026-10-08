# Active-set solution of the difference-constraint problem behind the
# `AbsLog{2}` minimal covers,
#
#     minimize ½ ∑ₑ dₑ(u)²   subject to   dₑ(u) ≥ 0,   dₑ(u) = u[p] - u[q] - c[e],
#
# over edges `e = (p, q)`. Constraint `e` has gradient `gₑ = e_p - e_q`, so a set
# of constraints is linearly independent exactly when its edges form a forest.
# The dual method of Goldfarb and Idnani (Math. Program. 27:1–33, 1983) keeps a
# forest `W` of active constraints whose multipliers are nonnegative and adds the
# most violated constraint until none is violated; the final point satisfies the
# KKT conditions to rounding error, which certifies it as the minimizer. Holding
# the edges of a tree tight fixes the differences of `u` along it, so each
# linear solve is a graph Laplacian with one unknown per tree.

# Trees of the forest `W`, each rooted at its smallest node.
struct _Forest{T}
    tree::Vector{Int}     # tree of each node
    order::Vector{Int}    # nodes, each listed after its parent
    parent::Vector{Int}   # parent node; 0 for a root
    pedge::Vector{Int}    # edge to the parent; 0 for a root
    off::Vector{T}        # `u[v] - u[root]` when every edge of `W` is tight
    ntree::Int
end

function _Forest(nV::Int, edges::Vector{Tuple{Int,Int}}, cvals::Vector{T}, inW::BitVector) where {T}
    W = findall(inW)
    ptr = zeros(Int, nV + 1)
    for e in W
        p, q = edges[e]
        ptr[p+1] += 1
        ptr[q+1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    adj = zeros(Int, ptr[end] - 1)
    cursor = ptr[1:nV]
    for e in W
        p, q = edges[e]
        adj[cursor[p]] = e
        cursor[p] += 1
        adj[cursor[q]] = e
        cursor[q] += 1
    end
    tree = zeros(Int, nV)
    parent = zeros(Int, nV)
    pedge = zeros(Int, nV)
    off = zeros(T, nV)
    order = Int[]
    sizehint!(order, nV)
    ntree = 0
    for r in 1:nV
        tree[r] == 0 || continue
        ntree += 1
        tree[r] = ntree
        head = length(order) + 1
        push!(order, r)
        while head <= length(order)
            v = order[head]
            head += 1
            for s in ptr[v]:ptr[v+1]-1
                e = adj[s]
                e == pedge[v] && continue
                p, q = edges[e]
                w = p == v ? q : p
                tree[w] == 0 || error("internal error: the active constraints contain a cycle")
                tree[w] = ntree
                parent[w] = v
                pedge[w] = e
                # Tight: `u[p] - u[q] == c[e]`.
                off[w] = p == v ? off[v] - cvals[e] : off[v] + cvals[e]
                push!(order, w)
            end
        end
    end
    return _Forest{T}(tree, order, parent, pedge, off, ntree)
end

# Flows `f` on the edges of `F` with `∑_w f[w] (e_p - e_q) = s`, written into `f`
# (entries off the forest are zeroed). `s` must sum to zero on every tree; the
# largest per-tree imbalance is returned.
function _route!(f::Vector{T}, s::Vector{T}, F::_Forest{T}, edges) where {T}
    fill!(f, zero(T))
    S = copy(s)
    imbalance = zero(T)
    for k in length(F.order):-1:1
        v = F.order[k]
        w = F.pedge[v]
        if w == 0
            imbalance = max(imbalance, abs(S[v]))
            continue
        end
        f[w] = edges[w][1] == v ? S[v] : -S[v]
        S[F.parent[v]] += S[v]
    end
    return imbalance
end

# Factorization of the Laplacian of the edges joining different trees of `F`,
# with one tree pinned to zero in each connected component of that graph.
struct _TreeLaplacian{T,Fac}
    idx::Vector{Int}   # unknown of each tree; 0 when pinned
    fac::Fac
    nfree::Int
end

function _TreeLaplacian(F::_Forest{T}, edges) where {T}
    nt = F.ntree
    # Each edge contributes to the Laplacian of the tree-contracted graph. When
    # that graph has few vertices, accumulating it densely in one pass over the
    # edges and factoring it densely is cheaper than sorting a triplet list of
    # three entries per edge and analyzing the sparsity pattern.
    if nt^2 <= length(edges)
        idx, L = _tree_laplacian_dense(F, edges)
    else
        idx, L = _tree_laplacian_triplets(F, edges)
    end
    fac = _laplacian_factor(L)
    return _TreeLaplacian{T,typeof(fac)}(idx, fac, size(L, 1))
end

# Upper triangle of the Laplacian of the free trees, as a dense matrix.
function _tree_laplacian_dense(F::_Forest{T}, edges) where {T}
    nt = F.ntree
    Lfull = zeros(T, nt, nt)
    deg = zeros(Int, nt)
    for (p, q) in edges
        tp, tq = F.tree[p], F.tree[q]
        tp == tq && continue
        deg[tp] += 1
        deg[tq] += 1
        Lfull[min(tp, tq), max(tp, tq)] -= one(T)
    end
    # Breadth-first search for the connected components; the smallest tree of
    # each is pinned and the others are the unknowns, numbered in tree order.
    seen = falses(nt)
    pinned = falses(nt)
    queue = Int[]
    for r in 1:nt
        seen[r] && continue
        seen[r] = pinned[r] = true
        push!(queue, r)
        while !isempty(queue)
            t = pop!(queue)
            for w in 1:nt
                (seen[w] || iszero(Lfull[min(w, t), max(w, t)])) && continue
                seen[w] = true
                push!(queue, w)
            end
        end
    end
    for t in 1:nt
        Lfull[t, t] = deg[t]
    end
    free = findall(.!pinned)
    idx = zeros(Int, nt)
    for (i, t) in enumerate(free)
        idx[t] = i
    end
    return idx, Lfull[free, free]
end

function _tree_laplacian_triplets(F::_Forest{T}, edges) where {T}
    nt = F.ntree
    # Union-find over trees to pick one pinned tree per connected component.
    uf = collect(1:nt)
    find(x) = (while uf[x] != x; uf[x] = uf[uf[x]]; x = uf[x]; end; x)
    for (p, q) in edges
        tp, tq = F.tree[p], F.tree[q]
        tp == tq && continue
        rp, rq = find(tp), find(tq)
        rp == rq || (uf[max(rp, rq)] = min(rp, rq))
    end
    idx = zeros(Int, nt)
    nfree = 0
    for t in 1:nt
        if find(t) != t
            nfree += 1
            idx[t] = nfree
        end
    end
    I = Int[]
    J = Int[]
    V = T[]
    for (p, q) in edges
        ip, iq = idx[F.tree[p]], idx[F.tree[q]]
        F.tree[p] == F.tree[q] && continue
        ip > 0 && (push!(I, ip); push!(J, ip); push!(V, one(T)))
        iq > 0 && (push!(I, iq); push!(J, iq); push!(V, one(T)))
        if ip > 0 && iq > 0
            push!(I, min(ip, iq)); push!(J, max(ip, iq)); push!(V, -one(T))
        end
    end
    return idx, sparse(I, J, V, nfree, nfree)
end

function _laplacian_factor(L::SparseMatrixCSC{Float64,Int})
    F = SparseCholesky()
    size(L, 1) == 0 && return F
    analyze!(F, L)
    factorize!(F, L)
    return F
end
# Wider types: dense Cholesky of the upper triangle.
_laplacian_factor(L::SparseMatrixCSC) = _laplacian_factor(Matrix(L))
_laplacian_factor(L::Matrix) = LinearAlgebra.cholesky!(Symmetric(L, :U))

_laplacian_solve!(x::Vector{Float64}, F::SparseCholesky, b::Vector{Float64}) =
    isempty(b) ? x : solve!(x, F, CHOLMOD_A, b)
_laplacian_solve!(x, F::Cholesky, b) = ldiv!(x, F, b)

# Per-tree values `y` minimizing `½‖L y - rhs‖` with pinned trees at zero; `rhs`
# is indexed by tree.
function _tree_solve(K::_TreeLaplacian, rhs::Vector{T}) where {T}
    b = zeros(T, K.nfree)
    for (t, i) in enumerate(K.idx)
        i > 0 && (b[i] = rhs[t])
    end
    x = similar(b)
    _laplacian_solve!(x, K.fac, b)
    y = zeros(T, length(K.idx))
    for (t, i) in enumerate(K.idx)
        i > 0 && (y[t] = x[i])
    end
    return y
end

# Minimizer of the objective with every edge of `F` tight.
function _tight_minimizer(F::_Forest{T}, K::_TreeLaplacian, edges, cvals::Vector{T}) where {T}
    rhs = zeros(T, F.ntree)
    for (e, (p, q)) in enumerate(edges)
        tp, tq = F.tree[p], F.tree[q]
        tp == tq && continue
        k = F.off[p] - F.off[q] - cvals[e]
        rhs[tp] -= k
        rhs[tq] += k
    end
    y = _tree_solve(K, rhs)
    return [y[F.tree[v]] + F.off[v] for v in eachindex(F.tree)]
end

# Objective gradient `∑ₑ dₑ gₑ` at `u`.
function _difference_gradient(u::Vector{T}, edges, cvals::Vector{T}) where {T}
    g = zeros(T, length(u))
    for (e, (p, q)) in enumerate(edges)
        d = u[p] - u[q] - cvals[e]
        g[p] += d
        g[q] -= d
    end
    return g
end

"""
    u, certified, nsteps = _polish_difference_qp(edges, cvals, u0; τ, maxsteps)

Minimize `½ ∑ₑ (u[p] - u[q] - cvals[e])²` subject to every term being
nonnegative, starting from the near-optimal `u0`. Edges with
`u0[p] - u0[q] - cvals[e] < τ` seed the active set, adding the tightest first and
skipping any that would close a cycle. `certified` reports that the returned `u`
satisfies the KKT conditions to tolerances proportional to the largest magnitude
in `cvals` and `u0`; when it is false, `u` is `u0`. Each step factors a graph
Laplacian, and `maxsteps` bounds their number. The constant on each connected component of the graph is
taken from `u0`.
"""
function _polish_difference_qp(edges::Vector{Tuple{Int,Int}}, cvals::Vector{T}, u0::Vector{T};
                               τ::Real=T(1e-4), maxsteps::Int=max(1000, 2 * length(u0))) where {T}
    nV = length(u0)
    E = length(edges)
    scale = max(oneunit(T), maximum(abs, cvals; init=zero(T)), maximum(abs, u0; init=zero(T)))
    ptol = 1000 * eps(T) * scale
    dtol = ptol * max(1, E)
    resid(u, e) = u[edges[e][1]] - u[edges[e][2]] - cvals[e]
    # Seed: Kruskal's algorithm on the nearly tight edges, tightest first.
    cand = [e for e in 1:E if resid(u0, e) < τ]
    sort!(cand; by=e -> resid(u0, e))
    uf = collect(1:nV)
    find(x) = (while uf[x] != x; uf[x] = uf[uf[x]]; x = uf[x]; end; x)
    inW = falses(E)
    for e in cand
        rp, rq = find(edges[e][1]), find(edges[e][2])
        rp == rq && continue
        uf[rp] = rq
        inW[e] = true
    end
    λ = zeros(T, E)
    nsteps = 0
    u = u0
    u, nsteps = _dual_feasible!(inW, λ, edges, cvals, nV, dtol, nsteps, maxsteps)
    r = zeros(T, E)
    certified = false
    while nsteps <= maxsteps
        # Most violated constraint outside `W`.
        e = 0
        de = -ptol
        for k in 1:E
            inW[k] && continue
            dk = resid(u, k)
            if dk < de
                e, de = k, dk
            end
        end
        if e == 0
            certified = all(k -> !inW[k] || λ[k] >= -dtol, 1:E)
            break
        end
        p, q = edges[e]
        λe = zero(T)
        added = false
        while !added && nsteps <= maxsteps
            nsteps += 1
            F = _Forest(nV, edges, cvals, inW)
            if F.tree[p] == F.tree[q]
                # `gₑ` is a combination of the tree path from `p` to `q`: shift
                # multiplier weight onto `e` until one on the path reaches zero.
                s = zeros(T, nV)
                s[p] -= one(T)
                s[q] += one(T)
                _route!(r, s, F, edges)
                t, wb = _ratio_test(λ, r, inW)
                wb == 0 && return u0, false, nsteps   # the constraints are infeasible
                @. λ += t * r
                λe += t
                λ[wb] = zero(T)
                inW[wb] = false
                continue
            end
            K = _TreeLaplacian(F, edges)
            rhs = zeros(T, F.ntree)
            rhs[F.tree[p]] += one(T)
            rhs[F.tree[q]] -= one(T)
            y = _tree_solve(K, rhs)
            z = [y[F.tree[v]] for v in 1:nV]
            a = z[p] - z[q]
            a > 0 || return u0, false, nsteps
            # Rates of the multipliers in `W`: route `L z - gₑ` onto the forest.
            s = zeros(T, nV)
            for (k, (pk, qk)) in enumerate(edges)
                δ = z[pk] - z[qk]
                s[pk] += δ
                s[qk] -= δ
            end
            s[p] -= one(T)
            s[q] += one(T)
            _route!(r, s, F, edges)
            tf = -resid(u, e) / a
            tp, wb = _ratio_test(λ, r, inW)
            t = min(tf, tp)
            u = u .+ t .* z
            @. λ += t * r
            λe += t
            if tf <= tp
                inW[e] = true
                λ[e] = λe
                added = true
            else
                λ[wb] = zero(T)
                inW[wb] = false
            end
        end
        added || break
        # Recompute from `W` alone, so rounding does not accumulate.
        u, nsteps = _dual_feasible!(inW, λ, edges, cvals, nV, dtol, nsteps, maxsteps)
    end
    certified || return u0, false, nsteps
    # Restore each component's constant from `u0`. (`u` is reassigned above, so
    # capturing it in a closure would box it; `ustar` is bound once.)
    ustar = u
    uf .= 1:nV
    for (p, q) in edges
        rp, rq = find(p), find(q)
        rp == rq || (uf[max(rp, rq)] = min(rp, rq))
    end
    shift = zeros(T, nV)
    for v in 1:nV
        rv = find(v)
        rv == v && (shift[v] = u0[v] - ustar[v])
    end
    return [ustar[v] + shift[find(v)] for v in 1:nV], true, nsteps
end

# Drop the edges of `W` with negative multipliers until none remain, and return
# the tight minimizer of `W` with the step count; `λ` holds the multipliers.
function _dual_feasible!(inW::BitVector, λ::Vector{T}, edges, cvals::Vector{T}, nV::Int, dtol,
                         nsteps::Int, maxsteps::Int) where {T}
    while true
        F = _Forest(nV, edges, cvals, inW)
        K = _TreeLaplacian(F, edges)
        u = _tight_minimizer(F, K, edges, cvals)
        _route!(λ, _difference_gradient(u, edges, cvals), F, edges)
        dropped = false
        for e in eachindex(λ)
            if inW[e] && λ[e] < -dtol
                inW[e] = false
                dropped = true
            end
        end
        (dropped && nsteps < maxsteps) || return u, nsteps
        nsteps += 1
    end
end

# Largest step `t` keeping `λ + t r ≥ 0` on `W`, and the edge that blocks it
# (0 when none does).
function _ratio_test(λ::Vector{T}, r::Vector{T}, inW::BitVector) where {T}
    t = T(Inf)
    wb = 0
    for w in eachindex(λ)
        inW[w] || continue
        if r[w] < 0
            tw = max(λ[w], zero(T)) / -r[w]
            if tw < t
                t, wb = tw, w
            end
        end
    end
    return t, wb
end

# Solve the `AbsLog{2}` problem of `_cover_min_abslog2` exactly from its
# augmented-Lagrangian iterate `x`, returning `(x, certified, nsteps)`. In the
# stacked layout `(α; β)` an entry's residual is `α[i] + β[j] - c`, which is a
# difference in `(α; -β)`; the transversal layout is a difference already.
function _polish_cover(x::Vector{T}, supp::EdgeList{T}, m::Int) where {T}
    supp.qsign < 0 && return _polish_difference_qp(supp.edges, supp.cvals, x)
    flip(u) = [k <= m ? u[k] : -u[k] for k in eachindex(u)]
    u, certified, nsteps = _polish_difference_qp(supp.edges, supp.cvals, flip(x))
    return flip(u), certified, nsteps
end

function _polish_cover(x::Vector{T}, supp::Grid{T}, m::Int) where {T}
    C = supp.C
    edges = Tuple{Int,Int}[]
    cvals = T[]
    for j in axes(C, 2), i in axes(C, 1)
        isfinite(C[i, j]) || continue
        push!(edges, (i, m + j))
        push!(cvals, C[i, j])
    end
    return _polish_cover(x, EdgeList{T}(edges, cvals), m)
end

function _polish_cover(x::Vector{T}, supp::DiffGrid{T}, m::Int) where {T}
    C = supp.C
    edges = Tuple{Int,Int}[]
    cvals = T[]
    for j in axes(C, 2), i in axes(C, 1)
        isfinite(C[i, j]) || continue
        push!(edges, (supp.rowidx[i], supp.colidx[j]))
        push!(cvals, C[i, j])
    end
    return _polish_difference_qp(edges, cvals, x)
end
