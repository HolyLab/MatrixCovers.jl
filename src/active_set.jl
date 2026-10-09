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

# Constraint supports. Constraint `e` is a position in an `EdgeList`, or a
# linear index into the matrix `C` of a difference grid; grid entries outside
# the support are never seeded, never violated, and never active.

# Difference grid with the per-row and per-column sums used to evaluate sums
# over all of its entries in O(m + n + #entries outside the support).
struct _DiffGridQP{T}
    C::Matrix{T}
    rowidx::Vector{Int}
    colidx::Vector{Int}
    outside::Vector{Tuple{Int,Int}}   # `(i, j)` of each entry outside the support
    nfr::Vector{Int}                  # number of supported entries in each row
    sr::Vector{T}                     # sum of the supported `C[i, j]` in each row
    nfc::Vector{Int}                  # number of supported entries in each column
    sc::Vector{T}                     # sum of the supported `C[i, j]` in each column
    cmax::T                           # largest supported `|C[i, j]|`
end

function _qp_support(supp::EdgeList{T}) where {T}
    supp.qsign == -oneunit(T) ||
        throw(ArgumentError("the active-set polish requires difference constraints (an `EdgeList` with `qsign = -1`), got `qsign = $(string(supp.qsign))`"))
    return supp
end

function _qp_support(supp::DiffGrid{T}) where {T}
    C = supp.C
    m, n = size(C)
    outside = Tuple{Int,Int}[]
    nfr, sr = zeros(Int, m), zeros(T, m)
    nfc, sc = zeros(Int, n), zeros(T, n)
    cmax = zero(T)
    for j in axes(C, 2), i in axes(C, 1)
        c = C[i, j]
        if isfinite(c)
            nfr[i] += 1
            sr[i] += c
            nfc[j] += 1
            sc[j] += c
            cmax = max(cmax, abs(c))
        else
            push!(outside, (i, j))
        end
    end
    return _DiffGridQP{T}(C, supp.rowidx, supp.colidx, outside, nfr, sr, nfc, sc, cmax)
end

_nconstraints(s::EdgeList) = length(s.edges)
_nconstraints(s::_DiffGridQP) = length(s.C)
_nsupported(s::EdgeList) = length(s.edges)
_nsupported(s::_DiffGridQP) = length(s.C) - length(s.outside)
_cmax(s::EdgeList{T}) where {T} = maximum(abs, s.cvals; init=zero(T))
_cmax(s::_DiffGridQP) = s.cmax

_edge(s::EdgeList, e::Int) = s.edges[e]
function _edge(s::_DiffGridQP, e::Int)
    ij = CartesianIndices(s.C)[e]
    return (s.rowidx[ij[1]], s.colidx[ij[2]])
end
_cval(s::EdgeList, e::Int) = s.cvals[e]
_cval(s::_DiffGridQP, e::Int) = s.C[e]

function _resid(u, s, e::Int)
    pq = _edge(s, e)
    return u[pq[1]] - u[pq[2]] - _cval(s, e)
end

# Flags marking the active constraints. The grid's flags are read inside a
# vectorized sweep, which a `BitArray` would prevent.
_active_flags(s::EdgeList) = falses(length(s.edges))
_active_flags(s::_DiffGridQP) = fill(false, size(s.C))

# Call `f(p, q)` for every supported constraint.
function _foreach_constraint(f, s::EdgeList)
    for pq in s.edges
        f(pq[1], pq[2])
    end
    return nothing
end
function _foreach_constraint(f, s::_DiffGridQP)
    C = s.C
    for j in axes(C, 2)
        q = s.colidx[j]
        for i in axes(C, 1)
            isfinite(C[i, j]) && f(s.rowidx[i], q)
        end
    end
    return nothing
end

# Constraints with `u[p] - u[q] - c < τ`.
function _seed_candidates(u, s::EdgeList, τ)
    return [e for e in 1:_nconstraints(s) if _resid(u, s, e) < τ]
end
function _seed_candidates(u, s::_DiffGridQP, τ)
    C = s.C
    L = LinearIndices(C)
    ur = u[s.rowidx]
    cand = Int[]
    for j in axes(C, 2)
        uj = u[s.colidx[j]]
        for i in eachindex(ur, view(C, :, j))
            # Entries outside the support have residual `+Inf`.
            ur[i] - uj - C[i, j] < τ && push!(cand, L[i, j])
        end
    end
    return cand
end

# Trees of the forest `W`, each rooted at its smallest node.
struct _Forest{T}
    tree::Vector{Int}     # tree of each node
    order::Vector{Int}    # nodes, each listed after its parent
    parent::Vector{Int}   # parent node; 0 for a root
    pedge::Vector{Int}    # edge to the parent; 0 for a root
    off::Vector{T}        # `u[v] - u[root]` when every edge of `W` is tight
    ntree::Int
end

# `W` lists the active constraints in increasing order.
function _Forest(nV::Int, supp::Union{EdgeList{T},_DiffGridQP{T}}, W::Vector{Int}) where {T}
    ptr = zeros(Int, nV + 1)
    for e in W
        p, q = _edge(supp, e)
        ptr[p+1] += 1
        ptr[q+1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    adj = zeros(Int, ptr[end] - 1)
    cursor = ptr[1:nV]
    for e in W
        p, q = _edge(supp, e)
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
                p, q = _edge(supp, e)
                w = p == v ? q : p
                tree[w] == 0 || error("internal error: the active constraints contain a cycle")
                tree[w] = ntree
                parent[w] = v
                pedge[w] = e
                # Tight: `u[p] - u[q] == c[e]`.
                c = _cval(supp, e)
                off[w] = p == v ? off[v] - c : off[v] + c
                push!(order, w)
            end
        end
    end
    return _Forest{T}(tree, order, parent, pedge, off, ntree)
end

# Flows `f` on the edges of `F` with `∑_w f[w] (e_p - e_q) = s`, written into the
# entries of `f` on the forest (other entries are left unchanged). `s` must sum
# to zero on every tree; the largest per-tree imbalance is returned.
function _route!(f::Vector{T}, s::Vector{T}, F::_Forest{T}, supp) where {T}
    S = copy(s)
    imbalance = zero(T)
    for k in length(F.order):-1:1
        v = F.order[k]
        w = F.pedge[v]
        if w == 0
            imbalance = max(imbalance, abs(S[v]))
            continue
        end
        f[w] = _edge(supp, w)[1] == v ? S[v] : -S[v]
        S[F.parent[v]] += S[v]
    end
    return imbalance
end

# Factorization (or, over budget, a CG solver; see `_LaplacianCG`) of the
# Laplacian of the edges joining different trees of `F`, with one tree pinned to
# zero in each connected component of that graph.
struct _TreeLaplacian{T,Fac}
    idx::Vector{Int}   # unknown of each tree; 0 when pinned
    fac::Fac
    nfree::Int
end

# `budgets` holds the `fillbudget` and `flopbudget` keywords of `_laplacian_factor`.
function _TreeLaplacian(F::_Forest{T}, supp; budgets::NamedTuple=(;)) where {T}
    nt = F.ntree
    # When the tree-contracted graph has few vertices, accumulating its
    # Laplacian densely and factoring it densely is cheaper than sorting a
    # triplet list of three entries per edge and analyzing the sparsity pattern.
    if nt^2 <= _nsupported(supp)
        idx, L = _tree_laplacian_dense(F, supp)
    else
        idx, L = _tree_laplacian_triplets(F, supp)
    end
    fac = L isa SparseMatrixCSC{Float64,Int} ? _laplacian_factor(L; budgets...) : _laplacian_factor(L)
    return _TreeLaplacian{T,typeof(fac)}(idx, fac, size(L, 1))
end

# Upper triangle of the Laplacian of the free trees, as a dense matrix.
function _tree_laplacian_dense(F::_Forest{T}, supp::EdgeList) where {T}
    nt = F.ntree
    Lfull = zeros(T, nt, nt)
    deg = zeros(Int, nt)
    for (p, q) in supp.edges
        tp, tq = F.tree[p], F.tree[q]
        tp == tq && continue
        deg[tp] += 1
        deg[tq] += 1
        Lfull[min(tp, tq), max(tp, tq)] -= one(T)
    end
    return _pin_tree_laplacian!(Lfull, deg)
end

# With `hr[t]` rows and `hc[t]` columns mapped into tree `t`, the grid has
# `hr[t]·hc[t'] + hr[t']·hc[t]` entries joining trees `t ≠ t'`, less those
# outside the support.
function _tree_laplacian_dense(F::_Forest{T}, s::_DiffGridQP) where {T}
    nt = F.ntree
    trow = F.tree[s.rowidx]
    tcol = F.tree[s.colidx]
    hr = zeros(Int, nt)
    hc = zeros(Int, nt)
    for t in trow
        hr[t] += 1
    end
    for t in tcol
        hc[t] += 1
    end
    nr, nc = length(trow), length(tcol)
    deg = [hr[t] * (nc - hc[t]) + hc[t] * (nr - hr[t]) for t in 1:nt]
    Lfull = zeros(T, nt, nt)
    for t2 in 1:nt, t1 in 1:t2-1
        Lfull[t1, t2] = -T(hr[t1] * hc[t2] + hr[t2] * hc[t1])
    end
    for (i, j) in s.outside
        tp, tq = trow[i], tcol[j]
        tp == tq && continue
        deg[tp] -= 1
        deg[tq] -= 1
        Lfull[min(tp, tq), max(tp, tq)] += one(T)
    end
    return _pin_tree_laplacian!(Lfull, deg)
end

# Given the strict upper triangle of the tree Laplacian in `Lfull` and the tree
# degrees, pin the smallest tree of each connected component and return the
# unknown of each tree with the upper triangle of the free block.
function _pin_tree_laplacian!(Lfull::Matrix{T}, deg::Vector{Int}) where {T}
    nt = length(deg)
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

function _tree_laplacian_triplets(F::_Forest{T}, supp) where {T}
    nt = F.ntree
    tree = F.tree
    # Union-find over trees to pick one pinned tree per connected component.
    uf = collect(1:nt)
    find(x) = (while uf[x] != x; uf[x] = uf[uf[x]]; x = uf[x]; end; x)
    _foreach_constraint(supp) do p, q
        tp, tq = tree[p], tree[q]
        if tp != tq
            rp, rq = find(tp), find(tq)
            rp == rq || (uf[max(rp, rq)] = min(rp, rq))
        end
        return nothing
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
    _foreach_constraint(supp) do p, q
        tp, tq = tree[p], tree[q]
        tp == tq && return nothing
        ip, iq = idx[tp], idx[tq]
        ip > 0 && (push!(I, ip); push!(J, ip); push!(V, one(T)))
        iq > 0 && (push!(I, iq); push!(J, iq); push!(V, one(T)))
        if ip > 0 && iq > 0
            push!(I, min(ip, iq)); push!(J, max(ip, iq)); push!(V, -one(T))
        end
        return nothing
    end
    return idx, sparse(I, J, V, nfree, nfree)
end

# Jacobi-preconditioned CG on a pinned tree Laplacian whose sparse Cholesky
# factor would exceed the fill or flop budget. Such Laplacians arise from
# expander-like supports, which fill in under elimination but are well
# conditioned, so CG converges in few iterations. A solve that CG does not
# finish falls back to the factorization.
struct _LaplacianCG
    L::SparseMatrixCSC{Float64,Int}   # upper triangle, as analyzed
    S::SparseMatrixCSC{Float64,Int}   # both triangles, for products
    dg::Vector{Float64}
    F::SparseCholesky
    factored::Base.RefValue{Bool}
    maxiter::Int                      # CG iterations per pass
end

function _laplacian_factor(L::SparseMatrixCSC{Float64,Int}; fillbudget::Real=LSQR_FILL_BUDGET,
                           flopbudget::Real=LSQR_FLOP_BUDGET, cgmaxiter::Int=10 * size(L, 1) + 100)
    F = SparseCholesky()
    size(L, 1) == 0 && return F
    analyze!(F, L)
    if sizeof(Float64) * factor_entries(F) > fillbudget || factor_flops(F) > flopbudget * nnz(L)
        S = L + triu(L, 1)'
        return _LaplacianCG(L, S, Vector(diag(S)), F, Ref(false), cgmaxiter)
    end
    factorize!(F, L)
    return F
end
# Wider types: dense Cholesky of the upper triangle.
_laplacian_factor(L::SparseMatrixCSC) = _laplacian_factor(Matrix(L))
_laplacian_factor(L::Matrix) = LinearAlgebra.cholesky!(Symmetric(L, :U))

_laplacian_solve!(x::Vector{Float64}, F::SparseCholesky, b::Vector{Float64}) =
    isempty(b) ? x : solve!(x, F, CHOLMOD_A, b)
_laplacian_solve!(x, F::Cholesky, b) = ldiv!(x, F, b)

function _laplacian_solve!(x::Vector{Float64}, C::_LaplacianCG, b::Vector{Float64})
    if !C.factored[]
        n = length(b)
        fill!(x, 0.0)
        work = ntuple(_ -> zeros(n), 4)
        # The multipliers are routed sums of the solution along the trees, so
        # certification needs a normwise backward error of a few units of
        # roundoff, as from a factorization. `2 max(dg)` bounds `‖S‖₂` for a
        # Laplacian; the first pass estimates `‖x‖`.
        Smul! = (y, v) -> mul!(y, C.S, v)
        _pcg!(Smul!, x, C.dg, b, work..., C.maxiter, sqrt(eps(Float64)) * norm(b))
        tol = 8 * eps(Float64) * (2 * maximum(C.dg) * norm(x) + norm(b))
        _, ok = _pcg!(Smul!, x, C.dg, b, work..., C.maxiter, tol)
        ok && return x
        factorize!(C.F, C.L)
        C.factored[] = true
    end
    return solve!(x, C.F, CHOLMOD_A, b)
end

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
function _tight_minimizer(F::_Forest{T}, K::_TreeLaplacian, supp::EdgeList) where {T}
    rhs = zeros(T, F.ntree)
    for (e, (p, q)) in enumerate(supp.edges)
        tp, tq = F.tree[p], F.tree[q]
        tp == tq && continue
        k = F.off[p] - F.off[q] - supp.cvals[e]
        rhs[tp] -= k
        rhs[tq] += k
    end
    y = _tree_solve(K, rhs)
    return [y[F.tree[v]] + F.off[v] for v in eachindex(F.tree)]
end

function _tight_minimizer(F::_Forest{T}, K::_TreeLaplacian, s::_DiffGridQP) where {T}
    # Edges within a tree contribute `-k` and `+k` to the same entry, so the
    # sum may run over every constraint.
    g = _difference_sums!(zeros(T, length(F.off)), F.off, s, true)
    rhs = zeros(T, F.ntree)
    for v in eachindex(F.tree, g)
        rhs[F.tree[v]] -= g[v]
    end
    y = _tree_solve(K, rhs)
    return [y[F.tree[v]] + F.off[v] for v in eachindex(F.tree)]
end

# Add `∑ₑ dₑ gₑ` to `g`, where `dₑ = v[p] - v[q] - c[e]` when `withc` and
# `dₑ = v[p] - v[q]` otherwise.
function _difference_sums!(g::Vector{T}, v::Vector{T}, s::EdgeList, withc::Bool) where {T}
    cvals = s.cvals
    for (e, (p, q)) in enumerate(s.edges)
        d = withc ? v[p] - v[q] - cvals[e] : v[p] - v[q]
        g[p] += d
        g[q] -= d
    end
    return g
end

# Row `i` contributes `∑ⱼ (v[rowidx[i]] - v[colidx[j]] - C[i, j])` over its
# supported entries, which is `nfr[i]·v[rowidx[i]]` less the sum of `v` over all
# columns, plus that sum over the row's unsupported entries, less `sr[i]`;
# columns are analogous.
function _difference_sums!(g::Vector{T}, v::Vector{T}, s::_DiffGridQP, withc::Bool) where {T}
    rowidx, colidx = s.rowidx, s.colidx
    # A uniform shift of `v` changes no difference; centering limits cancellation.
    vref = isempty(v) ? zero(T) : sum(v) / length(v)
    vr = [v[p] - vref for p in rowidx]
    vc = [v[q] - vref for q in colidx]
    Svr, Svc = sum(vr), sum(vc)
    rcorr = zeros(T, length(vr))
    ccorr = zeros(T, length(vc))
    for (i, j) in s.outside
        rcorr[i] += vc[j]
        ccorr[j] += vr[i]
    end
    for i in eachindex(rowidx, vr)
        d = s.nfr[i] * vr[i] - (Svc - rcorr[i])
        withc && (d -= s.sr[i])
        g[rowidx[i]] += d
    end
    for j in eachindex(colidx, vc)
        d = (Svr - ccorr[j]) - s.nfc[j] * vc[j]
        withc && (d -= s.sc[j])
        g[colidx[j]] -= d
    end
    return g
end

# Objective gradient `∑ₑ dₑ gₑ` at `u`.
_difference_gradient(u::Vector{T}, supp) where {T} = _difference_sums!(zeros(T, length(u)), u, supp, true)

# Most violated constraint outside `W` with residual below `de`, as `(e, de)`;
# `e == 0` when there is none.
function _most_violated(u::Vector{T}, inW, s::EdgeList, de::T) where {T}
    e = 0
    for k in 1:_nconstraints(s)
        inW[k] && continue
        dk = _resid(u, s, k)
        if dk < de
            e, de = k, dk
        end
    end
    return e, de
end

function _most_violated(u::Vector{T}, inW::Matrix{Bool}, s::_DiffGridQP, de::T) where {T}
    C = s.C
    ur = u[s.rowidx]
    jbest = 0
    # Column minima of the residuals, with active and unsupported entries at `+Inf`.
    for j in axes(C, 2)
        uj = u[s.colidx[j]]
        cj = view(C, :, j)
        wj = view(inW, :, j)
        mj = T(Inf)
        @simd for i in _eachindex(cj, wj, ur)
            d = ifelse(wj[i], T(Inf), ur[i] - uj - cj[i])
            mj = ifelse(d < mj, d, mj)
        end
        if mj < de
            jbest, de = j, mj
        end
    end
    jbest == 0 && return 0, de
    uj = u[s.colidx[jbest]]
    for i in axes(C, 1)
        if !inW[i, jbest] && ur[i] - uj - C[i, jbest] == de
            return LinearIndices(C)[i, jbest], de
        end
    end
    error("internal error: the most violated constraint was not found")
end

"""
    u, certified, nsteps, kkt = _polish_difference_qp(supp, u0; τ, maxsteps, seed, budgets)
    u, certified, nsteps, kkt = _polish_difference_qp(edges, cvals, u0; τ, maxsteps, seed, budgets)

Minimize `½ ∑ₑ (u[p] - u[q] - c[e])²` subject to every term being nonnegative,
starting from the near-optimal `u0`. The constraints are those of `supp`, an
`EdgeList` with `qsign = -1` or a `DiffGrid`, or the edges `(p, q)` with costs
`cvals`. The active set is seeded by adding candidate constraints in order and
skipping any that would close a cycle; `ptol` is `1000 eps` times the largest
magnitude in `c` and `u0`. Without `seed`, the candidates are the constraints
with `u0[p] - u0[q] - c[e] < max(τ, ptol)`, tightest first. With `seed`, a
vector of nonnegative weights indexed like the constraints (edge positions of
an `EdgeList`, linear indices of a `DiffGrid`'s `C`), such as the multipliers of
an augmented-Lagrangian iteration, the candidates are the constraints with
positive weight or residual below `ptol`, largest weight first and then
tightest first. `certified` reports that the returned `u` satisfies the KKT
conditions to tolerances proportional to the largest magnitude in `c` and `u0`;
when it is false, `u` is `u0`. `kkt = (; primal, ptol, dual, dtol)` gives the
largest constraint violation of the returned `u` and the most negative active
multiplier (as nonnegative magnitudes), with the tolerances they were certified
against; `primal` and `dual` are `NaN` when `certified` is false. Each step
solves a graph-Laplacian system, and `maxsteps` bounds their number. `budgets`, a `NamedTuple` with fields
`fillbudget` and/or `flopbudget`, sets the limits above which these systems are
solved by conjugate gradients rather than a sparse Cholesky factorization.
The constant on each connected component of the graph is taken from `u0`.
"""
function _polish_difference_qp(edges::Vector{Tuple{Int,Int}}, cvals::Vector{T}, u0::Vector{T}; kwargs...) where {T}
    return _polish_difference_qp(EdgeList{T}(edges, cvals, -oneunit(T)), u0; kwargs...)
end

function _polish_difference_qp(supp::Union{EdgeList{T},DiffGrid{T}}, u0::Vector{T};
                               τ::Real=T(1e-4), maxsteps::Int=max(1000, 2 * length(u0)),
                               seed::Union{Nothing,AbstractVector}=nothing,
                               budgets::NamedTuple=(;)) where {T}
    s = _qp_support(supp)
    nV = length(u0)
    scale = max(oneunit(T), _cmax(s), maximum(abs, u0; init=zero(T)))
    ptol = 1000 * eps(T) * scale
    dtol = ptol * max(1, _nsupported(s))
    # Seed: Kruskal's algorithm on the candidate constraints, in the order the
    # docstring describes.
    if seed === nothing
        cand = _seed_candidates(u0, s, max(T(τ), ptol))
        sort!(cand; by=e -> _resid(u0, s, e))
    else
        axes(seed, 1) == Base.OneTo(_nconstraints(s)) ||
            throw(DimensionMismatch("seed must have axes 1:$(_nconstraints(s)), one entry per constraint; got $(axes(seed, 1))"))
        # Entries outside the support have residual `+Inf` and are never candidates.
        cand = union([e for e in eachindex(seed) if seed[e] > 0 && isfinite(_resid(u0, s, e))],
                     _seed_candidates(u0, s, ptol))
        sort!(cand; by=e -> (-seed[e], _resid(u0, s, e)))
    end
    uf = collect(1:nV)
    find(x) = (while uf[x] != x; uf[x] = uf[uf[x]]; x = uf[x]; end; x)
    inW = _active_flags(s)
    W = Int[]   # the active constraints, in increasing order
    for e in cand
        p, q = _edge(s, e)
        rp, rq = find(p), find(q)
        rp == rq && continue
        uf[rp] = rq
        inW[e] = true
        push!(W, e)
    end
    sort!(W)
    # Multipliers and their rates; only the entries in `W` are meaningful.
    E = _nconstraints(s)
    λ = zeros(T, E)
    r = zeros(T, E)
    nsteps = 0
    u = u0
    u, nsteps = _dual_feasible!(inW, W, λ, s, nV, dtol, nsteps, maxsteps; budgets)
    certified = false
    nokkt = (; primal=T(NaN), ptol, dual=T(NaN), dtol)
    while nsteps <= maxsteps
        e, de = _most_violated(u, inW, s, -ptol)
        if e == 0
            certified = all(k -> λ[k] >= -dtol, W)
            break
        end
        p, q = _edge(s, e)
        λe = zero(T)
        added = false
        while !added && nsteps <= maxsteps
            nsteps += 1
            F = _Forest(nV, s, W)
            if F.tree[p] == F.tree[q]
                # `gₑ` is a combination of the tree path from `p` to `q`: shift
                # multiplier weight onto `e` until one on the path reaches zero.
                sv = zeros(T, nV)
                sv[p] -= one(T)
                sv[q] += one(T)
                _route!(r, sv, F, s)
                t, wb = _ratio_test(λ, r, W)
                wb == 0 && return u0, false, nsteps, nokkt   # the constraints are infeasible
                for w in W
                    λ[w] += t * r[w]
                end
                λe += t
                _deactivate!(inW, W, λ, wb)
                continue
            end
            K = _TreeLaplacian(F, s; budgets)
            rhs = zeros(T, F.ntree)
            rhs[F.tree[p]] += one(T)
            rhs[F.tree[q]] -= one(T)
            y = _tree_solve(K, rhs)
            z = [y[F.tree[v]] for v in 1:nV]
            a = z[p] - z[q]
            a > 0 || return u0, false, nsteps, nokkt
            # Rates of the multipliers in `W`: route `L z - gₑ` onto the forest.
            sv = _difference_sums!(zeros(T, nV), z, s, false)
            sv[p] -= one(T)
            sv[q] += one(T)
            _route!(r, sv, F, s)
            tf = -_resid(u, s, e) / a
            tp, wb = _ratio_test(λ, r, W)
            t = min(tf, tp)
            u = u .+ t .* z
            for w in W
                λ[w] += t * r[w]
            end
            λe += t
            if tf <= tp
                inW[e] = true
                insert!(W, searchsortedfirst(W, e), e)
                λ[e] = λe
                added = true
            else
                _deactivate!(inW, W, λ, wb)
            end
        end
        added || break
        # Recompute from `W` alone, so rounding does not accumulate.
        u, nsteps = _dual_feasible!(inW, W, λ, s, nV, dtol, nsteps, maxsteps; budgets)
    end
    certified || return u0, false, nsteps, nokkt
    # Restore each component's constant from `u0`. (`u` is reassigned above, so
    # capturing it in a closure would box it; `ustar` is bound once.)
    ustar = u
    kkt = (; primal=max(zero(T), -minimum(e -> _resid(ustar, s, e), 1:_nconstraints(s))), ptol,
           dual=max(zero(T), -minimum(k -> λ[k], W; init=zero(T))), dtol)
    uf .= 1:nV
    _foreach_constraint(s) do p, q
        rp, rq = find(p), find(q)
        rp == rq || (uf[max(rp, rq)] = min(rp, rq))
        return nothing
    end
    shift = zeros(T, nV)
    for v in 1:nV
        rv = find(v)
        rv == v && (shift[v] = u0[v] - ustar[v])
    end
    return [ustar[v] + shift[find(v)] for v in 1:nV], true, nsteps, kkt
end

function _deactivate!(inW, W::Vector{Int}, λ, w::Int)
    λ[w] = zero(eltype(λ))
    inW[w] = false
    deleteat!(W, searchsortedfirst(W, w))
    return nothing
end

# Drop the edges of `W` with negative multipliers until none remain, and return
# the tight minimizer of `W` with the step count; `λ` holds the multipliers.
function _dual_feasible!(inW, W::Vector{Int}, λ::Vector{T}, supp, nV::Int, dtol,
                         nsteps::Int, maxsteps::Int; budgets::NamedTuple=(;)) where {T}
    while true
        F = _Forest(nV, supp, W)
        K = _TreeLaplacian(F, supp; budgets)
        u = _tight_minimizer(F, K, supp)
        _route!(λ, _difference_gradient(u, supp), F, supp)
        nW = length(W)
        for e in W
            λ[e] < -dtol && (inW[e] = false)
        end
        filter!(e -> inW[e], W)
        (length(W) < nW && nsteps < maxsteps) || return u, nsteps
        nsteps += 1
    end
end

# Largest step `t` keeping `λ + t r ≥ 0` on `W`, and the edge that blocks it
# (0 when none does).
function _ratio_test(λ::Vector{T}, r::Vector{T}, W::Vector{Int}) where {T}
    t = T(Inf)
    wb = 0
    for w in W
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
# augmented-Lagrangian iterate `x`, returning `(x, certified, nsteps, kkt)`. In the
# stacked layout `(α; β)` an entry's residual is `α[i] + β[j] - c`, which is a
# difference in `(α; -β)`; the transversal layout is a difference already.
# `multipliers`, the iteration's multipliers in the layout of `supp`, seed the
# active set (see `_polish_difference_qp`, which also describes `budgets`).
function _polish_cover(x::Vector{T}, supp::EdgeList{T}, m::Int; multipliers=nothing,
                       budgets::NamedTuple=(;)) where {T}
    supp.qsign < 0 && return _polish_difference_qp(supp, x; seed=multipliers, budgets)
    flip(u) = [k <= m ? u[k] : -u[k] for k in eachindex(u)]
    u, certified, nsteps, kkt = _polish_difference_qp(EdgeList{T}(supp.edges, supp.cvals, -oneunit(T)), flip(x);
                                                      seed=multipliers, budgets)
    return flip(u), certified, nsteps, kkt
end

function _polish_cover(x::Vector{T}, supp::Grid{T}, m::Int; multipliers=nothing,
                       budgets::NamedTuple=(;)) where {T}
    C = supp.C
    edges = Tuple{Int,Int}[]
    cvals = T[]
    seed = multipliers === nothing ? nothing : T[]
    for j in axes(C, 2), i in axes(C, 1)
        isfinite(C[i, j]) || continue
        push!(edges, (i, m + j))
        push!(cvals, C[i, j])
        seed === nothing || push!(seed, multipliers[i, j])
    end
    return _polish_cover(x, EdgeList{T}(edges, cvals), m; multipliers=seed, budgets)
end

_polish_cover(x::Vector{T}, supp::DiffGrid{T}, m::Int; multipliers=nothing, budgets::NamedTuple=(;)) where {T} =
    _polish_difference_qp(supp, x; seed=multipliers === nothing ? nothing : vec(multipliers), budgets)

# Solve the `AbsLog{2}` problem of `_symcover_min_abslog2` exactly from its
# augmented-Lagrangian iterate `x`, returning `(x, certified, nsteps, kkt)`, with
# `kkt` describing the double-cover solution. The
# symmetric problem is the asymmetric one on the bipartite double cover: with
# unknowns `u = (α; β)` and start `(x; -x)`, each stored entry `(p, q)` gives the
# difference constraints `α[p] - β[q] ≥ c` and, off the diagonal, `α[q] - β[p] ≥ c`.
# Exchanging `α` with `-β` maps that problem to itself, so its minimizer has
# equal mirror residuals and `(α - β)/2` has those same residuals.
# Unknowns without support keep their values in `x`. `multipliers`, the
# iteration's multipliers in the layout of `supp`, seed the active set, each
# entry's value going to both of its constraints. `budgets` is passed to
# `_polish_difference_qp`.
function _polish_symcover(x::Vector{T}, supp::EdgeList{T}; multipliers=nothing,
                          budgets::NamedTuple=(;)) where {T}
    n = length(x)
    edges2 = Tuple{Int,Int}[]
    cvals2 = T[]
    seed = multipliers === nothing ? nothing : T[]
    sizehint!(edges2, 2 * length(supp.edges))
    sizehint!(cvals2, 2 * length(supp.edges))
    for (e, (p, q)) in pairs(supp.edges)
        c = supp.cvals[e]
        push!(edges2, (p, n + q))
        push!(cvals2, c)
        seed === nothing || push!(seed, multipliers[e])
        if p != q
            push!(edges2, (q, n + p))
            push!(cvals2, c)
            seed === nothing || push!(seed, multipliers[e])
        end
    end
    return _unfold_symcover(x, _polish_difference_qp(EdgeList{T}(edges2, cvals2, -one(T)), [x; -x]; seed, budgets))
end

# `supp.C` holds the upper triangle; the double cover needs both.
function _polish_symcover(x::Vector{T}, supp::Grid{T}; multipliers=nothing,
                          budgets::NamedTuple=(;)) where {T}
    n = length(x)
    C = copy(supp.C)
    for j in axes(C, 2), i in first(axes(C, 1)):j-1
        C[j, i] = C[i, j]
    end
    seed = nothing
    if multipliers !== nothing
        S = copy(multipliers)
        for j in axes(S, 2), i in first(axes(S, 1)):j-1
            S[j, i] = S[i, j]
        end
        seed = vec(S)
    end
    return _unfold_symcover(x, _polish_difference_qp(DiffGrid{T}(C, collect(1:n), collect(n+1:2n)), [x; -x];
                                                     seed, budgets))
end

# Symmetric scales `(α - β)/2` from the double-cover solution `u = (α; β)`.
function _unfold_symcover(x::Vector{T}, (u, certified, nsteps, kkt)) where {T}
    n = length(x)
    xnew = (u[1:n] .- u[n+1:2n]) ./ 2
    return xnew, certified, nsteps, kkt
end
