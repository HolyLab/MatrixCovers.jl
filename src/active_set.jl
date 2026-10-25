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
# linear solve is a graph Laplacian with one unknown per tree. Bulk steps add
# many constraints at once: the violated constraints, most violated first, that
# extend `W` to a larger forest. Like a single addition, each bulk step raises
# the dual function; they run while the number of violated constraints falls,
# within a fixed budget, and single additions finish the solve.

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
    fac = L isa SparseMatrixCSC ? _laplacian_factor(L; budgets...) : _laplacian_factor(L)
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

# Jacobi-preconditioned CG, with iterative refinement, on a pinned tree
# Laplacian: in `Float64`, one whose sparse Cholesky factor would exceed the
# fill or flop budget; in other element types, every sparse one, since CHOLMOD
# factors only `Float64`. Such Laplacians arise from expander-like supports,
# which fill in under elimination but have clustered spectra, so CG converges
# in few iterations. A solve that refinement does not finish falls back to a
# factorization: the analyzed CHOLMOD factor `F` in `Float64`, a dense Cholesky
# factorization otherwise.
struct _LaplacianCG{T}
    L::SparseMatrixCSC{T,Int}         # upper triangle
    S::SparseMatrixCSC{T,Int}         # both triangles, for products
    dg::Vector{T}
    F::Union{SparseCholesky,Nothing}  # analyzed, `Float64` only
    fallback::Base.RefValue{Any}      # the factorization once CG gave up; `nothing` before
    maxiter::Int                      # CG iterations per pass
    err::Base.RefValue{T}             # estimated 2-norm forward error of the last solve
end

# Estimated 2-norm forward error of the last `_laplacian_solve!`. A
# factorization is taken as exact at the certification floor of the finish.
_solve_error(::SparseCholesky) = 0.0
_solve_error(::Cholesky) = 0.0
_solve_error(C::_LaplacianCG) = C.err[]

function _laplacian_factor(L::SparseMatrixCSC{Float64,Int}; fillbudget::Real=LSQR_FILL_BUDGET,
                           flopbudget::Real=LSQR_FLOP_BUDGET, cgmaxiter::Int=10 * size(L, 1) + 100)
    F = SparseCholesky()
    size(L, 1) == 0 && return F
    analyze!(F, L)
    if sizeof(Float64) * factor_entries(F) > fillbudget || factor_flops(F) > flopbudget * nnz(L)
        S = L + triu(L, 1)'
        return _LaplacianCG{Float64}(L, S, Vector(diag(S)), F, Ref{Any}(nothing), cgmaxiter, Ref(0.0))
    end
    factorize!(F, L)
    return F
end
# Element types CHOLMOD does not factor; the budgets do not apply.
function _laplacian_factor(L::SparseMatrixCSC{T,Int}; fillbudget::Real=LSQR_FILL_BUDGET,
                           flopbudget::Real=LSQR_FLOP_BUDGET, cgmaxiter::Int=10 * size(L, 1) + 100) where {T}
    size(L, 1) == 0 && return _laplacian_factor(Matrix(L))
    return _laplacian_cg(L; cgmaxiter)
end
# Conjugate-gradient solver for `L`, built without a symbolic analysis, for a
# pattern presumed to exceed the budgets. The fallback factorization is analyzed
# only if CG fails (see `_cg_fallback`).
function _laplacian_cg(L::SparseMatrixCSC{T,Int}; cgmaxiter::Int=10 * size(L, 1) + 100) where {T}
    S = L + triu(L, 1)'
    return _LaplacianCG{T}(L, S, Vector(diag(S)), nothing, Ref{Any}(nothing), cgmaxiter, Ref(zero(T)))
end
# The factorization that replaces conjugate gradients once they fail. In
# `Float64` it reuses the symbolic analysis in `C.F` when there is one.
function _cg_fallback(C::_LaplacianCG{Float64})
    F = C.F
    if F === nothing
        F = SparseCholesky()
        analyze!(F, C.L)
    end
    return factorize!(F, C.L)
end
_cg_fallback(C::_LaplacianCG) = _laplacian_factor(Matrix(C.L))
_laplacian_factor(L::Matrix) = LinearAlgebra.cholesky!(Symmetric(L, :U))

# Solves `L x = b`. The keywords apply only to a `_LaplacianCG`: `rtol` is the
# relative residual at which conjugate gradients stop, and `refine` enables the
# iterative refinement that follows, which reduces the forward error to the
# resolution of the arithmetic. `_solve_error` then estimates the remaining
# error; it is `NaN` when refinement is skipped.
_laplacian_solve!(x::Vector{Float64}, F::SparseCholesky, b::Vector{Float64}; rtol=nothing, refine=true) =
    isempty(b) ? x : solve!(x, F, CHOLMOD_A, b)
_laplacian_solve!(x, F::Cholesky, b; rtol=nothing, refine=true) = ldiv!(x, F, b)

function _laplacian_solve!(x::Vector{T}, C::_LaplacianCG{T}, b::Vector{T}; rtol::Real=sqrt(eps(T)),
                           refine::Bool=true) where {T}
    if C.fallback[] === nothing
        n = length(b)
        fill!(x, zero(T))
        rdd, r, z, d, Ad, e = ntuple(_ -> zeros(T, n), 6)
        Smul! = (y, v) -> mul!(y, C.S, v)
        η = T(rtol)
        _, ok = _pcg!(Smul!, x, C.dg, b, r, z, d, Ad, C.maxiter, η * norm(b))
        if ok && !refine
            C.err[] = T(NaN)
            return x
        end
        # Iterative refinement with the residual in compensated arithmetic.
        # CG leaves a forward error of its normwise backward error times the
        # condition number, and the multipliers, routed sums of the solution
        # along the trees, need the solution to the certification floor. A
        # correction solved to relative residual `η` shrinks the error by a
        # factor of about `κ η`, so the size of the last correction estimates
        # the error that remains. Refinement that stops contracting hands the
        # system to the factorization.
        δprev = T(Inf)
        while ok
            _residual_dd!(rdd, b, C.S, x)
            fill!(e, zero(T))
            _, ok = _pcg!(Smul!, e, C.dg, rdd, r, z, d, Ad, C.maxiter, η * norm(rdd))
            ok || break
            x .+= e
            δ = norm(e)
            if δ <= 8 * eps(T) * norm(x)
                C.err[] = δ
                return x
            end
            δ <= δprev / 100 || break
            δprev = δ
        end
        C.fallback[] = _cg_fallback(C)
    end
    C.err[] = zero(T)
    return _laplacian_solve!(x, C.fallback[], b)
end

# `r = b - S x` with every product and sum carried in twice the working
# precision by error-free transformations, so that `r` is accurate to a few
# units of roundoff in its own magnitude rather than in that of `b` and `S x`.
# `fma` must be correctly rounded in `T`.
function _residual_dd!(r::Vector{T}, b::Vector{T}, S::SparseMatrixCSC{T,Int}, x::Vector{T}) where {T}
    axes(r) == axes(b) == axes(x) == (axes(S, 1),) == (axes(S, 2),) ||
        throw(DimensionMismatch("residual, right-hand side, matrix, and solution must share axes"))
    copyto!(r, b)
    lo = zeros(T, length(r))
    rv = rowvals(S)
    nz = nonzeros(S)
    for j in axes(S, 2)
        xj = x[j]
        for k in nzrange(S, j)
            i = rv[k]
            p = nz[k] * xj
            pe = fma(nz[k], xj, -p)   # `p + pe` is the exact product
            hi = r[i]
            s = hi - p
            t = s - hi
            lo[i] += (hi - (s - t)) + (-p - t) - pe
            r[i] = s
        end
    end
    r .+= lo
    return r
end

# Per-tree values `y` minimizing `½‖L y - rhs‖` with pinned trees at zero, and
# the estimated 2-norm error of the solve; `rhs` is indexed by tree.
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
    return y, _solve_error(K.fac)
end

# Minimizer of the objective with every edge of `F` tight, and the estimated
# error of the Laplacian solve behind it.
function _tight_minimizer(F::_Forest{T}, K::_TreeLaplacian, supp::EdgeList) where {T}
    rhs = zeros(T, F.ntree)
    for (e, (p, q)) in enumerate(supp.edges)
        tp, tq = F.tree[p], F.tree[q]
        tp == tq && continue
        k = F.off[p] - F.off[q] - supp.cvals[e]
        rhs[tp] -= k
        rhs[tq] += k
    end
    y, δ = _tree_solve(K, rhs)
    return [y[F.tree[v]] + F.off[v] for v in eachindex(F.tree)], δ
end

function _tight_minimizer(F::_Forest{T}, K::_TreeLaplacian, s::_DiffGridQP) where {T}
    # Edges within a tree contribute `-k` and `+k` to the same entry, so the
    # sum may run over every constraint.
    g = _difference_sums!(zeros(T, length(F.off)), F.off, s, true)
    rhs = zeros(T, F.ntree)
    for v in eachindex(F.tree, g)
        rhs[F.tree[v]] -= g[v]
    end
    y, δ = _tree_solve(K, rhs)
    return [y[F.tree[v]] + F.off[v] for v in eachindex(F.tree)], δ
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

# Constraints outside `W` with residual below `-ptol`, most violated first.
function _violated(u::Vector{T}, inW, s::EdgeList, ptol::T) where {T}
    viol = [e for e in 1:_nconstraints(s) if !inW[e] && _resid(u, s, e) < -ptol]
    return sort!(viol; by=e -> _resid(u, s, e))
end

function _violated(u::Vector{T}, inW::Matrix{Bool}, s::_DiffGridQP, ptol::T) where {T}
    C = s.C
    L = LinearIndices(C)
    ur = u[s.rowidx]
    viol = Int[]
    for j in axes(C, 2)
        uj = u[s.colidx[j]]
        for i in eachindex(ur, view(C, :, j))
            # Entries outside the support have residual `+Inf`.
            !inW[i, j] && ur[i] - uj - C[i, j] < -ptol && push!(viol, L[i, j])
        end
    end
    return sort!(viol; by=e -> _resid(u, s, e))
end

"""
    u, certified, nsteps, kkt = _polish_difference_qp(supp, u0; τ, maxsteps, maxbulk, seed, budgets)
    u, certified, nsteps, kkt = _polish_difference_qp(edges, cvals, u0; τ, maxsteps, maxbulk, seed, budgets)

Minimize `½ ∑ₑ (u[p] - u[q] - c[e])²` subject to every term being nonnegative,
starting from the near-optimal `u0`. The constraints are those of `supp`, an
`EdgeList` with `qsign = -1` or a `DiffGrid`, or the edges `(p, q)` with costs
`cvals`. Each step solves a graph-Laplacian system, and `maxsteps` bounds their
number. The constant on each connected component of the graph is taken from
`u0`.

The active set is seeded by adding candidate constraints in order and
skipping any that would close a cycle; `ptol` is `1000 eps` times the largest
magnitude in `c` and `u0`. Without `seed`, the candidates are the constraints
with `u0[p] - u0[q] - c[e] < max(τ, ptol)`, tightest first. With `seed`, a
vector of nonnegative weights indexed like the constraints (edge positions of
an `EdgeList`, linear indices of a `DiffGrid`'s `C`), such as the multipliers of
an augmented-Lagrangian iteration, the candidates are the constraints with
positive weight or residual below `ptol`, largest weight first and then
tightest first. Constraints on no cycle of the support graph are activated
ahead of the candidates: they are tight with zero multiplier at the minimizer,
so a solve error must not let them appear violated.

From the seed, each bulk step adds the violated constraints, most violated
first, that extend the active forest without closing a cycle, and drops active
constraints whose multipliers would become negative. Bulk steps repeat while
the number of violated constraints falls, up to `maxbulk` bulk steps in all;
single-constraint steps then finish the solve. A single-constraint step that
leaves more than 8 violated constraints re-enters the bulk steps while the
budget lasts. Each Laplacian solve of a bulk step counts toward `nsteps`.

`certified` reports that the
returned `u` satisfies the KKT conditions to tolerances proportional to the
largest magnitude in `c` and `u0`; when it is false, `u` is `u0`.
`kkt = (; primal, ptol, dual, dtol)` gives the largest constraint violation of
the returned `u` and the most negative active multiplier (as nonnegative
magnitudes), with the tolerances they were certified against; `primal` and
`dual` are `NaN` when `certified` is false.

`budgets`, a `NamedTuple` with
fields `fillbudget` and/or `flopbudget`, sets the limits above which these
systems are solved by conjugate gradients rather than a sparse Cholesky
factorization; the tolerances of the certificate then rise to twice the
estimated error of the solve, so that `certified` never claims more accuracy
than the solve delivers.
"""
function _polish_difference_qp(edges::Vector{Tuple{Int,Int}}, cvals::Vector{T}, u0::Vector{T}; kwargs...) where {T}
    return _polish_difference_qp(EdgeList{T}(edges, cvals, -oneunit(T)), u0; kwargs...)
end

function _polish_difference_qp(supp::Union{EdgeList{T},DiffGrid{T}}, u0::Vector{T};
                               τ::Real=T(1e-4), maxsteps::Int=max(1000, 2 * length(u0)), maxbulk::Int=20,
                               seed::Union{Nothing,AbstractVector}=nothing,
                               budgets::NamedTuple=(;)) where {T}
    s = _qp_support(supp)
    nV = length(u0)
    scale = max(oneunit(T), _cmax(s), maximum(abs, u0; init=zero(T)))
    ptol = 1000 * eps(T) * scale
    dtol = ptol * max(1, _nsupported(s))
    # Seed: Kruskal's algorithm on the candidate constraints, in the order the
    # docstring describes.
    inW = _active_flags(s)
    W = Int[]   # the active constraints, in increasing order
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
    # Constraints on no cycle come first; Kruskal then skips them if they recur.
    _activate!(inW, W, _forest_extension(W, vcat(_acyclic_constraints(s, nV), cand), s, nV))
    # Multipliers and their rates; only the entries in `W` are meaningful.
    E = _nconstraints(s)
    λ = zeros(T, E)
    r = zeros(T, E)
    nsteps = 0
    u = u0
    u, nsteps, ptol_eff, dtol_eff = _dual_feasible!(inW, W, λ, s, nV, ptol, nsteps, maxsteps; budgets)
    certified = false
    nokkt = (; primal=T(NaN), ptol, dual=T(NaN), dtol)
    nbulk = 0       # bulk steps taken
    bulkon = maxbulk > 0
    bulkmin = 0     # bulk steps run while more than this many constraints are violated
    while nsteps <= maxsteps
        if bulkon
            u, nsteps, ptol_eff, dtol_eff, nbulk, e, bulkon =
                _bulk_steps!(inW, W, λ, s, nV, u, ptol, ptol_eff, dtol_eff, nsteps, maxsteps, nbulk, maxbulk, bulkmin; budgets)
        else
            e, _ = _most_violated(u, inW, s, -ptol_eff)
        end
        e < 0 && break
        if e == 0
            certified = all(k -> λ[k] >= -dtol_eff, W)
            break
        end
        # Single-constraint step; bulk steps resume only when it leaves more
        # than `bulkmin` constraints violated.
        bulkmin = 8
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
            y, _ = _tree_solve(K, rhs)
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
        u, nsteps, ptol_eff, dtol_eff = _dual_feasible!(inW, W, λ, s, nV, ptol, nsteps, maxsteps; budgets)
    end
    certified || return u0, false, nsteps, nokkt
    # Restore each component's constant from `u0`. (`u` is reassigned above, so
    # capturing it in a closure would box it; `ustar` is bound once.)
    ustar = u
    kkt = (; primal=max(zero(T), -minimum(e -> _resid(ustar, s, e), 1:_nconstraints(s))), ptol=ptol_eff,
           dual=max(zero(T), -minimum(k -> λ[k], W; init=zero(T))), dtol=dtol_eff)
    uf = collect(1:nV)
    find(x) = (while uf[x] != x; uf[x] = uf[uf[x]]; x = uf[x]; end; x)
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

# Objective `½ ∑ₑ dₑ(u)²`.
_objective(u::Vector{T}, s::EdgeList) where {T} = sum(e -> _resid(u, s, e)^2, 1:_nconstraints(s); init=zero(T)) / 2
function _objective(u::Vector{T}, s::_DiffGridQP) where {T}
    C = s.C
    ur = u[s.rowidx]
    f = zero(T)
    for j in axes(C, 2)
        uj = u[s.colidx[j]]
        for i in eachindex(ur, view(C, :, j))
            c = C[i, j]
            isfinite(c) && (f += (ur[i] - uj - c)^2)
        end
    end
    return f / 2
end

# Bulk steps from the tight minimizer `u` of the dual-feasible forest `W`, run
# while more than `minviol` constraints are violated and their number falls.
# Each step adds `V`, the violated constraints, most violated first, that
# extend `W` to a larger forest. It first tries a primal-dual active-set step:
# activate all of `V`, then drop constraints with negative multipliers
# (`_dual_feasible!`). At the tight minimizer of a dual-feasible forest the
# objective equals the dual function, so the step is kept only if it raises
# the objective, as every single-constraint step does. Otherwise it is undone
# and `V` is added by `_bulk_ascent!`, which raises the dual function by
# construction, and the remaining steps of this call use `_bulk_ascent!`
# directly. Returns the updated `u`, `nsteps`, tolerances and `nbulk`; the most
# violated constraint (0 when none is violated, -1 when `maxsteps` interrupts
# `_bulk_ascent!`); and whether further bulk steps may run.
function _bulk_steps!(inW, W::Vector{Int}, λ::Vector{T}, s, nV::Int, u::Vector{T}, ptol, ptol_eff, dtol_eff,
                      nsteps::Int, maxsteps::Int, nbulk::Int, maxbulk::Int, minviol::Int; budgets) where {T}
    viol = _violated(u, inW, s, ptol_eff)
    f = _objective(u, s)
    pdas = true
    while length(viol) > minviol && nbulk < maxbulk && nsteps < maxsteps
        V = _forest_extension(W, viol, s, nV)
        isempty(V) && break
        nbulk += 1
        if pdas
            Wold, λold = copy(W), λ[W]
            _activate!(inW, W, V)
            nsteps += 1
            u1, nsteps, ptol1, dtol1 = _dual_feasible!(inW, W, λ, s, nV, ptol, nsteps, maxsteps; budgets)
            f1 = _objective(u1, s)
            if f1 > f
                u, f, ptol_eff, dtol_eff = u1, f1, ptol1, dtol1
            else
                for e in W
                    inW[e] = false
                end
                copyto!(resize!(W, length(Wold)), Wold)
                for (e, λe) in zip(Wold, λold)
                    inW[e] = true
                    λ[e] = λe
                end
                pdas = false
            end
        end
        if !pdas
            u, nsteps, ptol_eff, dtol_eff, done = _bulk_ascent!(inW, W, λ, V, s, nV, u, ptol, ptol_eff, dtol_eff,
                                                               nsteps, maxsteps; budgets)
            done || return u, nsteps, ptol_eff, dtol_eff, nbulk, -1, false
            f = _objective(u, s)
        end
        nprev = length(viol)
        viol = _violated(u, inW, s, ptol_eff)
        length(viol) < nprev || break
    end
    return u, nsteps, ptol_eff, dtol_eff, nbulk, isempty(viol) ? 0 : first(viol), nbulk < maxbulk
end

# Starting from the tight minimizer `u` of the dual-feasible forest `W`,
# activate the constraints `V`, which extend `W`, keeping the multipliers
# nonnegative. The segment from `u` to the tight minimizer of `W ∪ V`, with the multipliers
# interpolated linearly, satisfies stationarity throughout and raises the dual
# function. Each pass solves for the far end and moves along the segment until
# a multiplier reaches zero; that constraint leaves `W` or `V`. Constraints
# that leave `V` remain violated and are left for later steps. Returns `u`,
# `nsteps`, the tolerances (as `_dual_feasible!` does), and whether the
# remaining `V` became active; when `maxsteps` is reached first, `u` is a point
# on a segment, not the tight minimizer of `W`.
function _bulk_ascent!(inW, W::Vector{Int}, λ::Vector{T}, V::Vector{Int}, s, nV::Int, u::Vector{T}, ptol,
                       ptol_eff, dtol_eff, nsteps::Int, maxsteps::Int; budgets) where {T}
    E = max(1, _nsupported(s))
    λ1 = zeros(T, length(λ))
    inV = Set(V)
    for e in V
        λ[e] = zero(T)
    end
    while nsteps < maxsteps
        nsteps += 1
        A = sort!(vcat(W, V))
        F = _Forest(nV, s, A)
        K = _TreeLaplacian(F, s; budgets)
        u1, δ = _tight_minimizer(F, K, s)
        _route!(λ1, _difference_gradient(u1, s), F, s)
        t, kb = one(T), 0
        for k in A
            λk = max(λ[k], zero(T))
            if λ1[k] < λk
                tk = λk / (λk - λ1[k])
                if tk < t
                    t, kb = tk, k
                end
            end
        end
        if kb == 0
            _activate!(inW, W, V)
            for k in W
                λ[k] = λ1[k]
            end
            ptol_eff = max(ptol, T(2δ))
            return u1, nsteps, ptol_eff, ptol_eff * E, true
        end
        if t > 0
            u = u .+ t .* (u1 .- u)
            for k in A
                λ[k] += t * (λ1[k] - λ[k])
            end
            blocked = [kb]
        else
            # Every constraint at zero multiplier whose multiplier would
            # decrease blocks the step; none of them moves `u`.
            blocked = [k for k in A if max(λ[k], zero(T)) == 0 && λ1[k] < 0]
        end
        for k in blocked
            λ[k] = zero(T)
            k in inV ? delete!(inV, k) : (inW[k] = false)
        end
        filter!(in(inV), V)
        filter!(k -> inW[k], W)
    end
    return u, nsteps, ptol_eff, dtol_eff, false
end

# Add the constraints `V` to the active set `W`, keeping `W` sorted.
function _activate!(inW, W::Vector{Int}, V)
    for e in V
        inW[e] = true
    end
    append!(W, V)
    return sort!(W)
end

# The constraints of `cand`, in order, that extend the forest `W` without
# closing a cycle with `W` or with those before them.
function _forest_extension(W::Vector{Int}, cand, s, nV::Int)
    uf = collect(1:nV)
    find(x) = (while uf[x] != x; uf[x] = uf[uf[x]]; x = uf[x]; end; x)
    for e in W
        p, q = _edge(s, e)
        uf[find(p)] = find(q)
    end
    ext = Int[]
    for e in cand
        p, q = _edge(s, e)
        rp, rq = find(p), find(q)
        rp == rq && continue
        uf[rp] = rq
        push!(ext, e)
    end
    return ext
end

function _deactivate!(inW, W::Vector{Int}, λ, w::Int)
    λ[w] = zero(eltype(λ))
    inW[w] = false
    deleteat!(W, searchsortedfirst(W, w))
    return nothing
end

# Drop the edges of `W` with negative multipliers until none remain, and return
# the tight minimizer of `W`, the step count, and the primal and dual
# tolerances that minimizer can be certified against; `λ` holds the
# multipliers. The tolerances start from `ptol` and `ptol` times the number of
# constraints and rise with the error of the Laplacian solve: an error `δ` in
# the per-tree values moves a residual by at most `2δ`, and a routed
# multiplier, a sum of residuals over a subtree, by at most `2δ` per constraint.
function _dual_feasible!(inW, W::Vector{Int}, λ::Vector{T}, supp, nV::Int, ptol,
                         nsteps::Int, maxsteps::Int; budgets::NamedTuple=(;)) where {T}
    E = max(1, _nsupported(supp))
    while true
        F = _Forest(nV, supp, W)
        K = _TreeLaplacian(F, supp; budgets)
        u, δ = _tight_minimizer(F, K, supp)
        ptol_eff = max(ptol, T(2δ))
        dtol_eff = ptol_eff * E
        _route!(λ, _difference_gradient(u, supp), F, supp)
        nW = length(W)
        for e in W
            λ[e] < -dtol_eff && (inW[e] = false)
        end
        filter!(e -> inW[e], W)
        (length(W) < nW && nsteps < maxsteps) || return u, nsteps, ptol_eff, dtol_eff
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

# Constraints on no cycle of the support graph, found by peeling vertices of
# degree one. Each is tight with zero multiplier at the minimizer: the vertices
# beyond it enter no other constraint, so their scales make it tight at no cost
# and the gradient vanishes there. Seeding them keeps the finish from chasing
# roundoff-level violations of constraints that are tight by structure.
function _acyclic_constraints(s::EdgeList, nV::Int)
    E = length(s.edges)
    ptr = zeros(Int, nV + 1)
    for (p, q) in s.edges
        ptr[p+1] += 1
        ptr[q+1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    adj = zeros(Int, ptr[end] - 1)
    cursor = ptr[1:nV]
    for (e, (p, q)) in enumerate(s.edges)
        adj[cursor[p]] = e
        cursor[p] += 1
        adj[cursor[q]] = e
        cursor[q] += 1
    end
    deg = [ptr[v+1] - ptr[v] for v in 1:nV]
    alive = trues(E)
    stack = [v for v in 1:nV if deg[v] == 1]
    out = Int[]
    while !isempty(stack)
        v = pop!(stack)
        deg[v] == 1 || continue
        for k in ptr[v]:ptr[v+1]-1
            e = adj[k]
            alive[e] || continue
            alive[e] = false
            push!(out, e)
            deg[v] -= 1
            p, q = s.edges[e]
            w = p == v ? q : p
            deg[w] -= 1
            deg[w] == 1 && push!(stack, w)
        end
    end
    return out
end

function _acyclic_constraints(s::_DiffGridQP, nV::Int)
    C = s.C
    L = LinearIndices(C)
    m, n = size(C)
    degr = copy(s.nfr)
    degc = copy(s.nfc)
    alive = isfinite.(C)
    out = Int[]
    rows = [i for i in 1:m if degr[i] == 1]
    cols = [j for j in 1:n if degc[j] == 1]
    while !(isempty(rows) && isempty(cols))
        if !isempty(rows)
            i = pop!(rows)
            degr[i] == 1 || continue
            j = findfirst(view(alive, i, :))
            alive[i, j] = false
            push!(out, L[i, j])
            degr[i] -= 1
            degc[j] -= 1
            degc[j] == 1 && push!(cols, j)
        else
            j = pop!(cols)
            degc[j] == 1 || continue
            i = findfirst(view(alive, :, j))
            alive[i, j] = false
            push!(out, L[i, j])
            degc[j] -= 1
            degr[i] -= 1
            degr[i] == 1 && push!(rows, i)
        end
    end
    return out
end
