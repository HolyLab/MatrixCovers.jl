# Transversal-tight hard covers that push the coverage ratios of the entries off
# the transversal toward a target ratio.

# Stopping tolerance (see `BalancedCoverStats`).
const BALANCED_TOL = 1e-9
# Sufficient-decrease constant and the most halvings a line search may try.
const BALANCED_ARMIJO = 1e-4
const BALANCED_NBACKTRACK = 100
# A first trial step changes no slack by more than this many units of `1/p`, so
# that `exp(-p d)` stays in range.
const BALANCED_MAXSTEP = 64
# Roundoff allowance, in units of `eps`, for a change in the objective too small
# to resolve.
const BALANCED_NOISE = 64
# Newton systems on at most this many unknowns, or at least an eighth full, are
# factored densely.
const BALANCED_DENSE_MAX = 128
# Ridge retries for a Newton matrix that roundoff makes indefinite; each retry
# scales the ridge by `BALANCED_RIDGE_GROWTH`. The ridge is relative to each
# diagonal entry, so that it does not swamp the curvature of a group whose
# entries all have small `f''`.
const BALANCED_NRIDGE = 4
const BALANCED_RIDGE_GROWTH = 1e4

"""
    a, b, stats = cover_transversal(ϕ::PowerMean{p}, A; target=1e-2, maxbalance=false, tol=1e-9, maxiter, fillbudget, flopbudget)

Return the hard cover of the square matrix `A` that is tight on its
maximum-product transversals and pushes the coverage ratios
`r = |A[i,j]| / (a[i]*b[j])` of the other nonzero entries toward `target`: it
minimizes

    ∑ (r^p - 1 - p*log(r/target)*target^p) / p,

the sum taken over the nonzero entries off the transversals, among the hard
covers with `prod(a) * prod(b) == π*` (see [`cover_transversal(A)`](@ref
cover_transversal)). Each term is `target^p * φ(r/target)` up to a constant, with
`φ` the [`PowerMean{p}`](@ref) penalty, so it is least at `r = target`; the hard
constraint caps every ratio at one, and the transversal entries have ratio one.
`0 < target <= 1` and `0 < p < ∞` are required; otherwise an `ArgumentError` is
thrown. `cover_transversal(AbsLog{2}(), A)` is `cover_transversal(A)`.

The objective is strictly convex in the products `a[i]*b[j]`, so they are
unique on the support of `A`. As for `cover_transversal(A)`, they do not depend
on which maximum-product transversal is used, they are scale-covariant, and the
factors use the balance convention of [`cover_min`](@ref).

With `target = 1`, every ratio is pushed up to one, as far as the constraints
allow. With small `target`, the ratios within each block of a block-triangular
form of `A` balance in the sense of the `p`-power mean, and the entries joining
different blocks are pushed toward `target`; at the minimizer, the `p`-power
mean of the ratios of the entries crossing such a cut equals `target` whenever
none of them has ratio one. Large `p` approaches the cover that minimizes the
largest ratio within each block.

With `maxbalance=true`, the criterion within each fully indecomposable block of
`A` is the `p → ∞` limit of the objective (Hook, Pestana, Tisseur & Hogg, SIAM
J. Matrix Anal. Appl. 40:320–346, 2019, Theorem 3.2): the scales are
*max-balanced* within each block. Let `σ` be a maximum-product transversal,
and let an entry `A[i,j]` off it lead from row `i` to the row `k` with
`σ[k] == j`. For every partition of the rows of a block into two nonempty sets
`I` and `K`, the largest ratio of an entry leading from `I` to `K` equals the
largest ratio of an entry leading from `K` to `I`. This determines the products
`a[i]*b[j]` within each block uniquely. When `σ` is the only maximum-product
transversal, the largest ratio within a block equals the block's `θ`, the
largest geometric-mean ratio of a cycle of entries off `σ`, which is the least
largest ratio that any cover tight on `σ` can have there. `ϕ` and `target` fix
the scales between blocks: they minimize the objective above summed over the
entries joining different blocks, with each block's scales moving together. At
the minimizer, for every set of blocks that the entries joining it to the other
blocks all lead out of, the `p`-power mean of the ratios of those entries
equals `target` whenever none of them has ratio one.
The result is a hard cover tight on the maximum-product transversals, and its
products are unique on the support and scale-covariant.

`stats` is a [`BalancedCoverStats`](@ref MatrixCovers.BalancedCoverStats). The
solver warns when [`MatrixCovers.converged(stats)`](@ref MatrixCovers.converged)
is `false`; the result is then still a hard cover tight on the maximum-product
transversals.

# Extended help

As for `cover_transversal(A)`, tightness on a maximum-product transversal `σ`
leaves the row log scales as unknowns, with one constraint `d ≥ 0` on the log
slack `d = -log(r)` of each entry off the transversal. Starting from the
transversal's dual variables, which are feasible, a primal active-set Newton
method keeps every iterate feasible. Entries at zero slack are *active*, and the
rows they join form groups whose scales move together; the entries at zero
slack at the start are active from the outset. Each Newton step solves a
weighted graph-Laplacian system on the groups, a ratio test stops a step where
another entry reaches zero slack, and a backtracking line search ensures
descent. Where the groups are stationary, a maximum-flow problem on each group's
active entries either proves the KKT conditions or yields a set of rows whose
scales can decrease, releasing the active entries that enter it.

With `maxbalance=true`, each block is max-balanced as in Algorithm 3.10 of Hook
et al.: a cycle of largest geometric-mean ratio in the block is found, a
diagonal similarity brings every ratio of the block down to that mean, and the
cycle's rows, whose relative scales are then fixed, are merged into one; this
repeats until one row remains. The cycles are found by the parametric
shortest-path algorithm of Young, Tarjan & Orlin (Networks 21:205–221, 1991).
For a block with `n` rows and `m` entries, its cost is at most
`O(nm + n² log n)`. The cost is roughly linear in `m` when the block's
shortest-path tree stays shallow, as it does for most sparse matrices, and
about `O(nm)` for a banded block, whose tree is a path. The active-set Newton
method then runs with one scale per block.

`tol` (default `1e-9`) is the stopping tolerance: the groups are stationary when
the derivative of the objective with respect to each group's scale is at most
`tol` times the sum of the magnitudes of its terms, or when roundoff keeps the
Newton step from reducing the objective: the step changes no log slack by more
than `tol`, or the directional derivative of the objective along it is
nonnegative but within the roundoff of the groups' derivatives. `maxiter`
(default `100 + 10n` for `n×n` `A`) bounds the Newton steps plus the changes of
the active set.
`fillbudget` and `flopbudget` are as for [`cover_min`](@ref): a Newton system
whose sparse Cholesky factor would exceed them is solved by conjugate
gradients. The Newton systems have condition numbers of order `target^(-p)`
times that of the graph, and of order `r^(-p)` where cycles of entries hold
ratios `r` far below `target`, so a very small `target^p` or a large `p`
degrades the linear solves and the convergence.

`A` must be square and structurally nonsingular. Narrower element types than
`Float64` are computed in `Float64` and narrowed, with the coverage certified
afterwards.

See also: [`cover_transversal!`](@ref), [`PowerMean`](@ref).
"""
function cover_transversal(ϕ::PowerMean{p}, A::AbstractMatrix; target::Real=1e-2,
                           maxbalance::Bool=false,
                           tol::Real=BALANCED_TOL, maxiter::Union{Integer,Nothing}=nothing,
                           fillbudget::Real=LSQR_FILL_BUDGET,
                           flopbudget::Real=LSQR_FLOP_BUDGET) where {p}
    a, b, nt = _cover_transversal_balanced(A, p, target; maxbalance, tol, maxiter, fillbudget, flopbudget)
    return a, b, BalancedCoverStats{_auglag_stats_type(A)}(nt)
end

cover_transversal(::AbsLog{2}, A::AbstractMatrix; kwargs...) = cover_transversal(A; kwargs...)

"""
    a, b, stats = cover_transversal!(ϕ::PowerMean, a, b, A; kwargs...)

Mutating counterpart of [`cover_transversal(ϕ, A)`](@ref cover_transversal):
writes the cover into `a` and `b` and returns them with the solver's
statistics. The initial contents of `a` and `b` are ignored. `eachindex(a)` must
match `axes(A, 1)` and `eachindex(b)` must match `axes(A, 2)`.
"""
function cover_transversal!(ϕ::PowerMean, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...)
    _check_cover_axes(a, b, A)
    anew, bnew, stats = cover_transversal(ϕ, A; kwargs...)
    a .= anew
    b .= bnew
    return a, b, stats
end

cover_transversal!(::AbsLog{2}, a::AbstractVector, b::AbstractVector, A::AbstractMatrix; kwargs...) =
    cover_transversal!(a, b, A; kwargs...)

# Worker for `cover_transversal(::PowerMean{p}, A; target)`, returning `(a, b, nt)`
# with the statistics as a named tuple. `transversal` is as for
# `_transversal_system`.
function _cover_transversal_balanced(A::AbstractMatrix, p::Real, target::Real; maxbalance::Bool=false,
                                     tol::Real=BALANCED_TOL,
                                     maxiter::Union{Integer,Nothing}=nothing,
                                     fillbudget::Real=LSQR_FILL_BUDGET, flopbudget::Real=LSQR_FLOP_BUDGET,
                                     transversal::Union{Bool,AbstractVector{<:Integer}}=true,
                                     fname::Symbol=:cover_transversal)
    isfinite(p) || throw(ArgumentError("$fname requires a finite exponent, got PowerMean{$p}"))
    0 < target <= 1 || throw(ArgumentError("$fname: target must satisfy 0 < target <= 1, got $target"))
    tol > 0 || throw(ArgumentError("$fname: tol must be positive, got $tol"))
    maxiter === nothing || maxiter >= 0 ||
        throw(ArgumentError("$fname: maxiter must be nonnegative, got $maxiter"))
    T = float(real(eltype(A)))
    # The tolerances require at least Float64 resolution.
    if eps(T) > eps(Float64)
        a64, b64, nt = _cover_transversal_balanced(convert(AbstractMatrix{promote_type(eltype(A), Float64)}, A),
                                                   p, target; maxbalance, tol, maxiter, fillbudget, flopbudget,
                                                   transversal, fname)
        # Narrowing rounds to nearest and so can round a product below its entry.
        a, b = T.(a64), T.(b64)
        _certify_cover!(a, b, A, fname)
        return a, b, nt
    end
    pT = T(p)
    tp = T(target)^pT
    tp > 0 || throw(ArgumentError("$fname: target^p = $target^$p underflows in $T"))
    axr, axc = axes(A, 1), axes(A, 2)
    or, oc = first(axr) - 1, first(axc) - 1
    nzrow = zeros(Int, length(axr))
    nzcol = zeros(Int, length(axc))
    foreach_support(A) do i, j, v
        nzrow[i-or] += 1
        nzcol[j-oc] += 1
    end
    G = _row_support(A, T)
    ts = _transversal_system(G, axc, T, fname, transversal)
    sys = _contract_system(ts.dsupp, ts.comp, ts.offs)
    maxit = maxiter === nothing ? 100 + 10 * ts.nc : Int(maxiter)
    # The strongly connected components of the entries off the transversal are
    # the fully indecomposable blocks of `A`.
    ptr, arcs = _out_arcs(ts.nc, first.(sys.edges), _ -> true)
    scc, nblocks = _strong_components(ptr, [sys.edges[e][2] for e in arcs])
    if maxbalance
        # Each block's scales are fixed relative to one another by
        # max-balancing; the blocks' common scales minimize the objective over
        # the entries between blocks.
        yb = _maxbalance_components(sys, ts.y0, scc, nblocks)
        bsys = _contract_system(sys, scc, yb)
        Y, nt = _balanced_newton!(_feasible_component_scales(bsys, nblocks), bsys, pT, tp;
                                  tol=T(tol), maxiter=maxit, fillbudget, flopbudget)
        y = Y[scc] .+ yb
    else
        y, nt = _balanced_newton!(copy(ts.y0), sys, pT, tp; tol=T(tol), maxiter=maxit, fillbudget, flopbudget)
    end
    nt = (; nt..., maxbalanced=maxbalance, nblocks)
    _warn_balanced_unconverged(fname, nt, maxit)
    # The uniform shift of `_transversal_scales` absorbs roundoff violations.
    x = _transversal_scales(y[ts.comp] .+ ts.offs, ts.dsupp, ts.σ, ts.ctrans, true)
    a, b = _finish_cover(A, x, nzrow, nzcol, T, fname, true)
    return a, b, nt
end

function _check_cover_axes(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    axes(A, 1) == eachindex(a) || throw(DimensionMismatch("indices of `a` must match row-indexing of `A`, got eachindex(a)=$(string(eachindex(a))), axes(A, 1)=$(string(axes(A, 1)))"))
    axes(A, 2) == eachindex(b) || throw(DimensionMismatch("indices of `b` must match column-indexing of `A`, got eachindex(b)=$(string(eachindex(b))), axes(A, 2)=$(string(axes(A, 2)))"))
    return nothing
end

function _warn_balanced_unconverged(fname::Symbol, nt::NamedTuple, maxiter::Int)
    nt.converged && return nothing
    reason = nt.stalled ?
             "the line search could not reduce the objective; a larger `target` or smaller `p` improves the conditioning of the Newton systems" :
             "it reached maxiter=$maxiter; increase `maxiter`"
    @warn "$fname: the active-set Newton iteration stopped after $(nt.nnewton) Newton steps with relative reduced gradient $(nt.gradnorm) (tolerance $(nt.tol)) because $reason. The result covers `A` and is tight on a maximum-product transversal, but may not minimize the objective."
    return nothing
end

# The objective sums, over the entries off the transversal, `f` of the log slack `d`:
#
#     f(d) = (exp(-p d) - 1)/p + tp*d,   f'(d) = tp - exp(-p d),   f''(d) = p exp(-p d),
#
# with `tp = target^p`; `f` is least at `d = -log(target)`.

# Gradient `g[k]` of the objective with respect to a common shift of the scales
# of the rows `v` with `label[v] == k`, over the entries that join different
# labels when `between` is true and over all entries otherwise; the entries
# `e` with both rows labeled `k` cancel in the former. `f'` is `tp - w` with
# `w = exp(-p d)`, so `g[k] = tp * cnt[k] - (sum of ±w)`, where `cnt[k]` counts
# the entries leaving `k` minus those entering it. Summing the two parts
# separately avoids cancellation between them; `s[k] = tp * |cnt[k]| + (sum of
# w)` bounds the magnitudes summed and so scales the roundoff in `g[k]`.
function _balanced_gradient!(g, s, cnt, d, label, P, Q, p::T, tp::T, between::Bool) where {T}
    fill!(g, zero(T))
    fill!(s, zero(T))
    fill!(cnt, 0)
    for e in eachindex(d)
        G, H = label[P[e]], label[Q[e]]
        between && G == H && continue
        w = exp(-p * d[e])
        g[G] -= w
        g[H] += w
        s[G] += w
        s[H] += w
        cnt[G] += 1
        cnt[H] -= 1
    end
    for k in eachindex(g, s, cnt)
        g[k] += tp * cnt[k]
        s[k] += tp * abs(cnt[k])
    end
    return g
end

# Roundoff allowance for a zero slack of entry `e` at the scales `y`.
_balanced_tight(y, C, P, Q, e) =
    TIGHT_ULPS * eps(eltype(y)) * max(oneunit(eltype(y)), abs(y[P[e]]), abs(y[Q[e]]), abs(C[e]))

# Label the connected components of the graph on `1:nv` formed by the edges
# `(P[e], Q[e])` with `keep(e)`, numbered by their first vertex. Returns the
# number of components.
function _edge_components!(lab::Vector{Int}, keep, P::Vector{Int}, Q::Vector{Int})
    nv = length(lab)
    parent = collect(1:nv)
    function find(v)
        while parent[v] != v
            parent[v] = parent[parent[v]]
            v = parent[v]
        end
        return v
    end
    for e in eachindex(P, Q)
        keep(e) || continue
        rp, rq = find(P[e]), find(Q[e])
        rp == rq || (parent[max(rp, rq)] = min(rp, rq))
    end
    fill!(lab, 0)
    k = 0
    for v in 1:nv
        r = find(v)
        lab[r] == 0 && (lab[r] = (k += 1))
        lab[v] = lab[r]
    end
    return k
end

# Feasibility of a nonnegative flow on the uncapacitated arcs `tails[k] → heads[k]`
# among the vertices `1:nv` whose divergence (outflow minus inflow) at `v` is
# `g[v]`. Solved as a maximum flow from a source joined to each `v` with
# `g[v] > 0` (capacity `g[v]`) to a sink joined from each `v` with `g[v] < 0`
# (capacity `-g[v]`), by shortest augmenting paths (Edmonds–Karp).
#
# Returns `(deficit, S)`: `deficit` is the positive supply the maximum flow
# leaves unrouted, zero exactly when such a flow exists, and `S` marks the
# vertices reachable from the source in the final residual graph. No arc leaves
# `S`, no flow enters it, and `deficit == sum(g[S])` up to roundoff.
function _flow_deficit(nv::Int, tails::Vector{Int}, heads::Vector{Int}, g::AbstractVector{T}) where {T}
    length(g) == nv || throw(DimensionMismatch("g has length $(length(g)) but there are $nv vertices"))
    src, snk = nv + 1, nv + 2
    to = Int[]
    cap = T[]
    from = Int[]
    function addarc!(u, v, c)
        push!(from, u); push!(to, v); push!(cap, c)      # arc `2k-1`
        push!(from, v); push!(to, u); push!(cap, zero(T)) # its reverse, `2k`
    end
    for k in eachindex(tails, heads)
        addarc!(tails[k], heads[k], T(Inf))
    end
    supply = zero(T)
    for v in 1:nv
        if g[v] > 0
            addarc!(src, v, g[v])
            supply += g[v]
        elseif g[v] < 0
            addarc!(v, snk, -g[v])
        end
    end
    # Arcs leaving each node, in compressed form.
    ptr = zeros(Int, nv + 3)
    for u in from
        ptr[u+1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    adj = Vector{Int}(undef, length(from))
    cursor = ptr[1:end-1]
    for (a, u) in enumerate(from)
        adj[cursor[u]] = a
        cursor[u] += 1
    end
    pred = zeros(Int, nv + 2)   # arc reaching each node in the search; 0 = unreached
    queue = Int[]
    flow = zero(T)
    while true
        fill!(pred, 0)
        empty!(queue)
        push!(queue, src)
        pred[src] = -1
        head = 1
        while head <= length(queue) && pred[snk] == 0
            u = queue[head]
            head += 1
            for s in ptr[u]:ptr[u+1]-1
                a = adj[s]
                v = to[a]
                (pred[v] == 0 && cap[a] > 0) || continue
                pred[v] = a
                push!(queue, v)
            end
        end
        pred[snk] == 0 && break
        bottleneck = T(Inf)
        v = snk
        while v != src
            a = pred[v]
            bottleneck = min(bottleneck, cap[a])
            v = from[a]
        end
        v = snk
        while v != src
            a = pred[v]
            cap[a] -= bottleneck
            cap[isodd(a) ? a + 1 : a - 1] += bottleneck
            v = from[a]
        end
        flow += bottleneck
    end
    S = BitVector(pred[v] != 0 for v in 1:nv)
    return max(supply - flow, zero(T)), S
end

# Primal active-set Newton method for
#
#     minimize ∑ₑ f(y[P[e]] - y[Q[e]] - C[e])   subject to   y[P[e]] - y[Q[e]] - C[e] >= 0,
#
# over the component scales `y` of the contracted transversal-tight system `sys`
# (edges `(P[e], Q[e])`, constants `C[e]`), from a feasible `y`, which is
# updated in place. Every iterate is feasible up to roundoff.
#
# Active entries are held at zero slack. Groups are the connected components of
# the active entries; a step moves all scales of a group together, so an entry
# within a group keeps its slack. Newton steps on the group scales use the
# Laplacian of the entries between groups, weighted by `f''`. At a point where
# each group's reduced gradient vanishes, the KKT conditions require
# multipliers `μ >= 0` on the active entries of each group with divergence equal
# to the gradient `g[v]` of the objective at each of its rows. `g[v]` sums `f'`
# over every entry at row `v`: the inactive ones, including those within the
# group, and the active ones, where `f'(0) = tp - 1`. `_flow_deficit` decides
# whether the multipliers exist. If they do not, its set
# `S` has no active entry leaving it and `sum(g[S]) > 0`, so lowering the scales
# of `S` decreases the objective; the active entries entering `S` are released.
#
# Returns `(y, nt)`, with `nt` the fields of `BalancedCoverStats`.
function _balanced_newton!(y::Vector{T}, sys::EdgeList{T}, p::T, tp::T; tol::T, maxiter::Int,
                           fillbudget::Real=LSQR_FILL_BUDGET,
                           flopbudget::Real=LSQR_FLOP_BUDGET) where {T}
    nc = length(y)
    ne = length(sys.edges)
    P = first.(sys.edges)
    Q = last.(sys.edges)
    C = sys.cvals
    d = zeros(T, ne)
    Δd = zeros(T, ne)
    # Entries at zero slack at the start are active; the multiplier test
    # releases those that should not be.
    active = falses(ne)
    for e in 1:ne
        y[P[e]] - y[Q[e]] - C[e] <= _balanced_tight(y, C, P, Q, e) && (active[e] = true)
    end
    gid = zeros(Int, nc)
    wcomp = zeros(Int, nc)
    nw = _edge_components!(wcomp, _ -> true, P, Q)
    gv = zeros(T, nc)    # gradient over the inactive entries, by row
    sv = zeros(T, nc)    # sum of the magnitudes of its terms
    δv = zeros(T, nc)
    gG = T[]
    sG = T[]
    cnt = Int[]
    uidx = Int[]
    ng = 0
    nu = 0
    regroup = repin = true
    cache = Ref{Any}(nothing)   # the sparse factorization and its pattern, for reuse
    nnewton = nfactor = nreleases = iters = 0
    nactivations = count(active)
    converged = stalled = false
    gnorm = T(Inf)
    step = T(NaN)   # largest change of a slack in the last Newton step
    atfloor = false # the last Newton step was below the resolution of the arithmetic
    linsolve = :none
    while true
        if regroup
            ng = _edge_components!(gid, e -> active[e], P, Q)
            resize!(gG, ng)
            resize!(sG, ng)
            resize!(cnt, ng)
            regroup = false
            repin = true
        end
        for e in 1:ne
            d[e] = y[P[e]] - y[Q[e]] - C[e]
        end
        # Reduced gradient: the entries between groups.
        _balanced_gradient!(gG, sG, cnt, d, gid, P, Q, p, tp, true)
        if repin
            # Pin, in each weak component, the group whose gradient has the
            # largest terms; the others are the unknowns of the Newton systems.
            # Once their gradients vanish, the pinned group's is minus their
            # sum, and so holds their roundoff; that is smallest relative to
            # the pinned group's own terms when those are the largest.
            pin = zeros(Int, nw)
            for v in 1:nc
                G, k = gid[v], wcomp[v]
                (pin[k] == 0 || sG[G] > sG[pin[k]]) && (pin[k] = G)
            end
            uidx = ones(Int, ng)
            for k in 1:nw
                uidx[pin[k]] = 0
            end
            nu = 0
            for G in 1:ng
                uidx[G] == 0 || (uidx[G] = (nu += 1))
            end
            repin = false
        end
        gnorm = zero(T)
        for G in 1:ng
            sG[G] > 0 && (gnorm = max(gnorm, abs(gG[G]) / sG[G]))
        end
        if gnorm <= tol || atfloor
            atfloor = false
            # Entries at zero slack may carry multipliers; make them active.
            k = 0
            for e in 1:ne
                if !active[e] && d[e] <= _balanced_tight(y, C, P, Q, e)
                    active[e] = true
                    k += 1
                end
            end
            if k > 0
                (iters += 1) > maxiter && break
                nactivations += k
                regroup = true
                continue
            end
            inS = _balanced_release_set!(gv, sv, d, active, gid, ng, P, Q, p, tp, tol)
            if inS === nothing
                converged = true
                break
            end
            (iters += 1) > maxiter && break
            for e in 1:ne
                active[e] && !inS[P[e]] && inS[Q[e]] && (active[e] = false)
            end
            nreleases += 1
            regroup = true
            for v in 1:nc
                δv[v] = inS[v] ? -one(T) : zero(T)
            end
            status = _balanced_step!(y, δv, d, Δd, active, 1:nc, P, Q, C, p, tp, nothing)
            (status === :stalled || status === :floor) && (stalled = true; break)
            status isa Int && (nactivations += status)
            continue
        end
        (iters += 1) > maxiter && break
        linsolve = _balanced_direction!(δv, cache, d, gid, uidx, nu, gG, P, Q, p, tp,
                                        fillbudget, flopbudget)
        nnewton += 1
        nfactor += 1
        step = zero(T)
        for e in 1:ne
            step = max(step, abs(δv[P[e]] - δv[Q[e]]))
        end
        status = _balanced_step!(y, δv, d, Δd, active, gid, P, Q, C, p, tp, one(T))
        if status === :floor || status === :stalled
            # A Newton step whose slope is zero to roundoff, or that changes no
            # slack by more than `tol` but does not reduce the objective, is at
            # the resolution of the arithmetic: the reduced gradient is zero to
            # roundoff.
            status === :floor || step <= tol || (stalled = true; break)
            atfloor = true
        elseif status isa Int
            nactivations += status
            status > 0 && (regroup = true)
        end
    end
    return y, (; converged, stalled, tol, gradnorm=gnorm, step, nnewton, nfactor, nactivations,
               nreleases, nactive=count(active), linsolve)
end

# The multiplier test at a point where every group's reduced gradient vanishes.
# Returns `nothing` when multipliers exist for every group, and otherwise the
# union of the sets `S` of `_flow_deficit` over the groups that lack them.
function _balanced_release_set!(gv, sv, d, active, gid, ng, P, Q, p::T, tp::T, tol::T) where {T}
    nc = length(gv)
    _balanced_gradient!(gv, sv, zeros(Int, nc), d, 1:nc, P, Q, p, tp, false)
    # Rows and active entries of each group, in compressed form.
    vptr = zeros(Int, ng + 1)
    for v in 1:nc
        vptr[gid[v]+1] += 1
    end
    vptr[1] = 1
    cumsum!(vptr, vptr)
    verts = Vector{Int}(undef, nc)
    local_index = Vector{Int}(undef, nc)
    cursor = vptr[1:ng]
    for v in 1:nc
        G = gid[v]
        verts[cursor[G]] = v
        local_index[v] = cursor[G] - vptr[G] + 1
        cursor[G] += 1
    end
    eptr = zeros(Int, ng + 1)
    for e in eachindex(active)
        active[e] && (eptr[gid[P[e]]+1] += 1)
    end
    eptr[1] = 1
    cumsum!(eptr, eptr)
    aedges = Vector{Int}(undef, eptr[end] - 1)
    cursor = eptr[1:ng]
    for e in eachindex(active)
        active[e] || continue
        G = gid[P[e]]
        aedges[cursor[G]] = e
        cursor[G] += 1
    end
    inS = nothing
    for G in 1:ng
        er = eptr[G]:eptr[G+1]-1
        isempty(er) && continue
        vr = vptr[G]:vptr[G+1]-1
        tails = [local_index[P[aedges[k]]] for k in er]
        heads = [local_index[Q[aedges[k]]] for k in er]
        g = [gv[verts[k]] for k in vr]
        ctol = tol * sum(k -> sv[verts[k]], vr)
        deficit, S = _flow_deficit(length(vr), tails, heads, g)
        deficit <= ctol && continue
        inS === nothing && (inS = falses(nc))
        for (k, v) in enumerate(view(verts, vr))
            S[k] && (inS[v] = true)
        end
    end
    return inS
end

# Newton direction on the group scales, scattered to the rows in `δv`. Returns
# the kind of linear solve.
function _balanced_direction!(δv, cache, d, gid, uidx, nu, gG, P, Q, p::T, tp::T,
                              fillbudget, flopbudget) where {T}
    rhs = zeros(T, nu)
    for G in eachindex(uidx)
        uidx[G] > 0 && (rhs[uidx[G]] = -gG[G])
    end
    nb = 0
    for e in eachindex(d)
        gid[P[e]] == gid[Q[e]] || (nb += 1)
    end
    weight(e) = p * exp(-p * d[e])
    x = zeros(T, nu)
    kind = :none
    if nu == 0
    elseif nu <= BALANCED_DENSE_MAX || 8 * nb >= nu^2
        H = zeros(T, nu, nu)
        for e in eachindex(d)
            G, K = uidx[gid[P[e]]], uidx[gid[Q[e]]]
            gid[P[e]] == gid[Q[e]] && continue
            w = weight(e)
            G > 0 && (H[G, G] += w)
            K > 0 && (H[K, K] += w)
            G > 0 && K > 0 && (H[min(G, K), max(G, K)] -= w)
        end
        dmax = maximum(diag(H))
        dfloor = eps(T) * (dmax > 0 ? dmax : one(T))
        ridge = zero(T)
        for _ in 0:BALANCED_NRIDGE
            Hr = copy(H)
            for k in 1:nu
                Hr[k, k] += ridge * max(H[k, k], dfloor)
            end
            F = LinearAlgebra.cholesky!(Symmetric(Hr, :U); check=false)
            if LinearAlgebra.issuccess(F)
                ldiv!(x, F, rhs)
                kind = :dense
                break
            end
            ridge = _balanced_ridge(ridge)
        end
        kind === :none && throw(LinearAlgebra.PosDefException(1))
    else
        I = Int[]
        J = Int[]
        V = T[]
        dg = zeros(T, nu)
        for e in eachindex(d)
            gid[P[e]] == gid[Q[e]] && continue
            G, K = uidx[gid[P[e]]], uidx[gid[Q[e]]]
            w = weight(e)
            G > 0 && (dg[G] += w)
            K > 0 && (dg[K] += w)
            if G > 0 && K > 0
                push!(I, min(G, K)); push!(J, max(G, K)); push!(V, -w)
            end
        end
        dmax = maximum(dg)
        dfloor = eps(T) * (dmax > 0 ? dmax : one(T))
        ridge = zero(T)
        for _ in 0:BALANCED_NRIDGE
            L = sparse([I; 1:nu], [J; 1:nu], [V; dg .+ ridge .* max.(dg, dfloor)], nu, nu)
            F = _balanced_factor!(cache, L, fillbudget, flopbudget)
            if F !== nothing
                _laplacian_solve!(x, F, rhs)
                kind = F isa SparseCholesky ? :cholesky : :cg
                break
            end
            ridge = _balanced_ridge(ridge)
        end
        kind === :none && throw(LinearAlgebra.PosDefException(1))
    end
    for v in eachindex(δv)
        k = uidx[gid[v]]
        δv[v] = k > 0 ? x[k] : zero(T)
    end
    return kind
end

# The next ridge, relative to the diagonal, for a Newton matrix that roundoff
# made indefinite.
_balanced_ridge(ridge::T) where {T} = iszero(ridge) ? eps(T) : ridge * T(BALANCED_RIDGE_GROWTH)

# Factor the pinned Laplacian `L` (upper triangle), reusing the symbolic analysis
# in `cache` when the pattern is unchanged. Returns `nothing` when `L` is not
# numerically positive definite.
function _balanced_factor!(cache, L::SparseMatrixCSC{Float64,Int}, fillbudget, flopbudget)
    c = cache[]
    try
        if c !== nothing && c[2] == SparseArrays.getcolptr(L) && c[3] == rowvals(L)
            return factorize!(c[1], L)
        end
        F = _laplacian_factor(L; fillbudget, flopbudget)
        cache[] = F isa SparseCholesky ? (F, copy(SparseArrays.getcolptr(L)), copy(rowvals(L))) : nothing
        return F
    catch err
        err isa LinearAlgebra.PosDefException || rethrow()
        return nothing
    end
end
_balanced_factor!(cache, L::SparseMatrixCSC, fillbudget, flopbudget) = _laplacian_factor(L)

# Move the scales `y` along `δv`, which leaves the slack of every active entry
# unchanged, by a step that keeps every slack nonnegative and decreases the
# objective. `α0` is the first trial step (`nothing` for the minimizer of the
# quadratic model along `δv`), reduced by the ratio test. Returns the number of
# entries made active (an `Int`, possibly zero), `:floor` when the slope along
# `δv` is nonnegative but within the roundoff of the gradients of the labels, or
# `:stalled` when no step decreases the objective. Inactive entries at zero
# slack that `δv` would make negative are made active without moving. `δv` is
# constant on the sets of rows with equal `label`.
function _balanced_step!(y, δv, d, Δd, active, label, P, Q, C, p::T, tp::T, α0) where {T}
    # `f'` and `f(d + h) - f(d)` split into the parts in `tp` and in `exp(-p d)`,
    # summed separately: the first is `tp` times the sum `sΔ` of the slack
    # changes. As in `_balanced_gradient!`, `sΔ` sums `δv` on each label times
    # the count of inactive entries leaving it minus those entering it, so that
    # the changes of a label with as many entries leaving as entering cancel
    # exactly; a sum by entry leaves a rounding error proportional to their size.
    δl = zeros(T, length(δv))
    cnt = zeros(Int, length(δv))
    for v in eachindex(δv)
        δl[label[v]] = δv[v]
    end
    for v in eachindex(δv)
        δv[v] == δl[label[v]] || error("internal error: the step is not constant on a label")
    end
    sw = gscale = curv = zero(T)
    αmax = T(Inf)
    Δmax = zero(T)
    nblock = 0
    for e in eachindex(d)
        Δd[e] = δv[P[e]] - δv[Q[e]]
        if active[e]
            iszero(Δd[e]) || error("internal error: the step changes the slack of an active entry")
            continue
        end
        Δ = Δd[e]
        w = exp(-p * d[e])
        cnt[label[P[e]]] += 1
        cnt[label[Q[e]]] -= 1
        sw += w * Δ
        label[P[e]] == label[Q[e]] || (gscale += (tp + w) * (abs(δv[P[e]]) + abs(δv[Q[e]])))
        curv += p * w * Δ^2
        Δmax = max(Δmax, abs(Δ))
        if Δ < 0
            if d[e] <= _balanced_tight(y, C, P, Q, e)
                nblock += 1
            else
                αmax = min(αmax, d[e] / -Δ)
            end
        end
    end
    if nblock > 0
        for e in eachindex(d)
            if !active[e] && Δd[e] < 0 && d[e] <= _balanced_tight(y, C, P, Q, e)
                active[e] = true
            end
        end
        return nblock
    end
    sΔ = sΔabs = zero(T)
    for k in eachindex(cnt, δl)
        cnt[k] == 0 && continue
        sΔ += cnt[k] * δl[k]
        sΔabs += abs(cnt[k] * δl[k])
    end
    slope = tp * sΔ - sw
    if slope >= 0
        # The slope sums the gradient of each label times its change, and that
        # gradient is known only to within the roundoff of its terms; a slope
        # within the sum of that roundoff does not resolve the direction.
        slope <= BALANCED_NOISE * eps(T) * gscale && return :floor
        return :stalled
    end
    α = α0 === nothing ? -slope / curv : T(α0)
    α = min(α, αmax, BALANCED_MAXSTEP / (p * Δmax))
    for _ in 1:BALANCED_NBACKTRACK
        ΔF = scale = sw1 = zero(T)
        for e in eachindex(d)
            active[e] && continue
            h = α * Δd[e]
            iszero(h) && continue
            w = exp(-p * d[e])
            q = w * expm1(-p * h) / p
            ΔF += q
            scale += abs(q)
            sw1 += w * exp(-p * h) * Δd[e]
        end
        ΔF += tp * α * sΔ
        scale += tp * α * sΔabs
        slope1 = tp * sΔ - sw1
        accepted = ΔF <= BALANCED_ARMIJO * α * slope ||
                   # The objective cannot resolve this step: accept it if the
                   # directional derivative shrinks.
                   (abs(ΔF) <= BALANCED_NOISE * eps(T) * scale && abs(slope1) < abs(slope))
        if accepted
            y .+= α .* δv
            α == αmax || return 0
            k = 0
            for e in eachindex(d)
                active[e] && continue
                if Δd[e] < 0 && (d[e] / -Δd[e] == αmax ||
                                 y[P[e]] - y[Q[e]] - C[e] <= _balanced_tight(y, C, P, Q, e))
                    active[e] = true
                    k += 1
                end
            end
            return k
        end
        α /= 2
    end
    return :stalled
end
