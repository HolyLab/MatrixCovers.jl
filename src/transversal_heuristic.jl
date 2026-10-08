# Transversal-tight hard covers for `cover` on square, structurally nonsingular
# matrices.
#
# Write α = log a and β = log b. Every hard cover has `∑α + ∑β >= log π*`, with
# equality exactly when it is tight on every entry of every maximum-product
# transversal (see `cover_transversal`). Tightness on one such transversal `σ`
# fixes `β[σ[i]] = log|A[i,σ[i]]| - α[i]`, which leaves a difference system on
# the row scales (see `_cover_min_abslog2`). The rows that `_tight_components`
# merges are tight on one another in every transversal-tight cover, so each of
# these groups `g` carries a single unknown `u[g]`, measured from the
# transversal duals `α0`: `α[i] = α0[i] + u[g(i)]`. An entry joining group `h`
# to group `t` then requires `u[h] - u[t] >= d`, where `-d >= 0` is its slack
# under the duals; entries within a group are covered for every `u`. The groups
# depend on the set of maximum-product transversals, not on which one the
# matching returns.
#
# The heuristic approximately minimizes `∑ (u[h] - u[t] - d)²` subject to these
# constraints:
#   1. a layer-average start (`_tt_layer_start!`);
#   2. Jacobi-preconditioned CG iterations on the unconstrained least-squares
#      problem (`_tt_cg!`);
#   3. a deficit-splitting boost (`_tt_boost!`), then the least feasible point
#      above the result (`_tt_raise!`);
#   4. projected Gauss–Seidel sweeps within each unknown's feasible interval
#      (`_tt_sweep!`).
# The constants `d` are slacks, which are unchanged by rescaling `A`, and each
# stage commutes with a uniform shift of `u` on a support component, so the
# cover products are scale-covariant.

# Group graph in compressed form. Each constraint `u[h] - u[t] >= d` occupies a
# slot of `h` (`nbr = t`, `off = d`, `lower = true`) and a slot of `t`
# (`nbr = h`, `off = -d`, `lower = false`). Slot `s` of `v` asks for
# `u[v] ≈ u[nbr[s]] + off[s]`, as a lower bound when `lower[s]` and as an upper
# bound otherwise.
struct _GroupGraph{T}
    ptr::Vector{Int}
    nbr::Vector{Int}
    off::Vector{T}
    lower::Vector{Bool}
end

_tt_slots(Gg::_GroupGraph, v) = Gg.ptr[v]:Gg.ptr[v+1]-1

function _GroupGraph(nc::Int, eh::Vector{Int}, et::Vector{Int}, ed::Vector{T}) where {T}
    ptr = zeros(Int, nc + 1)
    for e in eachindex(eh, et)
        ptr[eh[e]+1] += 1
        ptr[et[e]+1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    ns = ptr[end] - 1
    nbr = Vector{Int}(undef, ns)
    off = Vector{T}(undef, ns)
    lower = Vector{Bool}(undef, ns)
    cursor = ptr[1:nc]
    for e in eachindex(eh, et, ed)
        h, t = eh[e], et[e]
        s = cursor[h]
        nbr[s], off[s], lower[s] = t, ed[e], true
        cursor[h] = s + 1
        s = cursor[t]
        nbr[s], off[s], lower[s] = h, -ed[e], false
        cursor[t] = s + 1
    end
    return _GroupGraph{T}(ptr, nbr, off, lower)
end

# Root each connected component at its highest-degree vertex (lowest index on
# ties), with value zero. Every other vertex averages `u[w] + off` over its
# neighbors `w` in the preceding breadth-first layer. Fills `gcomp` with
# component labels and returns their count.
function _tt_layer_start!(u::Vector{T}, gcomp::Vector{Int}, Gg::_GroupGraph{T}) where {T}
    nc = length(u)
    deg = [length(_tt_slots(Gg, v)) for v in 1:nc]
    order = sortperm(deg; rev=true, alg=MergeSort)   # stable: ties stay in index order
    dist = fill(-1, nc)
    queue = Vector{Int}(undef, nc)
    head, qend = 1, 0
    ncomp = 0
    fill!(u, zero(T))
    for r in order
        dist[r] == -1 || continue
        ncomp += 1
        dist[r] = 0
        gcomp[r] = ncomp
        qend += 1
        queue[qend] = r
        # Vertices leave the queue in order of distance, so the preceding layer
        # of each vertex has been assigned by the time it is reached.
        while head <= qend
            v = queue[head]
            head += 1
            if dist[v] > 0
                tot, cnt = zero(T), 0
                for s in _tt_slots(Gg, v)
                    w = Gg.nbr[s]
                    if dist[w] == dist[v] - 1
                        tot += u[w] + Gg.off[s]
                        cnt += 1
                    end
                end
                u[v] = tot / cnt
            end
            for s in _tt_slots(Gg, v)
                w = Gg.nbr[s]
                if dist[w] == -1
                    dist[w] = dist[v] + 1
                    gcomp[w] = ncomp
                    qend += 1
                    queue[qend] = w
                end
            end
        end
    end
    return ncomp
end

# `y = L x` for the graph Laplacian `L` of the group graph.
function _tt_laplacian!(y::Vector{T}, Gg::_GroupGraph{T}, x::Vector{T}) where {T}
    for v in eachindex(y, x)
        acc = zero(T)
        xv = x[v]
        for s in _tt_slots(Gg, v)
            acc += xv - x[Gg.nbr[s]]
        end
        y[v] = acc
    end
    return y
end

# `dot(x, y)` without BLAS, whose threading overhead dominates at the vector
# lengths seen here.
function _tt_dot(x::Vector{T}, y::Vector{T}) where {T}
    acc = zero(T)
    for i in eachindex(x, y)
        acc += x[i] * y[i]
    end
    return acc
end

# Up to `maxiter` Jacobi-preconditioned CG iterations on the normal equations
# `L u = rhs` of `∑ (u[h] - u[t] - d)²`, stopping once the residual, measured
# in the inverse of the preconditioner, falls below a fixed fraction of `rhs`.
# The preconditioner is the diagonal of `L` (the vertex degrees). Degrees can
# span orders of magnitude, and without it truncated CG amplifies roundoff
# enough to break scale-covariance; it depends only on the support, so the
# iteration remains covariant in exact arithmetic. `L` is singular, with each
# component's constants in its null space; the system is consistent, and the
# iterates keep each component's degree-weighted sum of `u`.
function _tt_cg!(u::Vector{T}, Gg::_GroupGraph{T}, maxiter::Int) where {T}
    maxiter == 0 && return u
    nc = length(u)
    rhs = zeros(T, nc)
    dinv = zeros(T, nc)   # inverse degrees; zero on isolated groups
    for v in 1:nc
        slots = _tt_slots(Gg, v)
        isempty(slots) || (dinv[v] = inv(T(length(slots))))
        for s in slots
            rhs[v] += Gg.off[s]
        end
    end
    r = similar(u)
    _tt_laplacian!(r, Gg, u)
    r .= rhs .- r
    z = dinv .* r
    p = copy(z)
    q = similar(u)
    rz = _tt_dot(r, z)
    # A test relative to `rhs` stays meaningful once the iterates have converged;
    # continuing past that point amplifies roundoff.
    tol = eps(T)^(T(3) / 2) * _tt_dot(rhs, dinv .* rhs)
    for _ in 1:maxiter
        rz <= tol && break
        _tt_laplacian!(q, Gg, p)
        pq = _tt_dot(p, q)
        pq > zero(T) || break
        γ = rz / pq
        u .+= γ .* p
        r .-= γ .* q
        z .= dinv .* r
        rznew = _tt_dot(r, z)
        p .= z .+ (rznew / rz) .* p
        rz = rznew
    end
    return u
end

# Split each constraint's deficit `-(u[h] - u[t] - d)` between raising `u[h]`
# and lowering `u[t]` in the ratio `sqrt(s[h]) : sqrt(r[t])`, where `s` and `r`
# total the deficits on each vertex's lower and upper bounds; every vertex then
# moves by its largest share in each direction. This mirrors `boost_feasible!`
# and need not reach feasibility, because a vertex can both rise and fall.
function _tt_boost!(u::Vector{T}, Gg::_GroupGraph{T}) where {T}
    nc = length(u)
    s = zeros(T, nc)
    r = zeros(T, nc)
    for v in 1:nc, k in _tt_slots(Gg, v)
        Gg.lower[k] || continue
        z = u[Gg.nbr[k]] + Gg.off[k] - u[v]
        z > zero(T) || continue
        s[v] += z
        r[Gg.nbr[k]] += z
    end
    up = zeros(T, nc)
    dn = zeros(T, nc)
    for v in 1:nc, k in _tt_slots(Gg, v)
        Gg.lower[k] || continue
        t = Gg.nbr[k]
        z = u[t] + Gg.off[k] - u[v]
        z > zero(T) || continue
        wh, wt = sqrt(s[v]), sqrt(r[t])
        up[v] = max(up[v], z * wh / (wh + wt))
        dn[t] = max(dn[t], z * wt / (wh + wt))
    end
    u .+= up .- dn
    return u
end

# Raise `u` to the least point above it satisfying every constraint: each
# vertex rises to the largest `u[s] + (sum of d along a path from s)` over all
# vertices `s`. Every `d <= 0`, so this is a shortest-path problem in `-u` with
# nonnegative weights, solved by Dijkstra from every vertex at once: vertices
# are settled in decreasing order of `u`, and when `u[v]` is settled, each `h`
# with a constraint `u[h] - u[v] >= d` rises to `u[v] + d` if needed. Rounding
# is monotone, so `u[v] + d <= u[v]` holds in floating point as well.
function _tt_raise!(u::Vector{T}, Gg::_GroupGraph{T}) where {T}
    heap = Tuple{T,Int}[]
    sizehint!(heap, length(u))
    for v in eachindex(u)
        _heappush!(heap, (-u[v], v))
    end
    while !isempty(heap)
        key, v = _heappop!(heap)
        key == -u[v] || continue    # superseded by a later raise
        uv = u[v]
        for k in _tt_slots(Gg, v)
            Gg.lower[k] && continue
            h = Gg.nbr[k]
            x = uv - Gg.off[k]
            if x > u[h]
                u[h] = x
                _heappush!(heap, (-x, h))
            end
        end
    end
    return u
end

# Projected Gauss–Seidel sweeps on `∑ (u[h] - u[t] - d)²`: each vertex moves to
# the mean of its targets, clamped to its feasible interval. When rounding
# empties the interval, the lower bound wins, which keeps the vertex's own
# entries covered.
function _tt_sweep!(u::Vector{T}, Gg::_GroupGraph{T}, sweeps::Int) where {T}
    for _ in 1:sweeps, v in eachindex(u)
        sl = _tt_slots(Gg, v)
        isempty(sl) && continue
        lo, hi, tot = T(-Inf), T(Inf), zero(T)
        for k in sl
            x = u[Gg.nbr[k]] + Gg.off[k]
            tot += x
            if Gg.lower[k]
                lo = max(lo, x)
            else
                hi = min(hi, x)
            end
        end
        u[v] = max(min(tot / length(sl), hi), lo)
    end
    return u
end

# Write the transversal-tight heuristic cover of the square matrix `A` into `a`
# and `b`, balanced per support component as in `cover_min` but not yet
# certified. Returns `false`, leaving `a` and `b` unchanged, when `A` has no
# transversal of nonzero entries.
function _cover_transversal_heuristic!(a::AbstractVector, b::AbstractVector, A::AbstractMatrix,
                                       ::Type{T}, cgiter::Int, sweeps::Int, fname::Symbol) where {T}
    axr, axc = axes(A, 1), axes(A, 2)
    n = length(axr)
    oc = first(axc) - 1
    G = _row_support(A, T)
    mpt = _max_product_transversal(G, axc, fname; throw_singular=false)
    mpt === nothing && return false
    σ, α0, _, _ = mpt
    τ = invperm(σ)
    # Log magnitudes of the transversal entries, by column.
    ctrans = Vector{T}(undef, n)
    nzrow = zeros(Int, n)
    nzcol = zeros(Int, n)
    for (ip, i) in enumerate(axr)
        for s in _slots(G, i)
            jp = G.idx[s] - oc
            jp == σ[ip] && (ctrans[jp] = log(G.val[s]))
            nzrow[ip] += 1
            nzcol[jp] += 1
        end
    end
    # Difference system on the row scales: entry `(i, j)` off the transversal
    # requires `α[i] - α[τ[j]] >= log|A[i,j]| - ctrans[j]`.
    edges = Tuple{Int,Int}[]
    cvals = T[]
    for (ip, i) in enumerate(axr)
        for s in _slots(G, i)
            jp = G.idx[s] - oc
            jp == σ[ip] && continue
            push!(edges, (ip, τ[jp]))
            push!(cvals, log(G.val[s]) - ctrans[jp])
        end
    end
    comp, _, nc = _tight_components(edges, cvals, α0)
    # Number the groups by their first row. Labels from `_tight_components`
    # follow the search order over near-tight entries, which roundoff can
    # change under rescaling; the start's tie-breaking and the sweep order
    # depend on the labels.
    relabel = zeros(Int, nc)
    k = 0
    for ip in 1:n
        c = comp[ip]
        relabel[c] == 0 && (relabel[c] = (k += 1))
        comp[ip] = relabel[c]
    end
    # Constraints between groups. `_tt_raise!` requires every `d <= 0`, which
    # holds up to roundoff in the duals; the clamp makes it exact.
    eh, et, ed = Int[], Int[], T[]
    for (e, (p, q)) in enumerate(edges)
        comp[p] == comp[q] && continue
        push!(eh, comp[p])
        push!(et, comp[q])
        push!(ed, min(zero(T), cvals[e] - α0[p] + α0[q]))
    end
    Gg = _GroupGraph(nc, eh, et, ed)
    u = Vector{T}(undef, nc)
    gcomp = Vector{Int}(undef, nc)
    ncomp = _tt_layer_start!(u, gcomp, Gg)
    _tt_cg!(u, Gg, cgiter)
    _tt_boost!(u, Gg)
    _tt_raise!(u, Gg)
    _tt_sweep!(u, Gg, sweeps)
    lα = [α0[ip] + u[comp[ip]] for ip in 1:n]
    lβ = [ctrans[jp] - lα[τ[jp]] for jp in 1:n]
    # Balance each support component; the groups of a component of the group
    # graph hold the rows and columns of one support component.
    Lα = zeros(T, ncomp)
    Lβ = zeros(T, ncomp)
    cnt = zeros(Int, ncomp)
    for ip in 1:n
        c = gcomp[comp[ip]]
        Lα[c] += nzrow[ip] * lα[ip]
        cnt[c] += nzrow[ip]
    end
    for jp in 1:n
        Lβ[gcomp[comp[τ[jp]]]] += nzcol[jp] * lβ[jp]
    end
    for ip in 1:n
        c = gcomp[comp[ip]]
        lα[ip] += (Lβ[c] - Lα[c]) / (2 * cnt[c])
    end
    for jp in 1:n
        c = gcomp[comp[τ[jp]]]
        lβ[jp] -= (Lβ[c] - Lα[c]) / (2 * cnt[c])
    end
    _check_representable(lα, _ -> true, lβ, _ -> true, T, fname; gauge=false)
    for (ip, i) in enumerate(axr)
        a[i] = exp(lα[ip])
    end
    for (jp, j) in enumerate(axc)
        b[j] = exp(lβ[jp])
    end
    return true
end
