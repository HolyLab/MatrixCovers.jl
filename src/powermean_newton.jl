# Newton refinement for the `PowerMean` soft covers.
#
# Over stacked log scales `x` (asymmetric: rows in `1:m`, columns in `m+1:m+n`;
# symmetric: one scale per row), the objective has one term per stored entry `e`
# joining unknowns `u[e] <= v[e]`:
#
#     E(x) = ∑_e mult[e] * ρ(x[u] + x[v] - L[e]),   ρ(z) = (exp(-p z) - 1 + p z)/p,
#
# with `L = log|A|`. An asymmetric entry has multiplicity one. A symmetric
# off-diagonal pair is stored once with multiplicity two and a diagonal entry once
# with multiplicity one, which is the full-grid sum of `cover_objective`. With
# weights `W[e] = exp(-p z[e])` (the coverage ratios to the power `p`),
#
#     ∇E = ∑_e mult[e] * (1 - W[e]) * (e_u + e_v),
#     ∇²E = p * ∑_e mult[e] * W[e] * (e_u + e_v) * (e_u + e_v)',
#
# i.e. `(n - W e, m - Wᵀ e)` and `p [diag(W e) W; Wᵀ diag(Wᵀ e)]` in the
# asymmetric case and `2(n - W e)` and `2p [diag(W e) + W]` in the symmetric one.
#
# The Hessian is singular exactly on the components of the support graph that are
# bipartite with no diagonal entry (every asymmetric component), with null vector
# `±1` on the two sides; the gradient is orthogonal to it. Each such component is
# grounded: one unknown's step is fixed at zero and its equation dropped, which
# leaves the rest of the system positive definite and consistent. The gauge of the
# result is fixed afterwards by the drivers' balance conventions.
#
# The asymmetric problem can carry the terms of `PowerMeanTerms`. Targets `t` in
# place of the counts `n` add `∑_q (t[q] - n[q]) x[q]` to `E`, which changes the
# gradient to `t - W e` but not the Hessian; the gradient stays orthogonal to the
# null vectors when the targets of each component's rows and columns have equal
# sums, as they do whenever the balance conditions have a solution. For `p = 1`, a
# rank-one term adds `κ (∑_i u[i] exp(-x[i])) (∑_j v[j] exp(-x[m+j]))`: with the
# weights `A[i] = u[i] exp(-x[i])` and `B[j] = v[j] exp(-x[m+j])` and their sums
# `SA` and `SB`, it adds `κ A SB` and `κ B SA` to the row and column sums and the
# diagonal of the Hessian, and `κ A Bᵀ` to its off-diagonal block. That block is
# dense, so it joins every row to every column: the whole problem is then one
# component, grounded at a single unknown.

struct PowerMeanSystem{T}
    u::Vector{Int}
    v::Vector{Int}
    L::Vector{T}
    mult::Vector{T}
    count::Vector{Int}      # stored entries in each unknown's row or column
    grid::Int               # number of cells of the full grid
    nrow::Int               # rows of the asymmetric problem (all unknowns for the symmetric one)
    target::Vector{T}       # target of each unknown's sum; empty when it is `count`
    lw::Vector{T}           # log rank-one weights; empty without the rank-one term
    lκ::T
end

PowerMeanSystem{T}(u, v, L, mult, count, grid, nrow) where {T} =
    PowerMeanSystem{T}(u, v, L, mult, count, grid, nrow, T[], T[], zero(T))

_powermean_hasterms(sys::PowerMeanSystem) = !(isempty(sys.target) && isempty(sys.lw))

# System of the asymmetric problem, from row-grouped `log|A|` and the column axis,
# with the additional `terms` if given.
function _powermean_system(R::GroupedSupport{T}, colax::AbstractUnitRange, ::Nothing) where {T}
    return _powermean_system(R, colax)
end

function _powermean_system(R::GroupedSupport{T}, colax::AbstractUnitRange, terms::PowerMeanTerms{T}) where {T}
    sys = _powermean_system(R, colax)
    return PowerMeanSystem{T}(sys.u, sys.v, sys.L, sys.mult, sys.count, sys.grid, sys.nrow,
                              exp.(terms.lt), terms.lw, terms.lκ)
end

function _powermean_system(R::GroupedSupport{T}, colax::AbstractUnitRange) where {T}
    m, n = length(R.ax), length(colax)
    oc = first(colax) - 1
    ne = length(R.val)
    u = Vector{Int}(undef, ne)
    v = Vector{Int}(undef, ne)
    L = Vector{T}(undef, ne)
    count = zeros(Int, m + n)
    e = 0
    for (ip, i) in enumerate(R.ax)      # `ip` is the position of row `i`
        for s in _slots(R, i)
            e += 1
            jp = m + R.idx[s] - oc
            u[e], v[e], L[e] = ip, jp, R.val[s]
            count[ip] += 1
            count[jp] += 1
        end
    end
    return PowerMeanSystem{T}(u, v, L, ones(T, ne), count, m * n, m)
end

# System of the symmetric problem, from the symmetric row-grouped `log|A|`.
function _powermean_symsystem(S::GroupedSupport{T}) where {T}
    off = first(S.ax) - 1
    n = length(S.ax)
    u, v, L, mult = Int[], Int[], T[], T[]
    count = zeros(Int, n)
    for g in S.ax
        gp = g - off
        count[gp] = _ngroup(S, g)
        for s in _slots(S, g)
            hp = S.idx[s] - off
            hp >= gp || continue
            push!(u, gp); push!(v, hp); push!(L, S.val[s]); push!(mult, hp == gp ? 1 : 2)
        end
    end
    return PowerMeanSystem{T}(u, v, L, mult, count, n * n, n)
end

# Unknowns whose step is fixed at zero: those without support, and one unknown (of
# largest count) in each component that is bipartite with no diagonal entry. With
# the rank-one term, the supported unknowns form a single such component.
function _powermean_pinned(sys::PowerMeanSystem)
    if !isempty(sys.lw)
        pinned = sys.count .== 0
        pinned[argmax(sys.count)] = true
        return pinned
    end
    N = length(sys.count)
    ptr = zeros(Int, N + 1)
    for e in eachindex(sys.u, sys.v)
        ptr[sys.u[e]+1] += 1
        ptr[sys.v[e]+1] += 1
    end
    ptr[1] = 1
    cumsum!(ptr, ptr)
    nbr = Vector{Int}(undef, ptr[end] - 1)
    cursor = ptr[1:N]
    for e in eachindex(sys.u, sys.v)
        a, b = sys.u[e], sys.v[e]
        nbr[cursor[a]] = b; cursor[a] += 1
        nbr[cursor[b]] = a; cursor[b] += 1
    end
    pinned = sys.count .== 0
    color = zeros(Int8, N)
    stack = Int[]
    for r in 1:N
        (color[r] != 0 || sys.count[r] == 0) && continue
        color[r] = 1
        push!(stack, r)
        bipartite = true
        best = r
        while !isempty(stack)
            q = pop!(stack)
            sys.count[q] > sys.count[best] && (best = q)
            for k in ptr[q]:ptr[q+1]-1
                h = nbr[k]
                if color[h] == 0
                    color[h] = -color[q]
                    push!(stack, h)
                elseif color[h] == color[q]
                    bipartite = false       # an odd cycle or a diagonal entry
                end
            end
        end
        bipartite && (pinned[best] = true)
    end
    return pinned
end

# Evaluate at `xt = x + t*d`: the weights `W`, gradient `g` and row sums `s` of the
# weights there, and the change of the objective from `x`, whose weights are `W0`.
# Returns the change and the sum of the magnitudes of its terms, the scale of its
# roundoff. Each term is `mult * (W0 * expm1(-p δ) + p δ)/p` with `δ = t*(d[u] + d[v])`,
# exact for the difference and free of the cancellation of subtracting two objectives.
#
# `Wr` and `Wr0` hold the rank-one weights `A` and `B` (see the header) at `xt` and
# `x`; they are unused without the rank-one term.
function _powermean_eval!(W, Wr, g, s, xt, d, t::T, W0, Wr0, sys::PowerMeanSystem{T}, p::T) where {T}
    fill!(g, zero(T))
    fill!(s, zero(T))
    ΔE = scale = zero(T)
    for e in eachindex(sys.u, sys.v, sys.L, sys.mult, W, W0)
        a, b, w = sys.u[e], sys.v[e], sys.mult[e]
        δ = t * (d[a] + d[b])
        q = W0[e] * expm1(-p * δ)
        ΔE += w * (q + p * δ)
        scale += w * (abs(q) + abs(p * δ))
        We = exp(-p * (xt[a] + xt[b] - sys.L[e]))
        W[e] = We
        g[a] += w * (1 - We)
        g[b] += w * (1 - We)
        s[a] += We
        a == b || (s[b] += We)
    end
    _powermean_hasterms(sys) && ((ΔE, scale) = _powermean_eval_terms!(Wr, g, s, xt, d, t, Wr0, sys, p, ΔE, scale))
    return ΔE / p, scale / p
end

# The targets' linear term and the rank-one term (see the header) for
# `_powermean_eval!`: added to its gradient `g`, sums `s`, and the change `ΔE` of
# `p` times the objective and its roundoff scale, which are returned.
function _powermean_eval_terms!(Wr, g, s, xt, d, t::T, Wr0, sys::PowerMeanSystem{T}, p::T, ΔE::T, scale::T) where {T}
    if !isempty(sys.target)
        for q in eachindex(g, d, sys.target, sys.count)
            sys.count[q] == 0 && continue
            c = sys.target[q] - sys.count[q]
            g[q] += c
            δ = p * c * t * d[q]
            ΔE += δ
            scale += abs(δ)
        end
    end
    isempty(sys.lw) && return ΔE, scale
    # The rank-one term exists only for `p = 1`.
    m = sys.nrow
    κ = exp(sys.lκ)
    SA = SB = SA0 = SB0 = ΔA = ΔB = zero(T)
    for q in eachindex(Wr, Wr0, xt, d, sys.lw)
        w0 = Wr0[q]
        w = exp(sys.lw[q] - xt[q])
        Wr[q] = w
        Δ = w0 * expm1(-t * d[q])
        if q <= m
            SA += w; SA0 += w0; ΔA += Δ
        else
            SB += w; SB0 += w0; ΔB += Δ
        end
    end
    for q in eachindex(Wr, g, s)
        h = κ * Wr[q] * (q <= m ? SB : SA)
        s[q] += h
        g[q] -= h
    end
    # `SA*SB - SA0*SB0` without cancellation.
    ΔE += κ * (ΔA * SB0 + SA0 * ΔB + ΔA * ΔB)
    scale += κ * (abs(ΔA) * SB0 + SA0 * abs(ΔB) + abs(ΔA * ΔB))
    return ΔE, scale
end

# Largest `|s/count - 1|`: the imbalance of the sweeps.
function _powermean_newton_imbalance(s, count)
    imb = zero(eltype(s))
    for q in eachindex(s, count)
        count[q] > 0 && (imb = max(imb, abs(s[q] / count[q] - 1)))
    end
    return imb
end

# The same with the targets of `sys`, if it has them, in place of the counts.
function _powermean_newton_imbalance(s, sys::PowerMeanSystem)
    isempty(sys.target) && return _powermean_newton_imbalance(s, sys.count)
    imb = zero(eltype(s))
    for q in eachindex(s, sys.count, sys.target)
        sys.count[q] > 0 && (imb = max(imb, abs(s[q] / sys.target[q] - 1)))
    end
    return imb
end

# Inner linear solvers for the Newton step:
#
# - `:dense` factorizes the dense Hessian (Bunch-Kaufman for `Float64`, LU otherwise);
# - `:cholesky` factorizes the sparse Hessian with CHOLMOD (`Float64` only), with
#   one symbolic analysis for all steps;
# - `:cg` runs Jacobi-preconditioned conjugate gradients with one pass over the
#   support per product, to the relative residual `min(1/2, sqrt(imbalance))`.
#
# `:auto` uses the dense factorization on support that fills at least
# `AUTO_LSQR_MAX_DENSITY` of the grid, otherwise the sparse one when its predicted
# storage and flop count fit the budgets of the LSQR preconditioner
# (`LSQR_FILL_BUDGET`, `LSQR_FLOP_BUDGET`); a factorization whose predicted flop
# count exceeds `LSQR_FLOP_BUDGET` times the number of stored entries gives way to CG.
struct PowerMeanLinsolve{T,F}
    kind::Symbol
    H::Matrix{T}                    # `:dense`
    M::SparseMatrixCSC{Float64,Int} # `:cholesky`: upper triangle of the Hessian
    F::F
    epos::Vector{Int}               # position in `nonzeros(M)` of each off-diagonal entry
    dpos::Vector{Int}               # position of each diagonal entry
    dg::Vector{T}                   # diagonal of the Hessian
    f::Vector{T}                    # right-hand side
    work::NTuple{4,Vector{T}}       # CG vectors
end

function _powermean_linsolve(sys::PowerMeanSystem{T}, linsolve::Symbol) where {T}
    N = length(sys.count)
    nstored = sum(sys.mult; init=zero(T))
    dense_ok = N^3 / 3 <= LSQR_FLOP_BUDGET * nstored
    kind = linsolve
    # The rank-one term's dense off-diagonal block has no sparse factorization;
    # conjugate gradients apply it as a product instead.
    if !isempty(sys.lw) && kind !== :dense
        kind = kind === :auto && dense_ok ? :dense : :cg
    end
    M = spzeros(Float64, 0, 0)
    F = nothing
    if kind === :cholesky || (kind === :auto && T === Float64 && nstored < AUTO_LSQR_MAX_DENSITY * sys.grid)
        T === Float64 ||
            throw(ArgumentError("linsolve=:cholesky requires Float64 arithmetic, but the problem works in $T; use :dense or :cg"))
        M = _powermean_pattern(sys)
        F, entries, flops = _precond_analysis(M)
        if kind === :auto
            kind = sizeof(T) * entries <= LSQR_FILL_BUDGET && flops <= LSQR_FLOP_BUDGET * nstored ? :cholesky :
                   dense_ok ? :dense : :cg
        end
    elseif kind === :auto
        kind = dense_ok ? :dense : :cg
    end
    H = kind === :dense ? zeros(T, N, N) : zeros(T, 0, 0)
    epos = zeros(Int, kind === :cholesky ? length(sys.u) : 0)
    # `M` is reassigned above, so the comprehension captures a binding made once.
    Mp = M
    dpos = kind === :cholesky ? [_nzindex(Mp, q, q) for q in 1:N] : Int[]
    if kind === :cholesky
        for e in eachindex(sys.u, sys.v)
            a, b = sys.u[e], sys.v[e]
            a == b || (epos[e] = _nzindex(M, a, b))
        end
    end
    nw = kind === :cg ? N : 0
    work = (zeros(T, nw), zeros(T, nw), zeros(T, nw), zeros(T, nw))
    return PowerMeanLinsolve{T,typeof(F)}(kind, H, M, F, epos, dpos, zeros(T, N), zeros(T, N), work)
end

# Upper-triangular pattern of the Hessian, rows sorted within each column.
function _powermean_pattern(sys::PowerMeanSystem)
    N = length(sys.count)
    I = collect(1:N)
    J = collect(1:N)
    for e in eachindex(sys.u, sys.v)
        sys.u[e] == sys.v[e] && continue
        push!(I, sys.u[e])
        push!(J, sys.v[e])
    end
    return sparse(I, J, ones(Float64, length(I)), N, N)
end

# Sums of the row and column rank-one weights `A` and `B`.
function _powermean_rankone_sums(Wr::AbstractVector{T}, m::Int) where {T}
    SA = SB = zero(T)
    for q in eachindex(Wr)
        q <= m ? (SA += Wr[q]) : (SB += Wr[q])
    end
    return SA, SB
end

# Solve `H d = -g` with the pinned unknowns' steps fixed at zero, `Wr` holding the
# rank-one weights if `sys` has the rank-one term. Returns the number of passes
# over the support and of factorizations.
function _powermean_direction!(d, ls::PowerMeanLinsolve{T}, sys::PowerMeanSystem{T}, W, Wr, g,
                               pinned, imb::T, p::T) where {T}
    N = length(d)
    f, dg = ls.f, ls.dg
    for q in eachindex(f, g, pinned)
        f[q] = pinned[q] ? zero(T) : -g[q]
    end
    fill!(dg, zero(T))
    for e in eachindex(sys.u, sys.v, W)
        a, b = sys.u[e], sys.v[e]
        h = p * sys.mult[e] * W[e]
        dg[a] += h
        dg[b] += h
        a == b && (dg[a] += 2h)
    end
    # The rank-one term (`p = 1`): diagonal `κ A SB` and `κ B SA`, off-diagonal `κ A Bᵀ`.
    rank1 = !isempty(sys.lw)
    m = sys.nrow
    κ = rank1 ? exp(sys.lκ) : zero(T)
    # Assigned once, so that the CG product below captures them unboxed.
    SA, SB = rank1 ? _powermean_rankone_sums(Wr, m) : (zero(T), zero(T))
    if rank1
        for q in eachindex(dg, Wr)
            dg[q] += κ * Wr[q] * (q <= m ? SB : SA)
        end
    end
    # A scale-relative ridge keeps the factorizations definite should weights underflow.
    dmax = maximum(dg; init=zero(T))
    ridge = (dmax > 0 ? dmax : one(T)) * eps(T)
    for q in eachindex(dg, pinned)
        dg[q] = pinned[q] ? one(T) : dg[q]
    end
    if ls.kind === :dense
        H = ls.H
        fill!(H, zero(T))
        for e in eachindex(sys.u, sys.v, W)
            a, b = sys.u[e], sys.v[e]
            (a == b || pinned[a] || pinned[b]) && continue
            h = p * sys.mult[e] * W[e]
            H[a, b] += h
            H[b, a] += h
        end
        if rank1
            for j in m+1:N, i in 1:m
                (pinned[i] || pinned[j]) && continue
                h = κ * Wr[i] * Wr[j]
                H[i, j] += h
                H[j, i] += h
            end
        end
        for q in 1:N
            H[q, q] = pinned[q] ? one(T) : dg[q] + ridge
        end
        Fd = T === Float64 ? bunchkaufman!(Symmetric(H)) : lu!(H)
        ldiv!(d, Fd, f)
        return 1, 1
    elseif ls.kind === :cholesky
        M = ls.M
        nzv = nonzeros(M)
        for e in eachindex(sys.u, sys.v, W, ls.epos)
            a, b = sys.u[e], sys.v[e]
            a == b && continue
            nzv[ls.epos[e]] = (pinned[a] || pinned[b]) ? zero(T) : p * sys.mult[e] * W[e]
        end
        for q in 1:N
            nzv[ls.dpos[q]] = pinned[q] ? one(T) : dg[q] + ridge
        end
        factorize!(ls.F, M)
        solve!(d, ls.F, CHOLMOD_A, f)
        return 1, 1
    end
    # Conjugate gradients from `d = 0`; pinned unknowns keep a zero step since
    # their residual starts and stays zero.
    nmul = Ref(0)
    Hmul! = function (y, xx)
        nmul[] += 1
        fill!(y, zero(T))
        for e in eachindex(sys.u, sys.v, W)
            a, b = sys.u[e], sys.v[e]
            h = p * sys.mult[e] * W[e]
            # Not named `z`: the CG work vector `z` bound below would make a
            # shared name a boxed capture.
            δ = h * (xx[a] + xx[b])
            y[a] += δ
            y[b] += δ
        end
        if rank1
            DA = DB = zero(T)
            for q in eachindex(Wr, xx)
                q <= m ? (DA += Wr[q] * xx[q]) : (DB += Wr[q] * xx[q])
            end
            for q in eachindex(y, Wr, xx)
                y[q] += κ * Wr[q] * (q <= m ? SB * xx[q] + DB : SA * xx[q] + DA)
            end
        end
        for q in eachindex(y, pinned)
            pinned[q] && (y[q] = xx[q])
        end
        return y
    end
    fill!(d, zero(T))
    r, z, dd, Ad = ls.work
    η = min(one(T) / 2, sqrt(imb))
    _pcg!(Hmul!, d, dg, f, r, z, dd, Ad, 2N + 100, η * norm(f))
    return nmul[], 0
end

# Damped Newton iteration from `x` on the objective of `sys`, with Armijo
# backtracking. The objective is convex and coercive up to the gauge, so the
# iteration converges from any start. Near the solution the change of the
# objective falls below its roundoff, and a step is then accepted if it reduces
# the imbalance. Stops when the imbalance is at most `tol`, after `maxiter`
# steps, or when no step along the Newton direction is acceptable. Returns
# `(converged, nsteps, npasses, nfactor, imbalance, linsolve)`, counting in
# `npasses` every traversal of the support: evaluations, assemblies, and products.
function _powermean_newton!(x0::Vector{T}, sys::PowerMeanSystem{T}, p::T, maxiter::Integer, tol::T,
                            linsolve::Symbol) where {T}
    x = x0
    N = length(x)
    ne = length(sys.u)
    pinned = _powermean_pinned(sys)
    ls = _powermean_linsolve(sys, linsolve)
    W, Wt = zeros(T, ne), zeros(T, ne)
    nr = isempty(sys.lw) ? 0 : N
    Wr, Wrt = zeros(T, nr), zeros(T, nr)
    g, gt, s, st = zeros(T, N), zeros(T, N), zeros(T, N), zeros(T, N)
    d, xt = zeros(T, N), similar(x)
    _powermean_eval!(W, Wr, g, s, x, d, zero(T), W, Wr, sys, p)
    npasses, nfactor = 1, 0
    imb = _powermean_newton_imbalance(s, sys)
    k = 0
    while imb > tol && k < maxiter
        k += 1
        np, nf = _powermean_direction!(d, ls, sys, W, Wr, g, pinned, imb, p)
        npasses += np
        nfactor += nf
        gd = dot(g, d)
        gd < 0 || break
        # Far from the solution, a nearly flat direction can give a step that
        # changes some weights by `exp(±p δ)` beyond the floating-point range;
        # start the search where every `p|δ|` is at most `POWERMEAN_MAXSTEP`.
        t = min(one(T), T(POWERMEAN_MAXSTEP) / (2p * maximum(abs, d)))
        accepted = false
        for _ in 1:POWERMEAN_NEWTON_NBACKTRACK
            @. xt = x + t * d
            ΔE, scale = _powermean_eval!(Wt, Wrt, gt, st, xt, d, t, W, Wr, sys, p)
            npasses += 1
            imbt = _powermean_newton_imbalance(st, sys)
            if ΔE <= POWERMEAN_ARMIJO * t * gd
                accepted = true
            elseif isfinite(scale) && abs(ΔE) <= POWERMEAN_NOISE * eps(T) * scale
                # The objective cannot resolve this step: judge it by the imbalance.
                accepted = imbt < imb
                accepted || break
            end
            if accepted
                imb = imbt
                break
            end
            t /= 2
        end
        accepted || break
        x, xt = xt, x
        W, Wt = Wt, W
        Wr, Wrt = Wrt, Wr
        g, gt = gt, g
        s, st = st, s
    end
    x === x0 || copyto!(x0, x)
    return imb <= tol, k, npasses, nfactor, imb, ls.kind
end

const POWERMEAN_NEWTON_NBACKTRACK = 40
const POWERMEAN_ARMIJO = 1e-4
const POWERMEAN_NOISE = 64
const POWERMEAN_MAXSTEP = 64

# ============================================================
# Polish of weakly joined clusters
# ============================================================

# After the Newton steps, the log products of entries whose weight `r^p` is small
# relative to the rest of their row and column are poorly determined. A cluster
# of rows and columns joined to the rest of the support only by entries of total
# weight `w` can shift as a whole while changing the row and column sums by only
# about `w` times the shift, so the Newton test fixes its shift only to about
# `max(tol, eps)/w`. The polish computes such shifts from the joining entries.
#
# It works on log scales in difference form: entry `e` joins node `u[e]`, which
# carries `+y`, to node `v[e]`, which carries `-y`, so with `L = log|A|` its log
# ratio is `z = y[u] - y[v] - L` and its log weight is `-p z`. The asymmetric
# problem has a row node per row and a column node per column, `y = [α; -β]`.
# The symmetric problem uses its double cover: a row node and a column node per
# index, `y = [α; -α]`, and one edge per stored orientation of each entry. The
# double cover's objective is the symmetric objective, and `α = (y_row - y_col)/2`.
#
# Clusters are the connected components of the entries at or above thresholds
# spaced `POWERMEAN_SWEEP_SPACING` apart in log weight, found by union-find over
# the entries in decreasing order of weight (Kruskal order). The unknowns of a
# level are the clusters of the next finer level (the nodes at the finest), and
# its entries are those crossing them. Over the shift `c` of one unknown, the
# objective is `(A exp(-p c) + B exp(p c) + p k c)/p` with `A` and `B` the weights
# of its out and in entries and `k` their difference in number (with targets, the
# sum of its members' row targets minus their column targets), so its exact
# minimizer is a root of a quadratic in `exp(p c)`, computed from the log weights
# at any scale. A pass sweeps these shifts Gauss-Seidel over each level, finest
# first. Level-by-level sweeps converge slowly along directions that couple two
# levels, i.e., those joined by entries near a threshold, so successive passes
# alternate between two hierarchies whose thresholds are offset by half a level.

const POWERMEAN_SWEEP_SPACING = log(10)   # `Δ`: threshold ratio `exp(Δ)` between levels
const POWERMEAN_SHIFT_SWEEPS = 20         # Gauss-Seidel sweeps of one level per pass, at most
const POWERMEAN_POLISH_ROUNDS = 4         # polishes, each followed by Newton steps, at most
const POWERMEAN_SETTLE_NOISE = 4          # roundoff of a shift, in `eps` times the largest `|y|` or `|L|`

struct PowerMeanGraph{T}
    N::Int                  # nodes
    u::Vector{Int}          # node carrying `+y` of each entry
    v::Vector{Int}          # node carrying `-y`
    L::Vector{T}            # `log|A|` of each entry
    k0::Vector{T}           # each node's `k` (see below); empty when it is out minus in entries
end

PowerMeanGraph{T}(N, u, v, L) where {T} = PowerMeanGraph{T}(N, u, v, L, T[])

# Asymmetric graph of the Newton system, whose unknowns are the row nodes `1:m`
# and the column nodes `m+1:m+n`, each entry joining its row and column node.
# With targets `t`, the linear term of a shift is `t` for a row node and `-t` for a
# column node.
function _powermean_graph(sys::PowerMeanSystem{T}) where {T}
    m = sys.nrow
    k0 = T[q <= m ? t : -t for (q, t) in pairs(sys.target)]
    return PowerMeanGraph{T}(length(sys.count), sys.u, sys.v, sys.L, k0)
end

# Double cover of the symmetric Newton system: row node `i`, column node `n+i`,
# both orientations of an off-diagonal pair (stored once, with multiplicity two),
# and a diagonal entry once.
function _powermean_symgraph(sys::PowerMeanSystem{T}) where {T}
    n = length(sys.count)
    ne = 0
    for e in eachindex(sys.u, sys.v)
        ne += sys.u[e] == sys.v[e] ? 1 : 2
    end
    u = Vector{Int}(undef, ne)
    v = Vector{Int}(undef, ne)
    L = Vector{T}(undef, ne)
    k = 0
    for e in eachindex(sys.u, sys.v, sys.L)
        a, b, l = sys.u[e], sys.v[e], sys.L[e]
        k += 1
        u[k], v[k], L[k] = a, n + b, l
        if a != b
            k += 1
            u[k], v[k], L[k] = b, n + a, l
        end
    end
    return PowerMeanGraph{T}(2n, u, v, L)
end

# Conversion between the Newton unknowns `x` and the difference form `y`: for
# the asymmetric problem with `m` rows, the column scales change sign (its own
# inverse); for the symmetric problem (`m === nothing`), `y = [x; -x]` and
# `x = (y_row - y_col)/2`.
function _powermean_toy!(y, x, m::Int)
    for q in eachindex(y, x)
        y[q] = q <= m ? x[q] : -x[q]
    end
    return y
end
_powermean_tox!(x, y, m::Int) = _powermean_toy!(x, y, m)

function _powermean_toy!(y, x, ::Nothing)
    n = length(x)
    for q in 1:n
        y[q] = x[q]
        y[n+q] = -x[q]
    end
    return y
end

function _powermean_tox!(x, y, ::Nothing)
    n = length(x)
    for q in 1:n
        x[q] = (y[q] - y[n+q]) / 2
    end
    return x
end

_powermean_polish_graph(sys, ::Int) = _powermean_graph(sys)
_powermean_polish_graph(sys, ::Nothing) = _powermean_symgraph(sys)

# Threshold below which a cluster shift counts as settled: `tol/p`, but no less
# than the roundoff of a shift computed from the log weights
# `-p (y[u] - y[v] - L)`, a few `eps` times the largest `|y|` or `|L|`.
function _powermean_settle(y::AbstractVector{T}, G::PowerMeanGraph, p, tol) where {T}
    scale = max(maximum(abs, y; init=zero(T)), maximum(abs, G.L; init=zero(T)))
    return max(tol / p, T(POWERMEAN_SETTLE_NOISE) * eps(T) * scale)
end

# Log weights `-p z` of every entry.
function _powermean_logweights!(lw, y, G::PowerMeanGraph, p)
    for e in eachindex(lw, G.u, G.v, G.L)
        lw[e] = -p * (y[G.u[e]] - y[G.v[e]] - G.L[e])
    end
    return lw
end

# One level of a hierarchy, stored in a `PowerMeanWorkspace`: its unknowns at
# positions `uoff+1:uoff+K` of the unknown arrays and its crossing entries at
# positions `eoff+1:eoff+ne` of the entry arrays.
struct PowerMeanLevel
    K::Int
    uoff::Int
    ne::Int
    eoff::Int
end

# Storage for the passes of one polish, allocated once and reused by every pass.
# The levels of a hierarchy are stored back to back.
struct PowerMeanWorkspace{T}
    lw::Vector{T}           # log weights of the graph's entries
    order::Vector{Int}      # entries by decreasing log weight
    scratch::Vector{Int}    # scratch space of the sort
    par::Vector{Int}        # union-find forest over a level's unknowns
    lab::Vector{Int}        # labels of the clusters a level forms
    levels::Vector{PowerMeanLevel}
    eid::Vector{Int}        # crossing entries: graph entry,
    u::Vector{Int}          # unknown carrying `+y`,
    v::Vector{Int}          # unknown carrying `-y`
    k::Vector{T}            # unknowns: linear coefficient of the shift (out minus in entries without targets),
    up::Vector{Int}         # unknown of the next level containing it (0 if none does),
    c::Vector{T}            # shift
    lwl::Vector{T}          # log weights of one level's entries
    optr::Vector{Int}       # out entries of unknown `P` of one level: `olist[optr[P]:optr[P+1]-1]`
    iptr::Vector{Int}       # in entries, likewise
    olist::Vector{Int}
    ilist::Vector{Int}
    cursor::Vector{Int}
end

function PowerMeanWorkspace(G::PowerMeanGraph{T}) where {T}
    N, ne = G.N, length(G.L)
    vi() = Int[]
    return PowerMeanWorkspace{T}(Vector{T}(undef, ne), Vector{Int}(undef, ne), Vector{Int}(undef, ne),
                                 Vector{Int}(undef, N), Vector{Int}(undef, N), PowerMeanLevel[],
                                 vi(), vi(), vi(), T[], vi(), T[], Vector{T}(undef, ne),
                                 Vector{Int}(undef, N + 1), Vector{Int}(undef, N + 1),
                                 Vector{Int}(undef, ne), Vector{Int}(undef, ne), Vector{Int}(undef, N + 1))
end

# Root of `q` in the union-find forest `parent`, halving the path on the way.
function _uf_root!(parent, q)
    while parent[q] != q
        parent[q] = parent[parent[q]]
        q = parent[q]
    end
    return q
end

# Levels of the hierarchy with thresholds `-(ℓ - offset) Δ`, `ℓ = 1, 2, …`, on the
# log weights `ws.lw`, from the finest; level `ℓ` joins the entries at or above
# its threshold. A threshold that joins no entry would repeat the next level and
# is skipped. The entries are sorted once; each level then takes time
# proportional to its unknowns and crossing entries, and only crossing entries
# pass on to the next level. An unknown none of whose entries still crosses
# after the joins drops out (`up` is 0).
function _powermean_hierarchy!(ws::PowerMeanWorkspace{T}, G::PowerMeanGraph, Δ, offset) where {T}
    lw, order, par, lab = ws.lw, ws.order, ws.par, ws.lab
    ne = length(lw)
    empty!(ws.levels)
    sortperm!(order, lw; rev=true, scratch=ws.scratch)
    resize!(ws.eid, ne); resize!(ws.u, ne); resize!(ws.v, ne)
    for q in 1:ne
        e = order[q]
        ws.eid[q], ws.u[q], ws.v[q] = e, G.u[e], G.v[e]
    end
    K, nk, eoff, uoff = G.N, ne, 0, 0
    ℓ = 0
    while nk > 0
        # The first threshold at or below the heaviest crossing entry.
        lw1 = lw[ws.eid[eoff+1]]
        ℓ = max(ℓ + 1, floor(Int, offset - lw1 / Δ))
        while lw1 < T(-(ℓ - offset) * Δ)
            ℓ += 1
        end
        thresh = T(-(ℓ - offset) * Δ)
        entries = eoff+1:eoff+nk
        resize!(ws.k, uoff + K); resize!(ws.up, uoff + K)
        for P in 1:K
            ws.k[uoff+P] = 0
            par[P] = P
            lab[P] = 0
        end
        if isempty(G.k0)
            for q in entries
                ws.k[uoff+ws.u[q]] += 1
                ws.k[uoff+ws.v[q]] -= 1
            end
        elseif uoff == 0
            copyto!(ws.k, 1, G.k0, 1, K)
        else
            # A cluster's coefficient is the sum of its members'.
            prev = ws.levels[end]
            for P in 1:prev.K
                up = ws.up[prev.uoff+P]
                up == 0 || (ws.k[uoff+up] += ws.k[prev.uoff+P])
            end
        end
        q = eoff + 1
        while q <= eoff + nk && lw[ws.eid[q]] >= thresh
            a, b = _uf_root!(par, ws.u[q]), _uf_root!(par, ws.v[q])
            a == b || (par[a] = b)
            q += 1
        end
        # Label the clusters that some entry still crosses (marked -1 first), in
        # the order of their unknowns.
        nkeep = 0
        for q in entries
            a, b = _uf_root!(par, ws.u[q]), _uf_root!(par, ws.v[q])
            if a != b
                lab[a] = lab[b] = -1
                nkeep += 1
            end
        end
        Knext = 0
        for P in 1:K
            r = _uf_root!(par, P)
            lab[r] == -1 && (lab[r] = Knext += 1)
            ws.up[uoff+P] = max(lab[r], 0)
        end
        push!(ws.levels, PowerMeanLevel(K, uoff, nk, eoff))
        # The next level's entries follow this level's.
        j = eoff + nk
        resize!(ws.eid, j + nkeep); resize!(ws.u, j + nkeep); resize!(ws.v, j + nkeep)
        for q in entries
            a, b = ws.up[uoff+ws.u[q]], ws.up[uoff+ws.v[q]]
            if a != b
                j += 1
                ws.eid[j], ws.u[j], ws.v[j] = ws.eid[q], a, b
            end
        end
        eoff, nk, uoff, K = eoff + nk, nkeep, uoff + K, Knext
    end
    resize!(ws.c, uoff)
    return ws.levels
end

# Log of the sum of `exp` over the log weights at the positions `idx`.
function _powermean_logsumexp(lw, idx)
    T = eltype(lw)
    m = typemin(T)
    for q in idx
        m = max(m, lw[q])
    end
    m == typemin(T) && return m
    t = zero(T)
    for q in idx
        t += exp(lw[q] - m)
    end
    return m + log(t)
end

# `p c` at the minimizer of `A exp(-p c) + B exp(p c) + p k c` over the shift `c` of
# one unknown, from `la = log A`, `lb = log B` (its out and in cut weights) and
# `k`: the root `exp(p c)` of `B u² + k u - A = 0`, in logs for any scale.
function _powermean_shift(la::T, lb::T, k::Real) where {T}
    k == 0 && return (la - lb) / 2
    lk = log(T(abs(k)))
    lq = la + lb + log(T(4)) - 2lk
    # For large `lq`, `exp(lq)` overflows; `log1p(sqrt(1 + exp(lq))) → lq/2`.
    t = lq > 700 ? log(T(2)) - lq / 2 : log(T(2)) - log1p(sqrt(1 + exp(lq)))
    return k > 0 ? la - lk + t : -(lb - lk + t)
end

# Gauss-Seidel sweeps of exact shifts over the unknowns of level `lev`, with
# their shifts in `ws.c` and the log weights of the level's entries in `ws.lwl`
# updated in place. Stops when a sweep's largest shift is below `settle`, or
# after `POWERMEAN_SHIFT_SWEEPS` sweeps. Returns the largest accumulated shift
# and the number of sweeps.
function _powermean_shift_sweeps!(ws::PowerMeanWorkspace{T}, lev::PowerMeanLevel, p, settle) where {T}
    K, uoff, ne, eoff = lev.K, lev.uoff, lev.ne, lev.eoff
    optr, iptr, olist, ilist, cursor, lwl, c = ws.optr, ws.iptr, ws.olist, ws.ilist, ws.cursor, ws.lwl, ws.c
    fill!(view(optr, 1:K+1), 0)
    fill!(view(iptr, 1:K+1), 0)
    for e in 1:ne
        optr[ws.u[eoff+e]+1] += 1
        iptr[ws.v[eoff+e]+1] += 1
    end
    optr[1] = 1; iptr[1] = 1
    for P in 1:K
        optr[P+1] += optr[P]
        iptr[P+1] += iptr[P]
    end
    copyto!(cursor, 1, optr, 1, K)
    for e in 1:ne
        a = ws.u[eoff+e]
        olist[cursor[a]] = e
        cursor[a] += 1
    end
    copyto!(cursor, 1, iptr, 1, K)
    for e in 1:ne
        b = ws.v[eoff+e]
        ilist[cursor[b]] = e
        cursor[b] += 1
    end
    nsweeps = 0
    for _ in 1:POWERMEAN_SHIFT_SWEEPS
        nsweeps += 1
        maxc = zero(T)
        for P in 1:K
            os = view(olist, optr[P]:optr[P+1]-1)
            is = view(ilist, iptr[P]:iptr[P+1]-1)
            (isempty(os) && isempty(is)) && continue
            pc = _powermean_shift(_powermean_logsumexp(lwl, os), _powermean_logsumexp(lwl, is), ws.k[uoff+P])
            isfinite(pc) || throw(DomainError(pc, "non-finite cluster shift; the log weights of the support are out of range"))
            cP = pc / p
            c[uoff+P] += cP
            for e in os
                lwl[e] -= pc
            end
            for e in is
                lwl[e] += pc
            end
            maxc = max(maxc, abs(cP))
        end
        maxc < settle && break
    end
    cmax = zero(T)
    for P in 1:K
        cmax = max(cmax, abs(c[uoff+P]))
    end
    return cmax, nsweeps
end

# One pass of exact shifts over every level of the hierarchy with the given
# offset, from the finest. `ws.lw` holds the log weights at `y` and is updated
# with it. Returns the largest cluster shift and the work, in entries traversed
# by the hierarchy build and the sweeps.
function _powermean_shift_pass!(y::AbstractVector{T}, ws::PowerMeanWorkspace{T}, G::PowerMeanGraph, p, offset, settle) where {T}
    levels = _powermean_hierarchy!(ws, G, POWERMEAN_SWEEP_SPACING, offset)
    lw, lwl, c = ws.lw, ws.lwl, ws.c
    fill!(c, zero(T))
    maxshift = zero(T)
    work = length(lw)
    for lev in levels
        for e in 1:lev.ne
            lwl[e] = lw[ws.eid[lev.eoff+e]]
        end
        cmax, nsweeps = _powermean_shift_sweeps!(ws, lev, p, settle)
        maxshift = max(maxshift, cmax)
        work += nsweeps * lev.ne
        for e in 1:lev.ne
            lw[ws.eid[lev.eoff+e]] = lwl[e]
        end
    end
    # Each unknown also moves with the clusters that contain it at coarser levels.
    for ℓ in length(levels)-1:-1:1
        lev, next = levels[ℓ], levels[ℓ+1]
        for P in 1:lev.K
            up = ws.up[lev.uoff+P]
            up == 0 || (c[lev.uoff+P] += c[next.uoff+up])
        end
    end
    if !isempty(levels)     # the finest level's unknowns are the nodes
        for i in 1:G.N
            y[i] += c[i]
        end
    end
    return maxshift, work
end

# Shift passes on `y`, alternating the hierarchies offset by 0 and 1/2 level, until
# a pass's largest shift is below `settle` or after `maxpasses` passes. Returns
# the number of passes, the largest shifts of the first and the last pass, and
# the work in entries traversed.
function _powermean_polish!(y::AbstractVector{T}, G::PowerMeanGraph{T}, p, settle, maxpasses::Integer) where {T}
    ws = PowerMeanWorkspace(G)
    npasses = work = 0
    shift1 = shift = zero(T)
    while npasses < maxpasses
        # Each pass updates the log weights incrementally; starting it from the
        # scales keeps that roundoff from accumulating.
        _powermean_logweights!(ws.lw, y, G, p)
        shift, w = _powermean_shift_pass!(y, ws, G, p, isodd(npasses) ? 0.5 : 0.0, settle)
        work += w
        npasses += 1
        npasses == 1 && (shift1 = shift)
        shift < settle && break
    end
    return npasses, shift1, shift, work
end

# ============================================================
# Refinement: Newton steps and polish
# ============================================================

# Largest imbalance of the Newton test at `x`, by one pass over the support.
function _powermean_system_imbalance(x::Vector{T}, sys::PowerMeanSystem{T}, p::T) where {T}
    N, ne = length(x), length(sys.u)
    W, s = zeros(T, ne), zeros(T, N)
    Wr = zeros(T, isempty(sys.lw) ? 0 : N)
    _powermean_eval!(W, Wr, zeros(T, N), s, x, zeros(T, N), zero(T), W, Wr, sys, p)
    return _powermean_newton_imbalance(s, sys)
end

# Newton steps from `x` on the objective of `sys`; once they converge, polish of
# the weakly joined clusters on the difference form of `x` (`m` rows for the
# asymmetric problem, `nothing` for the symmetric one; see `_powermean_toy!`).
# The returned `imbalance` and `shift` are both measured on the returned `x`.
#
# Each round polishes `x`, starting from an iterate that passes the imbalance
# test, and evaluates the imbalance. If the polish kept it within `tol`, the
# polished `x` is returned. Otherwise Newton steps repair it and the next round
# polishes again, for at most `POWERMEAN_POLISH_ROUNDS` repairs. When the rounds,
# the Newton steps (`maxnewton`), or the polish passes (`maxpolish`, 0 skips the
# polish) run out first, the last iterate that passed the test is returned, with
# the first-pass shift of the discarded polish as its residual along the weak
# joins. Newton steps after the first call use the linear solver it chose.
#
# Returns `(converged, nnewton, npasses, nfactor, imbalance, linsolve, npolish,
# shift, settle)`: `npasses` counts the polish's work in passes over the
# support, `npolish` the polish passes, `shift` the largest cluster shift of
# the last pass (zero when no polish ran), and `settle` the shift threshold.
function _powermean_refine!(x::Vector{T}, sys::PowerMeanSystem{T}, m::Union{Int,Nothing}, p::T,
                            maxnewton::Integer, maxpolish::Integer, tol::T, linsolve::Symbol) where {T}
    converged, nnewton, npasses, nfactor, imbalance, ls = _powermean_newton!(x, sys, p, maxnewton, tol, linsolve)
    npolish = 0
    shift = zero(T)
    settle = tol / p
    # The rank-one term joins every row to every column, so no cluster is joined
    # only by small entries and the polish does not apply.
    (converged && maxpolish > 0 && isempty(sys.lw)) ||
        return converged, nnewton, npasses, nfactor, imbalance, ls, npolish, shift, settle
    G = _powermean_polish_graph(sys, m)
    y = Vector{T}(undef, G.N)
    settle = _powermean_settle(_powermean_toy!(y, x, m), G, p, tol)
    xsave, imbsave = copy(x), imbalance     # the last iterate that passed the test
    work = 0
    for r in 1:POWERMEAN_POLISH_ROUNDS+1
        np, shift1, shift, w = _powermean_polish!(_powermean_toy!(y, x, m), G, p, settle, maxpolish - npolish)
        npolish += np
        work += w
        _powermean_tox!(x, y, m)
        imbalance = _powermean_system_imbalance(x, sys, p)
        npasses += 1
        imbalance <= tol && break
        if r > POWERMEAN_POLISH_ROUNDS || nnewton >= maxnewton || npolish >= maxpolish
            copyto!(x, xsave)
            imbalance, shift = imbsave, shift1
            break
        end
        converged, ns, np, nf, imbalance, ls = _powermean_newton!(x, sys, p, maxnewton - nnewton, tol, ls)
        nnewton += ns
        npasses += np
        nfactor += nf
        converged || break
        copyto!(xsave, x)
        imbsave = imbalance
    end
    npasses += work ÷ max(length(G.L), 1)
    return converged, nnewton, npasses, nfactor, imbalance, ls, npolish, shift, settle
end
